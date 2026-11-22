"""GPU (CUDA / Apple MPS / CPU) renderer for layered depth images."""
from __future__ import annotations

from typing import Iterator, List

import numpy as np
import torch
import torch.nn.functional as F

from .camera import Pose, get_pose
from .layers import Layer


def disp_to_z(d, z_far: float):
    """Disparity [0,1] -> depth [1, z_far] (1 = nearest)."""
    return 1.0 / (d * (1.0 - 1.0 / z_far) + 1.0 / z_far)


def default_focus(disp: np.ndarray) -> float:
    """Focus on the subject: the near-ish stuff in the central part of the frame.

    Keeping the subject still and swinging the background around it reads as a deliberate
    camera move (and hides edge artefacts on the thing people actually look at).
    """
    h, w = disp.shape
    c = disp[h // 5: 4 * h // 5, w // 5: 4 * w // 5]
    return float(np.clip(np.percentile(c, 85), 0.2, 0.9))


def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.3:
        return img
    k = _gauss_kernel(sigma, img.device)
    c = img.shape[1]
    r = k.numel() // 2
    x = F.pad(img, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


class ParallaxRenderer:
    def __init__(self, layers: List[Layer], device: torch.device, z_far: float = 8.0,
                 focus: float = 0.5, dof: float = 0.0):
        self.device = device
        self.z_far = z_far
        self.focus_d = focus
        self.focus_z = float(disp_to_z(focus, z_far))
        self.dof = dof
        h, w = layers[0].alpha.shape
        self.h, self.w = h, w
        self.f0 = float(w)  # ~53 degree horizontal FOV
        # premultiplied RGBA + depth per layer, uploaded once
        self.tex, self.zs = [], []
        for L in layers:
            rgba = np.concatenate([L.rgb * L.alpha[..., None], L.alpha[..., None]], axis=2)
            self.tex.append(torch.from_numpy(rgba.transpose(2, 0, 1).copy())[None].to(device))
            z = disp_to_z(L.disp, z_far).astype(np.float32)
            self.zs.append(torch.from_numpy(z)[None, None].to(device))
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32) + 0.5,
                                torch.arange(w, device=device, dtype=torch.float32) + 0.5, indexing="ij")
        self.ut, self.vt = xs, ys
        self.overscan = 1.0

    # ---- projection ---------------------------------------------------------------
    def _inverse(self, ut, vt, Z, p: Pose, s: float):
        """Target pixel + source depth -> source pixel (exact inverse of the forward projection)."""
        cx, cy, f0, zf = self.w / 2, self.h / 2, self.f0, self.focus_z
        comp_x = s * (-f0 * p.tx) / (zf - p.tz)  # keeps the focus plane centred (camera "looks at" it)
        comp_y = s * (-f0 * p.ty) / (zf - p.tz)
        us = ((ut - cx + comp_x) * (Z - p.tz) / s + f0 * p.tx) / Z + cx
        vs = ((vt - cy + comp_y) * (Z - p.tz) / s + f0 * p.ty) / Z + cy
        return us, vs

    def fit_overscan(self, motion: str, amplitude: float, border_disp: np.ndarray, steps: int = 90):
        """Smallest zoom that keeps the frame border inside the photo for the whole path."""
        Z = disp_to_z(border_disp, self.z_far)
        need = 1.0
        hw, hh = self.w / 2, self.h / 2
        for t in np.linspace(0, 1, steps):
            p = get_pose(motion, float(t), amplitude, self.focus_z)
            for Zc in (Z.min(), np.median(Z), Z.max()):
                for half, tr in ((hw, p.tx), (hh, p.ty)):
                    o = self.f0 * tr * (1.0 / Zc - (Zc - p.tz) / ((self.focus_z - p.tz) * Zc))
                    room = max(half - abs(o), 0.25 * half)
                    need = max(need, half * (Zc - p.tz) / (p.zoom * Zc) / room)
        self.overscan = float(min(need * 1.01, 1.6))
        return self.overscan

    # ---- rendering ----------------------------------------------------------------
    def _grid(self, us, vs):
        # clamp == "border" padding (not implemented for grid_sample on Apple MPS)
        us = us.clamp(0.5, self.w - 0.5)
        vs = vs.clamp(0.5, self.h - 0.5)
        return torch.stack([us / self.w * 2 - 1, vs / self.h * 2 - 1], dim=-1)[None]

    @torch.no_grad()
    def render(self, p: Pose) -> torch.Tensor:
        s = p.zoom * self.overscan
        out = torch.zeros(1, 3, self.h, self.w, device=self.device)
        out_d = torch.zeros(1, 1, self.h, self.w, device=self.device)
        for tex, zmap in zip(self.tex, self.zs):
            Z = zmap[0, 0]
            for it in range(5):  # damped fixed-point: which source depth lands on this target pixel?
                us, vs = self._inverse(self.ut, self.vt, Z, p, s)
                Zn = F.grid_sample(zmap, self._grid(us, vs), mode="bilinear",
                                   padding_mode="zeros", align_corners=False)[0, 0]
                Z = Zn if it == 0 else 0.5 * (Z + Zn)
            us, vs = self._inverse(self.ut, self.vt, Z, p, s)
            g = self._grid(us, vs)
            rgba = F.grid_sample(tex, g, mode="bilinear", padding_mode="zeros", align_corners=False)
            a = rgba[:, 3:4]
            out = rgba[:, :3] + out * (1 - a)  # premultiplied "over", back to front
            d = (1.0 / Z - 1.0 / self.z_far) / (1.0 - 1.0 / self.z_far)
            out_d = d[None, None] * a + out_d * (1 - a)
        if self.dof > 0:
            out = self._depth_of_field(out, out_d)
        return out.clamp(0, 1)

    def _depth_of_field(self, img, disp):
        """Cheap layered bokeh: blend a small blur pyramid by circle-of-confusion."""
        scale = max(self.h, self.w) / 1000.0
        sig = [0.0, 1.5 * scale, 3.5 * scale, 7.0 * scale]
        coc = ((disp - self.focus_d).abs() * self.dof * 3.0).clamp(0, len(sig) - 1)
        stack = [img] + [_blur(img, s_) for s_ in sig[1:]]
        out = torch.zeros_like(img)
        for i, b in enumerate(stack):
            w = (1 - (coc - i).abs()).clamp(min=0)
            out = out + b * w
        return out

    def frames(self, motion: str, n_frames: int, amplitude: float, loop: bool) -> Iterator[np.ndarray]:
        denom = n_frames if loop else max(n_frames - 1, 1)
        for i in range(n_frames):
            p = get_pose(motion, i / denom, amplitude, self.focus_z)
            img = self.render(p)[0].permute(1, 2, 0)
            yield (img * 255 + 0.5).byte().cpu().numpy()
