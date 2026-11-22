"""High-level API: photo in -> 3D parallax video out."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .camera import LOOPING
from .depth import DepthEstimator, colorize_depth
from .layers import build_layers, refine_disparity
from .render import ParallaxRenderer, default_focus
from .video import mp4_to_gif, write_mp4


@dataclass
class ParallaxConfig:
    motion: str = "orbit"
    duration: float = 4.0
    fps: int = 30
    strength: float = 1.0       # camera travel multiplier
    layers: int = 4             # depth slices
    z_far: float = 8.0          # how deep the scene feels (bigger = flatter background)
    focus: Optional[float] = None  # disparity in [0,1] that stays still; None = auto
    dof: float = 0.0            # depth-of-field strength (0 = off, 1 = cinematic)
    max_size: int = 1920        # long side of the output video
    gif: bool = False
    save_depth: bool = True


def _resize_long(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, max_size / max(h, w))
    nw, nh = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2  # even dims for yuv420p
    if (nw, nh) == (w, h):
        return img
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


class ParallaxPipeline:
    def __init__(self, model: str = "vits", device: str = "auto"):
        self.depth = DepthEstimator(model=model, device=device)

    def run(self, image_path: str | Path, out_dir: str | Path, cfg: ParallaxConfig = ParallaxConfig(),
            name: Optional[str] = None) -> dict:
        t0 = time.time()
        image_path, out_dir = Path(image_path), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = name or f"{image_path.stem}_{cfg.motion}"

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_path)
        img = _resize_long(img, cfg.max_size)
        h, w = img.shape[:2]

        disp = self.depth(img)
        t_depth = time.time() - t0
        focus = cfg.focus if cfg.focus is not None else default_focus(disp)

        # camera travel is expressed relative to the nearest object; 0.04 ~ a gentle handheld move
        amplitude = 0.04 * cfg.strength
        n_frames = max(2, int(round(cfg.duration * cfg.fps)))
        loop = cfg.motion in LOOPING

        band = self._max_shift_px(w, amplitude, cfg.z_far, focus) + 8
        layers = build_layers(img, disp, n_layers=cfg.layers, band_px=band)
        renderer = ParallaxRenderer(layers, self.depth.device, z_far=cfg.z_far, focus=focus, dof=cfg.dof)
        ref = refine_disparity(img, disp)
        border = np.concatenate([ref[0, ::4], ref[-1, ::4], ref[::4, 0], ref[::4, -1]])
        renderer.fit_overscan(cfg.motion, amplitude, border)

        mp4 = write_mp4(renderer.frames(cfg.motion, n_frames, amplitude, loop), out_dir / f"{stem}.mp4", cfg.fps)
        result = {"input": str(image_path), "motion": cfg.motion, "dof": cfg.dof, "video": str(mp4), "frames": n_frames, "size": f"{w}x{h}", "focus": round(focus, 3),
                  "layers": len(layers), "overscan": round(renderer.overscan, 3)}
        if cfg.save_depth:
            dpath = out_dir / f"{image_path.stem}_depth.png"
            cv2.imwrite(str(dpath), colorize_depth(disp))
            result["depth"] = str(dpath)
        if cfg.gif:
            result["gif"] = str(mp4_to_gif(mp4, out_dir / f"{stem}.gif"))
        result["seconds"] = round(time.time() - t0, 2)
        result["depth_seconds"] = round(t_depth, 2)
        return result

    @staticmethod
    def _max_shift_px(w: int, amplitude: float, z_far: float, focus_d: float) -> int:
        """Rough upper bound of how far the nearest layer slides relative to the farthest one."""
        # lateral moves: f0 * t * (1/Z_near - 1/Z_far); forward moves add a zoom-like spread
        lateral = w * 1.2 * amplitude * (1.0 - 1.0 / z_far)
        forward = w * 0.5 * (4.0 * amplitude) * (1.0 - 1.0 / z_far)
        return int(max(lateral, forward) * 1.15)
