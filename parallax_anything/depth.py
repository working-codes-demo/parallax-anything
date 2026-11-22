"""Monocular depth estimation with Depth Anything V2 (Small, Apache-2.0)."""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger("dinov2").setLevel(logging.ERROR)  # hush "xFormers not available"
from .third_party.depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

CHECKPOINTS = {
    # Only the Small model is Apache-2.0. Base/Large are CC-BY-NC-4.0 (non-commercial).
    "vits": {
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "file": "depth_anything_v2_vits.pth",
        "sha256": "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378",
        "config": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    },
    "vitb": {
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
        "file": "depth_anything_v2_vitb.pth",
        "sha256": None,
        "config": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    },
}

DEFAULT_CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_checkpoint(model: str = "vits", ckpt_dir: Path | str = DEFAULT_CKPT_DIR) -> Path:
    """Return the local checkpoint path, downloading (and verifying) it if missing."""
    spec = CHECKPOINTS[model]
    ckpt_dir = Path(ckpt_dir)
    path = ckpt_dir / spec["file"]
    if path.exists():
        return path
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"[parallax] downloading {spec['file']} ...")
    urllib.request.urlretrieve(spec["url"], tmp)
    if spec["sha256"] and _sha256(tmp) != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {spec['file']}")
    os.replace(tmp, path)
    return path


class DepthEstimator:
    """Wraps Depth Anything V2 and returns a normalised disparity map (1 = nearest)."""

    def __init__(self, model: str = "vits", device: str = "auto", ckpt_dir: Path | str = DEFAULT_CKPT_DIR):
        self.device = pick_device(device)
        spec = CHECKPOINTS[model]
        self.net = DepthAnythingV2(**spec["config"])
        state = torch.load(ensure_checkpoint(model, ckpt_dir), map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        self.net.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray, input_size: int = 518, flip_tta: bool = True) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        # keep aspect ratio, short side >= input_size, both sides multiple of 14 (ViT patch size)
        scale = input_size / min(h, w)
        nh = max(14, int(round(h * scale / 14)) * 14)
        nw = max(14, int(round(w * scale / 14)) * 14)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        disp = self.net(x)
        if flip_tta:  # horizontal-flip test-time augmentation: cleaner, more symmetric edges
            disp = 0.5 * (disp + self.net(torch.flip(x, dims=[3])).flip(dims=[2]))
        disp = F.interpolate(disp[:, None], (h, w), mode="bilinear", align_corners=False)[0, 0]
        disp = disp.float().cpu().numpy()

        lo, hi = np.percentile(disp, [0.5, 99.5])
        return np.clip((disp - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def colorize_depth(disp: np.ndarray, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Turn a [0,1] disparity map into a BGR visualisation."""
    return cv2.applyColorMap((disp * 255).astype(np.uint8), cmap)
