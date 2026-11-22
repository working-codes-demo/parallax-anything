"""Tiny ffmpeg wrappers (uses a system ffmpeg if present, else the one bundled with imageio-ffmpeg)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found: install ffmpeg or `pip install imageio-ffmpeg`") from e


def write_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: int, crf: int = 17) -> Path:
    path = Path(path)
    it = iter(frames)
    first = next(it)
    h, w = first.shape[:2]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        proc.stdin.write(first.tobytes())
        for f in it:
            proc.stdin.write(f.tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {path}")
    return path


def mp4_to_gif(src: str | Path, dst: str | Path, width: int = 640, fps: int = 20) -> Path:
    vf = (f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors=200:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a")
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-loop", "0", str(dst)],
                   check=True)
    return Path(dst)
