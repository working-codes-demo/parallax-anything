#!/usr/bin/env python3
"""Inference script: photo(s) -> 3D parallax video(s).

Examples
--------
  python infer.py --input assets/examples/street.jpg
  python infer.py --input assets/examples --motion all --gif
  python infer.py --input me.jpg --motion vertigo --dof 1 --duration 6 --strength 1.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parallax_anything import MOTIONS, ParallaxConfig, ParallaxPipeline

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(inp: Path):
    if inp.is_dir():
        return sorted(p for p in inp.iterdir() if p.suffix.lower() in IMG_EXT)
    if inp.is_file():
        return [inp]
    sys.exit(f"input not found: {inp}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, help="image file or folder of images")
    ap.add_argument("--output", "-o", default="outputs", help="output folder")
    ap.add_argument("--motion", "-m", default="orbit", help=f"one of {sorted(MOTIONS)}, a comma list, or 'all'")
    ap.add_argument("--duration", type=float, default=4.0, help="seconds")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--strength", type=float, default=1.0, help="camera travel (0.5 subtle ... 2 dramatic)")
    ap.add_argument("--layers", type=int, default=4, help="number of depth layers")
    ap.add_argument("--z-far", type=float, default=8.0, help="scene depth range; larger = flatter background")
    ap.add_argument("--focus", type=float, default=None, help="disparity 0..1 that stays fixed (default: auto)")
    ap.add_argument("--dof", type=float, default=0.0, help="depth-of-field blur strength (0 = off)")
    ap.add_argument("--max-size", type=int, default=1920, help="long side of output, px")
    ap.add_argument("--gif", action="store_true", help="also export a GIF")
    ap.add_argument("--model", default="vits", choices=["vits", "vitb"],
                    help="depth model (vits = Apache-2.0; vitb = CC-BY-NC, non-commercial only)")
    ap.add_argument("--manifest", default=None, help="append one JSON line per result to this file")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    a = ap.parse_args(argv)

    motions = sorted(MOTIONS) if a.motion == "all" else [m.strip() for m in a.motion.split(",")]
    bad = [m for m in motions if m not in MOTIONS]
    if bad:
        sys.exit(f"unknown motion(s) {bad}; choose from {sorted(MOTIONS)}")

    pipe = ParallaxPipeline(model=a.model, device=a.device)
    print(f"[parallax] device: {pipe.depth.device}")
    for img in collect(Path(a.input)):
        for m in motions:
            cfg = ParallaxConfig(motion=m, duration=a.duration, fps=a.fps, strength=a.strength, layers=a.layers,
                                 z_far=a.z_far, focus=a.focus, dof=a.dof, max_size=a.max_size, gif=a.gif)
            res = pipe.run(img, a.output, cfg)
            print(json.dumps(res))
            if a.manifest:
                with open(a.manifest, "a") as f:
                    f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
