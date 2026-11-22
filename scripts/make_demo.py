#!/usr/bin/env python3
"""Assemble the showcase video + README hero GIF from rendered outputs.

  python scripts/make_demo.py --gallery outputs/gallery --out demo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parallax_anything.video import mp4_to_gif, write_mp4  # noqa: E402

W, H, FPS = 1920, 1080, 30
ACCENT = (255, 196, 64)
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def font(size):
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size=size)


def read_frames(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def fit(img, w, h, cover=False):
    ih, iw = img.shape[:2]
    s = (max if cover else min)(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s + 0.5), int(ih * s + 0.5)), interpolation=cv2.INTER_AREA)
    if cover:
        y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
        return r[y:y + h, x:x + w]
    return r


def backdrop(frame):
    """Blurred, darkened full-bleed copy of the frame (no black bars)."""
    b = cv2.GaussianBlur(fit(frame, W // 4, H // 4, cover=True), (0, 0), 6)
    return (cv2.resize(b, (W, H)) * 0.35).astype(np.uint8)


def text(img, xy, s, size, color=(255, 255, 255), anchor="la", shadow=True):
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    f = font(size)
    if shadow:
        d.text((xy[0] + 3, xy[1] + 3), s, font=f, fill=(0, 0, 0), anchor=anchor)
    d.text(xy, s, font=f, fill=color, anchor=anchor)
    return np.asarray(pil).copy()


def rounded_paste(canvas, img, x, y, border=4):
    h, w = img.shape[:2]
    canvas[y - border:y + h + border, x - border:x + w + border] = 255
    canvas[y:y + h, x:x + w] = img


def clip_frames(meta, label):
    frames = read_frames(meta["video"])
    photo = cv2.cvtColor(cv2.imread(meta["input"]), cv2.COLOR_BGR2RGB)
    depth = cv2.cvtColor(cv2.imread(meta["depth"]), cv2.COLOR_BGR2RGB)
    main_h = H - 180
    inset_w = 300
    ph, dp = fit(photo, inset_w, 220), fit(depth, inset_w, 220)
    out = []
    for f in frames:
        c = backdrop(f)
        m = fit(f, W - 2 * 60 - inset_w - 40, main_h)
        x0 = 60 + (W - 2 * 60 - inset_w - 40 - m.shape[1]) // 2
        y0 = 90 + (main_h - m.shape[0]) // 2
        c[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        ix = W - 60 - inset_w
        rounded_paste(c, ph, ix, 170)
        rounded_paste(c, dp, ix, 170 + ph.shape[0] + 70)
        out.append(c)
    # draw static text once per clip on a template, then composite onto each frame
    overlay = np.zeros((H, W, 3), np.uint8)
    overlay = text(overlay, (60, 28), label, 44, ACCENT, shadow=False)
    overlay = text(overlay, (W - 60 - inset_w, 130), "INPUT PHOTO", 26, shadow=False)
    overlay = text(overlay, (W - 60 - inset_w, 170 + ph.shape[0] + 30), "AI DEPTH MAP", 26, shadow=False)
    overlay = text(overlay, (W - 60, H - 50), "Parallax Anything", 30, (200, 200, 200), anchor="rs", shadow=False)
    mask = overlay.max(axis=2, keepdims=True) > 0
    return [np.where(mask, overlay, f) for f in out]


def card(lines, seconds, bg_frame=None):
    base = backdrop(bg_frame) if bg_frame is not None else np.full((H, W, 3), 12, np.uint8)
    y = H // 2 - 60 * (len(lines) - 1)
    for s, size, color in lines:
        base = text(base, (W // 2, y), s, size, color, anchor="mm")
        y += int(size * 1.6)
    return [base] * int(seconds * FPS)


def crossfade(a, b, n=12):
    out = a[:-n]
    for i in range(n):
        t = (i + 1) / (n + 1)
        out.append((a[len(a) - n + i] * (1 - t) + b[i] * t).astype(np.uint8))
    return out + b[n:]


def hero_grid(metas, out_path, tile_w=300, cols=3):
    clips = [read_frames(m["video"]) for m in metas[: cols * 2]]
    n = min(len(c) for c in clips)
    th = int(tile_w * 2 / 3) // 2 * 2
    frames = []
    for i in range(n):
        tiles = [fit(c[i], tile_w, th, cover=True) for c in clips]
        rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)]
        frames.append(np.vstack(rows))
    tmp = out_path.with_suffix(".mp4")
    write_mp4(frames, tmp, FPS)
    mp4_to_gif(tmp, out_path, width=tile_w * cols, fps=12)
    tmp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="outputs/gallery")
    ap.add_argument("--manifest", default="outputs/gallery/manifest.jsonl")
    ap.add_argument("--out", default="demo")
    ap.add_argument("--url", default="", help="repo URL shown on the end card")
    ap.add_argument("--hero", default="", help="comma list of photo names for the hero GIF grid (default: first 6)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    metas = [json.loads(l) for l in open(a.manifest) if l.strip()]

    first = read_frames(metas[0]["video"])[0]
    seq = card([("Parallax Anything", 110, (255, 255, 255)),
                ("Any photo  \u2192  cinematic 3D camera motion", 52, ACCENT),
                ("Depth Anything V2  \u00b7  layered inpainting  \u00b7  runs on a laptop GPU", 34, (210, 210, 210))],
               3.5, first)
    for m in metas:
        title = m.get("title") or Path(m["input"]).stem.replace("_", " ").title()
        label = f"{title}   \u00b7   {m['motion'].upper()}" + ("  +  DEPTH OF FIELD" if m.get("dof") else "")
        seq = crossfade(seq, clip_frames(m, label))
    end = [("python infer.py --input photo.jpg --motion orbit", 46, ACCENT),
           ("7 camera moves  \u00b7  depth of field  \u00b7  GIF / MP4 export", 36, (220, 220, 220))]
    if a.url:
        end.append((a.url, 40, (255, 255, 255)))
    seq = crossfade(seq, card(end, 4.0, first))
    video = write_mp4(seq, out / "parallax_anything_demo.mp4", FPS, crf=20)
    print("demo video:", video, f"{len(seq) / FPS:.1f}s")
    hero = metas
    if a.hero:
        by_name = {Path(m["input"]).stem: m for m in metas}
        hero = [by_name[n] for n in a.hero.split(",") if n in by_name]
    hero_grid(hero, out / "hero.gif")
    print("hero gif:", out / "hero.gif")


if __name__ == "__main__":
    main()
