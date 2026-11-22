"""Turn (image, disparity) into a stack of RGBA-D layers with inpainted occlusions.

Each layer holds the pixels of one depth cluster plus a band of *hallucinated*
background behind nearer content. When the virtual camera moves, nearer layers
slide further than farther ones and the band is what gets revealed - instead of
the stretched "rubber sheet" artefacts of a naive single-mesh warp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Layer:
    rgb: np.ndarray    # H x W x 3 float32 in [0, 1]
    alpha: np.ndarray  # H x W float32 in [0, 1]
    disp: np.ndarray   # H x W float32 disparity in [0, 1] (1 = nearest)


def refine_disparity(image_bgr: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """Clean the raw depth: edge-preserving smoothing, then grow near objects a few pixels
    so their silhouette never stays glued to the background."""
    h, w = disp.shape
    s = max(h, w) / 1000.0
    d = cv2.bilateralFilter(disp.astype(np.float32), max(5, int(7 * s)) | 1, 0.05, 5 * s)
    k = max(3, int(7 * s)) | 1
    d = cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def depth_thresholds(disp: np.ndarray, n_layers: int, min_gap: float = 0.1) -> np.ndarray:
    """Pick cut points in the *valleys* of the disparity histogram.

    A cut through a populated depth range slices objects in half (they then tear apart when
    the camera moves); a cut through an empty range sits on a real depth discontinuity.
    """
    hist, edges = np.histogram(disp, bins=256, range=(0.0, 1.0))
    hist = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (0, 0), 3).ravel()
    hist /= hist.sum()
    centres = (edges[:-1] + edges[1:]) / 2
    cdf = np.cumsum(hist)
    # candidates: local minima that leave at least 3% of the pixels on each side
    cand = [i for i in range(2, 254) if hist[i] <= hist[i - 1] and hist[i] <= hist[i + 1]
            and 0.03 < cdf[i] < 0.97]
    cand.sort(key=lambda i: hist[i])
    cuts: list = []
    for i in cand:
        if all(abs(centres[i] - c) >= min_gap for c in cuts):
            cuts.append(float(centres[i]))
        if len(cuts) == n_layers - 1:
            break
    if not cuts:  # flat histogram: fall back to quantiles
        cuts = list(np.quantile(disp, np.linspace(0, 1, n_layers + 1)[1:-1]))
    return np.sort(np.asarray(cuts, np.float32))


def clean_labels(bins: np.ndarray, n_layers: int, min_frac: float = 0.002) -> np.ndarray:
    """Median-filter the label map and dissolve tiny islands into their surroundings."""
    h, w = bins.shape
    ks = max(5, int(max(h, w) / 200)) | 1
    bins = cv2.medianBlur(bins, ks)
    min_area = int(min_frac * h * w)
    for _ in range(2):
        changed = False
        for k in range(n_layers):
            n, lab, stats, _ = cv2.connectedComponentsWithStats((bins == k).astype(np.uint8), connectivity=8)
            for j in range(1, n):
                if stats[j, cv2.CC_STAT_AREA] < min_area:
                    m = lab == j
                    ring = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~m
                    if ring.any():
                        bins[m] = np.bincount(bins[ring], minlength=n_layers).argmax()
                        changed = True
        if not changed:
            break
    return bins


def _inpaint(img: np.ndarray, hole: np.ndarray, radius: int) -> np.ndarray:
    if not hole.any():
        return img
    return cv2.inpaint(img, hole.astype(np.uint8), radius, cv2.INPAINT_TELEA)


def _fill_layer(rgb8, disp, hole, band_px):
    """Inpaint colour + depth inside `hole`, coarse-to-fine so wide holes stay cheap."""
    h, w = hole.shape
    f = max(1, int(np.ceil(band_px / 48)))  # work at a scale where the band is <= ~48 px wide
    if f > 1:
        small = (max(1, w // f), max(1, h // f))
        hs = cv2.resize(hole.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0
        hs = cv2.dilate(hs.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        rs = _inpaint(cv2.resize(rgb8, small, interpolation=cv2.INTER_AREA), hs, 3)
        ds = _inpaint(cv2.resize(disp, small, interpolation=cv2.INTER_AREA), hs, 3)
        rgb_up = cv2.resize(rs, (w, h), interpolation=cv2.INTER_CUBIC)
        d_up = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb8 = np.where(hole[..., None], rgb_up, rgb8)
        disp = np.where(hole, d_up, disp)
        # one thin full-res pass along the seam keeps the transition crisp
        seam = hole & (cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 3) < 6)
        rgb8 = _inpaint(rgb8, seam, 3)
    else:
        rgb8 = _inpaint(rgb8, hole, 5)
        disp = _inpaint(disp, hole, 5)
    return rgb8, disp


def push_pull_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Smoothly extend `values` from `valid` pixels into the rest of the image (pyramid fill)."""
    if valid.all() or not valid.any():
        return values
    v, m = values.astype(np.float32) * valid, valid.astype(np.float32)
    h, w = v.shape
    if min(h, w) <= 2:
        return np.where(valid, values, v.sum() / max(m.sum(), 1e-6)).astype(np.float32)
    size = (max(1, w // 2), max(1, h // 2))
    vs, ms = cv2.resize(v, size, interpolation=cv2.INTER_AREA), cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    coarse = push_pull_fill(np.where(ms > 0, vs / np.maximum(ms, 1e-6), 0), ms > 0)
    up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.where(valid, values, up).astype(np.float32)


def build_layers(image_bgr: np.ndarray, disp: np.ndarray, n_layers: int = 4, band_px: int = 64) -> List[Layer]:
    """Split the scene into `n_layers` depth slices, farthest first."""
    disp = refine_disparity(image_bgr, disp)
    cuts = depth_thresholds(disp, n_layers)
    bins = np.digitize(disp, cuts).astype(np.uint8)  # 0 = farthest ... n_layers-1 = nearest
    # isolated label speckles would show up as "dust" flying at a different speed
    bins = clean_labels(bins, len(cuts) + 1)
    h, w = disp.shape
    feather = max(1.0, max(h, w) / 1500.0)
    band_px = int(band_px)
    ell = cv2.MORPH_ELLIPSE
    k_band = cv2.getStructuringElement(ell, (2 * band_px + 1, 2 * band_px + 1))
    k_safe = cv2.getStructuringElement(ell, (4 * band_px + 1, 4 * band_px + 1))

    layers: List[Layer] = []
    for k in range(len(cuts) + 1):
        own = bins == k
        if not own.any():
            continue
        nearer = bins > k
        if not layers:
            hole = nearer                      # back layer is fully opaque: no holes can ever show
            alpha = np.ones((h, w), np.float32)
            scale_hint = max(h, w) // 8
        else:
            own8 = own.astype(np.uint8)
            reach = cv2.dilate(own8, k_band) > 0
            # Inpaint a band twice as wide as we keep, so foreground colours sitting at the far
            # edge of the hole cannot bleed into the part that will actually be revealed.
            hole = (cv2.dilate(own8, k_safe) > 0) & nearer
            keep = own | (reach & nearer)
            alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), feather) * keep  # soft outer edge
            scale_hint = 2 * band_px
        # pixels of this layer and everything behind it are "known"; nearer ones are not
        l_rgb, l_disp = _fill_layer(image_bgr.copy(), disp.copy(), hole, scale_hint)
        # depth must never come *in front* of the layer's own range, or the band would swim
        l_disp = np.where(hole, np.minimum(l_disp, cuts[k] if k < len(cuts) else 1.0), l_disp)
        # A layer's depth must be *its own* depth everywhere (not the neighbours' depth just past
        # its silhouette), and smooth, so the per-pixel inverse warp converges instead of flickering.
        l_disp = push_pull_fill(l_disp, own | hole)
        l_disp = cv2.GaussianBlur(l_disp, (0, 0), max(1.5, max(h, w) / 400.0))
        layers.append(Layer(rgb=cv2.cvtColor(l_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                            alpha=alpha.astype(np.float32), disp=l_disp.astype(np.float32)))
    return layers
