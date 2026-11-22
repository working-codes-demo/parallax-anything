"""Fast unit tests (no checkpoint needed). Run: python -m pytest -q"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallax_anything.camera import LOOPING, MOTIONS, Pose, get_pose  # noqa: E402
from parallax_anything.layers import build_layers  # noqa: E402
from parallax_anything.render import ParallaxRenderer, disp_to_z  # noqa: E402


def _scene(h=96, w=128):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 120, 200)
    img[30:70, 40:90] = (20, 200, 60)  # a near box
    disp = np.full((h, w), 0.1, np.float32)
    disp[30:70, 40:90] = 0.9
    return img, disp


def _forward(us, Z, p, s, w, f0, zf):
    cx = w / 2
    comp = s * (-f0 * p.tx) / (zf - p.tz)
    return s * ((us - cx) * Z - f0 * p.tx) / (Z - p.tz) + cx - comp


def test_inverse_projection_roundtrip():
    img, disp = _scene()
    r = ParallaxRenderer(build_layers(img, disp, n_layers=2, band_px=8), torch.device("cpu"), focus=0.5)
    p = Pose(tx=0.04, ty=-0.02, tz=0.1, zoom=1.07)
    for Z in (1.0, 2.5, 7.0):
        us = torch.tensor([3.0, 64.0, 120.0])
        ut = _forward(us, Z, p, p.zoom, r.w, r.f0, r.focus_z)
        back, _ = r._inverse(ut, torch.zeros(3), torch.tensor(Z), p, p.zoom)
        assert torch.allclose(back, us, atol=1e-3)


def test_focus_plane_stays_still():
    img, disp = _scene()
    r = ParallaxRenderer(build_layers(img, disp, n_layers=2, band_px=8), torch.device("cpu"), focus=0.5)
    p = Pose(tx=0.05)
    ut = _forward(torch.tensor([r.w / 2]), r.focus_z, p, 1.0, r.w, r.f0, r.focus_z)
    assert abs(float(ut) - r.w / 2) < 1e-4


def test_layers_cover_frame_and_back_layer_opaque():
    img, disp = _scene()
    layers = build_layers(img, disp, n_layers=3, band_px=10)
    assert 1 <= len(layers) <= 3
    assert np.all(layers[0].alpha == 1.0)
    for L in layers:
        assert L.rgb.shape == img.shape and L.alpha.shape == disp.shape
        assert np.isfinite(L.disp).all()


def test_render_identity_pose_reproduces_image():
    img, disp = _scene()
    r = ParallaxRenderer(build_layers(img, disp, n_layers=2, band_px=8), torch.device("cpu"), focus=0.5)
    out = (r.render(Pose())[0].permute(1, 2, 0).numpy() * 255).round()
    ref = img[..., ::-1].astype(np.float32)
    assert np.abs(out[4:-4, 4:-4] - ref[4:-4, 4:-4]).mean() < 6.0


def test_looping_motions_start_where_they_end():
    for m in LOOPING:
        a, b = get_pose(m, 0.0, 0.05, 3.0), get_pose(m, 1.0, 0.05, 3.0)
        assert math.isclose(a.tx, b.tx, abs_tol=1e-9) and math.isclose(a.tz, b.tz, abs_tol=1e-9)


def test_all_motions_keep_camera_in_front_of_scene():
    for m in MOTIONS:
        for t in np.linspace(0, 1, 50):
            assert get_pose(m, float(t), 0.055 * 2, 3.0).tz < 0.9  # nearest depth is Z = 1


def test_disp_to_z_range():
    assert math.isclose(float(disp_to_z(1.0, 8.0)), 1.0)
    assert math.isclose(float(disp_to_z(0.0, 8.0)), 8.0)
