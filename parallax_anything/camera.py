"""Virtual camera paths. Every path maps t in [0, 1] to a camera pose.

Units: scene depth Z lives in [1, z_far] (1 = nearest pixel). Translations use the
same units, so tx = 0.05 moves the camera 5% of the way to the nearest object.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class Pose:
    tx: float = 0.0  # right
    ty: float = 0.0  # down
    tz: float = 0.0  # forward (into the scene)
    zoom: float = 1.0  # focal-length multiplier


def ease(t: float) -> float:
    """Smooth-step-ish cosine ease (zero velocity at both ends)."""
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def _orbit(t, a, zf):  # loops seamlessly
    th = 2 * math.pi * t
    return Pose(tx=a * math.sin(th), ty=0.45 * a * math.cos(th), tz=0.25 * a * math.sin(th))


def _swing(t, a, zf):  # loops seamlessly
    return Pose(tx=a * math.sin(2 * math.pi * t))


def _bob(t, a, zf):  # vertical swing, loops seamlessly
    return Pose(ty=0.8 * a * math.sin(2 * math.pi * t))


def _dolly(t, a, zf):  # push in towards the subject with a gentle drift
    e = ease(t)
    return Pose(tx=0.6 * a * (e - 0.5), tz=4.0 * a * e)


def _zoom_out(t, a, zf):  # reveal: start close, pull back
    e = 1.0 - ease(t)
    return Pose(ty=-0.4 * a * (e - 0.5), tz=4.0 * a * e)


def _crane(t, a, zf):  # rise up while pushing in slightly
    e = ease(t)
    return Pose(ty=1.2 * a * (0.5 - e), tz=1.5 * a * e)


def _vertigo(t, a, zf):
    """Hitchcock dolly-zoom: pull back while zooming in so the focus plane keeps its size."""
    e = math.sin(math.pi * t) ** 2  # out and back -> loops
    tz = -6.0 * a * e
    return Pose(tz=tz, zoom=(zf - tz) / zf)


MOTIONS: Dict[str, Callable[[float, float, float], Pose]] = {
    "orbit": _orbit,
    "swing": _swing,
    "bob": _bob,
    "dolly": _dolly,
    "zoom-out": _zoom_out,
    "crane": _crane,
    "vertigo": _vertigo,
}
LOOPING = {"orbit", "swing", "bob", "vertigo"}


def get_pose(motion: str, t: float, amplitude: float, focus_z: float) -> Pose:
    if motion not in MOTIONS:
        raise ValueError(f"unknown motion '{motion}', choose from {sorted(MOTIONS)}")
    return MOTIONS[motion](t, amplitude, focus_z)
