"""Parallax Anything: turn any photo into a cinematic 3D parallax video."""
from .camera import LOOPING, MOTIONS
from .pipeline import ParallaxConfig, ParallaxPipeline

__all__ = ["ParallaxPipeline", "ParallaxConfig", "MOTIONS", "LOOPING"]
__version__ = "0.1.0"
