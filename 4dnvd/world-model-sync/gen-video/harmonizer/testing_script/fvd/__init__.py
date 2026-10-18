"""PyTorch-only Frechet Video Distance backends."""

from pathlib import Path
from types import ModuleType


FVD_METHODS = ("styleganv", "videogpt")


def _backend(method: str) -> ModuleType:
    if method == "styleganv":
        from .styleganv import fvd

        return fvd
    if method == "videogpt":
        from .videogpt import fvd

        return fvd
    raise ValueError(f"Unknown PyTorch FVD method {method!r}; choose one of {FVD_METHODS}")


def default_model_path(method: str) -> Path:
    return _backend(method).DEFAULT_MODEL_PATH


def load_i3d_pretrained(method: str, device, model_path):
    return _backend(method).load_i3d_pretrained(device=device, model_path=model_path)


def extract_fvd_features(method: str, clips, detector, device, batch_size: int):
    return _backend(method).extract_fvd_features(
        clips,
        detector,
        device,
        batch_size=batch_size,
    )


def frechet_distance(method: str, features_fake, features_real):
    return _backend(method).frechet_distance(features_fake, features_real)


__all__ = [
    "FVD_METHODS",
    "default_model_path",
    "extract_fvd_features",
    "frechet_distance",
    "load_i3d_pretrained",
]
