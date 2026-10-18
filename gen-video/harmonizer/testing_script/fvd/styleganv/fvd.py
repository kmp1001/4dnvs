"""PyTorch-only FVD helpers adapted from the StyleGAN-V implementation.

The bundled TorchScript I3D network produces the 400-dimensional pre-softmax
features used by StyleGAN-V FVD. Each caller owns one detector on one CUDA
device; this module intentionally does not wrap the detector in DataParallel.
"""

import math
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import sqrtm
from torchvision.transforms import functional as TF


DEFAULT_MODEL_PATH = Path(__file__).with_name("i3d_torchscript.pt")


def load_i3d_pretrained(
    device: torch.device,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> torch.jit.ScriptModule:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"PyTorch FVD I3D weights not found: {path}")
    detector = torch.jit.load(str(path), map_location=device).eval().to(device)
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    return detector


def _load_preprocessed_clip(
    frame_paths: Sequence[str | Path],
    resolution: int = 224,
) -> torch.Tensor:
    """Load one video as CTHW float32 in [-1, 1]."""
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as handle:
            frames.append(TF.to_tensor(handle.convert("RGB")))

    frame_shapes = {tuple(frame.shape) for frame in frames}
    if len(frame_shapes) != 1:
        raise ValueError(f"All frames in an FVD clip must have one size, got {frame_shapes}")

    video = torch.stack(frames, dim=1)
    _, _, height, width = video.shape
    scale = resolution / min(height, width)
    target_size = (
        (resolution, math.ceil(width * scale))
        if height < width
        else (math.ceil(height * scale), resolution)
    )
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)
    _, _, height, width = video.shape
    left = (width - resolution) // 2
    top = (height - resolution) // 2
    video = video[:, :, top : top + resolution, left : left + resolution]
    return video.sub(0.5).mul(2.0).contiguous()


def _extract_batch(
    videos: list[torch.Tensor],
    detector: torch.jit.ScriptModule,
    device: torch.device,
) -> np.ndarray:
    batch = torch.stack(videos, dim=0).to(device)
    with torch.inference_mode():
        features = detector(
            x=batch,
            rescale=False,
            resize=False,
            return_features=True,
        )
    return features.detach().float().cpu().numpy()


def extract_fvd_features(
    clips: Sequence[Sequence[str | Path]],
    detector: torch.jit.ScriptModule,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """Extract I3D features, batching clips that have the same frame count."""
    pending: list[torch.Tensor] = []
    pending_frame_count = None
    feature_batches = []

    for clip in clips:
        video = _load_preprocessed_clip(clip)
        frame_count = video.shape[1]
        if pending and (frame_count != pending_frame_count or len(pending) >= batch_size):
            feature_batches.append(_extract_batch(pending, detector, device))
            pending = []
        pending.append(video)
        pending_frame_count = frame_count

    if pending:
        feature_batches.append(_extract_batch(pending, detector, device))

    if not feature_batches:
        return np.empty((0, 400), dtype=np.float32)
    return np.concatenate(feature_batches, axis=0)


def frechet_distance(features_fake: np.ndarray, features_real: np.ndarray) -> float | None:
    """Compute Frechet distance between two sets of I3D feature vectors."""
    if len(features_fake) == 0 or len(features_real) == 0:
        return None
    if features_fake.shape != features_real.shape:
        raise ValueError(
            f"FVD feature shapes must match, got {features_fake.shape} and {features_real.shape}"
        )

    mean_fake = features_fake.mean(axis=0)
    mean_real = features_real.mean(axis=0)
    mean_distance = np.square(mean_fake - mean_real).sum()
    if len(features_fake) == 1:
        return float(np.real(mean_distance))

    covariance_fake = np.cov(features_fake, rowvar=False)
    covariance_real = np.cov(features_real, rowvar=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covariance_mean, _ = sqrtm(covariance_fake @ covariance_real, disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_fake.shape[0]) * 1e-6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            covariance_mean = sqrtm((covariance_fake + offset) @ (covariance_real + offset))
    covariance_mean = np.real(covariance_mean)
    value = mean_distance + np.trace(covariance_fake + covariance_real - 2.0 * covariance_mean)
    return float(np.real(value))
