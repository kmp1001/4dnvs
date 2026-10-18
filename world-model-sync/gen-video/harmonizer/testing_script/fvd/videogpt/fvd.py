"""PyTorch-only VideoGPT FVD implementation.

This preserves the supplied VideoGPT backend's Kinetics-400 I3D logits and
PyTorch Fréchet calculation while adapting input to ordered frame paths. The
detector is deliberately not wrapped in DataParallel: one evaluator worker
owns one detector on exactly one CUDA device.
"""

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from .pytorch_i3d import InceptionI3d


DEFAULT_MODEL_PATH = Path(__file__).with_name("i3d_pretrained_400.pt")


def load_i3d_pretrained(
    device: torch.device,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> InceptionI3d:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"PyTorch VideoGPT I3D weights not found: {path}")

    detector = InceptionI3d(400, in_channels=3).eval()
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    detector.load_state_dict(state_dict)
    del state_dict
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    return detector.to(device)


def _load_preprocessed_clip(
    frame_paths: Sequence[str | Path],
    resolution: int = 224,
) -> torch.Tensor:
    """Load one VideoGPT input clip as CTHW float32 in [-1, 1]."""
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as handle:
            frames.append(TF.to_tensor(handle.convert("RGB")))

    frame_shapes = {tuple(frame.shape) for frame in frames}
    if len(frame_shapes) != 1:
        raise ValueError(f"All frames in an FVD clip must have one size, got {frame_shapes}")

    video = torch.stack(frames, dim=0)
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
    return video.permute(1, 0, 2, 3).sub(0.5).mul(2.0).contiguous()


def _extract_batch(
    videos: list[torch.Tensor],
    detector: InceptionI3d,
    device: torch.device,
) -> np.ndarray:
    batch = torch.stack(videos, dim=0).to(device)
    with torch.inference_mode():
        features = detector(batch)
    return features.detach().float().cpu().numpy()


def extract_fvd_features(
    clips: Sequence[Sequence[str | Path]],
    detector: InceptionI3d,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """Extract VideoGPT I3D logits, batching clips with equal frame counts."""
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


def _covariance(samples: torch.Tensor) -> torch.Tensor:
    samples = samples.flatten(start_dim=1)
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1).matmul(centered) / (samples.shape[0] - 1)


def _symmetric_matrix_square_root(matrix: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    left, singular_values, right = torch.svd(matrix)
    roots = torch.where(
        singular_values < eps,
        singular_values,
        torch.sqrt(singular_values),
    )
    return left.matmul(torch.diag(roots)).matmul(right.transpose(0, 1))


def _trace_sqrt_product(covariance: torch.Tensor, covariance_other: torch.Tensor) -> torch.Tensor:
    covariance_root = _symmetric_matrix_square_root(covariance)
    product = covariance_root.matmul(covariance_other).matmul(covariance_root)
    return torch.trace(_symmetric_matrix_square_root(product))


def frechet_distance(features_fake: np.ndarray, features_real: np.ndarray) -> float | None:
    """Compute VideoGPT's PyTorch Fréchet distance over I3D logits."""
    if len(features_fake) == 0 or len(features_real) == 0:
        return None
    if features_fake.shape != features_real.shape:
        raise ValueError(
            f"FVD feature shapes must match, got {features_fake.shape} and {features_real.shape}"
        )

    fake = torch.as_tensor(features_fake, dtype=torch.float32).flatten(start_dim=1)
    real = torch.as_tensor(features_real, dtype=torch.float32).flatten(start_dim=1)
    mean_distance = torch.sum((fake.mean(dim=0) - real.mean(dim=0)) ** 2)
    if len(fake) == 1:
        return float(mean_distance.item())

    covariance_fake = _covariance(fake)
    covariance_real = _covariance(real)
    trace = (
        torch.trace(covariance_fake + covariance_real)
        - 2.0 * _trace_sqrt_product(covariance_fake, covariance_real)
    )
    return float((mean_distance + trace).item())
