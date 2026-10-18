# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parsing and sample lookup for dataset_create camera calibrations.

The supplied files store pixel-coordinate intrinsics plus both camera-to-world
and world-to-camera matrices for every output frame. This module uses
``T_world_cam`` as the canonical pose and verifies that ``T_cam_world`` is its
inverse. Renderer-only coordinate flips are not applied a second time.

Parsed sequences are kept as compact float32 arrays. This is important for
shuffled multi-worker training over thousands of scenes: camera lookup remains
constant-time without caching hundreds of thousands of Python float objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


Matrix3 = tuple[tuple[float, float, float], ...]
Matrix4 = tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class CalibrationSequence:
    path: str
    role: str
    scene: str
    image_width: int
    image_height: int
    c2ws: np.ndarray
    normalized_Ks: np.ndarray
    native_frame_indices: np.ndarray
    timestamps_seconds: np.ndarray | None
    fps: float | None

    @property
    def num_frames(self) -> int:
        return int(self.c2ws.shape[0])

    def validate_frame_index(self, frame_index: int) -> None:
        if not 0 <= frame_index < self.num_frames:
            raise KeyError(
                f"Frame {frame_index} is outside [0, {self.num_frames}) in {self.path}"
            )


def _parse_matrix(lines: list[str], start: int, rows: int, cols: int, label: str):
    values = []
    for offset in range(rows):
        if start + offset >= len(lines):
            raise ValueError(f"Unexpected end of file while reading {label}")
        row = tuple(float(value) for value in lines[start + offset].split())
        if len(row) != cols:
            raise ValueError(f"{label} row {offset} must contain {cols} values, got {len(row)}")
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{label} row {offset} contains a non-finite value")
        values.append(row)
    return tuple(values), start + rows


def _matmul4(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def _validate_inverse(c2w: Matrix4, w2c: Matrix4, path: str, frame_index: int) -> None:
    product = _matmul4(c2w, w2c)
    max_error = max(
        abs(product[i][j] - (1.0 if i == j else 0.0))
        for i in range(4)
        for j in range(4)
    )
    if max_error > 2e-4:
        raise ValueError(
            f"Calibration matrices are not inverses in {path}, frame {frame_index}: "
            f"max identity error {max_error:.3g}"
        )


def _validate_intrinsics(K: Matrix3, path: str, frame_index: int) -> None:
    if K[0][0] <= 0 or K[1][1] <= 0:
        raise ValueError(f"Focal lengths must be positive in {path}, frame {frame_index}")
    expected_zero = (K[0][1], K[1][0], K[2][0], K[2][1])
    if max(abs(value) for value in expected_zero) > 1e-6 or abs(K[2][2] - 1.0) > 1e-6:
        raise ValueError(
            f"Expected standard pinhole intrinsics in {path}, frame {frame_index}; got {K}"
        )


def _normalized_prope_K(K: Matrix3, width: int, height: int) -> Matrix3:
    """Convert pixel intrinsics to ArtiFixer's normalized, centered K."""
    return (
        (K[0][0] / width, 0.0, K[0][2] / width - 0.5),
        (0.0, K[1][1] / height, K[1][2] / height - 0.5),
        (0.0, 0.0, 1.0),
    )


def _optional_timestamp_seconds(raw: Mapping, path: str, frame_index: int) -> float | None:
    """Read a supported per-frame timestamp and normalize it to seconds."""
    timestamp_fields = (
        ("timestamp_seconds", 1.0),
        ("timestamp_ns", 1e-9),
        ("timestamp_us", 1e-6),
        ("timestamp_ms", 1e-3),
        ("timestamp", 1.0),
    )
    present = [(name, scale) for name, scale in timestamp_fields if name in raw]
    if len(present) > 1:
        raise ValueError(
            f"Frame {frame_index} in {path} contains multiple timestamp fields: "
            f"{[name for name, _ in present]}"
        )
    if not present:
        return None
    name, scale = present[0]
    value = float(raw[name]) * scale
    if not math.isfinite(value):
        raise ValueError(f"Frame {frame_index} in {path} has non-finite {name}")
    return value


# The supplied manifest contains 1,380 scenes (2,760 files). A 4,096-entry
# bound keeps that working set resident in each worker while remaining finite.
@lru_cache(maxsize=4096)
def parse_calibration_file(
    path: str,
    parse_timing: bool = True,
) -> CalibrationSequence:
    """Parse one camera calibration file.

    ``parse_timing=False`` deliberately ignores FPS and per-frame timestamp
    fields. Camera-only training must not fail because optional timing metadata
    is incomplete or malformed when no temporal feature consumes it.
    """
    resolved = str(Path(path).resolve())
    lines = [line.strip() for line in Path(resolved).read_text().splitlines()]
    metadata: dict[str, str] = {}
    raw_frames: dict[int, dict] = {}
    current: dict | None = None
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line or line.startswith("#"):
            index += 1
            continue
        if line.startswith("[frame ") and line.endswith("]"):
            declared_index = int(line[len("[frame ") : -1])
            if declared_index in raw_frames:
                raise ValueError(f"Duplicate frame section {declared_index} in {resolved}")
            current = {"declared_index": declared_index}
            raw_frames[declared_index] = current
            index += 1
            continue
        if line in ("K:", "T_world_cam:", "T_cam_world:"):
            if current is None:
                raise ValueError(f"Matrix {line} appears before a frame in {resolved}")
            label = line[:-1]
            rows = 3 if label == "K" else 4
            matrix, index = _parse_matrix(lines, index + 1, rows, rows, label)
            current[label] = matrix
            continue
        if ":" not in line:
            raise ValueError(f"Unrecognized calibration line in {resolved}: {line!r}")
        key, value = (part.strip() for part in line.split(":", 1))
        if current is None:
            metadata[key] = value
        else:
            current[key] = value
        index += 1

    required_metadata = (
        "role",
        "output_scene_name",
        "num_frames",
        "image_width",
        "image_height",
        "matrix_layout",
        "intrinsics_convention",
        "extrinsics_saved_before_render_coord_flip_xy",
    )
    missing_metadata = [key for key in required_metadata if key not in metadata]
    if missing_metadata:
        raise ValueError(f"Missing calibration metadata in {resolved}: {missing_metadata}")
    if metadata["matrix_layout"] != "row_major":
        raise ValueError(f"Only row-major calibrations are supported, got {metadata['matrix_layout']!r}")
    if metadata["intrinsics_convention"] != "resized_pixel_coordinates":
        raise ValueError(
            "Expected resized-pixel-coordinate intrinsics, got "
            f"{metadata['intrinsics_convention']!r}"
        )
    if metadata["extrinsics_saved_before_render_coord_flip_xy"].lower() != "true":
        raise ValueError(
            "Expected camera extrinsics saved before the renderer-only XY flip; got "
            f"{metadata['extrinsics_saved_before_render_coord_flip_xy']!r}"
        )

    expected_count = int(metadata["num_frames"])
    width = int(metadata["image_width"])
    height = int(metadata["image_height"])
    if expected_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid calibration dimensions/count in {resolved}: "
            f"frames={expected_count}, size={(width, height)}"
        )
    if set(raw_frames) != set(range(expected_count)):
        raise ValueError(
            f"Calibration frame sections in {resolved} must be contiguous [0, {expected_count})"
        )

    fps = None
    if parse_timing:
        fps_text = metadata.get("fps", metadata.get("frame_rate"))
        fps = None if fps_text in (None, "") else float(fps_text)
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError(f"Invalid fps={fps_text!r} in {resolved}")

    c2ws = np.empty((expected_count, 4, 4), dtype=np.float32)
    normalized_Ks = np.empty((expected_count, 3, 3), dtype=np.float32)
    native_frame_indices = np.empty((expected_count,), dtype=np.int64)
    timestamps_seconds = np.full((expected_count,), np.nan, dtype=np.float64)
    for declared_index in range(expected_count):
        raw = raw_frames[declared_index]
        required_frame = (
            "output_frame_index",
            "native_frame_index",
            "K",
            "T_world_cam",
            "T_cam_world",
        )
        missing = [key for key in required_frame if key not in raw]
        if missing:
            raise ValueError(f"Frame {declared_index} in {resolved} is missing: {missing}")
        output_index = int(raw["output_frame_index"])
        if output_index != declared_index:
            raise ValueError(
                f"Frame section {declared_index} declares output_frame_index={output_index} in {resolved}"
            )
        K = raw["K"]
        c2w = raw["T_world_cam"]
        w2c = raw["T_cam_world"]
        _validate_intrinsics(K, resolved, output_index)
        _validate_inverse(c2w, w2c, resolved, output_index)
        c2ws[output_index] = c2w
        normalized_Ks[output_index] = _normalized_prope_K(K, width, height)
        native_frame_indices[output_index] = int(raw["native_frame_index"])
        if parse_timing:
            timestamp_seconds = _optional_timestamp_seconds(
                raw,
                resolved,
                output_index,
            )
            if timestamp_seconds is not None:
                timestamps_seconds[output_index] = timestamp_seconds

    timestamp_mask = np.isfinite(timestamps_seconds)
    if timestamp_mask.any() and not timestamp_mask.all():
        missing_timestamps = np.flatnonzero(~timestamp_mask).tolist()
        raise ValueError(
            f"Calibration {resolved} provides timestamps for only part of the sequence; "
            f"missing frames: {missing_timestamps[:16]}"
        )
    timestamps_or_none = timestamps_seconds if timestamp_mask.all() else None

    c2ws.setflags(write=False)
    normalized_Ks.setflags(write=False)
    native_frame_indices.setflags(write=False)
    if timestamps_or_none is not None:
        timestamps_or_none.setflags(write=False)
    return CalibrationSequence(
        path=resolved,
        role=metadata["role"],
        scene=metadata["output_scene_name"],
        image_width=width,
        image_height=height,
        c2ws=c2ws,
        normalized_Ks=normalized_Ks,
        native_frame_indices=native_frame_indices,
        timestamps_seconds=timestamps_or_none,
        fps=fps,
    )


def _frame_index_from_image(path: str) -> int:
    stem = Path(path).stem
    if not stem.isdigit():
        raise ValueError(f"Cannot infer calibration frame index from image name {path!r}")
    return int(stem)


def resolve_sample_calibration(
    entry: Mapping,
    manifest_path: str,
    reference_images: Sequence[str] | None = None,
    resolve_time: bool = True,
) -> dict[str, object]:
    """Resolve target and source camera tensors for one manifest sample.

    Explicit ``target_calibration`` / ``source_calibration`` entry fields take
    precedence. Relative overrides are resolved from the manifest directory.
    Otherwise files are inferred as
    ``<manifest-dir>/calibration/<scene>/{target,source}_calibration.txt``.
    When ``resolve_time`` is false, optional timestamps and FPS are ignored.
    """
    scene = str(entry.get("scene", ""))
    if not scene:
        scene = Path(str(entry["image"])).parent.name
    manifest_dir = Path(manifest_path).resolve().parent
    calibration_dir = manifest_dir / "calibration" / scene

    def resolve_path(field: str, default: Path) -> str:
        path = Path(str(entry.get(field, default)))
        if not path.is_absolute():
            path = manifest_dir / path
        return str(path)

    target_path = resolve_path("target_calibration", calibration_dir / "target_calibration.txt")
    source_path = resolve_path("source_calibration", calibration_dir / "source_calibration.txt")
    target_sequence = parse_calibration_file(target_path, parse_timing=resolve_time)
    source_sequence = parse_calibration_file(source_path, parse_timing=resolve_time)
    if source_sequence.role != "source":
        raise ValueError(
            f"Calibration roles must be target/source, got {target_sequence.role!r}/{source_sequence.role!r}"
        )
    if target_sequence.scene != scene or source_sequence.scene != scene:
        raise ValueError(
            f"Calibration scene mismatch for {scene!r}: "
            f"{target_sequence.scene!r}, {source_sequence.scene!r}"
        )

    target_index = (
        int(entry["frame_index"])
        if "frame_index" in entry
        else _frame_index_from_image(str(entry["image"]))
    )
    reference_paths = list(reference_images if reference_images is not None else entry["reference_images"])
    reference_indices = entry.get("reference_frame_indices")
    if reference_indices is None:
        reference_indices = [_frame_index_from_image(str(path)) for path in reference_paths]
    else:
        reference_indices = [int(index) for index in reference_indices]
    if len(reference_indices) != len(reference_paths):
        raise ValueError("reference_frame_indices must match reference_images length")

    target_sequence.validate_frame_index(target_index)
    for frame_index in reference_indices:
        source_sequence.validate_frame_index(frame_index)
    reference_index_array = np.asarray(reference_indices, dtype=np.int64)

    target_native_index = int(target_sequence.native_frame_indices[target_index])
    reference_native_indices = source_sequence.native_frame_indices[
        reference_index_array
    ].astype(np.int64, copy=True)

    result = {
        "target_c2w": target_sequence.c2ws[target_index],
        "target_K": target_sequence.normalized_Ks[target_index],
        "reference_c2ws": source_sequence.c2ws[reference_index_array],
        "reference_Ks": source_sequence.normalized_Ks[reference_index_array],
        "target_frame_index": target_index,
        "reference_frame_indices": reference_indices,
        "target_native_frame_index": target_native_index,
        "reference_native_frame_indices": reference_native_indices,
    }
    if not resolve_time:
        return result

    explicit_offsets = entry.get("reference_time_offsets")
    explicit_target_timestamp = entry.get("target_timestamp")
    explicit_reference_timestamps = entry.get("reference_timestamps")
    if explicit_offsets is not None:
        reference_time_offsets = np.asarray(explicit_offsets, dtype=np.float32)
        temporal_unit = str(entry.get("reference_time_unit", "frames"))
    elif (explicit_target_timestamp is None) != (explicit_reference_timestamps is None):
        raise ValueError(
            "target_timestamp and reference_timestamps must be provided together"
        )
    elif explicit_target_timestamp is not None:
        reference_time_offsets = np.asarray(
            explicit_reference_timestamps,
            dtype=np.float64,
        ) - float(explicit_target_timestamp)
        reference_time_offsets = reference_time_offsets.astype(np.float32)
        temporal_unit = "seconds"
    elif (
        target_sequence.timestamps_seconds is not None
        and source_sequence.timestamps_seconds is not None
    ):
        reference_time_offsets = (
            source_sequence.timestamps_seconds[reference_index_array]
            - target_sequence.timestamps_seconds[target_index]
        ).astype(np.float32)
        temporal_unit = "seconds"
    else:
        reference_time_offsets = (
            reference_native_indices - target_native_index
        ).astype(np.float32)
        fps_override = entry.get("fps")
        if fps_override is not None:
            fps = float(fps_override)
        else:
            sequence_fps = [
                value
                for value in (target_sequence.fps, source_sequence.fps)
                if value is not None
            ]
            if len(sequence_fps) == 2 and not math.isclose(
                sequence_fps[0], sequence_fps[1], rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError(
                    "Target/source calibration FPS values disagree: "
                    f"{sequence_fps}"
                )
            fps = sequence_fps[0] if sequence_fps else None
        if fps is not None:
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid sample fps={fps!r}")
            reference_time_offsets /= fps
            temporal_unit = "seconds"
        else:
            temporal_unit = "frames"

    if reference_time_offsets.shape != (len(reference_indices),):
        raise ValueError(
            "reference_time_offsets must match reference_images length; got "
            f"{reference_time_offsets.shape} for {len(reference_indices)} references"
        )
    if not np.isfinite(reference_time_offsets).all():
        raise ValueError("reference_time_offsets contain non-finite values")

    result.update(
        {
            "reference_time_offsets": reference_time_offsets,
            "reference_time_unit": temporal_unit,
        }
    )
    return result


def resolve_sample_time_offsets(
    entry: Mapping,
    manifest_path: str,
    reference_images: Sequence[str] | None = None,
) -> dict[str, object]:
    """Resolve signed reference times without requiring camera calibration.

    Explicit offsets/timestamps take precedence. If calibration files are
    available, their native indices, timestamps, and FPS are retained. When
    they are absent, output/native frame indices from the manifest or numeric
    filenames provide a camera-free fallback.
    """
    reference_paths = list(
        reference_images if reference_images is not None else entry["reference_images"]
    )
    target_index = (
        int(entry["frame_index"])
        if "frame_index" in entry
        else _frame_index_from_image(str(entry["image"]))
    )
    reference_indices = entry.get("reference_frame_indices")
    if reference_indices is None:
        reference_indices = [
            _frame_index_from_image(str(path)) for path in reference_paths
        ]
    else:
        reference_indices = [int(index) for index in reference_indices]
    if len(reference_indices) != len(reference_paths):
        raise ValueError("reference_frame_indices must match reference_images length")

    explicit_offsets = entry.get("reference_time_offsets")
    explicit_target_timestamp = entry.get("target_timestamp")
    explicit_reference_timestamps = entry.get("reference_timestamps")
    if (
        explicit_offsets is None
        and explicit_target_timestamp is None
        and explicit_reference_timestamps is None
    ):
        try:
            resolved = resolve_sample_calibration(
                entry,
                manifest_path,
                reference_images=reference_paths,
            )
        except FileNotFoundError:
            resolved = None
        if resolved is not None:
            return {
                key: resolved[key]
                for key in (
                    "target_frame_index",
                    "reference_frame_indices",
                    "target_native_frame_index",
                    "reference_native_frame_indices",
                    "reference_time_offsets",
                    "reference_time_unit",
                )
            }

    if explicit_offsets is not None:
        offsets = np.asarray(explicit_offsets, dtype=np.float32)
        temporal_unit = str(entry.get("reference_time_unit", "frames"))
    elif (explicit_target_timestamp is None) != (explicit_reference_timestamps is None):
        raise ValueError(
            "target_timestamp and reference_timestamps must be provided together"
        )
    elif explicit_target_timestamp is not None:
        offsets = (
            np.asarray(explicit_reference_timestamps, dtype=np.float64)
            - float(explicit_target_timestamp)
        ).astype(np.float32)
        temporal_unit = "seconds"
    else:
        offsets = (
            np.asarray(reference_indices, dtype=np.float32) - float(target_index)
        )
        fps = entry.get("fps")
        if fps is not None:
            fps = float(fps)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid sample fps={fps!r}")
            offsets /= fps
            temporal_unit = "seconds"
        else:
            temporal_unit = "frames"

    if offsets.shape != (len(reference_indices),):
        raise ValueError(
            "reference_time_offsets must match reference_images length; got "
            f"{offsets.shape} for {len(reference_indices)} references"
        )
    if not np.isfinite(offsets).all():
        raise ValueError("reference_time_offsets contain non-finite values")
    reference_native_indices = np.asarray(reference_indices, dtype=np.int64)
    return {
        "target_frame_index": target_index,
        "reference_frame_indices": reference_indices,
        "target_native_frame_index": target_index,
        "reference_native_frame_indices": reference_native_indices,
        "reference_time_offsets": offsets,
        "reference_time_unit": temporal_unit,
    }
