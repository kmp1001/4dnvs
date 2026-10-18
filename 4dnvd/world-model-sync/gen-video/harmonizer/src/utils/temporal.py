# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared temporal-sequence conventions for training and inference."""

from pathlib import Path


TEMPORAL_HISTORY_FRAMES = 4
TEMPORAL_TARGET_VIEWS = 1 + TEMPORAL_HISTORY_FRAMES


def temporal_frame_index(entry: dict, sample_key: str | None = None) -> int | None:
    """Resolve an output-frame index without confusing reference-frame time."""
    candidates = (
        entry.get("frame_index"),
        entry.get("frame_idx"),
        entry.get("idx"),
        sample_key,
        Path(str(entry.get("target_image", ""))).stem,
        Path(str(entry.get("image", ""))).stem,
    )
    for value in candidates:
        if value in (None, ""):
            continue
        stem = Path(str(value)).stem
        if stem.isdigit():
            return int(stem)
    return None


def temporal_sequence_key(entry: dict, sample_key: str | None = None) -> str:
    """Return the stream key used to prevent history leakage across cameras."""
    explicit = str(entry.get("temporal_sequence", "")).strip()
    if explicit:
        return explicit
    scene = str(entry.get("scene", "default")).strip() or "default"
    target_path = entry.get("target_image", entry.get("image", sample_key or "default"))
    return f"{scene}::{Path(str(target_path)).resolve().parent}"


def select_consecutive_prediction_history(
    history: list[dict],
    current_frame_index: int | None,
) -> list[dict]:
    """Return ``[t-1, ..., t-4]`` only when the complete history exists."""
    if current_frame_index is None or len(history) < TEMPORAL_HISTORY_FRAMES:
        return []
    candidates = list(reversed(history[-TEMPORAL_HISTORY_FRAMES:]))
    expected = [
        current_frame_index - offset
        for offset in range(1, TEMPORAL_HISTORY_FRAMES + 1)
    ]
    if [item.get("frame_index") for item in candidates] != expected:
        return []
    return candidates
