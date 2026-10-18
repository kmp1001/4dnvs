# SPDX-License-Identifier: Apache-2.0
"""Self-generated-history training for the existing image harmonizer.

This adapts Self Forcing's rollout conditioning, not its Wan KV cache or DMD loss.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import Dataset

from utils.temporal import TEMPORAL_HISTORY_FRAMES
from utils.training_utils import DrivenVSDataset


class SelfForcingWindowDataset(Dataset):
    """Consecutive frames, each loaded with the ordinary V=1 dataset path."""

    def __init__(
        self,
        *,
        manifest,
        split,
        image_prep,
        num_reference_images,
        camera_embed,
        time_embed,
        window_length,
        world_scenario=False,
    ):
        if window_length < TEMPORAL_HISTORY_FRAMES + 1:
            raise ValueError("window_length must be at least five")
        self.base = DrivenVSDataset(
            dataset_folder=manifest,
            split=split,
            image_prep=image_prep,
            num_reference_images=num_reference_images,
            camera_embed=camera_embed,
            time_embed=time_embed,
            training_stage="stage1",
            temporal_sample_probability=0.0,
            world_scenario=world_scenario,
        )
        groups = defaultdict(dict)
        for dataset_index, name in enumerate(self.base.img_names):
            entry = self.base.data[name]
            frame_index = entry["_frame_index"]
            if frame_index is None:
                continue
            group = (entry["_source_index"], entry["_sequence_key"])
            if frame_index in groups[group]:
                raise ValueError(f"Duplicate frame {frame_index} in {group}")
            groups[group][frame_index] = dataset_index
        self.windows = []
        for frames in groups.values():
            for final_index in sorted(frames):
                start = final_index - window_length + 1
                indices = range(start, final_index + 1)
                if all(index in frames for index in indices):
                    self.windows.append(tuple(frames[index] for index in indices))
        if not self.windows:
            raise ValueError("No consecutive windows; check manifest and window_length")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return tuple(self.base[i] for i in self.windows[index])


class SelfForcingRollout:
    """Replay inference with detached generated history; train the last frame."""

    def __init__(self, *, num_reference_images, camera_embed, time_embed,
                 num_inference_steps=4, timestep_shift=5.0):
        self.num_reference_images = int(num_reference_images)
        self.camera_embed = bool(camera_embed)
        self.time_embed = bool(time_embed)
        self.num_inference_steps = int(num_inference_steps)
        self.timestep_shift = float(timestep_shift)
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")

    def _prepare(self, frame, device, dtype):
        from train_drivenvs import prepare_training_batch
        return prepare_training_batch(
            frame, device=device, dtype=dtype,
            expected_reference_views=self.num_reference_images,
            camera_embed=self.camera_embed, time_embed=self.time_embed,
        )

    def _with_history(self, prepared, history):
        current_input, references, current_target, current_opacity, _, condition = prepared
        recent = list(reversed(history[-TEMPORAL_HISTORY_FRAMES:]))
        if len(recent) != TEMPORAL_HISTORY_FRAMES:
            return current_input, references, current_target, current_opacity, condition
        preds = [item["prediction"] for item in recent]
        source_path = torch.cat([current_input, *preds], dim=2)
        # GT is used only for the current target; generated history appears
        # identically in source and target paths, preventing teacher leakage.
        target_path = torch.cat([current_target, *preds], dim=2)
        opacity = torch.cat(
            [current_opacity, *[torch.ones_like(current_opacity) for _ in recent]],
            dim=2,
        )
        condition = dict(condition)
        if self.camera_embed:
            condition["target_c2w"] = torch.cat(
                [condition["target_c2w"], *[item["target_c2w"] for item in recent]], dim=1
            )
            condition["target_K"] = torch.cat(
                [condition["target_K"], *[item["target_K"] for item in recent]], dim=1
            )
        return source_path, references, target_path, opacity, condition

    def prepare_terminal(self, *, model, window, device, dtype, generator=None):
        if len(window) < TEMPORAL_HISTORY_FRAMES + 1:
            raise ValueError("Rollout window must contain at least five frames")
        history = []
        was_training = model.unet.training
        model.unet.eval()
        try:
            with torch.no_grad():
                for frame in window[:-1]:
                    prepared = self._prepare(frame, device, dtype)
                    source, references, _, opacity, condition = self._with_history(
                        prepared, history
                    )
                    prediction = model.sample_bridge(
                        source, reference_images=references, opacity=opacity,
                        num_steps=self.num_inference_steps,
                        timestep_shift=self.timestep_shift,
                        generator=generator, **condition,
                    )
                    item = {"prediction": prediction[:, :, :1].detach().to(dtype=dtype)}
                    if self.camera_embed:
                        item["target_c2w"] = condition["target_c2w"][:, :1].detach()
                        item["target_K"] = condition["target_K"][:, :1].detach()
                    history.append(item)
                    del history[:-TEMPORAL_HISTORY_FRAMES]
        finally:
            model.unet.train(was_training)
        return self._with_history(self._prepare(window[-1], device, dtype), history)
