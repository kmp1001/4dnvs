import argparse
import json
import math
import multiprocessing as mp
import os
import re
import tempfile
import traceback
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from reference_dedup import dedup_ranks, tile_vote_counts
from reference_dedup_trace import save_reference_trace
from reference_greedy_union import save_union_selection_trace
from dataset import (
    Py123dMonocularShiftedReconDataset,
    Py123dPairedCameraReconDataset,
    collate_fn,
    monocular_shifted_collate_fn,
)
from foundation.da3 import (
    load_da3_model,
    run_da3_depth_with_calibration,
    write_video_tensor_to_images,
)
from foundation.sam3 import (
    get_sam3_prompts,
    get_sam3_sky_prompts,
    load_sam3_video_predictor,
    run_sam3_vehicle_masks_video_based,
)
from utils.calib import apply_render_pose_convention, shift_camera_poses_local
from utils.pointcloud import (
    apply_sky_handling_like_utils,
    build_dynamic_instance_id_maps,
    build_seginstance_id_maps,
    build_pointcloud_visibility,
    build_temporal_instance_visibility,
    classify_static_instances_by_center_stability,
    compute_double_reprojection_source_frame_visibility_counts,
    compute_source_frame_visibility_counts,
    gather_virtual_render_to_source_grid,
    render,
    render_instance_ids,
    splat_global_static_points_to_source_holes,
    save_global_pointcloud_ply,
    split_dynamic_mask_using_static_vehicle_instances,
    unproject,
)
from utils.utils import (
    cfg_get,
    cleanup,
    configure_data_roots,
    fix_seed,
    get_batch_filter,
    load_config,
    apply_cli_overrides,
    make_output_dirs,
    build_scene_filter,
    tensor_to_rgb_numpy,
)
from utils.viz import (
    save_instance_id_overlay_frames,
    save_instance_id_overlay_image,
    save_rgb,
    save_visualization_grid_video,
)


def get_visualization_options(config):
    """Read the independent visualization-video toggle and its options."""
    raw = config.get("visualize", False)
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled", False))
        fps = float(raw.get("fps", 10))
        output_subdir = str(raw.get("output_subdir", "visualization"))
        overlay_alpha = float(raw.get("overlay_alpha", 0.50))
        draw_ids = bool(raw.get("draw_ids", True))
    else:
        enabled = bool(raw)
        fps = 10.0
        output_subdir = "visualization"
        overlay_alpha = 0.50
        draw_ids = True

    if fps <= 0:
        raise ValueError(f"visualize.fps must be positive, got {fps}")
    if not output_subdir.strip():
        raise ValueError("visualize.output_subdir must not be empty")
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError(
            "visualize.overlay_alpha must be between 0 and 1, got "
            f"{overlay_alpha}"
        )
    return {
        "enabled": enabled,
        "fps": fps,
        "output_subdir": output_subdir,
        "overlay_alpha": overlay_alpha,
        "draw_ids": draw_ids,
    }


def normalize_dataset_setup(raw):
    """Normalize the configured dataset/rendering setup."""
    key = str(raw or "stereo").strip().lower().replace("-", "_")
    if key in {"stereo", "paired", "paired_camera"}:
        return "stereo"
    if key in {
        "monocular_shifted",
        "monocular_shift",
        "mono_shifted",
        "mono_shift",
    }:
        return "monocular_shifted"
    raise ValueError(
        f"Unsupported dataset.setup={raw!r}. Expected 'stereo' or "
        "'monocular_shifted'."
    )


def make_storage_frame_indices(sampled_frame_indices):
    """Return the contiguous timeline used by every saved artifact.

    ``sampled_frame_indices`` records which native frames the dataset loader read
    (for example ``[0, 2, 4, ...]``).  Sampling is complete before this helper is
    called, so external filenames and metadata should use ``[0, 1, 2, ...]``
    regardless of the native stride.
    """
    return list(range(len(sampled_frame_indices)))


def normalize_render_directions(
    raw,
    dataset_setup="stereo",
    include_down_shift=False,
):
    """Normalize render directions for stereo or shifted-monocular setup."""
    dataset_setup = normalize_dataset_setup(dataset_setup)
    if raw is None:
        if dataset_setup == "stereo":
            return ["LR"]
        directions = ["L2M", "R2M", "U2M"]
        if include_down_shift:
            directions.append("D2M")
        return directions

    if isinstance(raw, str):
        # Accept both `both` and simple comma-separated CLI strings such as `LR,RL`.
        tokens = [
            x for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()
        ]
    else:
        tokens = list(raw)

    directions = []
    for token in tokens:
        key = str(token).strip().upper().replace("-", "").replace("_", "")
        if dataset_setup == "stereo" and key in {
            "BOTH",
            "BIDIRECTIONAL",
            "L2RR2L",
            "LRRL",
        }:
            candidates = ["LR", "RL"]
        elif dataset_setup == "stereo" and key in {
            "LR",
            "L2R",
            "LEFTRIGHT",
            "LEFTTORIGHT",
        }:
            candidates = ["LR"]
        elif dataset_setup == "stereo" and key in {
            "RL",
            "R2L",
            "RIGHTLEFT",
            "RIGHTTOLEFT",
        }:
            candidates = ["RL"]
        elif dataset_setup == "monocular_shifted" and key in {
            "ALL",
            "L2MR2MU2MD2M",
            "L2MR2MU2M",
        }:
            candidates = ["L2M", "R2M", "U2M"]
            if include_down_shift or key == "L2MR2MU2MD2M":
                candidates.append("D2M")
        elif dataset_setup == "monocular_shifted" and key in {
            "L2M",
            "LEFT2M",
            "LEFT2METER",
            "LEFT2METERS",
        }:
            candidates = ["L2M"]
        elif dataset_setup == "monocular_shifted" and key in {
            "R2M",
            "RIGHT2M",
            "RIGHT2METER",
            "RIGHT2METERS",
        }:
            candidates = ["R2M"]
        elif dataset_setup == "monocular_shifted" and key in {
            "U2M",
            "UP2M",
            "UP2METER",
            "UP2METERS",
        }:
            candidates = ["U2M"]
        elif dataset_setup == "monocular_shifted" and key in {
            "D2M",
            "DOWN2M",
            "DOWN2METER",
            "DOWN2METERS",
        }:
            candidates = ["D2M"]
        else:
            expected = (
                "LR, RL, both, or [LR, RL]"
                if dataset_setup == "stereo"
                else "L2M, R2M, U2M, D2M, or a YAML list of directions"
            )
            raise ValueError(
                f"Unsupported dataset.render_directions value {token!r} for "
                f"setup={dataset_setup}. Use {expected}."
            )

        for direction in candidates:
            if direction not in directions:
                directions.append(direction)

    if len(directions) == 0:
        raise ValueError(
            "dataset.render_directions must contain at least one direction."
        )
    return directions


def get_monocular_shift_offset(direction, distance_m=2.0):
    """Return the local OpenCV-camera translation for a virtual target pose."""
    distance_m = float(distance_m)
    if distance_m <= 0:
        raise ValueError(
            f"monocular_shift.distance_m must be positive, got {distance_m}"
        )

    direction = str(direction).strip().upper()
    offsets = {
        "L2M": (-distance_m, 0.0, 0.0),
        "R2M": (distance_m, 0.0, 0.0),
        "U2M": (0.0, -distance_m, 0.0),
        "D2M": (0.0, distance_m, 0.0),
    }
    if direction not in offsets:
        raise ValueError(f"Unsupported monocular shift direction: {direction}")
    return offsets[direction]


def select_monocular_shifted_data(
    direction,
    source_video,
    source_K_seq,
    source_T_world_cam_seq,
    source_cam_id,
    distance_m=2.0,
):
    """Create one synthetic target camera translated from the real source camera."""
    direction = str(direction).strip().upper()
    offset_xyz_m = get_monocular_shift_offset(direction, distance_m=distance_m)
    target_T_world_cam = shift_camera_poses_local(
        source_T_world_cam_seq,
        offset_xyz_m,
    )
    target_labels = {
        "L2M": "virtual_left_2m",
        "R2M": "virtual_right_2m",
        "U2M": "virtual_up_2m",
        "D2M": "virtual_down_2m",
    }
    return {
        "prefix": direction,
        "source_view_name": "front_source",
        "target_view_name": target_labels[direction],
        "source_video": source_video,
        "target_video": None,
        "source_K": source_K_seq,
        # The virtual camera retains the real camera's intrinsics and orientation.
        "target_K": source_K_seq.clone(),
        "source_T_world_cam": source_T_world_cam_seq,
        "target_T_world_cam": target_T_world_cam,
        "source_cam_id": source_cam_id,
        "target_cam_id": f"{source_cam_id}:{target_labels[direction]}",
        "virtual_shift_local_xyz_m": [float(x) for x in offset_xyz_m],
        "virtual_shift_distance_m": float(distance_m),
    }


def select_direction_data(
    direction,
    left,
    right,
    K_left_seq,
    K_right_seq,
    T_left_seq,
    T_right_seq,
    batch,
    sample_index,
):
    """Return source/target tensors for a configured render direction."""
    direction = direction.upper()
    if direction == "LR":
        return {
            "prefix": "LR",
            "source_view_name": "left",
            "target_view_name": "right",
            "source_video": left,
            "target_video": right,
            "source_K": K_left_seq,
            "target_K": K_right_seq,
            "source_T_world_cam": T_left_seq,
            "target_T_world_cam": T_right_seq,
            "source_cam_id": batch["left_cam_id"][sample_index],
            "target_cam_id": batch["right_cam_id"][sample_index],
        }
    if direction == "RL":
        return {
            "prefix": "RL",
            "source_view_name": "right",
            "target_view_name": "left",
            "source_video": right,
            "target_video": left,
            "source_K": K_right_seq,
            "target_K": K_left_seq,
            "source_T_world_cam": T_right_seq,
            "target_T_world_cam": T_left_seq,
            "source_cam_id": batch["right_cam_id"][sample_index],
            "target_cam_id": batch["left_cam_id"][sample_index],
        }
    raise ValueError(f"Unsupported render direction: {direction}")


def build_reconstruction_dataset(config):
    """Build the configured dataset without changing the legacy stereo defaults."""
    dataset_setup = normalize_dataset_setup(
        cfg_get(config, "dataset.setup", cfg_get(config, "dataset.mode", "stereo"))
    )
    common = {
        "scene_filter": build_scene_filter(config),
        "target_steps": int(cfg_get(config, "dataset.target_steps", 40)),
        "image_size": tuple(cfg_get(config, "dataset.image_size", [800, 450])),
        "frame_sampling": int(cfg_get(config, "dataset.frame_sampling", 1)),
    }

    if dataset_setup == "monocular_shifted":
        source_cam_id = cfg_get(
            config,
            "dataset.source_cam_id",
            "camera.pcam_f0",
        )
        dataset = Py123dMonocularShiftedReconDataset(
            source_cam_id=source_cam_id,
            **common,
        )
        return dataset_setup, dataset, monocular_shifted_collate_fn

    # Existing stereo aliases and defaults are intentionally unchanged.
    left_cam_id = cfg_get(
        config,
        "dataset.source_cam_id",
        cfg_get(config, "dataset.left_cam_id", "camera.pcam_f0"),
    )
    right_cam_id = cfg_get(
        config,
        "dataset.target_cam_id",
        cfg_get(config, "dataset.right_cam_id", "camera.pcam_r0"),
    )
    dataset = Py123dPairedCameraReconDataset(
        left_cam_id=left_cam_id,
        right_cam_id=right_cam_id,
        **common,
    )
    return dataset_setup, dataset, collate_fn


class SceneIndexSubset(Dataset):
    """A contiguous or explicit scene-index view over the full py123d dataset."""

    def __init__(self, dataset, scene_indices):
        self.dataset = dataset
        self.scene_indices = [int(i) for i in scene_indices]

    def __len__(self):
        return len(self.scene_indices)

    def __getitem__(self, local_idx):
        global_scene_index = self.scene_indices[int(local_idx)]
        item = self.dataset[global_scene_index]
        item["global_scene_index"] = int(global_scene_index)
        return item


def resolve_gpu_count(config):
    """Read the requested GPU-process count from config.

    Preferred key: runtime.gpu. A root-level gpu key is also accepted for CLI/config
    convenience, e.g. gpu=2 or gpu: 2.
    """
    raw = cfg_get(config, "runtime.gpu", cfg_get(config, "gpu", 1))
    return max(1, int(raw))


def split_indices_contiguous(indices, num_shards):
    """Split indices into contiguous shards using floor boundaries.

    For n scenes and 2 GPUs this gives:
      GPU 0: [0, ..., n//2 - 1]
      GPU 1: [n//2, ..., n - 1]
    """
    indices = list(indices)
    n = len(indices)
    shards = []
    for rank in range(int(num_shards)):
        start = math.floor(rank * n / num_shards)
        end = math.floor((rank + 1) * n / num_shards)
        shards.append(indices[start:end])
    return shards


def resolve_device(config, device_override=None):
    if device_override is not None:
        return device_override

    device = cfg_get(config, "runtime.device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def make_worker_prefix(worker_id, num_workers):
    if int(num_workers) <= 1:
        return ""
    return f"[worker {worker_id}/{num_workers}] "


def sanitize_scene_log_name(log_name):
    """Make a py123d scene/log name safe to use as one folder name."""
    name = str(log_name).strip()
    if not name:
        return "unknown_scene"

    # Some log names may come from paths. Keep the information, but avoid nested
    # directories or characters that are awkward on shared filesystems.
    name = name.replace(os.sep, "_")
    if os.altsep is not None:
        name = name.replace(os.altsep, "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("._-")
    return name or "unknown_scene"


def save_camera_calibration_txt(
    path,
    role,
    scene_uuid,
    scene_log_name,
    output_scene_name,
    direction,
    camera_id,
    K_seq,
    T_world_cam_seq,
    storage_frame_indices,
    frame_sampling,
    image_width,
    image_height,
    render_coord_flip_xy,
    is_virtual_camera=False,
    parent_camera_id=None,
    virtual_shift_local_xyz_m=None,
):
    """Save a complete sampled camera-calibration sequence as readable text.

    K_seq contains resized pixel-space intrinsics. T_world_cam_seq contains the
    camera-to-world poses produced by the dataset loader, before the renderer's
    optional global XY convention flip. The inverse T_cam_world is also written
    explicitly so downstream readers do not have to guess the extrinsic direction.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    K_cpu = K_seq.detach().to(dtype=torch.float64, device="cpu")
    T_world_cam_cpu = T_world_cam_seq.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    T_cam_world_cpu = torch.linalg.inv(T_world_cam_cpu)

    num_frames = int(K_cpu.shape[0])
    if tuple(K_cpu.shape) != (num_frames, 3, 3):
        raise ValueError(f"Expected K_seq shape [F, 3, 3], got {tuple(K_cpu.shape)}")
    if tuple(T_world_cam_cpu.shape) != (num_frames, 4, 4):
        raise ValueError(
            "Expected T_world_cam_seq shape [F, 4, 4], got "
            f"{tuple(T_world_cam_cpu.shape)}"
        )
    if len(storage_frame_indices) != num_frames:
        raise ValueError(
            "storage_frame_indices and calibration sequence length differ: "
            f"{len(storage_frame_indices)} != {num_frames}"
        )
    expected_storage_indices = list(range(num_frames))
    if [int(x) for x in storage_frame_indices] != expected_storage_indices:
        raise ValueError(
            "Saved calibration indices must be contiguous from zero: "
            f"expected {expected_storage_indices}, got {storage_frame_indices}"
        )

    def matrix_rows(matrix):
        return [
            " ".join(f"{float(value):.10f}" for value in row.tolist()) for row in matrix
        ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# dataset_create camera calibration sequence\n")
        f.write(f"role: {role}\n")
        f.write(f"scene_uuid: {scene_uuid}\n")
        f.write(f"scene_log_name: {scene_log_name}\n")
        f.write(f"output_scene_name: {output_scene_name}\n")
        f.write(f"direction: {direction}\n")
        f.write(f"camera_id: {camera_id}\n")
        f.write(f"num_frames: {num_frames}\n")
        f.write(f"frame_sampling: {int(frame_sampling)}\n")
        f.write("stored_frame_index_policy: contiguous_from_zero\n")
        f.write(f"image_width: {int(image_width)}\n")
        f.write(f"image_height: {int(image_height)}\n")
        f.write("intrinsics_name: K\n")
        f.write("intrinsics_convention: resized_pixel_coordinates\n")
        f.write("extrinsics_camera_to_world_name: T_world_cam\n")
        f.write("extrinsics_world_to_camera_name: T_cam_world\n")
        f.write("matrix_layout: row_major\n")
        f.write("extrinsics_saved_before_render_coord_flip_xy: true\n")
        f.write(
            "renderer_coord_flip_xy_enabled: "
            f"{str(bool(render_coord_flip_xy)).lower()}\n"
        )
        f.write(f"is_virtual_camera: {str(bool(is_virtual_camera)).lower()}\n")
        if is_virtual_camera:
            f.write(f"parent_camera_id: {parent_camera_id}\n")
            if virtual_shift_local_xyz_m is not None:
                shift_text = " ".join(
                    f"{float(x):.10f}" for x in virtual_shift_local_xyz_m
                )
                f.write("virtual_shift_coordinate_frame: local_opencv_camera\n")
                f.write("virtual_shift_axis_convention: x_right_y_down_z_forward\n")
                f.write(f"virtual_shift_local_xyz_m: {shift_text}\n")

        for output_frame_index in range(num_frames):
            stored_frame_index = int(storage_frame_indices[output_frame_index])
            f.write("\n")
            f.write(f"[frame {stored_frame_index}]\n")
            f.write(f"output_frame_index: {stored_frame_index}\n")
            # Keep the legacy key for downstream compatibility, but make it
            # follow the same contiguous stored timeline as every image file.
            f.write(f"native_frame_index: {stored_frame_index}\n")
            f.write("K:\n")
            for row in matrix_rows(K_cpu[output_frame_index]):
                f.write(f"{row}\n")
            f.write("T_world_cam:\n")
            for row in matrix_rows(T_world_cam_cpu[output_frame_index]):
                f.write(f"{row}\n")
            f.write("T_cam_world:\n")
            for row in matrix_rows(T_cam_world_cpu[output_frame_index]):
                f.write(f"{row}\n")


class SourcePreprocessingError(RuntimeError):
    """A source-stream preprocessing failure with a machine-readable stage."""

    def __init__(self, stage, cause):
        super().__init__(str(cause))
        self.stage = str(stage)
        self.cause = cause


def source_stream_cache_key(direction_data):
    """Identify source tensors that can safely share DA3/SAM3/PCD results."""

    def tensor_identity(value):
        if not torch.is_tensor(value):
            return id(value)
        return (
            str(value.device),
            int(value.data_ptr()),
            tuple(int(x) for x in value.shape),
            tuple(int(x) for x in value.stride()),
        )

    return (
        str(direction_data["source_cam_id"]),
        tensor_identity(direction_data["source_video"]),
        tensor_identity(direction_data["source_K"]),
        tensor_identity(direction_data["source_T_world_cam"]),
    )


@torch.no_grad()
def preprocess_source_stream(
    config,
    device,
    video_predictor,
    source_video,
    source_K_seq,
    source_T_world_cam_seq,
):
    """Run all target-independent preprocessing once for one source stream."""
    _, _, H, W = source_video.shape

    da3 = None
    try:
        da3 = load_da3_model(cfg_get(config, "models.da3_path"), device)
        with tempfile.TemporaryDirectory() as tmp:
            image_paths = write_video_tensor_to_images(source_video, tmp)
            source_depth_np = run_da3_depth_with_calibration(
                da3,
                image_paths,
                source_K_seq,
                source_T_world_cam_seq,
                output_size=(H, W),
            )
    except Exception as exc:
        raise SourcePreprocessingError(
            "run_da3_depth_with_calibration", exc
        ) from exc
    finally:
        if da3 is not None:
            del da3
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sam3_prompts = get_sam3_prompts(config)
    sky_prompts = get_sam3_sky_prompts(config)

    dynamic_mask, seg_frames, instance_id_maps = (
        run_sam3_vehicle_masks_video_based(
            video_predictor=video_predictor,
            left_video=source_video,
            prompts=sam3_prompts,
            score_thres=float(cfg_get(config, "sam3.score_thres", 0.5)),
            dilate_radius=int(cfg_get(config, "sam3.mask_dilate_radius", 3)),
            prompt_frame_index=int(
                cfg_get(config, "sam3.prompt_frame_index", 0)
            ),
            propagation_direction=cfg_get(
                config, "sam3.propagation_direction", "forward"
            ),
        )
    )

    if bool(cfg_get(config, "motion_static.enabled", True)):
        static_vehicle_ids, dynamic_vehicle_ids, vehicle_motion_stats = (
            classify_static_instances_by_center_stability(
                seg_frames=seg_frames,
                depths_np=source_depth_np,
                K_seq=source_K_seq,
                T_world_cam_seq=source_T_world_cam_seq,
                bev_prompts=cfg_get(
                    config, "motion_static.bev_prompts", ["vehicle"]
                ),
                center_motion_threshold_m=float(
                    cfg_get(
                        config,
                        "motion_static.center_motion_threshold_m",
                        0.5,
                    )
                ),
                center_percentile=float(
                    cfg_get(config, "motion_static.center_percentile", 90.0)
                ),
                min_points=int(cfg_get(config, "motion_static.min_points", 50)),
                min_observations=int(
                    cfg_get(config, "motion_static.min_observations", 3)
                ),
                max_gap=cfg_get(config, "motion_static.max_gap", None),
                ignore_border_observations=bool(
                    cfg_get(
                        config,
                        "motion_static.ignore_border_observations",
                        True,
                    )
                ),
                border_margin=int(
                    cfg_get(config, "motion_static.border_margin", 8)
                ),
                max_depth=float(cfg_get(config, "render.max_depth", 300.0)),
            )
        )
        dynamic_mask, static_vehicle_mask = (
            split_dynamic_mask_using_static_vehicle_instances(
                seg_frames=seg_frames,
                original_dynamic_mask=dynamic_mask,
                static_instance_ids=static_vehicle_ids,
            )
        )
        print(
            "Static vehicle ids by center stability: "
            f"{sorted(static_vehicle_ids)} | Dynamic vehicle ids: "
            f"{sorted(dynamic_vehicle_ids)}"
        )
    else:
        static_vehicle_ids = set()
        dynamic_vehicle_ids = set()
        vehicle_motion_stats = {}
        static_vehicle_mask = torch.zeros_like(dynamic_mask, dtype=torch.bool)

    sky_instance_id_maps = None
    if len(sky_prompts) > 0:
        sky_mask_raw, _, sky_instance_id_maps = (
            run_sam3_vehicle_masks_video_based(
                video_predictor=video_predictor,
                left_video=source_video,
                prompts=sky_prompts,
                score_thres=float(
                    cfg_get(
                        config,
                        "sam3.sky_score_thres",
                        cfg_get(config, "sam3.score_thres", 0.5),
                    )
                ),
                dilate_radius=int(
                    cfg_get(config, "sam3.sky_mask_dilate_radius", 0)
                ),
                prompt_frame_index=int(
                    cfg_get(config, "sam3.prompt_frame_index", 0)
                ),
                propagation_direction=cfg_get(
                    config, "sam3.propagation_direction", "forward"
                ),
            )
        )
    else:
        sky_mask_raw = torch.zeros_like(dynamic_mask, dtype=torch.bool)

    source_hwc_u8 = (
        source_video.permute(0, 2, 3, 1)
        .mul(255)
        .clamp(0, 255)
        .to(torch.uint8)
        .to(device)
    )
    source_depth_np_sky, dyn_cpu, static_cpu, sky_mask = (
        apply_sky_handling_like_utils(
            depths_np=source_depth_np,
            dynamic_mask_t=dynamic_mask,
            sky_mask_t=sky_mask_raw,
            render_moving=bool(cfg_get(config, "render.render_moving", True)),
            sky_depth=float(cfg_get(config, "render.sky_depth", 1000.0)),
            sky_contract_radius=int(
                cfg_get(config, "sam3.sky_contract_radius", 6)
            ),
            sky_contract_iterations=int(
                cfg_get(config, "sam3.sky_contract_iterations", 3)
            ),
        )
    )

    dynamic_instance_id_maps = build_dynamic_instance_id_maps(
        instance_id_maps=instance_id_maps,
        dynamic_mask=dyn_cpu,
    )
    seginstance_object_mode = (
        str(
            cfg_get(
                config, "render.seginstance_object_mode", "same_as_warped"
            )
        )
        .strip()
        .lower()
    )
    seginstance_id_maps = build_seginstance_id_maps(
        instance_id_maps=instance_id_maps,
        dynamic_mask=dyn_cpu,
        static_mask=static_cpu,
        sky_mask=sky_mask,
        object_mode=seginstance_object_mode,
    )

    source_depth = torch.from_numpy(source_depth_np_sky).float().to(device)
    source_K = source_K_seq.float().to(device)
    source_T_world_cam = apply_render_pose_convention(
        source_T_world_cam_seq.float().to(device),
        enabled=bool(cfg_get(config, "render.coord_flip_xy", False)),
    )
    color, xyz, visible, indices, point_dynamic = unproject(
        video=source_hwc_u8,
        depths=source_depth,
        cam_c2w=source_T_world_cam,
        K=source_K,
        dynamic_mask=dyn_cpu.to(device),
        static_mask=static_cpu.to(device),
        pixel_center_offset=bool(
            cfg_get(config, "render.pixel_center_offset", False)
        ),
        max_depth=cfg_get(config, "render.max_depth", 300.0),
    )

    dynamic_instance_id_maps = dynamic_instance_id_maps.to(
        device=device, dtype=torch.int64
    )
    seginstance_id_maps = seginstance_id_maps.to(
        device=device, dtype=torch.int64
    )
    point_instance_ids = seginstance_id_maps[
        indices[:, 0].long(), indices[:, 1].long(), indices[:, 2].long()
    ]

    return {
        "source_depth_np": source_depth_np,
        "sam3_prompts": sam3_prompts,
        "sky_prompts": sky_prompts,
        "dynamic_mask": dynamic_mask,
        "instance_id_maps": instance_id_maps,
        "sky_instance_id_maps": sky_instance_id_maps,
        "static_vehicle_ids": static_vehicle_ids,
        "dynamic_vehicle_ids": dynamic_vehicle_ids,
        "vehicle_motion_stats": vehicle_motion_stats,
        "static_vehicle_mask": static_vehicle_mask,
        "dyn_cpu": dyn_cpu,
        "sky_mask": sky_mask,
        "dynamic_instance_id_maps": dynamic_instance_id_maps,
        "seginstance_object_mode": seginstance_object_mode,
        "seginstance_id_maps": seginstance_id_maps,
        "source_depth": source_depth,
        "source_K": source_K,
        "source_T_world_cam": source_T_world_cam,
        "color": color,
        "xyz": xyz,
        "visible": visible,
        "indices": indices,
        "point_dynamic": point_dynamic,
        "point_instance_ids": point_instance_ids,
        "source_dynamic_mask": (dynamic_mask & ~sky_mask).to(device),
    }


@torch.no_grad()
def run_pipeline(
    config, device_override=None, scene_indices=None, worker_id=0, num_workers=1
):
    fix_seed(int(cfg_get(config, "runtime.seed", 42)))
    configure_data_roots(config)

    device = resolve_device(config, device_override=device_override)

    worker_prefix = make_worker_prefix(worker_id, num_workers)
    dataset_setup, dataset, active_collate_fn = build_reconstruction_dataset(config)
    out = make_output_dirs(config)
    visualization = get_visualization_options(config)
    visualization_root = (
        Path(cfg_get(config, "paths.output_dir")) / visualization["output_subdir"]
    )
    batch_filter = get_batch_filter(config)
    double_reprojection_enabled = bool(
        cfg_get(config, "double_reprojection.enabled", False)
    )
    render_directions = normalize_render_directions(
        cfg_get(
            config,
            "dataset.render_directions",
            cfg_get(config, "dataset.render_direction", None),
        ),
        dataset_setup=dataset_setup,
        include_down_shift=double_reprojection_enabled,
    )
    print(f"{worker_prefix}Dataset setup: {dataset_setup}")
    print(f"{worker_prefix}Render directions: {render_directions}")
    print(
        f"{worker_prefix}Double reprojection: "
        f"{'enabled' if double_reprojection_enabled else 'disabled'}"
    )

    if scene_indices is None:
        if batch_filter is None:
            selected_scene_indices = list(range(len(dataset)))
        else:
            selected_scene_indices = [
                i
                for i in sorted(int(x) for x in batch_filter)
                if 0 <= int(i) < len(dataset)
            ]
    else:
        selected_scene_indices = [
            int(i) for i in scene_indices if 0 <= int(i) < len(dataset)
        ]

    if len(selected_scene_indices) == 0:
        print(f"{worker_prefix}No scenes assigned; exiting.")
        return

    worker_dataset = SceneIndexSubset(dataset, selected_scene_indices)

    # Multi-GPU mode already partitions the dataset explicitly, so keep dataloader order stable.
    shuffle_dataloader = bool(cfg_get(config, "dataset.shuffle_dataloader", False))
    if int(num_workers) > 1 and shuffle_dataloader:
        print(
            f"{worker_prefix}dataset.shuffle_dataloader=true ignored in multi-GPU mode "
            "to preserve contiguous scene shards."
        )
        shuffle_dataloader = False

    loader = DataLoader(
        worker_dataset,
        batch_size=int(cfg_get(config, "dataset.batch_size", 1)),
        shuffle=shuffle_dataloader,
        collate_fn=active_collate_fn,
    )

    print(
        f"{worker_prefix}Device: {device} | scenes: "
        f"{selected_scene_indices[0]}..{selected_scene_indices[-1]} "
        f"({len(selected_scene_indices)} scenes)"
    )
    print(f"{worker_prefix}Loading official SAM3 video predictor...")
    video_predictor = load_sam3_video_predictor(
        checkpoint_path=cfg_get(config, "models.sam3_checkpoint_path")
    )
    for batch_idx, batch in enumerate(loader):
        if len(batch["unified_trajectory"][0]) == 1:  # required camera data missing
            continue
        primary_video_key = (
            "rgb_video_source"
            if dataset_setup == "monocular_shifted"
            else "rgb_video_left"
        )
        for b in range(batch[primary_video_key].shape[0]):
            scene_uuid = str(batch["scene_uuid"][b])
            scene_log_name = str(batch.get("scene_log_name", batch["scene_uuid"])[b])
            scene_output_base_name = sanitize_scene_log_name(scene_log_name)
            global_scene_index = batch.get(
                "global_scene_index", [None] * len(batch["scene_uuid"])
            )[b]
            print(
                f"\n{worker_prefix}Processing scene: {scene_log_name} "
                f"(uuid={scene_uuid}, scene_index={global_scene_index})"
            )
            if scene_output_base_name != scene_log_name:
                print(
                    f"{worker_prefix}Output folder scene name sanitized to: "
                    f"{scene_output_base_name}"
                )
            if dataset_setup == "monocular_shifted":
                source_base = batch["rgb_video_source"][b]
                K_source_base = batch["K_source"][b]
                T_source_base = batch["T_world_cam_source"][b]
            else:
                left_base = batch["rgb_video_left"][b]
                right_base = batch["rgb_video_right"][b]
                K_left_base = batch["K_left"][b]
                K_right_base = batch["K_right"][b]
                T_left_base = batch["T_world_cam_left"][b]
                T_right_base = batch["T_world_cam_right"][b]

            direction_items = []
            for direction in render_directions:
                if dataset_setup == "monocular_shifted":
                    direction_data = select_monocular_shifted_data(
                        direction=direction,
                        source_video=source_base,
                        source_K_seq=K_source_base,
                        source_T_world_cam_seq=T_source_base,
                        source_cam_id=batch["source_cam_id"][b],
                        distance_m=float(
                            cfg_get(config, "monocular_shift.distance_m", 2.0)
                        ),
                    )
                else:
                    direction_data = select_direction_data(
                        direction=direction,
                        left=left_base,
                        right=right_base,
                        K_left_seq=K_left_base,
                        K_right_seq=K_right_base,
                        T_left_seq=T_left_base,
                        T_right_seq=T_right_base,
                        batch=batch,
                        sample_index=b,
                    )
                direction_items.append(direction_data)

            source_preprocess_cache = {}
            source_uses_remaining = {}
            for direction_data in direction_items:
                cache_key = source_stream_cache_key(direction_data)
                source_uses_remaining[cache_key] = (
                    source_uses_remaining.get(cache_key, 0) + 1
                )

            for direction_data in direction_items:
                source_cache_key = source_stream_cache_key(direction_data)
                direction_prefix = direction_data["prefix"]
                scene_output_name = f"{direction_prefix}_{scene_output_base_name}"
                source_view_name = direction_data["source_view_name"]
                target_view_name = direction_data["target_view_name"]
                source_cam_id = direction_data["source_cam_id"]
                target_cam_id = direction_data["target_cam_id"]
                virtual_shift_local_xyz_m = direction_data.get(
                    "virtual_shift_local_xyz_m"
                )
                virtual_shift_distance_m = direction_data.get(
                    "virtual_shift_distance_m"
                )

                print(
                    f"{worker_prefix}Processing direction {direction_prefix}: "
                    f"{source_cam_id} -> {target_cam_id}"
                )

                # Keep the original rendering variable names: left means source;
                # right is target RGB for stereo and None for a virtual target.
                left = direction_data["source_video"]
                right = direction_data["target_video"]
                K_left_seq = direction_data["source_K"]
                K_right_seq = direction_data["target_K"]
                T_left_seq = direction_data["source_T_world_cam"]
                T_right_seq = direction_data["target_T_world_cam"]

                F = left.shape[0]
                H, W = left.shape[2], left.shape[3]
                native_sampled_frame_indices = (
                    batch["sampled_frame_indices"][b].detach().cpu().tolist()
                )
                if len(native_sampled_frame_indices) != F:
                    raise ValueError(
                        "Loaded frame count and sampled-index count differ: "
                        f"{F} != {len(native_sampled_frame_indices)}"
                    )
                storage_frame_indices = make_storage_frame_indices(
                    native_sampled_frame_indices
                )
                frame_sampling = int(batch["frame_sampling"][b])

                source_calibration_relpath = None
                target_calibration_relpath = None
                if out["calibration"] is not None:
                    calibration_scene_dir = out["calibration"] / scene_output_name
                    source_calibration_path = (
                        calibration_scene_dir / "source_calibration.txt"
                    )
                    target_calibration_path = (
                        calibration_scene_dir / "target_calibration.txt"
                    )
                    render_coord_flip_xy = bool(
                        cfg_get(config, "render.coord_flip_xy", False)
                    )
                    save_camera_calibration_txt(
                        path=source_calibration_path,
                        role="source",
                        scene_uuid=scene_uuid,
                        scene_log_name=scene_log_name,
                        output_scene_name=scene_output_name,
                        direction=direction_prefix,
                        camera_id=source_cam_id,
                        K_seq=K_left_seq,
                        T_world_cam_seq=T_left_seq,
                        storage_frame_indices=storage_frame_indices,
                        frame_sampling=frame_sampling,
                        image_width=W,
                        image_height=H,
                        render_coord_flip_xy=render_coord_flip_xy,
                    )
                    save_camera_calibration_txt(
                        path=target_calibration_path,
                        role=(
                            "target_virtual"
                            if dataset_setup == "monocular_shifted"
                            else "target"
                        ),
                        scene_uuid=scene_uuid,
                        scene_log_name=scene_log_name,
                        output_scene_name=scene_output_name,
                        direction=direction_prefix,
                        camera_id=target_cam_id,
                        K_seq=K_right_seq,
                        T_world_cam_seq=T_right_seq,
                        storage_frame_indices=storage_frame_indices,
                        frame_sampling=frame_sampling,
                        image_width=W,
                        image_height=H,
                        render_coord_flip_xy=render_coord_flip_xy,
                        is_virtual_camera=(
                            dataset_setup == "monocular_shifted"
                        ),
                        parent_camera_id=(
                            source_cam_id
                            if dataset_setup == "monocular_shifted"
                            else None
                        ),
                        virtual_shift_local_xyz_m=virtual_shift_local_xyz_m,
                    )
                    calibration_subdir = Path(
                        cfg_get(config, "paths.calibration_subdir", "calibration")
                    )
                    source_calibration_relpath = str(
                        (
                            calibration_subdir
                            / scene_output_name
                            / "source_calibration.txt"
                        ).as_posix()
                    )
                    target_calibration_relpath = str(
                        (
                            calibration_subdir
                            / scene_output_name
                            / "target_calibration.txt"
                        ).as_posix()
                    )

                source_preprocessing_reused = (
                    source_cache_key in source_preprocess_cache
                )
                if not source_preprocessing_reused:
                    print(
                        f"{worker_prefix}Preprocessing source stream once: "
                        f"{source_cam_id}"
                    )
                    try:
                        source_preprocess_cache[source_cache_key] = (
                            preprocess_source_stream(
                                config=config,
                                device=device,
                                video_predictor=video_predictor,
                                source_video=left,
                                source_K_seq=K_left_seq,
                                source_T_world_cam_seq=T_left_seq,
                            )
                        )
                    except SourcePreprocessingError as exc:
                        source_preprocess_cache[source_cache_key] = {
                            "failed": True,
                            "failed_stage": exc.stage,
                            "error_type": type(exc.cause).__name__,
                            "error_message": str(exc.cause),
                            "traceback": traceback.format_exc(),
                        }
                else:
                    print(
                        f"{worker_prefix}Reusing cached source preprocessing: "
                        f"{source_cam_id} for {direction_prefix}"
                    )

                source_state = source_preprocess_cache[source_cache_key]
                if source_state.get("failed", False):
                    print(
                        f"{worker_prefix}WARNING: source preprocessing failed for "
                        f"scene={scene_log_name} direction={direction_prefix} "
                        f"batch_idx={batch_idx} sample_idx={b}. "
                        "Skipping this item and continuing."
                    )
                    print(
                        f"{worker_prefix}{source_state['failed_stage']} error: "
                        f"{source_state['error_type']}: "
                        f"{source_state['error_message']}"
                    )
                    if bool(cfg_get(config, "runtime.print_da3_traceback", True)):
                        print(source_state["traceback"])

                    if out["metadata"] is not None:
                        metadata_scene_dir = out["metadata"] / scene_output_name
                        metadata_scene_dir.mkdir(parents=True, exist_ok=True)
                        failure_meta = {
                            "scene": {
                                "uuid": scene_uuid,
                                "log_name": scene_log_name,
                            },
                            "scene_uuid": scene_uuid,
                            "scene_log_name": scene_log_name,
                            "scene_output_base_name": scene_output_base_name,
                            "output_scene_name": scene_output_name,
                            "dataset_setup": dataset_setup,
                            "worker_id": int(worker_id),
                            "num_workers": int(num_workers),
                            "global_scene_index": (
                                None
                                if global_scene_index is None
                                else int(global_scene_index)
                            ),
                            "direction": direction_prefix,
                            "source_view": source_view_name,
                            "target_view": target_view_name,
                            "batch_idx": int(batch_idx),
                            "sample_idx": int(b),
                            "frame_sampling": frame_sampling,
                            "stored_frame_index_policy": "contiguous_from_zero",
                            "sampled_frame_indices": [
                                int(x) for x in storage_frame_indices
                            ],
                            "source_cam_id": source_cam_id,
                            "target_cam_id": target_cam_id,
                            "target_is_virtual": (
                                dataset_setup == "monocular_shifted"
                            ),
                            "virtual_shift_local_xyz_m": virtual_shift_local_xyz_m,
                            "virtual_shift_distance_m": virtual_shift_distance_m,
                            "status": "skipped",
                            "failed_stage": source_state["failed_stage"],
                            "error_type": source_state["error_type"],
                            "error_message": source_state["error_message"],
                        }
                        with (
                            metadata_scene_dir
                            / f"batch_{batch_idx:04d}_sample_{b:02d}_{direction_prefix}_source_preprocessing_failed.json"
                        ).open("w", encoding="utf-8") as f:
                            json.dump(failure_meta, f, indent=2)

                    source_uses_remaining[source_cache_key] -= 1
                    if source_uses_remaining[source_cache_key] == 0:
                        del source_preprocess_cache[source_cache_key]
                        del source_uses_remaining[source_cache_key]
                    continue

                left_depth_np = source_state["source_depth_np"]
                sam3_prompts = source_state["sam3_prompts"]
                sky_prompts = source_state["sky_prompts"]
                dynamic_mask = source_state["dynamic_mask"]
                instance_id_maps = source_state["instance_id_maps"]
                sky_instance_id_maps = source_state["sky_instance_id_maps"]
                static_vehicle_ids = source_state["static_vehicle_ids"]
                dynamic_vehicle_ids = source_state["dynamic_vehicle_ids"]
                vehicle_motion_stats = source_state["vehicle_motion_stats"]
                static_vehicle_mask = source_state["static_vehicle_mask"]
                dyn_cpu = source_state["dyn_cpu"]
                sky_mask = source_state["sky_mask"]
                dynamic_instance_id_maps = source_state[
                    "dynamic_instance_id_maps"
                ]
                seginstance_object_mode = source_state[
                    "seginstance_object_mode"
                ]
                seginstance_id_maps = source_state["seginstance_id_maps"]
                left_depth = source_state["source_depth"]
                K_left = source_state["source_K"]
                T_left = source_state["source_T_world_cam"]
                color = source_state["color"]
                xyz = source_state["xyz"]
                visible = source_state["visible"]
                indices = source_state["indices"]
                point_dynamic = source_state["point_dynamic"]
                point_instance_ids = source_state["point_instance_ids"]
                source_dynamic_mask = source_state["source_dynamic_mask"]

                K_right = K_right_seq.float().to(device)
                T_right = apply_render_pose_convention(
                    T_right_seq.float().to(device),
                    enabled=bool(cfg_get(config, "render.coord_flip_xy", False)),
                )

                if out["metadata"] is not None:
                    save_global_pointcloud_ply(
                        out["metadata"]
                        / scene_output_name
                        / f"batch_{batch_idx:04d}_sample_{b:02d}_{direction_prefix}_source_global.ply",
                        video_chw=left,
                        depths_np=left_depth_np,
                        K_seq=K_left_seq,
                        T_world_cam_seq=T_left_seq,
                        stride=2,
                        max_depth=cfg_get(config, "render.max_depth", 300.0),
                    )
                    save_instance_id_overlay_frames(
                        left_video=left,
                        instance_id_maps=instance_id_maps,
                        out_dir=out["metadata"]
                        / scene_output_name
                        / f"batch_{batch_idx:04d}_sample_{b:02d}_{direction_prefix}_sam3_dynamic_instances",
                    )
                    if sky_instance_id_maps is not None:
                        save_instance_id_overlay_frames(
                            left_video=left,
                            instance_id_maps=sky_instance_id_maps,
                            out_dir=out["metadata"]
                            / scene_output_name
                            / f"batch_{batch_idx:04d}_sample_{b:02d}_{direction_prefix}_sam3_sky_instances",
                        )
                visibility_mode = cfg_get(
                    config,
                    "render.visibility_mode",
                    "target_frustum_static_dynamic_same_frame",
                )
                if bool(cfg_get(config, "render.no_temporal_persistence", False)):
                    visibility_mode = "same_frame"

                render_visible = build_pointcloud_visibility(
                    source_visible=visible,
                    indices=indices,
                    point_dynamic=point_dynamic,
                    points_world=xyz,
                    target_cam_c2w=T_right,
                    target_K=K_right,
                    H=H,
                    W=W,
                    mode=visibility_mode,
                )
                render_outputs = render(
                    points_color=color.float() / 255.0,
                    points_pos=xyz,
                    visible=render_visible,
                    indices=indices,
                    cam_c2w=T_right,
                    K=K_right,
                    H=H,
                    W=W,
                    # Always provide point motion labels: the second pass needs
                    # them even when dynamic-mask visualization is disabled.
                    dynamic_mask=point_dynamic,
                    verbose=bool(cfg_get(config, "runtime.verbose_render", False)),
                    z_tolerance=float(cfg_get(config, "render.z_tolerance", 0.10)),
                    occlusion_sharpness=float(
                        cfg_get(config, "render.occlusion_sharpness", 10.0)
                    ),
                    batch_size=int(cfg_get(config, "render.batch_size", 10000000)),
                    source_cam_c2w=T_left,
                    source_K=K_left,
                    source_depth=left_depth,
                    # Keep the semantic motion mask for hole classification even
                    # when render.render_moving=false removes moving points from
                    # the local PCD.
                    source_dynamic_mask=source_dynamic_mask,
                    global_fill_static_holes_only=bool(
                        cfg_get(config, "render.global_fill_static_holes_only", True)
                    ),
                    dynamic_hole_depth_abs_tolerance=float(
                        cfg_get(
                            config,
                            "render.dynamic_hole_depth_abs_tolerance",
                            0.10,
                        )
                    ),
                    dynamic_hole_depth_rel_tolerance=float(
                        cfg_get(
                            config,
                            "render.dynamic_hole_depth_rel_tolerance",
                            0.02,
                        )
                    ),
                    dynamic_hole_mask_dilation_radius=int(
                        cfg_get(
                            config,
                            "render.dynamic_hole_mask_dilation_radius",
                            0,
                        )
                    ),
                    invalid_dynamic_depth_is_occluded=bool(
                        cfg_get(
                            config,
                            "render.invalid_dynamic_depth_is_occluded",
                            True,
                        )
                    ),
                    return_hole_masks=True,
                    return_local_render=False,
                    return_local_correspondence=False,
                    local_only=double_reprojection_enabled,
                )
                (
                    warped,
                    depths_pc,
                    coverage,
                    dyn_pc,
                    static_hole_mask,
                    dynamic_hole_mask,
                ) = render_outputs[:6]

                first_warped_ntp = None
                first_depth_ntp = None
                first_coverage_ntp = None
                first_dynamic_ntp = None
                source_frame_visibility_counts = None
                dedup_rankings = None
                if double_reprojection_enabled:
                    # local_only=True makes these the current-frame-only
                    # virtual render. After its dense return, the global PCD is
                    # splatted directly into remaining source-camera holes.
                    first_warped_ntp = warped
                    first_depth_ntp = depths_pc
                    first_coverage_ntp = coverage
                    first_dynamic_ntp = dyn_pc

                    (
                        local_return_rgb,
                        local_return_depth,
                        local_return_coverage,
                        local_return_dynamic,
                    ) = gather_virtual_render_to_source_grid(
                        first_warped=first_warped_ntp,
                        first_depth=first_depth_ntp,
                        first_coverage=first_coverage_ntp,
                        points_pos=xyz,
                        indices=indices,
                        virtual_cam_c2w=T_right,
                        virtual_K=K_right,
                        source_depth=left_depth,
                        source_dynamic_mask=source_dynamic_mask,
                        point_visibility=render_visible,
                        z_tolerance=float(
                            cfg_get(config, "render.z_tolerance", 0.10)
                        ),
                    )

                    global_return_visible = build_pointcloud_visibility(
                        source_visible=visible,
                        indices=indices,
                        point_dynamic=point_dynamic,
                        points_world=xyz,
                        target_cam_c2w=T_left,
                        target_K=K_left,
                        H=H,
                        W=W,
                        mode=visibility_mode,
                    )
                    allowed_global_return = (
                        ~local_return_coverage & ~source_dynamic_mask
                    )
                    global_return_outputs = (
                        splat_global_static_points_to_source_holes(
                            points_color=color.float() / 255.0,
                            points_pos=xyz,
                            visible=global_return_visible,
                            indices=indices,
                            point_dynamic=point_dynamic,
                            source_cam_c2w=T_left,
                            source_K=K_left,
                            source_depth=left_depth,
                            source_dynamic_mask=source_dynamic_mask,
                            allowed_return_mask=allowed_global_return,
                            z_tolerance=float(
                                cfg_get(config, "render.z_tolerance", 0.10)
                            ),
                            occlusion_sharpness=float(
                                cfg_get(config, "render.occlusion_sharpness", 10.0)
                            ),
                            batch_size=int(
                                cfg_get(config, "render.batch_size", 10000000)
                            ),
                            dynamic_hole_depth_abs_tolerance=float(
                                cfg_get(
                                    config,
                                    "render.dynamic_hole_depth_abs_tolerance",
                                    0.10,
                                )
                            ),
                            dynamic_hole_depth_rel_tolerance=float(
                                cfg_get(
                                    config,
                                    "render.dynamic_hole_depth_rel_tolerance",
                                    0.02,
                                )
                            ),
                            dynamic_hole_mask_dilation_radius=int(
                                cfg_get(
                                    config,
                                    "render.dynamic_hole_mask_dilation_radius",
                                    0,
                                )
                            ),
                            invalid_dynamic_depth_is_occluded=bool(
                                cfg_get(
                                    config,
                                    "render.invalid_dynamic_depth_is_occluded",
                                    True,
                                )
                            ),
                            return_source_frame_provenance=(
                                out["top20_reference"] is not None
                            ),
                        )
                    )
                    (
                        global_return_rgb,
                        global_return_depth,
                        global_return_coverage,
                        global_return_blocked_dynamic,
                    ) = global_return_outputs[:4]

                    if out["top20_reference"] is not None:
                        global_return_source_frame = global_return_outputs[4]
                        source_frame_visibility_counts = (
                            compute_double_reprojection_source_frame_visibility_counts(
                                local_return_coverage=local_return_coverage,
                                global_return_coverage=global_return_coverage,
                                global_return_source_frame=(
                                    global_return_source_frame
                                ),
                            )
                        )
                        if bool(cfg_get(config, "reference.dedup_enabled", False)):
                            tile_counts = tile_vote_counts(
                                local_return_coverage,
                                global_return_coverage,
                                global_return_source_frame,
                            )
                            dedup_rankings = {
                                tenth: dedup_ranks(
                                    source_frame_visibility_counts.numpy(),
                                    tile_counts,
                                    tenth,
                                    max(1, int(cfg_get(config, "reference.top_k", 20))),
                                )
                                for tenth in (3, 5, 7)
                            }

                    warped = torch.where(
                        local_return_coverage[..., None],
                        local_return_rgb,
                        global_return_rgb,
                    )
                    depths_pc = torch.where(
                        local_return_coverage,
                        local_return_depth,
                        global_return_depth,
                    )
                    coverage = (
                        local_return_coverage | global_return_coverage
                    )
                    dyn_pc = local_return_dynamic

                    return_holes = ~coverage
                    dynamic_hole_mask = return_holes & (
                        source_dynamic_mask | global_return_blocked_dynamic
                    )
                    static_hole_mask = return_holes & ~dynamic_hole_mask

                    if dedup_rankings is not None:
                        for trace_frame in cfg_get(
                            config, "reference.trace_target_frames", []
                        ):
                            save_reference_trace(
                                output_root=cfg_get(config, "paths.output_dir"),
                                scene_output_name=scene_output_name,
                                target_index=trace_frame,
                                source_video=left,
                                virtual_rgb=first_warped_ntp,
                                virtual_coverage=first_coverage_ntp,
                                local_rgb=local_return_rgb,
                                local_coverage=local_return_coverage,
                                global_rgb=global_return_rgb,
                                global_coverage=global_return_coverage,
                                final_rgb=warped,
                                final_coverage=coverage,
                                source_dynamic_mask=source_dynamic_mask,
                                global_source_frame=global_return_source_frame,
                                global_counts=source_frame_visibility_counts.numpy(),
                                tile_counts=tile_counts,
                                top_k=cfg_get(config, "reference.top_k", 20),
                            )

                    for trace_frame in cfg_get(
                        config, "reference.union_trace_target_frames", []
                    ):
                        if source_frame_visibility_counts is None:
                            raise ValueError(
                                "Union reference trace requires top20_reference output"
                            )
                        save_union_selection_trace(
                            output_root=cfg_get(config, "paths.output_dir"),
                            scene_output_name=scene_output_name,
                            target_index=trace_frame,
                            source_video=left,
                            points_pos=xyz,
                            indices=indices,
                            point_dynamic=point_dynamic,
                            target_visibility=global_return_visible,
                            target_cam_c2w=T_left,
                            target_K=K_left,
                            source_depth=left_depth,
                            source_dynamic_mask=source_dynamic_mask,
                            local_return_coverage=local_return_coverage,
                            local_return_depth=local_return_depth,
                            global_return_depth=global_return_depth,
                            final_coverage=coverage,
                            global_counts=source_frame_visibility_counts.numpy(),
                            top_k=cfg_get(config, "reference.top_k", 20),
                            select_k=8,
                            z_tolerance=cfg_get(config, "render.z_tolerance", 0.10),
                            occlusion_sharpness=cfg_get(
                                config, "render.occlusion_sharpness", 10.0
                            ),
                            batch_size=cfg_get(
                                config, "render.batch_size", 10000000
                            ),
                        )

                    del local_return_rgb, local_return_depth
                    del local_return_coverage, local_return_dynamic
                    del global_return_rgb, global_return_depth
                    del global_return_coverage, global_return_blocked_dynamic
                    del global_return_outputs
                    if out["top20_reference"] is not None:
                        del global_return_source_frame
                    del global_return_visible, allowed_global_return, return_holes
                del render_outputs

                if (
                    out["top20_reference"] is not None
                    and not double_reprojection_enabled
                ):
                    source_frame_visibility_counts = (
                        compute_source_frame_visibility_counts(
                            points_source_frame=indices[:, 0].long(),
                            points_pos=xyz,
                            visible=render_visible,
                            cam_c2w=T_right,
                            K=K_right,
                            H=H,
                            W=W,
                            verbose=bool(
                                cfg_get(
                                    config, "runtime.verbose_reference_ranking", False
                                )
                            ),
                            z_tolerance=float(
                                cfg_get(config, "render.z_tolerance", 0.10)
                            ),
                            occlusion_sharpness=float(
                                cfg_get(config, "render.occlusion_sharpness", 10.0)
                            ),
                            batch_size=int(
                                cfg_get(config, "render.batch_size", 10000000)
                            ),
                        )
                    )

                seginstance_warped = None
                seginstance_visible = None
                seginstance_visibility_mode = None
                seginstance_temporal_persistence = False

                # Preferred config location:
                #   render.seginstance_visibility_mode
                # Backward compatible fallback for older ZIPs/overrides:
                #   seginstance_warped.visibility_mode
                seginstance_max_past_frames = cfg_get(
                    config,
                    "render.seginstance_max_past_frames",
                    cfg_get(config, "seginstance_warped.max_past_frames", None),
                )

                if out["seginstance_warped"] is not None or visualization["enabled"]:
                    if double_reprojection_enabled:
                        seginstance_visibility_mode = "source_coordinate_visibility"
                        seginstance_temporal_persistence = False
                        seginstance_visible = torch.zeros_like(visible)
                        seginstance_visible.scatter_(1, indices[:, 0:1], True)
                        seginstance_warped = seginstance_id_maps.clone()
                        seginstance_warped[~coverage] = 0
                    else:
                        seginstance_visibility_mode = cfg_get(
                            config,
                            "render.seginstance_visibility_mode",
                            cfg_get(
                                config,
                                "seginstance_warped.visibility_mode",
                                "same_as_warped",
                            ),
                        )
                        seginstance_visibility_mode = (
                            str(seginstance_visibility_mode).strip().lower()
                        )
                        seginstance_temporal_persistence = (
                            seginstance_visibility_mode
                            not in {
                                "same_frame",
                                "sameframe",
                            }
                        )

                        if seginstance_visibility_mode in {
                            "same_as_warped",
                            "same_as_rgb",
                            "warped",
                            "rgb",
                        }:
                            seginstance_visible = render_visible
                        else:
                            seginstance_visible = build_temporal_instance_visibility(
                                source_visible=visible,
                                indices=indices,
                                points_world=xyz,
                                target_cam_c2w=T_right,
                                target_K=K_right,
                                H=H,
                                W=W,
                                mode=seginstance_visibility_mode,
                                max_past_frames=seginstance_max_past_frames,
                            )

                        seginstance_warped = render_instance_ids(
                            points_instance_id=point_instance_ids,
                            points_pos=xyz,
                            visible=seginstance_visible,
                            cam_c2w=T_right,
                            K=K_right,
                            H=H,
                            W=W,
                            verbose=bool(
                                cfg_get(
                                    config,
                                    "runtime.verbose_render_instance",
                                    False,
                                )
                            ),
                            z_tolerance=float(
                                cfg_get(config, "render.z_tolerance", 0.10)
                            ),
                            occlusion_sharpness=float(
                                cfg_get(
                                    config,
                                    "render.occlusion_sharpness",
                                    10.0,
                                )
                            ),
                            batch_size=int(
                                cfg_get(config, "render.batch_size", 10000000)
                            ),
                        )

                if visualization["enabled"]:
                    visualization_path = (
                        visualization_root / f"{scene_output_name}.mp4"
                    )
                    save_visualization_grid_video(
                        path=visualization_path,
                        source_video=left,
                        warped_video=warped,
                        coverage_video=coverage,
                        dynamic_instance_id_maps=dynamic_instance_id_maps,
                        seginstance_warped=seginstance_warped,
                        fps=visualization["fps"],
                        overlay_alpha=visualization["overlay_alpha"],
                        draw_ids=visualization["draw_ids"],
                    )
                    print(
                        f"{worker_prefix}Saved visualization video: "
                        f"{visualization_path}"
                    )

                for i in trange(F, desc="Saving"):
                    stored_frame_index = int(storage_frame_indices[i])
                    stem = f"{stored_frame_index:05d}"

                    source_rgb = tensor_to_rgb_numpy(left[i]).astype("float32") / 255.0
                    target_rgb = (
                        None
                        if right is None
                        else tensor_to_rgb_numpy(right[i]).astype("float32") / 255.0
                    )
                    warped_rgb = warped[i].detach().cpu().numpy().astype("float32")
                    alpha = coverage[i].detach().cpu().numpy().astype("float32")

                    os.makedirs(out["source"] / scene_output_name, exist_ok=True)
                    os.makedirs(out["warped"] / scene_output_name, exist_ok=True)
                    os.makedirs(out["coverage"] / scene_output_name, exist_ok=True)
                    save_rgb(
                        out["source"] / scene_output_name / f"{stem}.png", source_rgb
                    )
                    if out["target"] is not None and target_rgb is not None:
                        os.makedirs(
                            out["target"] / scene_output_name,
                            exist_ok=True,
                        )
                        save_rgb(
                            out["target"] / scene_output_name / f"{stem}.png",
                            target_rgb,
                        )
                    save_rgb(
                        out["warped"] / scene_output_name / f"{stem}.png", warped_rgb
                    )

                    cv2.imwrite(
                        str(out["coverage"] / scene_output_name / f"{stem}.png"),
                        (alpha * 255).clip(0, 255).astype("uint8"),
                    )
                    if out["static_holes"] is not None:
                        os.makedirs(
                            out["static_holes"] / scene_output_name, exist_ok=True
                        )
                        cv2.imwrite(
                            str(
                                out["static_holes"]
                                / scene_output_name
                                / f"{stem}.png"
                            ),
                            static_hole_mask[i]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype("uint8")
                            * 255,
                        )
                    if out["dynamic_holes"] is not None:
                        os.makedirs(
                            out["dynamic_holes"] / scene_output_name, exist_ok=True
                        )
                        cv2.imwrite(
                            str(
                                out["dynamic_holes"]
                                / scene_output_name
                                / f"{stem}.png"
                            ),
                            dynamic_hole_mask[i]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype("uint8")
                            * 255,
                        )
                    metadata_scene_dir = None
                    if out["metadata"] is not None:
                        metadata_scene_dir = out["metadata"] / scene_output_name
                        metadata_scene_dir.mkdir(parents=True, exist_ok=True)

                    seginstance_warped_overlay_enabled = bool(
                        out["metadata"] is not None
                        and out["seginstance_warped"] is not None
                        and cfg_get(config, "save.seginstance_warped_overlay", True)
                    )
                    seginstance_warped_overlay_background = (
                        str(
                            cfg_get(
                                config,
                                "metadata.seginstance_warped_overlay_background",
                                "warped",
                            )
                        )
                        .strip()
                        .lower()
                    )
                    seginstance_warped_overlay_relpath = None

                    if out["seginstance_warped"] is not None:
                        os.makedirs(
                            out["seginstance_warped"] / scene_output_name,
                            exist_ok=True,
                        )
                        seg_ids_np = (
                            seginstance_warped[i]
                            .detach()
                            .cpu()
                            .numpy()
                            .clip(0, 65535)
                            .astype("uint16")
                        )
                        cv2.imwrite(
                            str(
                                out["seginstance_warped"]
                                / scene_output_name
                                / f"{stem}.png"
                            ),
                            seg_ids_np,
                        )

                        if seginstance_warped_overlay_enabled:
                            overlay_subdir = cfg_get(
                                config,
                                "paths.seginstance_warped_overlay_subdir",
                                "seginstance_warped_overlay",
                            )
                            overlay_dir = metadata_scene_dir / overlay_subdir
                            overlay_path = overlay_dir / f"{stem}.png"
                            if seginstance_warped_overlay_background in {
                                "target",
                                "target_rgb",
                                "front_target",
                            } and target_rgb is not None:
                                overlay_background = target_rgb
                                seginstance_warped_overlay_background = "target"
                            else:
                                overlay_background = warped_rgb
                                seginstance_warped_overlay_background = "warped"
                            save_instance_id_overlay_image(
                                overlay_path,
                                rgb=overlay_background,
                                instance_id_map=seg_ids_np,
                                alpha=float(
                                    cfg_get(
                                        config,
                                        "metadata.seginstance_warped_overlay_alpha",
                                        0.50,
                                    )
                                ),
                                draw_ids=bool(
                                    cfg_get(
                                        config,
                                        "metadata.seginstance_warped_overlay_draw_ids",
                                        True,
                                    )
                                ),
                            )
                            seginstance_warped_overlay_relpath = str(
                                overlay_path.relative_to(metadata_scene_dir)
                            )
                    if out["vehicle_masks"] is not None:
                        os.makedirs(
                            out["vehicle_masks"] / scene_output_name, exist_ok=True
                        )
                        cv2.imwrite(
                            str(
                                out["vehicle_masks"] / scene_output_name / f"{stem}.png"
                            ),
                            dynamic_mask[i].detach().cpu().numpy().astype("uint8")
                            * 255,
                        )
                    if out["sky_masks"] is not None:
                        os.makedirs(out["sky_masks"] / scene_output_name, exist_ok=True)
                        cv2.imwrite(
                            str(out["sky_masks"] / scene_output_name / f"{stem}.png"),
                            sky_mask[i].detach().cpu().numpy().astype("uint8") * 255,
                        )

                    if out["top20_reference"] is not None:
                        reference_scene_dir = out["top20_reference"] / scene_output_name
                        reference_scene_dir.mkdir(parents=True, exist_ok=True)

                        frame_visibility_counts = source_frame_visibility_counts[
                            i
                        ].tolist()
                        # Rank by the final occlusion-aware dominant-pixel count.
                        # Use temporal distance and then frame index only as stable
                        # tie breakers, so repeated runs produce identical JSON.
                        ranked_source_frames = sorted(
                            range(F),
                            key=lambda j: (
                                -int(frame_visibility_counts[j]),
                                abs(int(j) - int(i)),
                                int(j),
                            ),
                        )
                        top_k_requested = max(
                            1, int(cfg_get(config, "reference.top_k", 20))
                        )
                        ranked_source_frames = ranked_source_frames[
                            : min(top_k_requested, F)
                        ]

                        total_visible_pixels = int(sum(frame_visibility_counts))
                        all_reference_frames = []
                        for rank, source_frame_index in enumerate(
                            range(F), start=1
                        ):
                            source_stored_frame_index = int(
                                storage_frame_indices[source_frame_index]
                            )
                            source_stem = f"{source_stored_frame_index:05d}"
                            visible_pixel_count = int(
                                frame_visibility_counts[source_frame_index]
                            )
                            all_reference_frames.append(
                                {
                                    "rank": int(rank),
                                    "source_output_frame_index": int(
                                        source_frame_index
                                    ),
                                    "source_native_frame_index": source_stored_frame_index,
                                    "source_frame_offset": int(source_frame_index - i),
                                    "source_native_frame_offset": int(
                                        source_stored_frame_index
                                        - stored_frame_index
                                    ),
                                    "source_path": str(
                                        (
                                            Path(
                                                cfg_get(
                                                    config,
                                                    "paths.front_source",
                                                    "front_source",
                                                )
                                            )
                                            / scene_output_name
                                            / f"{source_stem}.png"
                                        ).as_posix()
                                    ),
                                    "visible_pixel_count": visible_pixel_count,
                                    "visibility_score": (
                                        0.0
                                        if total_visible_pixels == 0
                                        else float(visible_pixel_count)
                                        / float(total_visible_pixels)
                                    ),
                                }
                            )

                        reference_frames = [
                            dict(all_reference_frames[j], rank=rank)
                            for rank, j in enumerate(ranked_source_frames, start=1)
                        ]

                        reference_meta = {
                            "scene": {
                                "uuid": scene_uuid,
                                "log_name": scene_log_name,
                            },
                            "scene_uuid": scene_uuid,
                            "scene_log_name": scene_log_name,
                            "output_scene_name": scene_output_name,
                            "dataset_setup": dataset_setup,
                            "direction": direction_prefix,
                            "source_cam_id": source_cam_id,
                            "target_cam_id": target_cam_id,
                            "target_is_virtual": (
                                dataset_setup == "monocular_shifted"
                            ),
                            "virtual_shift_local_xyz_m": virtual_shift_local_xyz_m,
                            "virtual_shift_distance_m": virtual_shift_distance_m,
                            "warped_output_frame_index": int(i),
                            "warped_native_frame_index": stored_frame_index,
                            "warped_path": str(
                                (
                                    Path(
                                        cfg_get(
                                            config,
                                            "paths.front_warped",
                                            "front_warped",
                                        )
                                    )
                                    / scene_output_name
                                    / f"{stem}.png"
                                ).as_posix()
                            ),
                            "source_calibration_path": source_calibration_relpath,
                            "target_calibration_path": target_calibration_relpath,
                            "frame_sampling": frame_sampling,
                            "stored_frame_index_policy": "contiguous_from_zero",
                            "ranking_metric": (
                                "local_roundtrip_plus_direct_global_coverage"
                                if double_reprojection_enabled
                                else "occlusion_aware_dominant_pixel_count"
                            ),
                            "ranking_order": "highest_first",
                            "ranking_includes_matching_source_frame": True,
                            "top_k_requested": top_k_requested,
                            "top_k_returned": len(reference_frames),
                            "total_visible_pixels": total_visible_pixels,
                            "total_image_pixels": int(H * W),
                            "coverage_ratio": float(total_visible_pixels)
                            / float(H * W),
                            "reference_frames": reference_frames,
                        }
                        with (reference_scene_dir / f"{stem}.json").open(
                            "w", encoding="utf-8"
                        ) as f:
                            json.dump(reference_meta, f, indent=2)
                        if dedup_rankings is not None:
                            for tenth, all_ranks in dedup_rankings.items():
                                dedup_scene_dir = (
                                    out["top20_reference"].parent
                                    / f"top20_reference_dedup_0p{tenth}"
                                    / scene_output_name
                                )
                                dedup_scene_dir.mkdir(parents=True, exist_ok=True)
                                chosen = [int(j) for j in all_ranks[i]]
                                dedup_refs = [
                                    dict(
                                        all_reference_frames[j],
                                        rank=rank,
                                        visible_pixels_by_tile=[
                                            int(v) for v in tile_counts[i, :, j]
                                        ],
                                    )
                                    for rank, j in enumerate(chosen, start=1)
                                ]
                                dedup_meta = dict(reference_meta)
                                dedup_meta.update(
                                    ranking_metric="dedup_2x2_final_provenance",
                                    dedup_lambda_global=1.0,
                                    dedup_lambda_tile=tenth / 10.0,
                                    dedup_tile_grid=[2, 2],
                                    reference_frames=dedup_refs,
                                )
                                with (dedup_scene_dir / f"{stem}.json").open(
                                    "w", encoding="utf-8"
                                ) as f:
                                    json.dump(dedup_meta, f, indent=2)

                    if out["metadata"] is None:
                        continue

                    meta = {
                        "scene": {
                            "uuid": scene_uuid,
                            "log_name": scene_log_name,
                        },
                        "scene_uuid": scene_uuid,
                        "scene_log_name": scene_log_name,
                        "scene_output_base_name": scene_output_base_name,
                        "output_scene_name": scene_output_name,
                        "dataset_setup": dataset_setup,
                        "worker_id": int(worker_id),
                        "num_workers": int(num_workers),
                        "global_scene_index": (
                            None
                            if global_scene_index is None
                            else int(global_scene_index)
                        ),
                        "direction": direction_prefix,
                        "source_view": source_view_name,
                        "target_view": target_view_name,
                        "batch_idx": int(batch_idx),
                        "sample_idx": int(b),
                        "output_frame_index": int(i),
                        "native_frame_index": stored_frame_index,
                        "source_native_frame": stored_frame_index,
                        "target_native_frame": stored_frame_index,
                        "frame_sampling": frame_sampling,
                        "stored_frame_index_policy": "contiguous_from_zero",
                        "source_cam_id": source_cam_id,
                        "target_cam_id": target_cam_id,
                        "front_target_available": target_rgb is not None,
                        "target_is_virtual": (
                            dataset_setup == "monocular_shifted"
                        ),
                        "virtual_shift_local_xyz_m": virtual_shift_local_xyz_m,
                        "virtual_shift_distance_m": virtual_shift_distance_m,
                        "sam3_prompts": sam3_prompts,
                        "sam3_sky_prompts": sky_prompts,
                        "source_preprocessing_reused": source_preprocessing_reused,
                        "sam3_dynamic_pixels_raw": int(dynamic_mask[i].sum().item()),
                        "static_vehicle_ids_by_center_stability": sorted(
                            [int(x) for x in static_vehicle_ids]
                        ),
                        "dynamic_vehicle_ids_by_center_stability": sorted(
                            [int(x) for x in dynamic_vehicle_ids]
                        ),
                        "static_vehicle_pixels_by_center_stability": int(
                            static_vehicle_mask[i].sum().item()
                        ),
                        "vehicle_motion_stats_by_center_stability": vehicle_motion_stats,
                        "sam3_dynamic_pixels_after_sky": int(dyn_cpu[i].sum().item()),
                        "sam3_sky_pixels": int(sky_mask[i].sum().item()),
                        "sky_depth": float(cfg_get(config, "render.sky_depth", 1000.0)),
                        "coverage_pixels": int(coverage[i].sum().item()),
                        "double_reprojection_enabled": double_reprojection_enabled,
                        "double_reprojection_algorithm": (
                            "local_dense_gather_plus_direct_static_global_pcd_splat"
                            if double_reprojection_enabled
                            else None
                        ),
                        "warped_output_camera_role": (
                            "source_return"
                            if double_reprojection_enabled
                            else "target"
                        ),
                        "local_static_hole_pixels": int(
                            static_hole_mask[i].sum().item()
                        ),
                        "local_dynamic_hole_pixels": int(
                            dynamic_hole_mask[i].sum().item()
                        ),
                        "source_dynamic_instance_ids": sorted(
                            [
                                int(x)
                                for x in torch.unique(dynamic_instance_id_maps[i])
                                .detach()
                                .cpu()
                                .tolist()
                                if int(x) > 0
                            ]
                        ),
                        "source_seginstance_ids": sorted(
                            [
                                int(x)
                                for x in torch.unique(seginstance_id_maps[i])
                                .detach()
                                .cpu()
                                .tolist()
                                if int(x) > 0
                            ]
                        ),
                        "seginstance_warped_enabled": out["seginstance_warped"]
                        is not None,
                        "seginstance_warped_object_mode": seginstance_object_mode,
                        "seginstance_warped_temporal_persistence": seginstance_temporal_persistence,
                        "seginstance_warped_visibility_mode": seginstance_visibility_mode,
                        "seginstance_warped_max_past_frames": seginstance_max_past_frames,
                        "seginstance_warped_visible_points_for_frame": (
                            0
                            if seginstance_visible is None
                            else int(seginstance_visible[:, i].sum().item())
                        ),
                        "seginstance_warped_pixels": (
                            0
                            if seginstance_warped is None
                            else int((seginstance_warped[i] > 0).sum().item())
                        ),
                        "seginstance_warped_ids": (
                            []
                            if seginstance_warped is None
                            else sorted(
                                [
                                    int(x)
                                    for x in torch.unique(seginstance_warped[i])
                                    .detach()
                                    .cpu()
                                    .tolist()
                                    if int(x) > 0
                                ]
                            )
                        ),
                        "seginstance_warped_overlay_enabled": seginstance_warped_overlay_enabled,
                        "seginstance_warped_overlay_background": (
                            seginstance_warped_overlay_background
                            if seginstance_warped_overlay_enabled
                            else None
                        ),
                        "seginstance_warped_overlay_file": seginstance_warped_overlay_relpath,
                        "render_coord_flip_xy": bool(
                            cfg_get(config, "render.coord_flip_xy", True)
                        ),
                        "render_visibility_mode": visibility_mode,
                        "render_no_temporal_persistence": bool(
                            cfg_get(config, "render.no_temporal_persistence", False)
                        ),
                        "render_render_moving": bool(
                            cfg_get(config, "render.render_moving", True)
                        ),
                        "render_global_fill_static_holes_only": bool(
                            cfg_get(
                                config,
                                "render.global_fill_static_holes_only",
                                True,
                            )
                        ),
                        "render_dynamic_hole_depth_abs_tolerance": float(
                            cfg_get(
                                config,
                                "render.dynamic_hole_depth_abs_tolerance",
                                0.10,
                            )
                        ),
                        "render_dynamic_hole_depth_rel_tolerance": float(
                            cfg_get(
                                config,
                                "render.dynamic_hole_depth_rel_tolerance",
                                0.02,
                            )
                        ),
                        "render_dynamic_hole_mask_dilation_radius": int(
                            cfg_get(
                                config,
                                "render.dynamic_hole_mask_dilation_radius",
                                0,
                            )
                        ),
                        "render_pixel_center_offset": bool(
                            cfg_get(config, "render.pixel_center_offset", False)
                        ),
                    }

                    with (metadata_scene_dir / f"{stem}.json").open(
                        "w", encoding="utf-8"
                    ) as f:
                        json.dump(meta, f, indent=2)

                del color, xyz, visible, indices, point_dynamic
                del render_visible, warped, depths_pc, coverage, dyn_pc
                del static_hole_mask, dynamic_hole_mask
                if first_warped_ntp is not None:
                    del first_warped_ntp
                if first_depth_ntp is not None:
                    del first_depth_ntp
                if first_coverage_ntp is not None:
                    del first_coverage_ntp
                if first_dynamic_ntp is not None:
                    del first_dynamic_ntp
                if seginstance_visible is not None:
                    del seginstance_visible
                if seginstance_warped is not None:
                    del seginstance_warped
                if source_frame_visibility_counts is not None:
                    del source_frame_visibility_counts
                del point_instance_ids, dynamic_instance_id_maps, seginstance_id_maps
                del dyn_cpu, sky_mask, dynamic_mask, static_vehicle_mask
                del left_depth, K_left, T_left, source_dynamic_mask
                del left_depth_np, instance_id_maps, sky_instance_id_maps
                del sam3_prompts, sky_prompts
                del static_vehicle_ids, dynamic_vehicle_ids, vehicle_motion_stats
                del seginstance_object_mode
                del source_state

                source_uses_remaining[source_cache_key] -= 1
                if source_uses_remaining[source_cache_key] == 0:
                    cached_source_state = source_preprocess_cache.pop(
                        source_cache_key
                    )
                    del cached_source_state
                    del source_uses_remaining[source_cache_key]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            source_preprocess_cache.clear()
            source_uses_remaining.clear()

            if bool(
                cfg_get(config, "runtime.process_first_sample_per_batch_only", True)
            ):
                break

    del video_predictor
    cleanup()


def run_pipeline_worker(config, scene_indices, gpu_id, num_workers):
    """Entry point for one spawned GPU worker."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))

    run_pipeline(
        config=config,
        device_override=f"cuda:{int(gpu_id)}",
        scene_indices=scene_indices,
        worker_id=int(gpu_id),
        num_workers=int(num_workers),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/stereo/default.yaml", type=str)
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Optional dotted key overrides, e.g. render.render_moving=false "
            "dataset.target_steps=80 runtime.gpu=2"
        ),
    )
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config), args.overrides)
    gpu_count = resolve_gpu_count(config)

    # gpu: 1 keeps the original single-process behavior.
    if gpu_count <= 1:
        run_pipeline(config=config)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("runtime.gpu >= 2 requires CUDA, but CUDA is not available.")

    available_gpus = torch.cuda.device_count()
    if gpu_count > available_gpus:
        raise RuntimeError(
            f"runtime.gpu={gpu_count} was requested, but only {available_gpus} CUDA GPUs are visible."
        )

    # Build the dataset once in the parent only to determine the stable scene count/order.
    # Each child process rebuilds its own dataset and then consumes only its assigned shard.
    fix_seed(int(cfg_get(config, "runtime.seed", 42)))
    configure_data_roots(config)

    _, dataset, _ = build_reconstruction_dataset(config)

    batch_filter = get_batch_filter(config)
    if batch_filter is None:
        scene_indices = list(range(len(dataset)))
    else:
        scene_indices = [
            i for i in sorted(int(x) for x in batch_filter) if 0 <= i < len(dataset)
        ]

    shards = split_indices_contiguous(scene_indices, gpu_count)
    for gpu_id, shard in enumerate(shards):
        if len(shard) == 0:
            print(f"GPU {gpu_id}: no assigned scenes")
        else:
            print(
                f"GPU {gpu_id}: assigned scene indices "
                f"{shard[0]}..{shard[-1]} ({len(shard)} scenes)"
            )

    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id, shard in enumerate(shards):
        if len(shard) == 0:
            continue
        p = ctx.Process(
            target=run_pipeline_worker,
            args=(config, shard, gpu_id, gpu_count),
        )
        p.start()
        processes.append((gpu_id, p))

    failed = []
    for gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append((gpu_id, p.exitcode))

    if failed:
        raise RuntimeError(f"One or more GPU workers failed: {failed}")

    cleanup()


if __name__ == "__main__":
    main()
