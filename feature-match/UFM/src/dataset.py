import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from py123d.api import SceneAPI, SceneFilter, get_filtered_scenes
from py123d.geometry.transform import abs_to_rel_se2_array
from py123d.visualization.matplotlib.plots import add_scene_on_ax
from torch.utils.data import Dataset

from utils.calib import (
    extract_camera_intrinsics,
    extract_camera_to_ego,
    make_se3_from_se2_array,
    rescale_intrinsics,
)



def _get_nested_attr(obj, attr_path, default=None):
    """Safely read a possibly nested attribute path from py123d objects."""
    cur = obj
    for part in attr_path.split("."):
        if cur is None:
            return default
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def get_scene_log_name(scene: SceneAPI, meta=None) -> str:
    """Return a human-readable py123d scene/log name when available.

    py123d-backed datasets expose slightly different metadata fields depending on
    the source dataset. Prefer explicit log/scene names and keep the UUID only as
    a last-resort fallback so output folders are easier to inspect manually.
    """
    if meta is None:
        meta = getattr(scene, "scene_metadata", None)

    candidates = [
        _get_nested_attr(scene, "log_name"),
        _get_nested_attr(scene, "name"),
        _get_nested_attr(scene, "scene_name"),
        _get_nested_attr(scene, "log.name"),
        _get_nested_attr(scene, "metadata.log_name"),
        _get_nested_attr(scene, "metadata.name"),
        _get_nested_attr(meta, "log_name"),
        _get_nested_attr(meta, "name"),
        _get_nested_attr(meta, "scene_name"),
        _get_nested_attr(meta, "initial_log_name"),
        _get_nested_attr(meta, "initial_name"),
        _get_nested_attr(meta, "initial_scene_name"),
        _get_nested_attr(meta, "initial_uuid"),
    ]
    for value in candidates:
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        value = str(value).strip()
        if value:
            return value
    return "unknown_scene"


def configure_default_data_roots(
    temp_path="/project/data",
    av2_path="/project/data/raw_data/av2",
    nusc_path="/project/data/raw_data/nusc",
):
    os.makedirs(temp_path, exist_ok=True)
    os.environ["PY123D_DATA_ROOT"] = temp_path
    os.environ["AV2_DATA_ROOT"] = av2_path
    os.environ["NUSCENES_DATA_ROOT"] = nusc_path
    breakpoint()


class Py123dDataset(Dataset):
    def __init__(
        self,
        scene_filter: SceneFilter,
        target_steps: int = 150,
        cam_id: str = None,
        image_size: tuple = (400, 300),
        frame_sampling: int = 1,
    ):
        """
        Initializes the dataset.

        Args:
            scene_filter: Configured SceneFilter instance.
            target_steps: Uniform sequence frame limit per batch item (e.g., 150 frames).
            cam_id: Explicit camera sensor ID string to capture and track.
            image_size: (width, height) resolution tuple for video downscaling.
            frame_sampling: Temporal sampling stride in native dataset frames. For example,
                frame_sampling=2 reads native frames 0, 2, 4, ...
        """
        self.scenes = get_filtered_scenes(scene_filter)
        self.target_steps = int(target_steps)
        self.cam_id = cam_id
        self.image_size = image_size  # (Width, Height) format for cv2.resize
        self.frame_sampling = max(1, int(frame_sampling))

    def __len__(self) -> int:
        return len(self.scenes)

    def _get_active_cam_id(self, scene: SceneAPI):
        """Resolves the intended target sensor identifier, safely mapping strings to Enum items."""
        if len(scene.available_camera_ids) == 0:
            return None

        if self.cam_id is None:
            return scene.available_camera_ids[0]

        # If user passed a string, dynamically locate the matching Enum object
        if isinstance(self.cam_id, str):
            clean_target = self.cam_id.lower()
            for cam in scene.available_camera_ids:
                # Safely matches if 'pcam_f0' is found within 'camera.pcam_f0' or vice-versa
                if cam.name.lower() in clean_target or clean_target in cam.name.lower():
                    return cam

        if self.cam_id in scene.available_camera_ids:
            return self.cam_id

        return scene.available_camera_ids[0]

    def __getitem__(self, idx: int) -> dict:
        scene: SceneAPI = self.scenes[idx]
        meta = scene.scene_metadata
        scene_uuid = str(meta.initial_uuid)
        scene_log_name = get_scene_log_name(scene, meta)

        # Direct extraction of strict framework boundary limitations
        num_history = scene.number_of_history_iterations
        max_iterations = scene.number_of_iterations

        # Ground starting point as original (0,0) at the absolute beginning of the clip (-num_history)
        origin_ego_state = scene.get_ego_state_se3_at_iteration(iteration=-num_history)
        if origin_ego_state is None:
            return self.__getitem__((idx + 1) % len(self))

        origin_pose_se2 = origin_ego_state.center_se3.pose_se2
        active_cam_id = self._get_active_cam_id(scene)

        global_poses = []
        rgb_frames = []
        out_w, out_h = self.image_size

        sampled_frame_indices = [it * self.frame_sampling for it in range(self.target_steps)]

        # Sequential processing sweep mapped cleanly over the framework's time offset
        for sampled_it in sampled_frame_indices:
            framework_it = -num_history + sampled_it

            # --- 1. Position extraction guarded strictly by framework assertion rules ---
            if -num_history <= framework_it < max_iterations:
                ego_state = scene.get_ego_state_se3_at_iteration(iteration=framework_it)
            else:
                ego_state = None

            if ego_state is not None:
                global_poses.append(ego_state.center_se3.pose_se2.array)
            else:
                # Dynamic edge duplication padding fallback
                if len(global_poses) > 0:
                    global_poses.append(global_poses[-1])
                else:
                    global_poses.append(origin_pose_se2.array)

            # --- 2. Camera frame extraction guarded strictly by framework assertion rules ---
            if (
                active_cam_id is not None
                and -num_history <= framework_it < max_iterations
            ):
                camera_obj = scene.get_camera_at_iteration(
                    iteration=framework_it, camera_id=active_cam_id
                )
            else:
                camera_obj = None

            if camera_obj is not None:
                raw_image = camera_obj.image
                resized_image = cv2.resize(
                    raw_image, self.image_size, interpolation=cv2.INTER_LINEAR
                )
                img_tensor = (
                    torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
                )
                rgb_frames.append(img_tensor)
            else:
                # Video padding fallback
                if len(rgb_frames) > 0:
                    rgb_frames.append(rgb_frames[-1])
                else:
                    rgb_frames.append(
                        torch.zeros((3, out_h, out_w), dtype=torch.float32)
                    )

        # Vectorized coordinate normalizations relative to the starting frame (0,0)
        global_track_arr = np.array(global_poses, dtype=np.float32)
        local_unified_arr = abs_to_rel_se2_array(origin_pose_se2, global_track_arr)

        return {
            "unified_trajectory": torch.from_numpy(local_unified_arr),
            "rgb_video": torch.stack(rgb_frames),
            "sampled_frame_indices": torch.tensor(sampled_frame_indices, dtype=torch.long),
            "frame_sampling": self.frame_sampling,
            "scene_uuid": scene_uuid,
            "scene_log_name": scene_log_name,
            "scene": {"uuid": scene_uuid, "log_name": scene_log_name},
        }

    def visualize_scene_gif(
        self,
        idx: int,
        output_dir: str = "./visualizations",
        filename: str = "scene_movie.gif",
        radius: float = 60.0,
    ):
        """Compiles the continuous timelines into an auto-scaling 3-panel animation loop."""
        scene: SceneAPI = self.scenes[idx]
        meta = scene.scene_metadata

        num_history = scene.number_of_history_iterations
        max_iterations = scene.number_of_iterations

        tensor_data = self.__getitem__(idx)
        unified_traj = tensor_data["unified_trajectory"].numpy()
        video_frames = tensor_data["rgb_video"].permute(0, 2, 3, 1).numpy()
        sampled_frame_indices = tensor_data["sampled_frame_indices"].numpy().tolist()

        frame_indices = list(range(len(unified_traj)))

        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        full_output_path = save_path / filename

        active_cam_id = self._get_active_cam_id(scene)
        cam_name_str = active_cam_id.name if active_cam_id else "Unknown"

        # Dynamic view boundary setup
        margin = 10.0
        x_min, x_max = (
            min(unified_traj[:, 0].min() - margin, -margin),
            max(unified_traj[:, 0].max() + margin, margin),
        )
        y_min, y_max = (
            min(unified_traj[:, 1].min() - margin, -margin),
            max(unified_traj[:, 1].max() + margin, margin),
        )

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        def update(frame_idx):
            raw_framework_it = -num_history + sampled_frame_indices[frame_idx]
            framework_it = max(-num_history, min(raw_framework_it, max_iterations - 1))

            for ax in axes:
                ax.clear()

            # PANEL 1: BEV HD Map Updates
            axes[0].set_title(f"Environment Map (Iteration {framework_it})")
            add_scene_on_ax(axes[0], scene, iteration=framework_it, radius=radius)

            # PANEL 2: Unified Coordinate Map
            ax2 = axes[1]
            ax2.set_title("Ego-Centric Unified Trajectory Frame")
            ax2.plot(
                unified_traj[:, 0],
                unified_traj[:, 1],
                color="purple",
                marker="o",
                markersize=4,
                linewidth=2,
                label="Unified Trajectory Line",
                alpha=0.7,
            )
            ax2.plot(0, 0, "ro", markersize=10, label="Sequence Start Origin (0,0)")

            current_ego = scene.get_ego_state_se3_at_iteration(iteration=framework_it)
            origin_ego = scene.get_ego_state_se3_at_iteration(iteration=-num_history)
            if current_ego and origin_ego:
                frame_local = abs_to_rel_se2_array(
                    origin_ego.center_se3.pose_se2,
                    np.array([current_ego.center_se3.pose_se2.array]),
                )[0]
                ax2.plot(
                    frame_local[0],
                    frame_local[1],
                    color="red",
                    marker="X",
                    markersize=14,
                    label="Active Position",
                )

            ax2.set_xlabel("Ego X-Forward [meters]")
            ax2.set_ylabel("Ego Y-Left [meters]")
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)
            ax2.set_aspect("equal")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right")

            # PANEL 3: Step Video Frame Array
            ax3 = axes[2]
            ax3.imshow(video_frames[frame_idx])
            ax3.set_title(f"Resized Video Stream: {cam_name_str}")
            ax3.axis("off")
            plt.tight_layout()

        native_fps = 1 / meta.iteration_duration_s
        sampled_fps = native_fps / self.frame_sampling
        interval_ms = int(1000 / sampled_fps)

        print(f"Compiling dataset video loop layout -> {full_output_path}")
        anim = FuncAnimation(
            fig, update, frames=len(frame_indices), interval=interval_ms
        )
        anim.save(str(full_output_path), writer="pillow", fps=max(1, int(sampled_fps)))
        plt.close(fig)
        print("Multi-panel synchronized .gif successfully generated!")

    def visualize_sample(self, idx: int, radius: float = 60.0):
        """Generates checking graphics validating unified output pairs."""
        scene: SceneAPI = self.scenes[idx]
        num_history = scene.number_of_history_iterations

        tensor_data = self.__getitem__(idx)
        unified_traj = tensor_data["unified_trajectory"].numpy()
        first_frame_img = tensor_data["rgb_video"][0].permute(1, 2, 0).numpy()
        active_cam_id = self._get_active_cam_id(scene)
        cam_name_str = active_cam_id.name if active_cam_id else "Unknown"

        margin = 10.0
        x_min, x_max = (
            min(unified_traj[:, 0].min() - margin, -margin),
            max(unified_traj[:, 0].max() + margin, margin),
        )
        y_min, y_max = (
            min(unified_traj[:, 1].min() - margin, -margin),
            max(unified_traj[:, 1].max() + margin, margin),
        )

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        axes[0].set_title(f"Environment Render (Sample {idx})")
        add_scene_on_ax(axes[0], scene, iteration=-num_history, radius=radius)

        ax2 = axes[1]
        ax2.set_title("Ego-Centric Unified Trajectory Frame")
        ax2.plot(
            unified_traj[:, 0],
            unified_traj[:, 1],
            color="purple",
            marker="o",
            linewidth=2,
            label="Unified Trajectory",
        )
        ax2.plot(0, 0, "ro", markersize=10, label="Sequence Start Origin (0,0)")
        ax2.set_xlabel("Ego X-Forward [meters]")
        ax2.set_ylabel("Ego Y-Left [meters]")
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = axes[2]
        ax3.set_title(f"Resized RGB Video (Frame 0): {cam_name_str}")
        ax3.imshow(first_frame_img)
        ax3.axis("off")

        plt.tight_layout()
        plt.savefig("test_unified_trajectory_pipeline.png")
        print(
            "Static verification plot written to 'test_unified_trajectory_pipeline.png'"
        )


def py123d_collate_fn(batch: list) -> dict:
    """Collates sequential dictionary inputs into packed batch volumes."""
    return {
        "unified_trajectory": torch.stack(
            [item["unified_trajectory"] for item in batch]
        ),
        "rgb_video": torch.stack([item["rgb_video"] for item in batch]),
        "sampled_frame_indices": torch.stack([item["sampled_frame_indices"] for item in batch]),
        "frame_sampling": [item["frame_sampling"] for item in batch],
        "scene_uuid": [item["scene_uuid"] for item in batch],
        "scene_log_name": [item["scene_log_name"] for item in batch],
        "scene": [item["scene"] for item in batch],
    }


class Py123dPairedCameraReconDataset(Py123dDataset):
    def __init__(
        self,
        scene_filter,
        target_steps=150,
        left_cam_id="camera.pcam_f0",
        right_cam_id="camera.pcam_r0",
        image_size=(800, 450),
        frame_sampling=1,
    ):
        super().__init__(
            scene_filter=scene_filter,
            target_steps=target_steps,
            cam_id=None,
            image_size=image_size,
            frame_sampling=frame_sampling,
        )
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id

    @staticmethod
    def _camera_name(camera_id):
        return getattr(camera_id, "name", str(camera_id)).lower()

    def _resolve_camera_id(self, scene: SceneAPI, query):
        if len(scene.available_camera_ids) == 0:
            return None
        if query in scene.available_camera_ids:
            return query

        q = str(query).lower().replace("camera.", "")
        for cid in scene.available_camera_ids:
            name = self._camera_name(cid).replace("camera.", "")
            if name == q:
                return cid
        for cid in scene.available_camera_ids:
            name = self._camera_name(cid).replace("camera.", "")
            if q in name or name in q:
                return cid
        return None

    def _empty_camera_state(self, out_w, out_h):
        return {
            "rgb": torch.zeros((3, out_h, out_w), dtype=torch.float32),
            "K": torch.eye(3, dtype=torch.float32),
            "T_ego_cam": torch.eye(4, dtype=torch.float32),
        }

    def _read_camera_state(self, scene, framework_it, camera_id, prev, out_w, out_h):
        num_history = scene.number_of_history_iterations
        max_iterations = scene.number_of_iterations

        cam = None
        if camera_id is not None and -num_history <= framework_it < max_iterations:
            cam = scene.get_camera_at_iteration(
                iteration=framework_it, camera_id=camera_id
            )

        if cam is None:
            return prev if prev is not None else self._empty_camera_state(out_w, out_h)

        old_w, old_h = cam.metadata.width, cam.metadata.height
        resized = cv2.resize(cam.image, self.image_size, interpolation=cv2.INTER_LINEAR)
        rgb = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        K = extract_camera_intrinsics(cam, old_w, old_h)
        K = rescale_intrinsics(K, old_w, old_h, out_w, out_h)

        return {
            "rgb": rgb,
            "K": torch.from_numpy(K),
            "T_ego_cam": torch.from_numpy(extract_camera_to_ego(cam)),
        }

    def __getitem__(self, idx):
        scene: SceneAPI = self.scenes[idx]
        meta = scene.scene_metadata
        scene_uuid = str(meta.initial_uuid)
        scene_log_name = get_scene_log_name(scene, meta)

        num_history = scene.number_of_history_iterations
        max_iterations = scene.number_of_iterations

        origin_ego = scene.get_ego_state_se3_at_iteration(iteration=-num_history)
        if origin_ego is None:
            return self.__getitem__((idx + 1) % len(self))

        origin_pose_se2 = origin_ego.center_se3.pose_se2

        left_id = self._resolve_camera_id(scene, self.left_cam_id)
        right_id = self._resolve_camera_id(scene, self.right_cam_id)
        if left_id is None or right_id is None:
            available = [self._camera_name(c) for c in scene.available_camera_ids]
            print(
                f"Could not resolve cameras: left={self.left_cam_id}, right={self.right_cam_id}, available={available}, scene={scene_log_name}, uuid={scene_uuid}"
            )
            return {
                "unified_trajectory": torch.tensor([0]),
                "rgb_video_left": torch.tensor([0]),
                "rgb_video_right": torch.tensor([0]),
                "rgb_video": torch.tensor([0]),
                "K_left": torch.tensor([0]),
                "K_right": torch.tensor([0]),
                "K": torch.tensor([0]),
                "T_ego_cam_left": torch.tensor([0]),
                "T_ego_cam_right": torch.tensor([0]),
                "T_ego_cam": torch.tensor([0]),
                "T_world_ego": torch.tensor([0]),
                "T_world_cam_left": torch.tensor([0]),
                "T_world_cam_right": torch.tensor([0]),
                "T_world_cam": torch.tensor([0]),
                "sampled_frame_indices": torch.tensor([0]),
                "frame_sampling": self.frame_sampling,
                "left_cam_id": -1,
                "right_cam_id": -1,
                "scene_uuid": scene_uuid,
                "scene_log_name": scene_log_name,
                "scene": {"uuid": scene_uuid, "log_name": scene_log_name},
            }

            # raise ValueError(
            #     f"Could not resolve cameras: left={self.left_cam_id}, "
            #     f"right={self.right_cam_id}, available={available}"
            # )

        out_w, out_h = self.image_size
        global_poses = []
        left_states = []
        right_states = []
        last_left = None
        last_right = None

        sampled_frame_indices = [it * self.frame_sampling for it in range(self.target_steps)]

        for sampled_it in sampled_frame_indices:
            framework_it = -num_history + sampled_it

            if -num_history <= framework_it < max_iterations:
                ego = scene.get_ego_state_se3_at_iteration(iteration=framework_it)
            else:
                ego = None

            if ego is not None:
                pose = ego.center_se3.pose_se2.array
            elif global_poses:
                pose = global_poses[-1]
            else:
                pose = origin_pose_se2.array

            global_poses.append(pose)

            last_left = self._read_camera_state(
                scene, framework_it, left_id, last_left, out_w, out_h
            )
            last_right = self._read_camera_state(
                scene, framework_it, right_id, last_right, out_w, out_h
            )

            left_states.append(last_left)
            right_states.append(last_right)

        local_traj = abs_to_rel_se2_array(
            origin_pose_se2, np.asarray(global_poses, dtype=np.float32)
        )

        T_world_ego_list = []
        T_world_left_list = []
        T_world_right_list = []

        for it in range(self.target_steps):
            T_world_ego = make_se3_from_se2_array(local_traj[it])
            T_left = T_world_ego @ left_states[it]["T_ego_cam"].numpy()
            T_right = T_world_ego @ right_states[it]["T_ego_cam"].numpy()

            T_world_ego_list.append(torch.from_numpy(T_world_ego.astype(np.float32)))
            T_world_left_list.append(torch.from_numpy(T_left.astype(np.float32)))
            T_world_right_list.append(torch.from_numpy(T_right.astype(np.float32)))

        rgb_left = torch.stack([s["rgb"] for s in left_states])
        rgb_right = torch.stack([s["rgb"] for s in right_states])
        K_left = torch.stack([s["K"] for s in left_states])
        K_right = torch.stack([s["K"] for s in right_states])
        T_ego_left = torch.stack([s["T_ego_cam"] for s in left_states])
        T_ego_right = torch.stack([s["T_ego_cam"] for s in right_states])

        return {
            "unified_trajectory": torch.from_numpy(local_traj),
            "rgb_video_left": rgb_left,
            "rgb_video_right": rgb_right,
            "rgb_video": torch.stack([rgb_left, rgb_right], dim=0),
            "K_left": K_left,
            "K_right": K_right,
            "K": torch.stack([K_left, K_right], dim=0),
            "T_ego_cam_left": T_ego_left,
            "T_ego_cam_right": T_ego_right,
            "T_ego_cam": torch.stack([T_ego_left, T_ego_right], dim=0),
            "T_world_ego": torch.stack(T_world_ego_list),
            "T_world_cam_left": torch.stack(T_world_left_list),
            "T_world_cam_right": torch.stack(T_world_right_list),
            "T_world_cam": torch.stack(
                [torch.stack(T_world_left_list), torch.stack(T_world_right_list)], dim=0
            ),
            "sampled_frame_indices": torch.tensor(sampled_frame_indices, dtype=torch.long),
            "frame_sampling": self.frame_sampling,
            "left_cam_id": self._camera_name(left_id),
            "right_cam_id": self._camera_name(right_id),
            "scene_uuid": scene_uuid,
            "scene_log_name": scene_log_name,
            "scene": {"uuid": scene_uuid, "log_name": scene_log_name},
        }


class Py123dMonocularShiftedReconDataset(Py123dPairedCameraReconDataset):
    """Load one real camera for rendering into shifted virtual cameras.

    This intentionally exposes only source-camera tensors. Virtual target poses
    are created later by ``preprocess.py`` so no physical target camera or
    target RGB stream is required.
    """

    def __init__(
        self,
        scene_filter,
        target_steps=150,
        source_cam_id="camera.pcam_f0",
        image_size=(800, 450),
        frame_sampling=1,
    ):
        Py123dDataset.__init__(
            self,
            scene_filter=scene_filter,
            target_steps=target_steps,
            cam_id=None,
            image_size=image_size,
            frame_sampling=frame_sampling,
        )
        self.source_cam_id = source_cam_id

    def __getitem__(self, idx):
        scene: SceneAPI = self.scenes[idx]
        meta = scene.scene_metadata
        scene_uuid = str(meta.initial_uuid)
        scene_log_name = get_scene_log_name(scene, meta)

        num_history = scene.number_of_history_iterations
        max_iterations = scene.number_of_iterations

        origin_ego = scene.get_ego_state_se3_at_iteration(iteration=-num_history)
        if origin_ego is None:
            return self.__getitem__((idx + 1) % len(self))

        origin_pose_se2 = origin_ego.center_se3.pose_se2
        source_id = self._resolve_camera_id(scene, self.source_cam_id)
        if source_id is None:
            available = [self._camera_name(c) for c in scene.available_camera_ids]
            print(
                "Could not resolve monocular source camera: "
                f"source={self.source_cam_id}, available={available}, "
                f"scene={scene_log_name}, uuid={scene_uuid}"
            )
            return {
                "unified_trajectory": torch.tensor([0]),
                "rgb_video_source": torch.tensor([0]),
                "K_source": torch.tensor([0]),
                "T_ego_cam_source": torch.tensor([0]),
                "T_world_ego": torch.tensor([0]),
                "T_world_cam_source": torch.tensor([0]),
                "sampled_frame_indices": torch.tensor([0]),
                "frame_sampling": self.frame_sampling,
                "source_cam_id": -1,
                "scene_uuid": scene_uuid,
                "scene_log_name": scene_log_name,
                "scene": {"uuid": scene_uuid, "log_name": scene_log_name},
            }

        out_w, out_h = self.image_size
        global_poses = []
        source_states = []
        last_source = None
        sampled_frame_indices = [
            it * self.frame_sampling for it in range(self.target_steps)
        ]

        for sampled_it in sampled_frame_indices:
            framework_it = -num_history + sampled_it

            if -num_history <= framework_it < max_iterations:
                ego = scene.get_ego_state_se3_at_iteration(iteration=framework_it)
            else:
                ego = None

            if ego is not None:
                pose = ego.center_se3.pose_se2.array
            elif global_poses:
                pose = global_poses[-1]
            else:
                pose = origin_pose_se2.array
            global_poses.append(pose)

            last_source = self._read_camera_state(
                scene,
                framework_it,
                source_id,
                last_source,
                out_w,
                out_h,
            )
            source_states.append(last_source)

        local_traj = abs_to_rel_se2_array(
            origin_pose_se2,
            np.asarray(global_poses, dtype=np.float32),
        )

        T_world_ego_list = []
        T_world_source_list = []
        for it in range(self.target_steps):
            T_world_ego = make_se3_from_se2_array(local_traj[it])
            T_source = T_world_ego @ source_states[it]["T_ego_cam"].numpy()
            T_world_ego_list.append(torch.from_numpy(T_world_ego.astype(np.float32)))
            T_world_source_list.append(
                torch.from_numpy(T_source.astype(np.float32))
            )

        rgb_source = torch.stack([s["rgb"] for s in source_states])
        K_source = torch.stack([s["K"] for s in source_states])
        T_ego_source = torch.stack([s["T_ego_cam"] for s in source_states])

        return {
            "unified_trajectory": torch.from_numpy(local_traj),
            "rgb_video_source": rgb_source,
            "K_source": K_source,
            "T_ego_cam_source": T_ego_source,
            "T_world_ego": torch.stack(T_world_ego_list),
            "T_world_cam_source": torch.stack(T_world_source_list),
            "sampled_frame_indices": torch.tensor(
                sampled_frame_indices,
                dtype=torch.long,
            ),
            "frame_sampling": self.frame_sampling,
            "source_cam_id": self._camera_name(source_id),
            "scene_uuid": scene_uuid,
            "scene_log_name": scene_log_name,
            "scene": {"uuid": scene_uuid, "log_name": scene_log_name},
        }


def collate_fn(batch):
    keys = [
        "unified_trajectory",
        "rgb_video_left",
        "rgb_video_right",
        "rgb_video",
        "K_left",
        "K_right",
        "K",
        "T_ego_cam_left",
        "T_ego_cam_right",
        "T_ego_cam",
        "T_world_ego",
        "T_world_cam_left",
        "T_world_cam_right",
        "T_world_cam",
        "sampled_frame_indices",
    ]
    out = {k: torch.stack([x[k] for x in batch]) for k in keys}
    out["frame_sampling"] = [x["frame_sampling"] for x in batch]
    out["left_cam_id"] = [x["left_cam_id"] for x in batch]
    out["right_cam_id"] = [x["right_cam_id"] for x in batch]
    out["scene_uuid"] = [x["scene_uuid"] for x in batch]
    out["scene_log_name"] = [x["scene_log_name"] for x in batch]
    out["scene"] = [x["scene"] for x in batch]

    if "global_scene_index" in batch[0]:
        out["global_scene_index"] = [int(x["global_scene_index"]) for x in batch]

    return out


def monocular_shifted_collate_fn(batch):
    keys = [
        "unified_trajectory",
        "rgb_video_source",
        "K_source",
        "T_ego_cam_source",
        "T_world_ego",
        "T_world_cam_source",
        "sampled_frame_indices",
    ]
    out = {k: torch.stack([x[k] for x in batch]) for k in keys}
    out["frame_sampling"] = [x["frame_sampling"] for x in batch]
    out["source_cam_id"] = [x["source_cam_id"] for x in batch]
    out["scene_uuid"] = [x["scene_uuid"] for x in batch]
    out["scene_log_name"] = [x["scene_log_name"] for x in batch]
    out["scene"] = [x["scene"] for x in batch]

    if "global_scene_index" in batch[0]:
        out["global_scene_index"] = [
            int(x["global_scene_index"]) for x in batch
        ]

    return out
