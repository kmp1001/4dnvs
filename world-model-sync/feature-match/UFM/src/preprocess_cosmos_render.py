#!/usr/bin/env python3
"""Render hdmap, world_scenario, and videos directly from Py123D Arrow logs.

One standalone script; no Cosmos checkout, RDS tar conversion, LiDAR, DA3,
SAM3, torch, or GPU renderer is required. Uses native Py123D calibration.

    pixi run python src/visualize_py123d_hdmap.py \
        --config src/config_geometry/monocular_shift/nuscene.yaml
    # Retain the interactive 3D map/box viewer:
    pixi run python src/visualize_py123d_hdmap.py --config config.yaml --viewer

Default exports three synchronized MP4 categories on the native source camera
timeline: hdmap/<camera>/<clip>_0.mp4 (colored map + wireframe cuboids),
world_scenario/<camera>/<clip>_0.mp4 (lane styles + filled, depth-tested cuboids),
videos/<camera>/<clip>_0.mp4 (source RGB with 3D box overlays).
--no-rgb-boxes saves clean source RGB. --comparison also saves
RGB | HDMap | world_scenario under
videos/comparison/. --save-frames also saves PNGs.
--frames-only saves PNGs WITHOUT MP4 encoding or an FFmpeg requirement.
All PNG exports default to the preprocessing folders, for example:
    hdmap/R2M_scene-0571/00000.png
    world_scenario/R2M_scene-0571/00000.png
    videos/scene-0571/00000.png
Frames use the selected sequence index (0, 1, ...); native iterations and
timestamps are retained in metadata.json. No layout option is needed.
--frame-layout stream optionally selects the older
camera/clip_frames/000000.png layout. Duplicate log names within one run
receive scene-index suffixes to avoid overwriting one another.
Every export also saves ONE scene_data/<scene>.json with queried map objects,
render geometry, triangles, styles, per-frame boxes/ego/camera poses, native
calibration, render parameters and view offsets. Map snapshots/calibrations
are shared across frames. This contains rendered frames/queried map regions,
not the entire city map. RGB pixel data remains in the image/video exports.
--no-scene-data disables this additional file. To recreate both control
streams offline (Py123D calibration code required; no Arrow/data/models):
    pixi run python src/visualize_py123d_hdmap.py \
        --replay-scene hdmap_render/scene_data/scene-0571.json \
        --frames-only --output-dir ./offline_render
With double_reprojection.enabled=true, both controls use the original pose,
regardless of render_directions. Otherwise, dataset.setup=monocular_shifted
renders each dataset.render_directions view under hdmap/<camera>/<direction>/
and world_scenario/<camera>/<direction>/. Source RGB is saved ONCE at
videos/<camera>/; it is never mislabeled as a virtual camera image.
Offsets are local OpenCV camera meters: L2M=(-d,0,0), R2M=(d,0,0),
U2M=(0,-d,0), D2M=(0,d,0), with d=monocular_shift.distance_m.
Each frame uses t_target=t_source+R_source@offset; its rotation/calibration
remain unchanged. The requested distance is preserved in global meters.
--chunk-frames 0 keeps the whole selected sequence; positive values split it.
Trailing partial chunks are retained unless --drop-last is specified.

Extra dependencies: Pillow, numpy, scipy, PyYAML, OpenCV and mapbox-earcut
(the last two are already Py123D dependencies), FFmpeg on PATH for MP4 only.
--viewer additionally needs viser==1.0.24. Py123D must be installed.
The CPU rasterizer follows Cosmos's geometry/color conventions; it is not
a byte-identical replacement for Cosmos's OpenGL renderer. World-scenario
occlusion uses the annotation geometry, not measured scene depth.
Projection is automatic: nuScenes uses its pinhole calibration; f-theta
Arrow cameras use their stored polynomials. No generic f-theta calibration
is silently substituted. Nonlinear projections tessellate curves/faces.
Only map layers and semantic attributes actually stored in Arrow are used;
this script does not infer poles, traffic-light cuboids, or lane styles.
Viewer: frame selection, autoplay, map-layer visibility, source RGB overlay,
ego/viewer-oriented BEV, follow source camera, pose recording and export.
Viser's native browser navigation supplies free camera movement.

Map vertices retain their stored elevation. 2D maps are displayed at z=0;
nuScenes maps do not become elevation-aware through this renderer. All
annotations keep global coordinates; one fixed translation rebases EVERY
displayed element for precision. Pose exports undo that translation.
Recorded trajectories export JSON + NPZ, not rewritten Arrow logs. They
describe virtual camera poses only; source RGB always remains original.
--frames defaults to dataset.target_steps; --frames 0 shows the whole scene.
Scene indices are deterministic (shuffle_filter/shuffle_dataloader ignored).
Export processes all matching scenes in dataset.split_types by default,
sequentially. --splits overrides the configured splits. Nonempty
dataset.batch_indices or explicit scene-index options restrict selection;
--max-scenes optionally limits export. The viewer opens one scene by default.
Model and render.coord_flip_xy config sections are ignored: annotations
already share the dataset's native global frame. No double-pass RGB warp
is performed; double_reprojection only selects the final original viewpoint.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import inspect
import json
import math
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import yaml


# Same corner topology as Py123D's native camera overlay, not Cosmos ordering.
BOX_EDGES = np.array([(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2),
                      (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)])
BOX_FACES = [(0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4),
             (1, 2, 6, 5), (4, 5, 6, 7), (0, 1, 2, 3)]
HD_COLORS = {"LANE": (98, 183, 249), "LANE_CENTER": (56, 103, 221),
             "ROAD_LINE": (98, 183, 249), "ROAD_EDGE": (200, 36, 35),
             "CROSSWALK": (206, 131, 63), "STOP_ZONE": (185, 63, 34),
             "SPEED_BUMP": (126, 204, 205)}
WORLD_COLORS = {"LANE": (255, 255, 255), "ROAD_LINE": (181, 164, 71),
                "ROAD_EDGE": (253, 1, 232), "CROSSWALK": (139, 93, 255),
                "STOP_ZONE": (108, 179, 59), "SPEED_BUMP": (20, 254, 185)}
# Coplanar semantic areas overlap in 2D maps. Later surfaces win depth ties;
# draw markings after surfaces, without displacing any annotation geometry.
MAP_SURFACE_ORDER = {"CROSSWALK": 0, "STOP_ZONE": 1, "SPEED_BUMP": 2}
DEPTH_TIE_RTOL = 1e-7
HD_BOX_COLORS = {"Car": (255, 0, 0), "Truck": (0, 0, 255),
                 "Pedestrian": (0, 255, 0), "Cyclist": (255, 255, 0), "Others": (255, 255, 255)}
WORLD_BOX_COLORS = {"Car": ((0, 46, 136), (126, 206, 255)),
                    "Truck": ((204, 55, 0), (255, 192, 64)),
                    "Pedestrian": ((148, 0, 62), (255, 124, 171)),
                    "Cyclist": ((0, 80, 66), (102, 208, 198)),
                    "Others": ((53, 26, 20), (166, 136, 125))}
LAYER_COLORS = {
    "LANE": (145, 195, 215), "LANE_GROUP": (240, 140, 120),
    "INTERSECTION": (210, 180, 195), "CROSSWALK": (240, 170, 75),
    "WALKWAY": (170, 145, 220), "CARPARK": (180, 185, 155),
    "GENERIC_DRIVABLE": (185, 190, 185), "STOP_ZONE": (220, 110, 110),
    "ROAD_EDGE": (235, 115, 115), "ROAD_LINE": (225, 185, 70),
    "SPEED_BUMP": (200, 130, 70),
}


def load_config(path):
    if path is None:
        return {}
    with Path(path).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return value


def configure_roots(paths):
    for key, env in {"temp_path": "PY123D_DATA_ROOT", "nusc_path": "NUSCENES_DATA_ROOT",
                     "av2_path": "AV2_DATA_ROOT", "waymo_path": "WOD_PERCEPTION_DATA_ROOT",
                     "kitti360_path": "KITTI360_DATA_ROOT"}.items():
        if paths.get(key) is not None:
            os.environ[env] = str(Path(paths[key]).expanduser().resolve())


def scene_name(scene):
    return str(getattr(scene, "log_name", None) or
               getattr(scene.scene_metadata, "initial_uuid", "scene"))


def resolve_map(scene):
    map_api = scene.get_map_api()
    if map_api is not None:
        return map_api
    # Some Py123D versions resolve maps using the log dataset alias instead of
    # map_metadata.dataset. nuscenes-interpolated logs share maps/nuscenes/.
    from py123d.api.map.arrow.arrow_map_api import get_lru_cached_map_api
    from py123d.common.runtime import get_dataset_paths

    log = scene.get_log_metadata()
    metadata = getattr(log, "map_metadata", None)
    location = getattr(metadata, "location", None) or log.location
    if location is None:
        return None
    datasets = [getattr(metadata, "dataset", None), log.dataset]
    if str(log.dataset).startswith("nuscenes"):
        datasets.append("nuscenes")
    root = os.environ.get("PY123D_DATA_ROOT")
    maps_root = Path(root) / "maps" if root else get_dataset_paths().py123d_maps_root
    if maps_root is None:
        return None
    for dataset in dict.fromkeys(name for name in datasets if name):
        path = Path(maps_root) / dataset / f"{dataset}_{location}.arrow"
        if path.is_file():
            print(f"Resolved shared map: {path}")
            return get_lru_cached_map_api(path)
    return None


def matrix(pose):
    value = np.asarray(pose.transformation_matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Invalid SE3 pose.")
    return value.copy()


@dataclass(frozen=True)
class RenderView:
    name: str = "original"
    offset_local_xyz_m: tuple = (0., 0., 0.)

    @property
    def distance_m(self):
        return float(np.linalg.norm(self.offset_local_xyz_m))


def select_render_views(config):
    """Double-reprojection's final original view takes priority over all shifts."""
    enabled = (config.get("double_reprojection") or {}).get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("double_reprojection.enabled must be a YAML boolean (true/false).")
    if enabled:
        return [RenderView()]
    dataset = config.get("dataset") or {}
    setup = str(dataset.get("setup", "")).strip().lower().replace("-", "_")
    if setup not in {"monocular_shifted", "monocular_shift"}:
        return [RenderView()]
    shift = config.get("monocular_shift") or {}
    convention = str(shift.get("coordinate_convention", "opencv")).strip().lower()
    if convention != "opencv":
        raise ValueError("Monocular shift requires coordinate_convention: opencv.")
    # Match preprocess.py's fallback, but always honor an explicit YAML value.
    distance = float(shift.get("distance_m", 2.))
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("monocular_shift.distance_m must be positive and finite.")
    directions = dataset.get("render_directions", ["L2M", "R2M", "U2M"])
    if isinstance(directions, str):
        directions = [directions]
    if not isinstance(directions, (list, tuple)) or not directions:
        raise ValueError("dataset.render_directions must be a nonempty list of shift directions.")
    offsets = {"L2M": (-distance, 0., 0.), "R2M": (distance, 0., 0.),
               "U2M": (0., -distance, 0.), "D2M": (0., distance, 0.)}
    names = []
    for value in directions:
        name = str(value).strip().upper()
        for direction in (["L2M", "R2M", "U2M"] if name == "ALL" else [name]):
            if direction not in offsets:
                raise ValueError(f"Unsupported monocular direction {value!r}; use L2M/R2M/U2M/D2M.")
            if direction not in names:
                names.append(direction)
    return [RenderView(name, offsets[name]) for name in names]


def shifted_camera_pose(source_pose, view):
    """Apply a metric camera-local translation, not a global-axis translation."""
    pose = np.asarray(source_pose, dtype=np.float64).copy()
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Source camera pose must be a finite 4x4 camera-to-world transform.")
    if view.distance_m == 0:
        return pose
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation), 1., atol=1e-6):
        raise ValueError("Camera rotation is not a rigid rotation; a metric shift would change scale.")
    world_offset = rotation @ np.asarray(view.offset_local_xyz_m, dtype=float)
    pose[:3, 3] += world_offset
    if not np.isclose(np.linalg.norm(world_offset), view.distance_m, rtol=1e-6, atol=1e-8):
        raise ValueError("Camera transform did not preserve the requested shift distance.")
    return pose


def view_subdirectory(view):
    return Path() if view.name == "original" else Path(view.name)


def to_segments(points, close=False):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"Expected Nx2 or Nx3 geometry, got {points.shape}.")
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    if close and len(points) > 2 and not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float64)
    segments = np.stack((points[:-1], points[1:]), axis=1)
    return segments[np.isfinite(segments).all(axis=(1, 2))]


def points3d(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("Map geometry must be Nx2 or Nx3.")
    return np.column_stack((points, np.zeros(len(points)))) if points.shape[1] == 2 else points.copy()


def polygon_triangles(points):
    """Earcut handles concave map outlines; a triangle fan would fill outside them."""
    import mapbox_earcut

    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3 or not np.isfinite(points).all():
        return np.empty((0, 3, 3))
    # Subtract a local origin for stable triangulation at large map coordinates.
    xy = np.ascontiguousarray(points[:, :2] - points[0, :2])
    indices = mapbox_earcut.triangulate_float64(xy, np.array([len(points)], np.uint32))
    return points[indices.reshape(-1, 3)]


def box_class(box):
    name = (box["label"] + " " + box["default_label"]).upper()
    if any(word in name for word in ("PEDESTRIAN", "PERSON")):
        return "Pedestrian"
    if any(word in name for word in ("CYCLE", "CYCLIST", "RIDER")):
        return "Cyclist"
    if any(word in name for word in ("TRUCK", "BUS", "TRAILER", "CONSTRUCTION")):
        return "Truck"
    if any(word in name for word in ("CAR", "VEHICLE", "AUTOMOBILE")):
        return "Car"
    return "Others"


def sample_polyline(points, step, dash=None):
    """Sample in world meters; dashed patterns remain stable across frames."""
    output, offset = [], 0.
    for a, b in zip(points[:-1], points[1:]):
        length = np.linalg.norm(b - a)
        if not math.isfinite(length) or length < 1e-9:
            continue
        distances = list(np.arange(0., length, step)) + [length]
        if dash is not None:
            period, on = dash
            # Include each pattern boundary, so no segment spans a dash/gap.
            boundaries = []
            for cycle in range(int(offset // period), int((offset + length) // period) + 1):
                for boundary in (cycle * period, cycle * period + on):
                    relative = boundary - offset
                    if 0 < relative < length:
                        boundaries.append(relative)
            distances = sorted(set(distances + boundaries))
        for start, stop in zip(distances[:-1], distances[1:]):
            if dash is None or (offset + (start + stop) / 2) % dash[0] < dash[1]:
                output.append(np.stack((a + (b - a) * start / length, a + (b - a) * stop / length)))
        offset += length
    return np.asarray(output).reshape(-1, 2, 3)


def marking_segments(feature, step):
    points, style = feature["points"], feature["style"]
    color = (255, 255, 0) if "YELLOW" in style else (255, 255, 255) if "WHITE" in style else (
        WORLD_COLORS["LANE"] if feature["layer"] == "LANE" else WORLD_COLORS["ROAD_LINE"])
    if style == "NONE" or len(points) < 2:
        return [], color
    double = any(word in style for word in ("DOUBLE", "DASH_SOLID", "SOLID_DASH"))
    if not double:
        dash = (6., 3.) if "DASH" in style else None
        return sample_polyline(points, step, dash), color
    # Two markings offset in the local XY plane; preserve stored elevation.
    directions = np.zeros_like(points)
    directions[:-1] = points[1:] - points[:-1]
    directions[-1] = directions[-2]
    norm = np.linalg.norm(directions[:, :2], axis=1)
    lateral = np.column_stack((-directions[:, 1], directions[:, 0], np.zeros(len(points))))
    lateral /= np.maximum(norm[:, None], 1e-9)
    patterns = ["DASH" in style, "DASH" in style]
    if "DASH_SOLID" in style:
        patterns = [True, False]
    elif "SOLID_DASH" in style:
        patterns = [False, True]
    segments = [sample_polyline(points + sign * .12 * lateral, step, (6., 3.) if dashed else None)
                for sign, dashed in zip((-1, 1), patterns)]
    return np.concatenate(segments), color


def clip_polygon_depth(points, colors, near, far):
    """Clip in 3D before projection; interpolate attributes at clipping vertices."""
    values = np.column_stack((points, colors))
    for limit, sign in ((near, 1), (far, -1)):
        output = []
        for previous, current in zip(np.roll(values, 1, axis=0), values):
            a, b = sign * (previous[2] - limit) >= 0, sign * (current[2] - limit) >= 0
            if a != b:
                t = (limit - previous[2]) / (current[2] - previous[2])
                output.append(previous + t * (current - previous))
            if b:
                output.append(current)
        if not output:
            return np.empty((0, 3)), np.empty((0, 3))
        values = np.array(output)
    return values[:, :3], values[:, 3:]


class CameraRasterizer:
    """CPU rasterization in native camera pixels, with optional shared z-buffer."""
    def __init__(self, camera, size, depth_test, max_depth=200., step=1., map_z=None, camera_pose=None):
        self.metadata = camera.metadata
        self.pose = matrix(camera.camera_to_global_se3) if camera_pose is None else np.asarray(camera_pose).copy()
        self.width, self.height = size
        self.scale = np.array([self.width / self.metadata.width, self.height / self.metadata.height])
        self.depth_test, self.max_depth, self.step, self.map_z = depth_test, max_depth, step, map_z
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth = np.full((self.height, self.width), np.inf, dtype=np.float64)
        model = str(getattr(self.metadata.camera_model, "name", self.metadata.camera_model)).upper()
        distortion = getattr(self.metadata, "distortion", None)
        self.nonlinear = "PINHOLE" not in model or (
            distortion is not None and not getattr(self.metadata, "is_undistorted", False)
            and np.any(np.asarray(distortion.array) != 0))

    def camera_points(self, points, is_map=False):
        points = np.asarray(points, dtype=np.float64).copy()
        if is_map and self.map_z is not None:
            points[:, 2] = self.map_z
        return (points - self.pose[:3, 3]) @ self.pose[:3, :3]

    def project(self, points):
        pixels, _, _ = self.metadata.project_to_image(points)
        return pixels * self.scale

    def visible_segments(self, segments, is_map=False):
        if not len(segments):
            return segments
        points = self.camera_points(segments.reshape(-1, 3), is_map).reshape(-1, 2, 3)
        eligible = (points[:, :, 2].max(axis=1) >= .05) & (points[:, :, 2].min(axis=1) <= self.max_depth)
        # Edges crossing the near plane need exact clipping in line().
        crossing = (points[:, :, 2].min(axis=1) < .05) & eligible
        front = eligible & ~crossing
        if front.any():
            projected = self.project(points[front].reshape(-1, 3)).reshape(-1, 2, 2)
            bounds = (projected.min(axis=1) <= [self.width + 5, self.height + 5]).all(axis=1)
            bounds &= (projected.max(axis=1) >= [-5, -5]).all(axis=1)
            bounds &= np.isfinite(projected).all(axis=(1, 2))
            eligible[np.flatnonzero(front)] = bounds
        return segments[eligible]

    def paint(self, y, x, z, colors, bias=0.):
        if not len(y):
            return
        fade = np.clip(1. - z / self.max_depth, .15, 1.)
        colors = np.clip(np.asarray(colors) * fade[:, None], 0, 255).astype(np.uint8)
        if self.depth_test:
            # Different triangulations of one plane differ at roundoff scale.
            # Treat those depths as ties instead of generating colored speckles.
            tolerance = DEPTH_TIE_RTOL * np.maximum(1., np.abs(z))
            visible = z <= self.depth[y, x] + bias + tolerance
            y, x, z, colors = y[visible], x[visible], z[visible], colors[visible]
            self.depth[y, x] = np.minimum(self.depth[y, x], z)
            self.image[y, x] = colors
        else:
            # Match the original HDMap renderer's per-channel maximum merge.
            self.image[y, x] = np.maximum(self.image[y, x], colors)

    def line(self, world, color, width=3, is_map=False, bias=0.):
        import cv2

        a, b = self.camera_points(world, is_map)
        low, high = 0., 1.
        dz = b[2] - a[2]
        if abs(dz) < 1e-12:
            if a[2] < .05 or a[2] > self.max_depth:
                return
        else:
            t0, t1 = (.05 - a[2]) / dz, (self.max_depth - a[2]) / dz
            low, high = max(low, min(t0, t1)), min(high, max(t0, t1))
        if low > high:
            return
        a, b = a + low * (b - a), a + high * (b - a)
        uv = self.project(np.stack((a, b)))
        clipped = clip_line(*uv, self.width, self.height)
        if clipped is None:
            return
        c, d = np.asarray(clipped[0]), np.asarray(clipped[1])
        margin = int(math.ceil(width / 2)) + 1
        xmin, ymin = np.maximum(np.floor(np.minimum(c, d)).astype(int) - margin, 0)
        xmax, ymax = np.minimum(np.ceil(np.maximum(c, d)).astype(int) + margin,
                                [self.width - 1, self.height - 1])
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
        cv2.line(mask, tuple(np.rint(c - [xmin, ymin]).astype(int)),
                 tuple(np.rint(d - [xmin, ymin]).astype(int)), 255, width)
        y, x = np.nonzero(mask)
        y, x = y + ymin, x + xmin
        vector = uv[1] - uv[0]
        denominator = max(float(vector @ vector), 1e-12)
        t = np.clip(((np.column_stack((x, y)) - uv[0]) @ vector) / denominator, 0., 1.)
        z = 1. / ((1. - t) / a[2] + t / b[2])
        self.paint(y, x, z, np.broadcast_to(color, (len(y), 3)), bias)

    def triangle(self, world, colors, is_map=False, subdivision=0):
        points = self.camera_points(world, is_map)
        if np.max(points[:, 2]) < .05 or np.min(points[:, 2]) > self.max_depth:
            return
        # Tessellation approximates curved faces for f-theta/distorted cameras.
        if self.nonlinear and subdivision < 5:
            lengths = np.linalg.norm(np.roll(world, -1, axis=0) - world, axis=1)
            edge = int(np.argmax(lengths))
            if lengths[edge] > self.step:
                a, b, other = edge, (edge + 1) % 3, (edge + 2) % 3
                middle = (world[a] + world[b]) / 2
                color_middle = (colors[a] + colors[b]) / 2
                self.triangle(np.stack((world[a], middle, world[other])),
                              np.stack((colors[a], color_middle, colors[other])), is_map, subdivision + 1)
                self.triangle(np.stack((middle, world[b], world[other])),
                              np.stack((color_middle, colors[b], colors[other])), is_map, subdivision + 1)
                return
        clipped, colors = clip_polygon_depth(points, colors, .05, self.max_depth)
        for index in range(1, len(clipped) - 1):
            ids = [0, index, index + 1]
            self.projected_triangle(self.project(clipped[ids]), clipped[ids, 2], colors[ids])

    def projected_triangle(self, pixels, depth, colors):
        if not np.isfinite(pixels).all():
            return
        xmin, ymin = np.floor(np.clip(pixels.min(axis=0), [0, 0], [self.width, self.height])).astype(int)
        xmax, ymax = np.ceil(np.clip(pixels.max(axis=0), [-1, -1],
                                     [self.width - 1, self.height - 1])).astype(int)
        if xmin > xmax or ymin > ymax:
            return
        a, b, c = pixels
        determinant = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(determinant) < 1e-9:
            return
        x, y = np.meshgrid(np.arange(xmin, xmax + 1) + .5, np.arange(ymin, ymax + 1) + .5)
        w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / determinant
        w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / determinant
        w2 = 1. - w0 - w1
        inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
        yy, xx = np.nonzero(inside)
        weights = np.column_stack((w0[inside], w1[inside], w2[inside])) / depth
        total = weights.sum(axis=1)
        z = 1. / total
        rgb = (weights @ colors) / total[:, None]
        self.paint(yy + ymin, xx + xmin, z, rgb)


def render_controls(reader, frame, args, camera_pose=None):
    if frame.camera is None:
        raise ValueError(f"Camera observation unavailable at iteration {frame.iteration}.")
    hd = CameraRasterizer(frame.camera, args.image_size, False, args.max_depth, args.geometry_step, args.map_z, camera_pose)
    world = CameraRasterizer(frame.camera, args.image_size, True, args.max_depth, args.geometry_step, args.map_z, camera_pose)
    width = max(1, round(args.image_size[0] / 320))
    features = sorted(reader.map_features, key=lambda feature: (
        0 if feature["kind"] == "polygon" else 1,
        MAP_SURFACE_ORDER.get(feature["layer"], 0) if feature["kind"] == "polygon" else 0))
    for feature in features:
        layer = feature["layer"]
        if layer not in HD_COLORS:
            continue
        if layer in {"CROSSWALK", "STOP_ZONE", "CARPARK"}:
            continue
        
        if feature["kind"] == "polygon":
            for triangle in feature["triangles"]:
                hd.triangle(triangle, np.tile(HD_COLORS[layer], (3, 1)).astype(float), is_map=True)
                world.triangle(triangle, np.tile(WORLD_COLORS[layer], (3, 1)).astype(float), is_map=True)
        else:
            if feature["style"] == "NONE":
                continue
            if feature.get("_step") != args.geometry_step:
                feature["_dense"] = sample_polyline(feature["points"], args.geometry_step)
                feature["_step"] = args.geometry_step
                feature.pop("_markings", None)
            for segment in hd.visible_segments(feature["_dense"], is_map=True):
                hd.line(segment, HD_COLORS[layer], width, is_map=True)
            #### Remove Redundancy
            if layer in {"LANE_CENTER", "LANE"}:
                continue
            ######################
            if layer in {"ROAD_LINE", "LANE"}:
                if "_markings" not in feature:
                    feature["_markings"] = marking_segments(feature, args.geometry_step)
                segments, color = feature["_markings"]
            else:
                segments, color = feature["_dense"], WORLD_COLORS[layer]
            for segment in world.visible_segments(np.asarray(segments).reshape(-1, 2, 3), is_map=True):
                world.line(segment, color, width, is_map=True, bias=.005)
    if not args.no_boxes:
        # Filled annotation cuboids share a depth buffer with the road geometry.
        for box in frame.boxes:
            corners = np.asarray(box["corners_xyz"])
            category = box_class(box)
            dark, light = WORLD_BOX_COLORS[category]
            colors = np.tile(light, (8, 1)).astype(float)
            colors[[0, 1, 4, 5]] = dark  # Native Py123D front face, not Cosmos corner order.
            for face in BOX_FACES:
                for ids in ([face[0], face[1], face[2]], [face[0], face[2], face[3]]):
                    world.triangle(corners[ids], colors[ids])
            for segment in sample_polyline_segments(corners[BOX_EDGES], args.geometry_step):
                hd.line(segment, HD_BOX_COLORS[category], width)
        # Draw edges after ALL filled boxes, so nearer boxes occlude far edges.
        for box in frame.boxes:
            for segment in sample_polyline_segments(np.asarray(box["corners_xyz"])[BOX_EDGES], args.geometry_step):
                world.line(segment, (100, 100, 100), max(1, width // 2), bias=.02)
    return hd.image, world.image


def sample_polyline_segments(segments, step):
    pieces = [sample_polyline(segment, step) for segment in segments]
    return np.concatenate(pieces) if pieces else np.empty((0, 2, 3))


def box_color(label):
    name = str(label).upper()
    if "PEDESTRIAN" in name or "PERSON" in name:
        return (65, 155, 75)
    if any(word in name for word in ("CYCLE", "CYCLIST", "BICYCLE", "MOTORCYCLE")):
        return (220, 170, 45)
    if any(word in name for word in ("TRUCK", "BUS", "TRAILER")):
        return (75, 110, 215)
    return (120, 175, 95)


def read_boxes(detections):
    boxes = []
    if detections is None:
        return boxes
    for detection in detections:
        box = detection.bounding_box_se3
        corners = np.asarray(box.corners_array, dtype=np.float64)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            continue
        label = detection.attributes.default_label.name
        boxes.append({"track_token": str(detection.attributes.track_token),
                      "label": detection.attributes.label.name, "default_label": label,
                      "center_xyz": matrix(box.center_se3)[:3, 3].tolist(),
                      "dimensions_lwh": [float(box.length), float(box.width), float(box.height)],
                      "quaternion_wxyz": np.asarray(box.center_se3.quaternion.array).tolist(),
                      "corners_xyz": corners.tolist(), "color": box_color(label)})
    return boxes


@dataclass
class Frame:
    iteration: int
    timestamp_us: int
    ego_pose: np.ndarray
    camera: object
    boxes: list
    boxes_available: bool


def json_value(value):
    """Convert geometry/semantic values to portable JSON, without Python pickles."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, "name"):
        return value.name
    if hasattr(value, "array"):
        return json_value(value.array)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} into scene data.")


def serialize_map_object(obj, layer):
    result = {"object_id": str(obj.object_id), "layer": layer, "geometry": {}, "attributes": {}}
    for name in ("left_boundary", "right_boundary", "centerline", "polyline", "outline"):
        value = getattr(obj, name, None)
        if value is not None:
            result["geometry"][name] = json_value(value.array)
    for name in ("lane_type", "road_line_type", "road_edge_type", "intersection_type", "stop_zone_type",
                 "speed_bump_type", "lane_ids", "lane_group_ids", "intersection_id", "lane_group_id",
                 "left_lane_id", "right_lane_id", "predecessor_ids", "successor_ids", "speed_limit_mps"):
        if hasattr(obj, name):
            result["attributes"][name] = json_value(getattr(obj, name))
    return result


class SceneReader:
    def __init__(self, scene, iterations, camera_query, radius):
        self.scene, self.iterations, self.radius = scene, list(iterations), radius
        self.map_api = resolve_map(scene)
        if self.map_api is None:
            raise ValueError("This scene has no available converted HDMap.")
        self.layers = list(self.map_api.get_available_map_layers())
        self.camera_id = None
        query = camera_query.lower().removeprefix("camera.")
        for cid in scene.available_camera_ids:
            if getattr(cid, "name", str(cid)).lower().removeprefix("camera.") == query:
                self.camera_id = cid
                break
        if self.camera_id is None:
            print(f"Camera {camera_query!r} unavailable; map/box viewer still works.")
        self.cache, self.map_segments, self.query_xy = {}, {}, None
        self.map_features, self.map_objects, self.map_revision = [], [], -1
        initial = self.frame(0)
        self.origin = initial.ego_pose[:3, 3].copy()
        self.update_map(initial.ego_pose[:2, 3])

    def frame(self, index):
        if index in self.cache:
            return self.cache[index]
        iteration = self.iterations[index]
        ego = self.scene.get_ego_state_se3_at_iteration(iteration=iteration)
        if ego is None:
            raise ValueError(f"No ego pose at iteration {iteration}.")
        camera = None if self.camera_id is None else self.scene.get_camera_at_iteration(
            iteration=iteration, camera_id=self.camera_id)
        detections = self.scene.get_box_detections_se3_at_iteration(iteration=iteration)
        frame = Frame(iteration, int(self.scene.get_timestamp_at_iteration(iteration).time_us),
                      matrix(ego.center_se3), camera, read_boxes(detections), detections is not None)
        # Bounded image cache; do not preload all video frames into memory.
        if len(self.cache) >= 4:
            self.cache.pop(next(iter(self.cache)))
        self.cache[index] = frame
        return frame

    def update_map(self, xy):
        from py123d.geometry import Point2D
        if self.query_xy is not None and np.linalg.norm(xy - self.query_xy) < self.radius / 3:
            return False
        objects = self.map_api.get_map_objects_in_radius(
            point=Point2D.from_array(np.asarray(xy)), radius=self.radius * 2, layers=self.layers)
        output, features, native_objects = {}, [], []
        for layer, entries in objects.items():
            pieces = []
            for obj in entries:
                native_objects.append(serialize_map_object(obj, layer.name))
                feature_start = len(features)
                if layer.name == "LANE":
                    # Keep native dimensionality, especially for 2D nuScenes maps.
                    for attr in ("left_boundary", "right_boundary", "centerline"):
                        polyline = getattr(obj, attr, None)
                        if polyline is not None:
                            pieces.append(to_segments(polyline.array))
                            features.append({"layer": "LANE_CENTER" if attr == "centerline" else "LANE",
                                             "kind": "line", "points": points3d(polyline.array),
                                             "style": "UNKNOWN"})
                elif hasattr(obj, "polyline"):
                    pieces.append(to_segments(obj.polyline.array))
                    kind = getattr(obj, "road_line_type", None)
                    features.append({"layer": layer.name, "kind": "line",
                                     "points": points3d(obj.polyline.array),
                                     "style": getattr(kind, "name", "UNKNOWN")})
                elif hasattr(obj, "outline"):
                    pieces.append(to_segments(obj.outline.array, close=True))
                    if layer.name in {"CROSSWALK", "STOP_ZONE", "SPEED_BUMP"}:
                        points = points3d(obj.outline.array)
                        features.append({"layer": layer.name, "kind": "polygon", "points": points,
                                         "triangles": polygon_triangles(points), "style": "UNKNOWN"})
                for feature in features[feature_start:]:
                    feature["object_id"] = str(obj.object_id)
                    feature["source_layer"] = layer.name
            output[layer.name] = np.concatenate(pieces) if pieces else np.empty((0, 2, 3))
        self.map_segments, self.map_features, self.query_xy = output, features, np.asarray(xy).copy()
        self.map_objects = native_objects
        self.map_revision += 1
        return True


def clip_line(a, b, width, height):
    """Liang-Barsky clipping also retains edges with both endpoints offscreen."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not np.isfinite([a, b]).all():
        return None
    dx, dy = b - a
    low, high = 0., 1.
    for p, q in zip((-dx, dx, -dy, dy),
                    (a[0], width - 1 - a[0], a[1], height - 1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            low = max(low, q / p)
        else:
            high = min(high, q / p)
        if low > high:
            return None
    return tuple(a + low * (b - a)), tuple(a + high * (b - a))


def source_overlay(frame, size=(800, 450), include_boxes=True):
    if frame.camera is None:
        image = Image.new("RGB", size, (235, 235, 235))
        ImageDraw.Draw(image).text((20, 20), "Camera frame unavailable", fill="black")
        return np.asarray(image)
    camera = frame.camera
    image = Image.fromarray(np.asarray(camera.rgb_image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    if include_boxes:
        c2w = matrix(camera.camera_to_global_se3)
        for box in frame.boxes:
            world = np.asarray(box["corners_xyz"])
            points = (world - c2w[:3, 3]) @ c2w[:3, :3]
            for edge in BOX_EDGES:
                a, b = points[edge].copy()
                near = .05
                if max(a[2], b[2]) < near:
                    continue
                if a[2] < near:
                    a += (b - a) * ((near - a[2]) / (b[2] - a[2]))
                if b[2] < near:
                    b += (a - b) * ((near - b[2]) / (a[2] - b[2]))
                # Native projection includes the camera's own distortion model.
                uv, _, _ = camera.metadata.project_to_image(np.stack((a, b)))
                clipped = clip_line(uv[0], uv[1], image.width, image.height)
                if clipped:
                    draw.line(clipped, fill=box["color"], width=3)
    image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def planar_pose(pose, opencv=False, height=None):
    direction = pose[:3, 2 if opencv else 0].copy()
    norm = np.linalg.norm(direction[:2])
    if norm < 1e-8:
        raise ValueError("Camera points vertically; choose a horizontal viewing direction.")
    forward = np.array([direction[0] / norm, direction[1] / norm, 0.])
    up, left = np.array([0., 0., 1.]), np.cross([0., 0., 1.], forward)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((-left, -up, forward) if opencv else (forward, left, up))
    result[:3, 3] = pose[:3, 3]
    if height is not None:
        result[2, 3] = height
    return result


def bev_image(reader, frame, visible_layers=None, view_pose=None, recorded=None,
              radius=40., size=500, include_boxes=True):
    pose = planar_pose(frame.ego_pose if view_pose is None else view_pose, opencv=view_pose is not None)
    forward = pose[:3, 2] if view_pose is not None else pose[:3, 0]
    left = -pose[:3, 0] if view_pose is not None else pose[:3, 1]
    center = pose[:3, 3]
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    def pixels(points):
        delta = np.asarray(points) - center
        return np.column_stack((size / 2 - delta @ left * size / (2 * radius),
                                size / 2 - delta @ forward * size / (2 * radius)))

    def lines(segments, color, width):
        for segment in segments:
            xy = pixels(segment)
            clipped = clip_line(*xy, size, size)
            if clipped:
                draw.line(clipped, fill=color, width=width)

    for layer, segments in reader.map_segments.items():
        if visible_layers is None or layer in visible_layers:
            lines(segments, LAYER_COLORS.get(layer, (130, 130, 130)), 2)
    if include_boxes:
        for box in frame.boxes:
            # Flatten all 12 cuboid edges; avoids assuming a corner-plane order.
            lines(np.asarray(box["corners_xyz"])[BOX_EDGES], box["color"], 2)
    ego = frame.ego_pose[:3, 3]
    lines(np.array([[ego, ego + frame.ego_pose[:3, 0] * 4]]), (220, 40, 40), 3)
    if recorded and len(recorded) > 1:
        positions = [recorded[key][:3, 3] for key in sorted(recorded)]
        lines(to_segments(positions), (220, 40, 40), 3)
    draw.ellipse((size / 2 - 3, size / 2 - 3, size / 2 + 3, size / 2 + 3), fill="black")
    draw.text((10, 10), f"BEV: +/-{radius:g} m | iteration {frame.iteration}", fill="black")
    return np.asarray(image)


def interpolate_trajectory(recorded, timestamps):
    keys = sorted(recorded)
    if not keys or keys[0] != 0 or keys[-1] != len(timestamps) - 1:
        raise ValueError("Record the first and last displayed frames before exporting.")
    times = (np.asarray(timestamps, dtype=np.int64) - timestamps[0]) / 1e6
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Scene timestamps must be strictly increasing.")
    poses = np.stack([recorded[key] for key in keys])
    output = np.repeat(np.eye(4)[None], len(times), axis=0)
    if len(times) == 1:
        return poses.copy()
    for axis in range(3):
        output[:, axis, 3] = np.interp(times, times[keys], poses[:, axis, 3])
    output[:, :3, :3] = Slerp(times[keys], Rotation.from_matrix(poses[:, :3, :3]))(times).as_matrix()
    return output


def export_poses(reader, recorded, output_dir):
    timestamps = [int(reader.scene.get_timestamp_at_iteration(i).time_us) for i in reader.iterations]
    poses = interpolate_trajectory(recorded, timestamps)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "novel_camera_poses.npz", iterations=reader.iterations,
                        timestamps_us=timestamps, camera_to_world=poses,
                        keyframe_indices=sorted(recorded))
    payload = {"scene": scene_name(reader.scene), "coordinate_frame": "global",
               "camera_convention": "opencv", "transform": "camera_to_world",
               "quaternion_order": "wxyz", "interpolation": "linear translation + SLERP",
               "source_camera_id": str(reader.camera_id), "iterations": reader.iterations,
               "timestamps_us": timestamps, "keyframe_indices": sorted(recorded),
               "camera_to_world": poses.tolist()}
    (output_dir / "novel_camera_poses.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    return output_dir / "novel_camera_poses.npz"


def add_lines(server, name, points, colors, width=2, visible=True):
    # Py123D pins Viser 1.0.24; newer Viser names the width argument thickness.
    kwargs = {"line_width": width} if "line_width" in inspect.signature(
        server.scene.add_line_segments).parameters else {"thickness": width, "thickness_units": "screen"}
    return server.scene.add_line_segments(name, points=np.asarray(points, dtype=np.float32),
                                          colors=colors, visible=visible, **kwargs)


def run_viewer(reader, args, output_dir):
    import viser

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(dark_mode=False)
    lock = threading.RLock()
    recorded, handles, layer_checks = {}, {}, {}
    current = reader.frame(0)
    initial_cam = matrix(current.camera.camera_to_global_se3) if current.camera is not None else None
    mount_height = 0. if initial_cam is None else initial_cam[2, 3] - current.ego_pose[2, 3]
    render_views = {view.name: view for view in getattr(args, "render_views", [RenderView()])}
    state = {"index": -1, "map_refresh": True, "preview_refresh": True, "last_play": time.monotonic()}

    with server.gui.add_folder("Control"):
        slider = server.gui.add_slider("Frame Selector", min=0, max=max(1, len(reader.iterations) - 1),
                                       step=1, initial_value=0, disabled=len(reader.iterations) == 1)
        autoplay = server.gui.add_checkbox("Auto Play", initial_value=False)
        speed = server.gui.add_slider("Playback FPS", min=1., max=60., step=.5, initial_value=min(args.fps, 60.))
        target_view = server.gui.add_dropdown("Control view", tuple(render_views), initial_value=next(iter(render_views)))
        follow = server.gui.add_checkbox("Follow selected control camera", initial_value=False)
        record_button = server.gui.add_button("Record Current Pose")
        reset_button = server.gui.add_button("Reset Recorded Poses")
        ground = server.gui.add_checkbox("Ground recorded poses", initial_value=True)
        export_button = server.gui.add_button("Export Poses")
        status = server.gui.add_markdown("")
    with server.gui.add_folder("Map layers"):
        for layer in reader.layers:
            checked = layer.name not in {"LANE_GROUP", "GENERIC_DRIVABLE"} or len(reader.layers) == 1
            layer_checks[layer.name] = server.gui.add_checkbox(layer.name, initial_value=checked)
    with server.gui.add_folder("Previews"):
        boxes_check = server.gui.add_checkbox("Show 3D boxes", initial_value=not args.no_boxes)
        rgb_check = server.gui.add_checkbox("Show source RGB", initial_value=not args.no_rgb)
        bev_mode = server.gui.add_dropdown("BEV reference", ("ego", "target", "viewer"), initial_value="ego")
        blank = np.full((450, 800, 3), 240, np.uint8)
        rgb_handle = server.gui.add_image(blank, label="Source RGB + 3D boxes", visible=not args.no_rgb)
        bev_handle = server.gui.add_image(np.full((500, 500, 3), 255, np.uint8), label="BEV Projection")

    def changed(_):
        with lock:
            state["preview_refresh"] = True

    for handle in (slider, boxes_check, rgb_check, bev_mode, target_view, follow):
        handle.on_update(changed)
    for handle in layer_checks.values():
        handle.on_update(changed)

    def camera_pose(client):
        pose = np.eye(4)
        # Viser uses wxyz; scipy uses xyzw.
        pose[:3, :3] = Rotation.from_quat(np.asarray(client.camera.wxyz)[[1, 2, 3, 0]]).as_matrix()
        pose[:3, 3] = np.asarray(client.camera.position) + reader.origin
        return pose

    def setup_client(client):
        with lock:
            frame = reader.frame(min(int(slider.value), len(reader.iterations) - 1))
            ego = frame.ego_pose
            center = ego[:3, 3] - reader.origin
            client.camera.position = center - ego[:3, 0] * 12 + np.array([0., 0., 8.])
            client.camera.look_at = center + ego[:3, 0] * 15
            client.camera.up_direction = np.array([0., 0., 1.])
            client.camera.fov = math.pi / 3
        client.camera.on_update(changed)

    server.on_client_connect(setup_client)

    def notify(client, message, error=False):
        if client is not None:
            client.add_notification(title="HDMap viewer", body=message, loading=False,
                                    with_close_button=True, auto_close=5000,
                                    color="red" if error else "green")

    @record_button.on_click
    def record(event):
        if event.client is None:
            return
        with lock:
            try:
                index = min(int(slider.value), len(reader.iterations) - 1)
                pose = camera_pose(event.client)
                if ground.value:
                    frame = reader.frame(index)
                    height = frame.ego_pose[2, 3] + mount_height
                    if frame.camera is not None:
                        height = shifted_camera_pose(matrix(frame.camera.camera_to_global_se3),
                                                     render_views[target_view.value])[2, 3]
                    pose = planar_pose(pose, opencv=True, height=height)
                recorded[index] = pose
                state["preview_refresh"] = True
                notify(event.client, f"Recorded frames: {sorted(recorded)}")
            except ValueError as error:
                notify(event.client, str(error), error=True)

    @reset_button.on_click
    def reset(event):
        with lock:
            recorded.clear()
            state["preview_refresh"] = True
        notify(event.client, "Recorded poses cleared.")

    @export_button.on_click
    def export(event):
        with lock:
            try:
                path = export_poses(reader, recorded, output_dir)
                print(f"Saved {path}")
                notify(event.client, f"Saved trajectory: {path}")
            except (ValueError, OSError) as error:
                notify(event.client, str(error), error=True)

    def set_lines(key, points, color, width=2, visible=True):
        old = handles.pop(key, None)
        if old is not None:
            old.remove()
        if len(points):
            handles[key] = add_lines(server, key, np.asarray(points) - reader.origin,
                                     color, width, visible)

    print(f"Viewer: http://localhost:{args.port} (forward port {args.port} for a remote server)")
    print("Map and boxes use native Arrow annotations; source RGB preview uses native camera calibration.")
    try:
        while True:
            with lock:
                index = min(int(slider.value), len(reader.iterations) - 1)
                frame_changed = index != state["index"]
                frame = reader.frame(index)
                if frame_changed:
                    state["map_refresh"] |= reader.update_map(frame.ego_pose[:2, 3])
                if state["map_refresh"]:
                    for layer, segments in reader.map_segments.items():
                        set_lines(f"/map/{layer}", segments, LAYER_COLORS.get(layer, (130, 130, 130)),
                                  visible=layer_checks[layer].value)
                    state["map_refresh"] = False
                for layer, checkbox in layer_checks.items():
                    if f"/map/{layer}" in handles:
                        handles[f"/map/{layer}"].visible = checkbox.value
                if frame_changed or state["preview_refresh"]:
                    segments, colors = [], []
                    for box in frame.boxes:
                        segments.extend(np.asarray(box["corners_xyz"])[BOX_EDGES])
                        colors.extend([box["color"]] * len(BOX_EDGES))
                    set_lines("/boxes", np.asarray(segments).reshape(-1, 2, 3),
                              np.repeat(np.asarray(colors, dtype=np.uint8)[:, None, :], 2, axis=1)
                              if colors else (120, 175, 95), 3, boxes_check.value)
                    ego = frame.ego_pose
                    set_lines("/ego", np.array([[ego[:3, 3], ego[:3, 3] + ego[:3, 0] * 4]]), (230, 40, 40), 4)
                    old = handles.pop("/source_camera", None)
                    if old is not None:
                        old.remove()
                    old = handles.pop("/target_camera", None)
                    if old is not None:
                        old.remove()
                    target_pose = None
                    if frame.camera is not None:
                        camera = frame.camera
                        pose = matrix(camera.camera_to_global_se3)
                        target_pose = shifted_camera_pose(pose, render_views[target_view.value])
                        # Frustum is a pinhole approximation for display only.
                        width, height = camera.metadata.width, camera.metadata.height
                        fov = getattr(camera.metadata, "fov_y", None)
                        fov = math.pi / 3 if fov is None else float(fov)
                        fov = float(np.clip(fov, .1, math.pi - .1))
                        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                        handles["/source_camera"] = server.scene.add_camera_frustum(
                            "/source_camera", fov=fov, aspect=width / height, scale=1., color=(230, 40, 40),
                            position=pose[:3, 3] - reader.origin, wxyz=quat)
                        if target_view.value != "original":
                            handles["/target_camera"] = server.scene.add_camera_frustum(
                                "/target_camera", fov=fov, aspect=width / height, scale=1., color=(40, 110, 230),
                                position=target_pose[:3, 3] - reader.origin, wxyz=quat)
                        if follow.value:
                            for client in server.get_clients().values():
                                client.camera.position = target_pose[:3, 3] - reader.origin
                                client.camera.wxyz = quat
                                client.camera.fov = fov
                    rgb_handle.visible = rgb_check.value
                    if rgb_check.value:
                        rgb_handle.image = source_overlay(frame, include_boxes=boxes_check.value)
                    view = None
                    if bev_mode.value == "target" and target_pose is not None:
                        view = planar_pose(target_pose, opencv=True)
                    elif bev_mode.value == "viewer":
                        clients = list(server.get_clients().values())
                        if clients:
                            try:
                                view = planar_pose(camera_pose(clients[0]), opencv=True)
                            except ValueError:
                                pass  # Vertical view: keep usable ego-oriented BEV.
                    if view is not None:
                        # Query around the free camera as well as the current ego.
                        if reader.update_map(view[:2, 3]):
                            state["map_refresh"] = True
                    visible_layers = {name for name, cb in layer_checks.items() if cb.value}
                    bev_handle.image = bev_image(reader, frame, visible_layers, view, recorded,
                                                  args.radius, include_boxes=boxes_check.value)
                    if len(recorded) > 1:
                        set_lines("/recorded_path", to_segments([recorded[k][:3, 3] for k in sorted(recorded)]),
                                  (230, 40, 40), 3)
                    elif "/recorded_path" in handles:
                        handles.pop("/recorded_path").remove()
                    status.content = (f"**{scene_name(reader.scene)}**\n\nIteration {frame.iteration}; "
                                      f"{len(frame.boxes)} boxes; control view {target_view.value} "
                                      f"({render_views[target_view.value].distance_m:g} m); recorded frames {sorted(recorded)}" +
                                      ("\n\nBox annotations unavailable for this frame." if not frame.boxes_available else ""))
                    state["preview_refresh"] = False
                    state["index"] = index
                now = time.monotonic()
                if autoplay.value and now - state["last_play"] >= 1. / speed.value:
                    slider.value = (index + 1) % len(reader.iterations)
                    state["last_play"] = now
            time.sleep(.025)
    except KeyboardInterrupt:
        print("Viewer stopped.")
    finally:
        server.stop()


class MP4Writer:
    def __init__(self, path, size, fps):
        self.path, self.size = path, tuple(size)
        path.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                   "-threads", "2", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, image):
        image = np.asarray(image, dtype=np.uint8)
        if image.shape != (self.size[1], self.size[0], 3):
            raise ValueError(f"Unexpected video frame shape {image.shape}.")
        self.process.stdin.write(image.tobytes())

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is not None:
            self.process.terminate()
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        code = self.process.wait()
        if code and error_type is None:
            raise RuntimeError(f"FFmpeg failed writing {self.path} (exit code {code}).")


def video_chunks(count, length=0, overlap=0, drop_last=False):
    if length == 0:
        return [(0, count)] if count else []
    if length < 1 or overlap < 0 or overlap >= length:
        raise ValueError("Chunk length must be positive and overlap must be smaller than it.")
    chunks = []
    for first in range(0, count, length - overlap):
        stop = min(first + length, count)
        if stop - first < length and drop_last:
            break
        chunks.append((first, stop))
        if stop == count:
            break
    return chunks


class SceneBundle:
    """One file per scene; enough geometry/calibration to replay the controls."""
    def __init__(self, reader, args, scene_id, clip_id):
        self.snapshots, self.calibrations, self.frames = [], [], {}
        self.revisions, self.calibration_ids = {}, {}
        log_reader = getattr(reader.scene, "get_log_metadata", None)
        log = log_reader() if log_reader else None
        metadata = getattr(log, "map_metadata", None)
        self.data = {
            "schema": "py123d_scene_bundle", "schema_version": 1,
            "scene": scene_name(reader.scene), "scene_id": scene_id, "clip_id": clip_id,
            "dataset": getattr(log, "dataset", None), "split": getattr(log, "split", None),
            "map_metadata": metadata.to_dict() if metadata is not None else None,
            "coordinates": {"frame": "native dataset global", "units": "meters",
                "camera_axes": "+x right, +y down, +z forward (OpenCV)",
                "pose_convention": "4x4 camera/ego-to-world matrices; column vectors",
                "dimensions_order": "length,width,height", "quaternion_order": "w,x,y,z",
                "viewer_origin_xyz": reader.origin.tolist(),
                "map_2d_policy": "Stored XY preserved in native_objects; render geometry uses Z=0."},
            "coverage": "Frames actually rendered and map regions queried along them; not the entire city map.",
            "rgb_pixels_embedded": False,
            "offline_dependencies": "This script and Py123D calibration classes; no Arrow logs, raw datasets, LiDAR or models.",
            "render_settings": {"image_size_wh": list(args.image_size), "fps": args.fps,
                "max_depth": args.max_depth, "geometry_step": args.geometry_step,
                "map_z": args.map_z, "no_boxes": args.no_boxes,
                "map_query_radius": reader.radius, "map_query_multiplier": 2.,
                "near_plane_m": .05, "pixel_center_offset": .5,
                "depth_tie_rtol": DEPTH_TIE_RTOL},
            "render_style": {"hd_colors": HD_COLORS, "world_colors": WORLD_COLORS,
                "hd_box_colors": HD_BOX_COLORS, "world_box_colors": WORLD_BOX_COLORS,
                "box_edges": BOX_EDGES, "box_faces": BOX_FACES,
                "map_surface_order": MAP_SURFACE_ORDER},
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def record(self, reader, frame, index, record):
        # Capture query snapshots, rather than using a union which could change visibility.
        revision = reader.map_revision
        if revision not in self.revisions:
            self.revisions[revision] = len(self.snapshots)
            self.snapshots.append(json_value({
                "query_center_xy": reader.query_xy, "native_objects": reader.map_objects,
                "features": [{k: v for k, v in feature.items() if not k.startswith("_")}
                             for feature in reader.map_features],
                "segments_by_layer": reader.map_segments}))
        calibration = json_value(frame.camera.metadata.to_dict())
        key = json.dumps(calibration, sort_keys=True, allow_nan=False)
        if key not in self.calibration_ids:
            self.calibration_ids[key] = len(self.calibrations)
            self.calibrations.append(calibration)
        self.frames[index] = json_value({**record, "selected_frame_index": index,
            "ego_to_world": frame.ego_pose, "boxes": frame.boxes,
            "calibration_id": self.calibration_ids[key],
            "map_snapshot_id": self.revisions[revision]})

    def save(self, path, render_metadata):
        data = json_value({**self.data, "calibrations": self.calibrations,
            "map_snapshots": self.snapshots, "frames": [self.frames[k] for k in sorted(self.frames)],
            "render_metadata": render_metadata})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(data, handle, separators=(",", ":"), allow_nan=False)
        temporary.replace(path)
        print(f"Saved offline scene data: {path}")


def replay_scene(args):
    """Recreate HDMap/world_scenario using ONLY a scene JSON and calibration code."""
    path = Path(args.replay_scene).expanduser().resolve()
    with path.open() as handle:
        data = json.load(handle)
    if data.get("schema") != "py123d_scene_bundle" or data.get("schema_version") != 1:
        raise ValueError("Expected a py123d_scene_bundle schema_version=1 JSON file.")
    if args.viewer or args.comparison or args.save_boxes:
        raise ValueError("Offline replay exports the two control streams; viewer/comparison/save-boxes are live-mode options.")
    # Uses native calibration deserialization, including f-theta polynomials/distortion.
    # Does not construct a SceneAPI or touch any Arrow/raw sensor files.
    from py123d.datatypes.sensors.base_camera import camera_metadata_from_dict
    from py123d.datatypes.sensors import pinhole_camera, ftheta_camera, fisheye_mei_camera  # noqa: F401
    settings, metadata = data["render_settings"], data["render_metadata"]
    args.image_size = tuple(args.image_size or settings["image_size_wh"])
    stride = args.stride if args.stride is not None else 1
    if stride < 1 or args.frame_offset < 0 or (args.frames is not None and args.frames < 0):
        raise ValueError("Replay stride must be positive; frames and offset must be nonnegative.")
    args.fps = args.fps if args.fps is not None else settings["fps"] / stride
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("FPS must be positive and finite.")
    if len(args.image_size) != 2 or any(v < 1 for v in args.image_size):
        raise ValueError("Image dimensions must be positive.")
    if not args.frames_only and any(v % 2 for v in args.image_size):
        raise ValueError("MP4 dimensions must be even; use --frames-only for odd dimensions.")
    if not args.frames_only and not shutil.which("ffmpeg"):
        raise RuntimeError("MP4 replay needs FFmpeg; use --frames-only for PNG export.")
    for name in ("max_depth", "geometry_step", "map_z"):
        if getattr(args, name) is None:
            setattr(args, name, settings[name])
    args.no_boxes = settings["no_boxes"] if args.no_boxes is None else args.no_boxes
    args.frame_layout = args.frame_layout or metadata.get("frame_layout") or "original"
    if (not math.isfinite(args.max_depth) or args.max_depth <= .05 or
        not math.isfinite(args.geometry_step) or args.geometry_step <= 0 or
        (args.map_z is not None and not math.isfinite(args.map_z))):
        raise ValueError("Max depth must exceed 0.05; geometry step must be positive; map Z must be finite.")
    preserve_chunks = (args.chunk_frames is None and args.overlap is None and not args.drop_last
                       and args.frame_offset == 0 and stride == 1 and not args.frames)
    args.chunk_frames = args.chunk_frames if args.chunk_frames is not None else 0
    args.overlap = args.overlap if args.overlap is not None else 0
    if args.chunk_frames < 0 or args.overlap < 0 or (args.overlap and args.overlap >= args.chunk_frames):
        raise ValueError("Chunk frames must be nonnegative; overlap must be nonnegative and smaller than a positive chunk length.")
    # Restore the saved palette/topology, even if module defaults have changed.
    style = data["render_style"]
    for target, key in ((HD_COLORS, "hd_colors"), (WORLD_COLORS, "world_colors"),
                        (HD_BOX_COLORS, "hd_box_colors"), (WORLD_BOX_COLORS, "world_box_colors")):
        target.clear()
        target.update(style[key])
    global BOX_EDGES, BOX_FACES, DEPTH_TIE_RTOL, MAP_SURFACE_ORDER
    BOX_EDGES, BOX_FACES = np.asarray(style["box_edges"], dtype=int), style["box_faces"]
    DEPTH_TIE_RTOL = settings.get("depth_tie_rtol", DEPTH_TIE_RTOL)
    MAP_SURFACE_ORDER = style.get("map_surface_order", MAP_SURFACE_ORDER)
    records = data["frames"][args.frame_offset::stride]
    if args.frames:
        records = records[:args.frames]
    if not records:
        raise ValueError("No frames selected in the scene JSON.")
    if preserve_chunks and metadata.get("chunks"):
        positions = {record["selected_frame_index"]: i for i, record in enumerate(records)}
        chunks = []
        for saved in metadata["chunks"]:
            positions_in_chunk = [positions[i] for i in saved["selected_frame_indices"]]
            if not positions_in_chunk or positions_in_chunk != list(range(positions_in_chunk[0], positions_in_chunk[-1] + 1)):
                raise ValueError("Saved chunk frame indices must form a contiguous sequence.")
            chunks.append((positions_in_chunk[0], positions_in_chunk[-1] + 1))
    else:
        chunks = video_chunks(len(records), args.chunk_frames, args.overlap, args.drop_last)
    if not chunks:
        raise ValueError("No replay chunks selected.")
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else path.parent / (path.stem + "_replay")
    # Default directory is separate from the source outputs; RGB remains in the original export.
    views = metadata["render_views"]
    camera_folder = f"{metadata['camera_model']}_{metadata['source_camera'].lower()}"
    calibrations = [camera_metadata_from_dict(value) for value in data["calibrations"]]
    save_frames = args.frames_only or args.save_frames
    written, snapshot_cache = set(), {}
    for chunk_index, (first, stop) in enumerate(chunks):
        with ExitStack() as stack:
            writers = {} if args.frames_only else {(kind, view["name"]): stack.enter_context(MP4Writer(
                output / kind / camera_folder / view_subdirectory(RenderView(view["name"])) /
                f"{data['clip_id']}_{chunk_index}.mp4", args.image_size, args.fps))
                for kind in ("hdmap", "world_scenario") for view in views}
            for record in records[first:stop]:
                snapshot_id = record["map_snapshot_id"]
                if snapshot_id not in snapshot_cache:
                    snapshot = data["map_snapshots"][snapshot_id]
                    features = [{k: np.asarray(v, dtype=float) if k in {"points", "triangles"} else v
                                 for k, v in feature.items()} for feature in snapshot["features"]]
                    snapshot_cache[snapshot_id] = SimpleNamespace(map_features=features)
                source_pose = np.asarray(record["source_camera_to_world"], dtype=float)
                camera = SimpleNamespace(metadata=calibrations[record["calibration_id"]],
                    camera_to_global_se3=SimpleNamespace(transformation_matrix=source_pose))
                frame = Frame(record["iteration"], record["timestamp_us"], np.asarray(record["ego_to_world"]),
                              camera, record["boxes"], record["boxes_available"])
                index = record["selected_frame_index"]
                for view_data in views:
                    view = RenderView(view_data["name"], tuple(view_data["offset_local_xyz_m"]))
                    target = np.asarray(record["target_camera_to_world"][view.name], dtype=float)
                    images = render_controls(snapshot_cache[snapshot_id], frame, args, target)
                    for kind, pixels in zip(("hdmap", "world_scenario"), images):
                        if not args.frames_only:
                            writers[kind, view.name].write(pixels)
                        if save_frames and index not in written:
                            folder = frame_output_directory(output, kind, camera_folder, data["clip_id"],
                                                            data["scene_id"], args.frame_layout, view)
                            folder.mkdir(parents=True, exist_ok=True)
                            digits = 5 if args.frame_layout == "original" else 6
                            Image.fromarray(pixels).save(folder / f"{index:0{digits}d}.png")
                written.add(index)
    print(f"Offline replay complete: {len(records)} selected frames, {len(views)} views under {output}")


def frame_output_directory(output_dir, category, camera_folder, clip_id, scene_id, layout, view=None):
    if layout == "original":
        name = scene_id if view is None else f"{view.name}_{scene_id}"
        return output_dir / category / name
    return output_dir / category / camera_folder / (view_subdirectory(view) if view else Path()) / f"{clip_id}_frames"


def render_video(reader, args, output_dir, clip_id=None):
    """Export videos and/or image sequences at the selected camera poses."""
    frames_only = getattr(args, "frames_only", False)
    save_frames = args.save_frames or frames_only
    layout = getattr(args, "frame_layout", None) or "original"
    digits = 5 if layout == "original" else 6
    if not frames_only and not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required for MP4 export (pixi add ffmpeg), or use --frames-only.")
    if reader.camera_id is None:
        raise ValueError("The requested camera is unavailable. Select an available --camera-id.")
    frame = reader.frame(0)
    if frame.camera is None:
        raise ValueError("Initial camera observation is unavailable.")
    model = frame.camera.metadata.camera_model.name.lower()
    camera_folder = f"{model}_{reader.camera_id.name.lower()}"
    views = getattr(args, "render_views", [RenderView()])
    clip_id = clip_id or re.sub(r"[^A-Za-z0-9._-]+", "_", scene_name(reader.scene)).strip("._-")
    scene_id = getattr(args, "frame_scene_id", None) or re.sub(
        r"[^A-Za-z0-9._-]+", "_", scene_name(reader.scene)).strip("._-") or "scene"
    rgb_frame_dir = frame_output_directory(output_dir, "videos", camera_folder, clip_id, scene_id, layout)
    control_frame_dirs = {(name, view.name): frame_output_directory(
        output_dir, name, camera_folder, clip_id, scene_id, layout, view)
        for view in views for name in ("hdmap", "world_scenario")}
    comparison_frame_dirs = {view.name: frame_output_directory(
        output_dir, Path("videos") / "comparison", camera_folder, clip_id, scene_id, layout, view) for view in views}
    chunks = video_chunks(len(reader.iterations), args.chunk_frames, args.overlap, args.drop_last)
    if not chunks:
        raise ValueError("No complete chunk is available; reduce --chunk-frames or remove --drop-last.")
    print(f"Rendering {model} camera {reader.camera_id.name}: {args.image_size}, {len(chunks)} chunk(s).")
    print("Control views: " + ", ".join(f"{view.name} ({view.distance_m:g} m)" for view in views))
    metadata_path = (rgb_frame_dir / "metadata.json" if frames_only else
                     output_dir / "videos" / camera_folder / f"{clip_id}_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scene": scene_name(reader.scene), "source_camera": reader.camera_id.name,
               "camera_model": model, "image_size_wh": list(args.image_size), "fps": args.fps,
               "camera_calibration": frame.camera.metadata.to_dict(),
               "camera_calibration_role": "source; target views share intrinsics and have their own poses below",
               "projection": "native Py123D calibration scaled to output resolution",
               "map_z_override": args.map_z, "world_occlusion": "annotation geometry z-buffer",
               "double_reprojection_enabled": getattr(args, "double_reprojection_enabled", False),
               "source_rgb_view": "original", "source_rgb_is_shifted": False,
               "target_translation_rule": "t_target = t_source + R_source @ offset_local_xyz_m",
               "target_rotation_rule": "R_target = R_source",
               "camera_to_world_role": "source (legacy alias for source_camera_to_world)",
               "render_views": [{"name": view.name, "offset_local_xyz_m": list(view.offset_local_xyz_m),
                                  "distance_m": view.distance_m, "offset_convention": "opencv",
                                  "target_camera_to_imu": shifted_camera_pose(
                                      matrix(frame.camera.metadata.camera_to_imu_se3), view).tolist(),
                                  "controls_relative_subdir": str(Path(camera_folder) / view_subdirectory(view)),
                                  "rgb_is_same_view": view.name == "original"} for view in views],
               "source_rgb_box_overlay": args.rgb_boxes and not args.no_boxes, "chunks": []}
    payload["output_formats"] = (["png"] if frames_only else ["mp4", "png"] if save_frames else ["mp4"])
    payload["frame_layout"] = layout if save_frames else None
    payload["frame_digits"] = digits if save_frames else None
    if save_frames:
        payload["frame_directories"] = {"source_rgb": str(rgb_frame_dir.relative_to(output_dir)),
            "controls": {view.name: {name: str(control_frame_dirs[name, view.name].relative_to(output_dir))
                         for name in ("hdmap", "world_scenario")} for view in views}}
    bundle = None if getattr(args, "no_scene_data", False) else SceneBundle(reader, args, scene_id, clip_id)
    frame_records = {}
    png_written = set()
    # Reuse frames across overlapped chunks and avoid rerendering their controls.
    overlap_cache = {}
    for chunk_index, (first, stop) in enumerate(chunks):
        filename = f"{clip_id}_{chunk_index}.mp4"
        with ExitStack() as stack:
            rgb_writer = stack.enter_context(MP4Writer(output_dir / "videos" / camera_folder / filename,
                                                       args.image_size, args.fps)) if not frames_only else None
            writers = {(name, view.name): stack.enter_context(MP4Writer(
                output_dir / name / camera_folder / view_subdirectory(view) / filename, args.image_size, args.fps))
                for view in views for name in ("hdmap", "world_scenario")} if not frames_only else {}
            comparisons = {view.name: stack.enter_context(MP4Writer(
                output_dir / "videos" / "comparison" / camera_folder / view_subdirectory(view) / filename,
                (args.image_size[0] * 3, args.image_size[1]), args.fps)) for view in views} if args.comparison and not frames_only else {}
            for index in range(first, stop):
                if index in overlap_cache:
                    controls, rgb = overlap_cache[index]
                else:
                    frame = reader.frame(index)
                    reader.update_map(frame.ego_pose[:2, 3])
                    if frame.camera is None:
                        raise ValueError(f"Camera observation missing at iteration {frame.iteration}.")
                    source_pose = matrix(frame.camera.camera_to_global_se3)
                    targets = {view.name: shifted_camera_pose(source_pose, view) for view in views}
                    controls = {view.name: render_controls(reader, frame, args, targets[view.name]) for view in views}
                    rgb = source_overlay(frame, size=args.image_size, include_boxes=args.rgb_boxes and not args.no_boxes)
                    offsets = {name: target[:3, 3] - source_pose[:3, 3] for name, target in targets.items()}
                    frame_records[index] = {"iteration": frame.iteration, "timestamp_us": frame.timestamp_us,
                                            "boxes_available": frame.boxes_available,
                                            "box_count": len(frame.boxes),
                                            "camera_to_world": source_pose.tolist(),
                                            "source_camera_to_world": source_pose.tolist(),
                                            "target_camera_to_world": {name: target.tolist() for name, target in targets.items()},
                                            "world_shift_xyz_m": {name: offset.tolist() for name, offset in offsets.items()},
                                            "world_shift_distance_m": {name: float(np.linalg.norm(offset)) for name, offset in offsets.items()}}
                    if save_frames:
                        frame_records[index]["frame_filename"] = f"{index:0{digits}d}.png"
                    if bundle is not None:
                        bundle.record(reader, frame, index, frame_records[index])
                    if args.save_boxes:
                        box_path = output_dir / "videos" / camera_folder / f"{clip_id}_boxes" / f"{index:06d}.json"
                        box_path.parent.mkdir(parents=True, exist_ok=True)
                        box_path.write_text(json.dumps({**frame_records[index], "coordinate_frame": "global",
                                                       "dimension_order": "lwh", "quaternion_order": "wxyz",
                                                       "boxes": frame.boxes}, allow_nan=False))
                if rgb_writer is not None:
                    rgb_writer.write(rgb)
                write_png = save_frames and index not in png_written
                if write_png:
                    image_path = rgb_frame_dir / f"{index:0{digits}d}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(image_path)
                for view in views:
                    hdmap, world = controls[view.name]
                    for name, image in (("hdmap", hdmap), ("world_scenario", world)):
                        if not frames_only:
                            writers[name, view.name].write(image)
                        if write_png:
                            image_path = control_frame_dirs[name, view.name] / f"{index:0{digits}d}.png"
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(image).save(image_path)
                    if args.comparison:
                        panel = Image.fromarray(np.concatenate((rgb, hdmap, world), axis=1))
                        draw = ImageDraw.Draw(panel)
                        for column, label in enumerate(("RGB: original", f"HDMap: {view.name} ({view.distance_m:g} m)",
                                                        f"World scenario: {view.name} ({view.distance_m:g} m)")):
                            x = column * args.image_size[0]
                            draw.rectangle((x, 0, x + args.image_size[0] - 1, 19), fill="black")
                            draw.text((x + 5, 4), label, fill="white")
                        if view.name in comparisons:
                            comparisons[view.name].write(np.asarray(panel))
                        if write_png:
                            image_path = comparison_frame_dirs[view.name] / f"{index:0{digits}d}.png"
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            panel.save(image_path)
                if write_png:
                    png_written.add(index)
                if args.overlap and index >= stop - args.overlap and chunk_index + 1 < len(chunks):
                    overlap_cache[index] = (controls, rgb)
                else:
                    overlap_cache.pop(index, None)
                if (index - first) % 10 == 0 or index == stop - 1:
                    print(f"  Chunk {chunk_index}: frame {index - first + 1}/{stop - first}")
        payload["chunks"].append({"filename": None if frames_only else filename, "frame_count": stop - first,
                                   "selected_frame_indices": list(range(first, stop)),
                                   "frames": [frame_records[i] for i in range(first, stop)]})
    if bundle is not None:
        bundle_path = output_dir / "scene_data" / f"{scene_id}.json"
        payload["offline_scene_data"] = str(bundle_path.relative_to(output_dir))
        bundle.save(bundle_path, payload)
    metadata_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"Saved {', '.join(payload['output_formats'])}: hdmap/, world_scenario/, videos/ under {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Existing preprocessing YAML.")
    parser.add_argument("--replay-scene", metavar="JSON", help="Recreate hdmap/world_scenario from a saved scene_data JSON; no Arrow logs or config needed.")
    parser.add_argument("--no-scene-data", action="store_true", help="Disable automatic scene_data/<scene>.json export.")
    parser.add_argument("--data-root", help="Converted Py123D root containing logs/ and maps/.")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--splits", nargs="+", help="Override dataset.split_types; default uses all configured splits.")
    parser.add_argument("--log-names", nargs="+")
    parser.add_argument("--scene-index", type=int, help="Process only this deterministic scene index.")
    parser.add_argument("--scene-indices", nargs="+", type=int, help="Explicit deterministic scene indices to export.")
    parser.add_argument("--max-scenes", type=int, default=0, help="Maximum scenes to export; default=0 (all selected scenes). Viewer defaults to one scene.")
    parser.add_argument("--frames", type=int, help="Number of sampled frames; 0=all; default=dataset.target_steps.")
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--stride", type=int, help="Native stride; default=dataset.frame_sampling.")
    parser.add_argument("--camera-id", help="Default=dataset.source_cam_id, or camera.pcam_f0.")
    parser.add_argument("--radius", type=float, default=40., help="BEV half-extent in meters.")
    parser.add_argument("--map-radius", type=float, default=100., help="Map query radius parameter in meters.")
    parser.add_argument("--max-depth", type=float, help="Control clipping/fade depth in meters; default=200 live, saved value in replay.")
    parser.add_argument("--geometry-step", type=float, help="3D curve/face sampling scale in meters; default=1 live, saved value in replay.")
    parser.add_argument("--map-z", type=float, help="Optional constant map plane in GLOBAL meters; default keeps stored Z.")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="Default=dataset.image_size, or 800 450.")
    parser.add_argument("--chunk-frames", type=int, help="0=one full video; positive values split each stream identically. Default=0 live; saved chunks in full replay.")
    parser.add_argument("--overlap", type=int, help="Overlap between chunks, in selected frames; default=0 live, saved chunks in full replay.")
    parser.add_argument("--drop-last", action="store_true", help="Discard a trailing incomplete chunk.")
    rgb_boxes = parser.add_mutually_exclusive_group()
    rgb_boxes.add_argument("--rgb-boxes", dest="rgb_boxes", action="store_true", help="Overlay 3D boxes on videos/ RGB (default).")
    rgb_boxes.add_argument("--no-rgb-boxes", dest="rgb_boxes", action="store_false", help="Save clean source RGB without box overlays.")
    parser.set_defaults(rgb_boxes=True)
    parser.add_argument("--comparison", action="store_true", help="Also save RGB | HDMap | world_scenario under videos/comparison/.")
    parser.add_argument("--save-boxes", action="store_true", help="Save matching per-frame global 3D box JSONs under videos/.")
    parser.add_argument("--fps", type=float, help="Playback/output rate; default=scene FPS divided by stride.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--output-dir", help="Default=<paths.output_dir>/hdmap_render, or ./hdmap_render.")
    boxes = parser.add_mutually_exclusive_group()
    boxes.add_argument("--no-boxes", dest="no_boxes", action="store_true", help="Disable box rendering.")
    boxes.add_argument("--boxes", dest="no_boxes", action="store_false", help="Enable box rendering, including when disabled in saved replay settings.")
    parser.set_defaults(no_boxes=None)
    parser.add_argument("--no-rgb", action="store_true", help="Start viewer with RGB preview disabled.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--viewer", action="store_true", help="Open the interactive Viser viewer instead of exporting.")
    modes.add_argument("--video", action="store_true", help="Export all three MP4 streams (also the default).")
    modes.add_argument("--frames-only", action="store_true", help="Save individual PNG frames without MP4 encoding; no FFmpeg required.")
    parser.add_argument("--save-frames", action="store_true", help="Also save PNG frames alongside MP4s; implied by --frames-only.")
    parser.add_argument("--frame-layout", choices=("original", "stream"), help="Optional override. Default=original live, saved layout in replay; original: hdmap/R2M_scene-0571/00000.png; stream: older camera/clip_frames layout.")
    args = parser.parse_args()
    args.save_frames = args.save_frames or args.frames_only
    if args.replay_scene:
        replay_scene(args)
        return
    args.max_depth = args.max_depth if args.max_depth is not None else 200.
    args.geometry_step = args.geometry_step if args.geometry_step is not None else 1.
    args.chunk_frames = args.chunk_frames if args.chunk_frames is not None else 0
    args.overlap = args.overlap if args.overlap is not None else 0
    args.frame_layout = args.frame_layout or "original"
    args.no_boxes = bool(args.no_boxes)
    config = load_config(args.config)
    try:
        args.render_views = select_render_views(config)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    args.double_reprojection_enabled = (config.get("double_reprojection") or {}).get("enabled", False)
    paths, dataset = dict(config.get("paths") or {}), config.get("dataset") or {}
    if args.data_root:
        paths["temp_path"] = args.data_root
    configure_roots(paths)  # Must happen before Py123D imports.
    stride = args.stride if args.stride is not None else int(dataset.get("frame_sampling", 1))
    frames = args.frames if args.frames is not None else int(dataset.get("target_steps", 0))
    args.image_size = tuple(args.image_size or dataset.get("image_size", [800, 450]))
    camera_query = args.camera_id or dataset.get("source_cam_id", dataset.get("left_cam_id", "camera.pcam_f0"))
    if args.scene_index is not None and args.scene_indices is not None:
        parser.error("Choose --scene-index or --scene-indices.")
    if stride < 1 or frames < 0 or args.frame_offset < 0 or args.max_scenes < 0:
        parser.error("Stride must be positive; frames, offset, and maximum scenes must be nonnegative.")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.radius, args.map_radius, args.geometry_step)) or args.max_depth <= .05 or not math.isfinite(args.max_depth):
        parser.error("Radii and geometry step must be positive/finite; max depth must be greater than 0.05.")
    if not 1 <= args.port <= 65535 or (args.map_z is not None and not math.isfinite(args.map_z)):
        parser.error("Port must be in 1..65535; map Z must be finite.")
    if len(args.image_size) != 2 or any(not isinstance(v, int) or v < 1 for v in args.image_size):
        parser.error("Image width and height must be positive integers.")
    if not args.frames_only and not args.viewer and any(v % 2 for v in args.image_size):
        parser.error("Image width and height must be even for H264/yuv420p; --frames-only allows odd sizes.")
    if args.chunk_frames < 0 or args.overlap < 0 or (args.overlap and args.chunk_frames <= args.overlap):
        parser.error("Chunk frames must be nonnegative; overlap must be smaller than a positive chunk length.")
    from py123d.api import SceneFilter, get_filtered_scenes
    scene_filter = SceneFilter(datasets=args.datasets or dataset.get("datasets", ["nuscenes-interpolated"]),
                               split_types=args.splits or dataset.get("split_types", ["val"]),
                               log_names=args.log_names or dataset.get("log_names"), has_map=None,
                               required_scene_modalities=["ego_state_se3"] + ([] if args.viewer else
                                   ["camera." + str(camera_query).lower().removeprefix("camera.")]), shuffle=False)
    data_root = os.environ.get("PY123D_DATA_ROOT")
    scenes = get_filtered_scenes(scene_filter, data_root=None if data_root is None else Path(data_root))
    if not scenes:
        raise SystemExit("No scenes found. Check data root, dataset/split, and source-camera availability.")
    indices = args.scene_indices or ([args.scene_index] if args.scene_index is not None else
                                    dataset.get("batch_indices") or list(range(len(scenes))))
    indices = list(dict.fromkeys(int(i) for i in indices))
    if any(i < 0 or i >= len(scenes) for i in indices):
        parser.error(f"Scene indices must be in 0..{len(scenes) - 1}.")
    if args.max_scenes:
        indices = indices[:args.max_scenes]
    elif args.viewer and args.scene_indices is None and args.scene_index is None and not dataset.get("batch_indices"):
        indices = indices[:1]
    if args.viewer and len(indices) != 1:
        parser.error("The viewer opens one scene; choose --scene-index or --max-scenes 1.")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("FPS must be positive and finite.")
    output_dir = (Path(args.output_dir).expanduser() if args.output_dir else
                  Path(paths.get("output_dir", ".")).expanduser() / "hdmap_render").resolve()
    if any(name.startswith("nuscenes") for name in scene_filter.datasets) and "test" in scene_filter.split_types:
        print("nuScenes test has no public GT boxes; use --splits val or train for annotations.")
    print(f"Selected {len(indices)} of {len(scenes)} matching scenes from splits {scene_filter.split_types}.")
    requested_fps = args.fps
    frame_scene_names = {}
    for scene_index in indices:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", scene_name(scenes[scene_index])).strip("._-") or "scene"
        frame_scene_names.setdefault(safe_name, []).append(scene_index)
    for scene_index in indices:
        scene = scenes[scene_index]
        first = -int(scene.number_of_history_iterations) + args.frame_offset
        iterations = range(first, int(scene.number_of_iterations), stride)
        if frames:
            iterations = iterations[:frames]
        if not iterations:
            raise SystemExit("No selected frames; check --frame-offset and scene duration.")
        interval = float(scene.scene_metadata.iteration_duration_s)
        args.fps = requested_fps if requested_fps is not None else (
            1. / (interval * stride) if math.isfinite(interval) and interval > 0 else 10.)
        # Expand the query margin for virtual cameras instead of moving map objects.
        map_margin = max(view.distance_m for view in args.render_views)
        reader = SceneReader(scene, iterations, camera_query, args.map_radius + map_margin)
        initial = reader.frame(0)
        print(f"3D box annotations at first selected frame: {len(initial.boxes)} boxes"
              if initial.boxes_available else "3D box annotations unavailable at first selected frame in this Arrow log.")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", scene_name(scene)).strip("._-") or "scene"
        args.frame_scene_id = safe_name if len(frame_scene_names[safe_name]) == 1 else f"{safe_name}_{scene_index:06d}"
        clip_id = f"{scene_index:06d}_{safe_name}"
        print(f"Scene: {scene_name(scene)} | frames: {len(iterations)} | layers: {[layer.name for layer in reader.layers]}")
        if args.viewer:
            run_viewer(reader, args, output_dir / "videos" / "viewer" / clip_id)
        else:
            render_video(reader, args, output_dir, clip_id)


if __name__ == "__main__":
    main()