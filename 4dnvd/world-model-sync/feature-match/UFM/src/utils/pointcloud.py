from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import trange


def save_global_pointcloud_ply(
    ply_path,
    video_chw,  # torch [F, 3, H, W], RGB in [0, 1]
    depths_np,  # numpy [F, H, W]
    K_seq,  # torch [F, 3, 3]
    T_world_cam_seq,  # torch [F, 4, 4], camera-to-world
    stride=2,
    max_depth=300.0,
):
    ply_path = Path(ply_path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    video_np = video_chw.detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    K_np = K_seq.detach().cpu().numpy().astype(np.float32)
    T_np = T_world_cam_seq.detach().cpu().numpy().astype(np.float32)

    F, H, W = depths_np.shape
    all_xyz = []
    all_rgb = []

    yy, xx = np.meshgrid(
        np.arange(0, H, stride, dtype=np.float32),
        np.arange(0, W, stride, dtype=np.float32),
        indexing="ij",
    )

    pix = np.stack(
        [xx.reshape(-1), yy.reshape(-1), np.ones(xx.size, dtype=np.float32)],
        axis=1,
    )

    for t in range(F):
        depth = depths_np[t, ::stride, ::stride].reshape(-1).astype(np.float32)
        rgb = video_np[t, ::stride, ::stride].reshape(-1, 3)

        valid = np.isfinite(depth) & (depth > 1e-4) & (depth < max_depth)
        if valid.sum() == 0:
            continue

        K_inv = np.linalg.inv(K_np[t])
        pts_cam = (pix[valid] @ K_inv.T) * depth[valid, None]

        R = T_np[t, :3, :3]
        trans = T_np[t, :3, 3]
        pts_world = pts_cam @ R.T + trans[None]

        all_xyz.append(pts_world.astype(np.float32))
        all_rgb.append((rgb[valid] * 255.0).clip(0, 255).astype(np.uint8))

    if len(all_xyz) == 0:
        print(f"No valid points to save: {ply_path}")
        return

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(xyz, rgb):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

    print(f"Saved global point cloud: {ply_path} with {xyz.shape[0]} points")


def contract_mask(mask, radius=6, iterations=3):
    """Erode/contract a [F, H, W] bool mask, matching the sky/static-mask cleanup style in utils."""
    mask = mask.astype(bool).copy()
    if radius <= 0 or iterations <= 0:
        return mask

    kernel = np.ones((int(radius), int(radius)), dtype=np.uint8)
    for i in range(mask.shape[0]):
        mask[i] = cv2.erode(
            mask[i].astype(np.uint8),
            kernel,
            iterations=int(iterations),
        ).astype(bool)
    return mask


def apply_sky_handling_like_utils(
    depths_np,
    dynamic_mask_t,
    sky_mask_t,
    render_moving=True,
    sky_depth=1000.0,
    sky_contract_radius=6,
    sky_contract_iterations=3,
):
    """
    Match utils.point_cloud.preprocess sky semantics, but using SAM3 sky masks:

      1. Contract/erode sky mask to avoid horizon/tree/building boundary artifacts.
      2. Remove sky from dynamic mask.
      3. Union sky into static mask.
      4. Set sky depth to a far constant.

    There is intentionally no sky dilation/padding. If the SAM3 sky mask is used, it is
    either unchanged or contracted, never expanded.
    """
    depths_out = depths_np.copy().astype(np.float32)

    dynamic_np = dynamic_mask_t.detach().cpu().numpy().astype(bool)
    sky_np = sky_mask_t.detach().cpu().numpy().astype(bool)

    sky_np = contract_mask(
        sky_np,
        radius=int(sky_contract_radius),
        iterations=int(sky_contract_iterations),
    )

    # Sky should never be dynamic.
    dynamic_no_sky = dynamic_np & ~sky_np

    # Sky is static/far background. This mirrors:
    # dynamic_mask = dynamic_mask & ~sky_mask
    # static_mask = static_mask | sky_mask
    # depths[sky_mask] = SKY_DEPTH
    depths_out[sky_np] = float(sky_depth)

    if render_moving:
        dyn_np = dynamic_no_sky
        static_np = (~dynamic_no_sky) | sky_np
    else:
        # Remove dynamic objects before unprojection, while preserving sky as static.
        dyn_np = np.zeros_like(dynamic_no_sky, dtype=bool)
        static_np = (~dynamic_no_sky) | sky_np

    dyn_t = torch.from_numpy(dyn_np).bool()
    static_t = torch.from_numpy(static_np).bool()
    sky_t = torch.from_numpy(sky_np).bool()

    return depths_out, dyn_t, static_t, sky_t


def unproject_mask_to_world_points_numpy(
    depth,
    mask,
    K,
    T_world_cam,
    max_depth=300.0,
):
    """
    Convert one instance mask in one frame into world-space 3D points.

    depth: [H, W] numpy float
    mask: [H, W] bool
    K: [3, 3] numpy
    T_world_cam: [4, 4] numpy camera-to-world

    Returns:
        points_world: [N, 3] float32
    """
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    K = np.asarray(K, dtype=np.float32)
    T_world_cam = np.asarray(T_world_cam, dtype=np.float32)

    valid = mask & np.isfinite(depth) & (depth > 1e-4)
    if max_depth is not None:
        valid = valid & (depth < float(max_depth))

    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs].astype(np.float32)
    pix = np.stack(
        [
            xs.astype(np.float32),
            ys.astype(np.float32),
            np.ones_like(xs, dtype=np.float32),
        ],
        axis=1,
    )

    points_cam = (pix @ np.linalg.inv(K).T) * z[:, None]
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    points_world = points_cam @ R.T + t[None]
    return points_world.astype(np.float32)


def robust_bev_box_from_points(
    points_world,
    q_low=0.05,
    q_high=0.95,
    min_points=50,
):
    """
    Build robust BEV box [xmin, ymin, xmax, ymax] from world-space 3D points.
    Uses x/y only, so it is less sensitive to DA3 height/depth noise than 3D IoU.
    """
    points_world = np.asarray(points_world, dtype=np.float32)
    if points_world.shape[0] < int(min_points):
        return None

    xy = points_world[:, :2]
    lo = np.quantile(xy, float(q_low), axis=0)
    hi = np.quantile(xy, float(q_high), axis=0)

    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        return None
    if np.any(hi <= lo):
        return None

    return np.array([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32)


def bev_iou_xyxy(box_a, box_b):
    """
    center stability for boxes [xmin, ymin, xmax, ymax].
    """
    if box_a is None or box_b is None:
        return None

    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)

    inter_min = np.maximum(a[:2], b[:2])
    inter_max = np.minimum(a[2:], b[2:])
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_area = float(inter_size[0] * inter_size[1])

    area_a = float(max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0))
    area_b = float(max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0))
    denom = area_a + area_b - inter_area
    if denom <= 1e-8:
        return 0.0
    return inter_area / denom


def mask_touches_border(mask, margin=8):
    """
    Return True if an instance mask touches the image border.
    Border-touching observations are often partial objects.
    """
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2 or mask.size == 0:
        return True

    margin = int(max(margin, 0))
    if margin <= 0:
        return False

    H, W = mask.shape
    m = min(margin, H, W)
    return bool(
        mask[:m, :].any()
        or mask[-m:, :].any()
        or mask[:, :m].any()
        or mask[:, -m:].any()
    )


def classify_static_instances_by_center_stability(
    seg_frames,
    depths_np,
    K_seq,
    T_world_cam_seq,
    bev_prompts=("vehicle",),
    center_motion_threshold_m=0.5,
    center_percentile=90.0,
    min_points=50,
    min_observations=3,
    max_gap=None,
    ignore_border_observations=True,
    border_margin=8,
    max_depth=300.0,
):
    """
    Classify tracked SAM3 vehicle instances as static using offline center stability.

    For each valid vehicle observation:
        SAM3 mask + DA3 depth + K + T_world_cam -> world-space 3D points
        center_t = median(points_world[:, :2])

    For the whole valid track:
        median_center = median(center_t over time)
        distances = ||center_t - median_center||_2
        motion_score = percentile(distances, center_percentile)

    Units:
        center_t: meters, because T_world_cam/depth are metric world coordinates
        distances: meters
        motion_score: meters
        raw center variance: meters^2

    Rule:
        static if motion_score < center_motion_threshold_m

    Only prompts in bev_prompts are checked. With the default config, this means
    vehicle is checked, while pedestrian remains dynamic foreground.

    Returns:
        static_instance_ids: set[int]
        dynamic_instance_ids: set[int]
        instance_stats: JSON-safe dict keyed by string instance id
    """
    bev_prompts = set(str(x).lower() for x in bev_prompts)
    K_np = K_seq.detach().cpu().numpy().astype(np.float32)
    T_np = T_world_cam_seq.detach().cpu().numpy().astype(np.float32)

    instance_records = {}
    skipped_border = {}
    skipped_low_points = {}

    for frame_idx, objs in enumerate(seg_frames):
        for obj in objs:
            prompt = str(obj.get("prompt", obj.get("keyword", ""))).lower()

            # Only prompts in bev_prompts are eligible for static-vehicle classification.
            # With the default config, vehicle is checked, pedestrian is not.
            if prompt not in bev_prompts:
                continue

            obj_id = int(obj["id"])
            mask = np.asarray(obj["mask"]).astype(bool)

            if bool(ignore_border_observations) and mask_touches_border(
                mask, margin=int(border_margin)
            ):
                skipped_border[obj_id] = skipped_border.get(obj_id, 0) + 1
                continue

            points_world = unproject_mask_to_world_points_numpy(
                depth=depths_np[frame_idx],
                mask=mask,
                K=K_np[frame_idx],
                T_world_cam=T_np[frame_idx],
                max_depth=max_depth,
            )

            if points_world.shape[0] < int(min_points):
                skipped_low_points[obj_id] = skipped_low_points.get(obj_id, 0) + 1
                continue

            center_xy = np.median(points_world[:, :2], axis=0).astype(np.float32)
            if np.any(~np.isfinite(center_xy)):
                continue

            instance_records.setdefault(obj_id, []).append(
                {
                    "frame_idx": int(frame_idx),
                    "center_xy": center_xy,
                    "num_points": int(points_world.shape[0]),
                    "prompt": prompt,
                }
            )

    static_instance_ids = set()
    dynamic_instance_ids = set()
    instance_stats = {}

    all_instance_ids = (
        set(instance_records.keys())
        | set(skipped_border.keys())
        | set(skipped_low_points.keys())
    )

    for obj_id in sorted(all_instance_ids):
        records = sorted(instance_records.get(obj_id, []), key=lambda x: x["frame_idx"])
        obj_key = str(int(obj_id))
        frame_ids = [int(r["frame_idx"]) for r in records]

        if len(records) < int(min_observations):
            dynamic_instance_ids.add(obj_id)
            instance_stats[obj_key] = {
                "status": "dynamic_insufficient_observations",
                "num_observations": int(len(records)),
                "frames": frame_ids,
                "skipped_border_observations": int(skipped_border.get(obj_id, 0)),
                "skipped_low_point_observations": int(
                    skipped_low_points.get(obj_id, 0)
                ),
                "motion_score_m": None,
            }
            continue

        # Optional track-quality guard. This does not decide motion; it only rejects
        # broken tracks with large missing-frame gaps if max_gap is not None.
        gaps = [int(b) - int(a) for a, b in zip(frame_ids[:-1], frame_ids[1:])]
        if max_gap is not None and any(g > int(max_gap) for g in gaps):
            dynamic_instance_ids.add(obj_id)
            instance_stats[obj_key] = {
                "status": "dynamic_large_tracking_gap",
                "num_observations": int(len(records)),
                "frames": frame_ids,
                "gaps": [int(g) for g in gaps],
                "skipped_border_observations": int(skipped_border.get(obj_id, 0)),
                "skipped_low_point_observations": int(
                    skipped_low_points.get(obj_id, 0)
                ),
                "motion_score_m": None,
            }
            continue

        centers = np.stack([r["center_xy"] for r in records], axis=0).astype(np.float32)
        median_center = np.median(centers, axis=0)
        distances = np.linalg.norm(centers - median_center[None], axis=1)

        motion_score = float(np.percentile(distances, float(center_percentile)))
        center_std_xy = np.std(centers, axis=0)
        center_std_norm_m = float(np.linalg.norm(center_std_xy))
        center_var_xy_m2 = np.var(centers, axis=0)
        center_var_sum_m2 = float(np.sum(center_var_xy_m2))

        is_static = motion_score < float(center_motion_threshold_m)

        if is_static:
            static_instance_ids.add(obj_id)
            status = "static"
        else:
            dynamic_instance_ids.add(obj_id)
            status = "dynamic"

        instance_stats[obj_key] = {
            "status": status,
            "method": "center_stability",
            "num_observations": int(len(records)),
            "frames": frame_ids,
            "first_frame": int(frame_ids[0]),
            "last_frame": int(frame_ids[-1]),
            "frame_span": int(frame_ids[-1] - frame_ids[0]),
            "gaps": [int(g) for g in gaps],
            "num_points": [int(r["num_points"]) for r in records],
            "centers_xy_m": [
                [float(v) for v in r["center_xy"].tolist()] for r in records
            ],
            "median_center_xy_m": [float(x) for x in median_center.tolist()],
            "distances_to_median_m": [float(x) for x in distances.tolist()],
            "motion_score_m": float(motion_score),
            "center_percentile": float(center_percentile),
            "center_motion_threshold_m": float(center_motion_threshold_m),
            "center_std_xy_m": [float(x) for x in center_std_xy.tolist()],
            "center_std_norm_m": float(center_std_norm_m),
            "center_var_xy_m2": [float(x) for x in center_var_xy_m2.tolist()],
            "center_var_sum_m2": float(center_var_sum_m2),
            "skipped_border_observations": int(skipped_border.get(obj_id, 0)),
            "skipped_low_point_observations": int(skipped_low_points.get(obj_id, 0)),
        }

    return static_instance_ids, dynamic_instance_ids, instance_stats


def split_dynamic_mask_using_static_vehicle_instances(
    seg_frames,
    original_dynamic_mask,
    static_instance_ids,
):
    """
    Remove static vehicle instances from the dynamic mask so they become background.
    """
    dynamic_np = original_dynamic_mask.detach().cpu().numpy().astype(bool)
    static_vehicle_np = np.zeros_like(dynamic_np, dtype=bool)

    for frame_idx, objs in enumerate(seg_frames):
        for obj in objs:
            obj_id = int(obj["id"])
            if obj_id not in static_instance_ids:
                continue
            static_vehicle_np[frame_idx] |= np.asarray(obj["mask"]).astype(bool)

    dynamic_refined_np = dynamic_np & ~static_vehicle_np

    return (
        torch.from_numpy(dynamic_refined_np).bool(),
        torch.from_numpy(static_vehicle_np).bool(),
    )


def inverse(t, dtype=torch.float32):
    return t.to(dtype=dtype).inverse().to(dtype=t.dtype)


@torch.autocast(device_type="cuda", enabled=False)
def unproject(
    video,
    depths,
    cam_c2w,
    K,
    dynamic_mask=None,
    static_mask=None,
    pixel_center_offset=False,
    max_depth=None,
):
    F, H, W, _ = video.shape
    dtype, device = depths.dtype, depths.device

    depths = depths.to(dtype=dtype, device=device)
    cam_c2w = cam_c2w.to(dtype=dtype, device=device)
    K = K.to(dtype=dtype, device=device)

    ff, yy, xx = torch.meshgrid(
        torch.arange(F, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )

    pix = torch.stack(
        [xx, yy, torch.ones_like(yy)],
        dim=-1,
    ).to(dtype=dtype, device=device)

    if pixel_center_offset:
        pix[..., :2] += 0.5

    points_cam = (pix.flatten(1, 2) @ inverse(K).mT) * depths.flatten(1, 2)[..., None]

    R = cam_c2w[:, :3, :3]
    T = cam_c2w[:, None, :3, 3]
    points_world = (points_cam @ R.mT) + T

    true_mask = torch.ones_like(depths, dtype=torch.bool, device=device)

    if dynamic_mask is None:
        dynamic_mask = true_mask
    else:
        dynamic_mask = dynamic_mask.to(device=device, dtype=torch.bool)

    if static_mask is None:
        static_mask = ~true_mask
    else:
        static_mask = static_mask.to(device=device, dtype=torch.bool)

    valid_depth = torch.isfinite(depths) & (depths > 1e-4)

    # Important:
    # This removes sky if you previously set depths[sky_mask] = 1000
    # and max_depth = 300.
    if max_depth is not None:
        valid_depth = valid_depth & (depths < float(max_depth))

    keep = (valid_depth & (dynamic_mask | static_mask)).view(-1)

    frame_idx = ff.reshape(-1)[keep]

    visible = torch.zeros(frame_idx.shape[0], F, dtype=torch.bool, device=device)
    visible.scatter_(1, frame_idx.unsqueeze(1), True)

    point_dynamic = dynamic_mask.view(-1)[keep]
    point_static = static_mask.view(-1)[keep]

    visible = (visible & point_dynamic[:, None]) | point_static[:, None]

    indices = torch.stack([ff, yy, xx], dim=-1).view(-1, 3)[keep]

    return (
        video.reshape(-1, 3)[keep],
        points_world.reshape(-1, 3)[keep],
        visible,
        indices,
        point_dynamic,
    )


def make_same_frame_visible(visible, indices):
    visible_ntp = torch.zeros_like(visible)
    visible_ntp.scatter_(1, indices[:, 0:1], True)
    return visible_ntp


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def classify_points_occluded_by_current_dynamic_mask(
    points_world,
    source_cam_c2w,
    source_K,
    source_depth,
    source_dynamic_mask,
    depth_abs_tolerance=0.10,
    depth_rel_tolerance=0.02,
    dynamic_mask_dilation_radius=0,
    invalid_dynamic_depth_is_occluded=True,
):
    """Classify nonlocal background points hidden by a current-frame dynamic object.

    A global static point is a dynamic-disocclusion candidate when it reprojects
    inside the current source image, lands on the current dynamic mask, and lies
    behind the observed current-frame depth. If DA3 has filtered the depth at a
    dynamic pixel, ``invalid_dynamic_depth_is_occluded`` conservatively treats the
    point as occluded.

    The returned mask is point-wise. During the target-view fill pass, occluded
    points still participate in the z-buffer but are not allowed to contribute
    color. This prevents a farther background point from filling through a
    dynamic-caused hole.
    """
    device = points_world.device
    num_points = int(points_world.shape[0])
    if num_points == 0:
        return torch.zeros((0,), dtype=torch.bool, device=device)

    points_world = points_world.to(device=device, dtype=torch.float32)
    source_cam_c2w = source_cam_c2w.to(device=device, dtype=torch.float32)
    source_K = source_K.to(device=device, dtype=torch.float32)
    source_depth = source_depth.to(device=device, dtype=torch.float32)
    source_dynamic_mask = source_dynamic_mask.to(device=device, dtype=torch.bool)

    if source_depth.ndim != 2:
        raise ValueError(
            f"source_depth must have shape [H, W], got {tuple(source_depth.shape)}"
        )
    if source_dynamic_mask.shape != source_depth.shape:
        raise ValueError(
            "source_dynamic_mask and source_depth must have the same shape: "
            f"{tuple(source_dynamic_mask.shape)} != {tuple(source_depth.shape)}"
        )

    dilation_radius = max(0, int(dynamic_mask_dilation_radius))
    if dilation_radius > 0:
        kernel_size = 2 * dilation_radius + 1
        source_dynamic_mask = (
            torch.nn.functional.max_pool2d(
                source_dynamic_mask[None, None].to(torch.float32),
                kernel_size=kernel_size,
                stride=1,
                padding=dilation_radius,
            )[0, 0]
            > 0.0
        )

    source_cam_w2c = torch.linalg.inv(source_cam_c2w)
    points_cam = (
        points_world @ source_cam_w2c[:3, :3].T
        + source_cam_w2c[:3, 3][None]
    )
    point_depth = points_cam[:, 2]
    projection = points_cam @ source_K.T
    u = projection[:, 0] / projection[:, 2]
    v = projection[:, 1] / projection[:, 2]

    H, W = source_depth.shape
    u_idx = torch.round(u).to(torch.int64)
    v_idx = torch.round(v).to(torch.int64)
    inside = (
        (point_depth > 1e-5)
        & torch.isfinite(u)
        & torch.isfinite(v)
        & (u_idx >= 0)
        & (u_idx < W)
        & (v_idx >= 0)
        & (v_idx < H)
    )

    dynamic_occluded = torch.zeros(num_points, dtype=torch.bool, device=device)
    if not inside.any():
        return dynamic_occluded

    inside_indices = torch.nonzero(inside, as_tuple=False).flatten()
    sampled_dynamic = source_dynamic_mask[
        v_idx[inside_indices], u_idx[inside_indices]
    ]
    if not sampled_dynamic.any():
        return dynamic_occluded

    dynamic_indices = inside_indices[sampled_dynamic]
    observed_depth = source_depth[v_idx[dynamic_indices], u_idx[dynamic_indices]]
    observed_depth_valid = (
        torch.isfinite(observed_depth) & (observed_depth > 1e-4)
    )
    depth_margin = torch.maximum(
        torch.full_like(observed_depth, float(depth_abs_tolerance)),
        observed_depth.abs() * float(depth_rel_tolerance),
    )
    behind_dynamic = observed_depth_valid & (
        point_depth[dynamic_indices] > observed_depth + depth_margin
    )
    if bool(invalid_dynamic_depth_is_occluded):
        behind_dynamic = behind_dynamic | ~observed_depth_valid

    dynamic_occluded[dynamic_indices] = behind_dynamic
    return dynamic_occluded


@torch.autocast(device_type="cuda", enabled=False)
def render_frame(
    points_color,
    points_pos,
    cam_c2w,
    K,
    H,
    W,
    dynamic_mask=None,
    z_tolerance=0.02,
    occlusion_sharpness=10.0,
    batch_size=int(1e7),
    allowed_pixel_mask=None,
    point_splat_mask=None,
    return_blocked_mask=False,
    point_ids=None,
    return_correspondence=False,
):
    device = points_pos.device
    out_dtype = points_color.dtype
    N = points_pos.shape[0]
    batch_size = int(batch_size)

    # Important: all geometry/projection math should be fp32.
    points_pos = points_pos.to(device=device, dtype=torch.float32)
    points_color = points_color.to(device=device, dtype=torch.float32)
    cam_c2w = cam_c2w.to(device=device, dtype=torch.float32)
    K = K.to(device=device, dtype=torch.float32)

    if dynamic_mask is not None:
        dynamic_mask = dynamic_mask.to(device=device, dtype=torch.bool)

    if allowed_pixel_mask is not None:
        allowed_pixel_mask = allowed_pixel_mask.to(device=device, dtype=torch.bool)
        if tuple(allowed_pixel_mask.shape) != (H, W):
            raise ValueError(
                "allowed_pixel_mask must have shape [H, W], got "
                f"{tuple(allowed_pixel_mask.shape)}"
            )
        allowed_pixel_flat = allowed_pixel_mask.reshape(-1)
    else:
        allowed_pixel_flat = None

    if point_splat_mask is not None:
        point_splat_mask = point_splat_mask.to(device=device, dtype=torch.bool)
        if tuple(point_splat_mask.shape) != (N,):
            raise ValueError(
                "point_splat_mask must have shape [N], got "
                f"{tuple(point_splat_mask.shape)} for N={N}"
            )

    if return_correspondence and point_ids is None:
        raise ValueError("return_correspondence=True requires point_ids")
    if point_ids is not None:
        point_ids = point_ids.to(device=device, dtype=torch.int64)
        if tuple(point_ids.shape) != (N,):
            raise ValueError(
                f"point_ids must have shape [N], got {tuple(point_ids.shape)} "
                f"for N={N}"
            )

    accum_color = torch.zeros((H * W, 3), dtype=torch.float32, device=device)
    accum_depth = torch.zeros((H * W,), dtype=torch.float32, device=device)
    accum_weight = torch.zeros((H * W,), dtype=torch.float32, device=device)
    z_buffer = torch.full((H * W,), float("inf"), dtype=torch.float32, device=device)
    min_z_batch = torch.empty((H * W,), dtype=torch.float32, device=device)

    has_dyn = dynamic_mask is not None
    accum_dyn = (
        torch.zeros((H * W,), dtype=torch.float32, device=device) if has_dyn else None
    )
    if return_correspondence:
        correspondence_id = torch.full(
            (H * W,), -1, dtype=torch.int64, device=device
        )
        correspondence_score = torch.zeros(
            (H * W,), dtype=torch.float32, device=device
        )
        local_max_correspondence_score = torch.empty(
            (H * W,), dtype=torch.float32, device=device
        )
        local_correspondence_id = torch.empty(
            (H * W,), dtype=torch.int64, device=device
        )

    cam_w2c = torch.linalg.inv(cam_c2w)
    R = cam_w2c[:3, :3]
    T = cam_w2c[:3, 3]

    for start in range(0, N, batch_size):
        pos = points_pos[start : start + batch_size]
        col = points_color[start : start + batch_size]
        dyn = dynamic_mask[start : start + batch_size] if has_dyn else None
        splat = (
            point_splat_mask[start : start + batch_size]
            if point_splat_mask is not None
            else None
        )
        ids = point_ids[start : start + batch_size] if point_ids is not None else None

        pc = (pos @ R.T) + T
        d = pc[:, 2]
        valid = d > 1e-5
        if not valid.any():
            continue

        pc = pc[valid]
        col = col[valid]
        d = d[valid]
        if has_dyn:
            dyn = dyn[valid]
        if splat is not None:
            splat = splat[valid]
        if ids is not None:
            ids = ids[valid]

        uvz = (K @ pc.T).T
        u = uvz[:, 0] / uvz[:, 2]
        v = uvz[:, 1] / uvz[:, 2]

        u0 = torch.floor(u).to(torch.int64)
        v0 = torch.floor(v).to(torch.int64)
        du = u - u0
        dv = v - v0

        u_frag = torch.cat([u0, u0, u0 + 1, u0 + 1])
        v_frag = torch.cat([v0, v0 + 1, v0, v0 + 1])
        w_geom = torch.cat(
            [
                (1 - du) * (1 - dv),
                (1 - du) * dv,
                du * (1 - dv),
                du * dv,
            ]
        )

        d_frag = d.repeat(4)
        c_frag = col.repeat(4, 1)
        dyn_frag = dyn.repeat(4).float() if has_dyn else None
        splat_frag = splat.repeat(4) if splat is not None else None
        id_frag = ids.repeat(4) if ids is not None else None

        w_depth = torch.pow(
            d_frag.to(torch.float32).clamp(min=1e-1),
            -occlusion_sharpness,
        ).clamp(min=1e-10)
        w_final = w_geom.to(torch.float32) * w_depth

        valid_uv = (
            (u_frag >= 0) & (u_frag < W) & (v_frag >= 0) & (v_frag < H) & (w_final > 0)
        )
        if not valid_uv.any():
            continue

        idx = v_frag[valid_uv] * W + u_frag[valid_uv]
        d_frag = d_frag[valid_uv]
        c_frag = c_frag[valid_uv]
        w_final = w_final[valid_uv]
        if has_dyn:
            dyn_frag = dyn_frag[valid_uv]
        if splat_frag is not None:
            splat_frag = splat_frag[valid_uv]
        if id_frag is not None:
            id_frag = id_frag[valid_uv]

        if allowed_pixel_flat is not None:
            allowed_fragment = allowed_pixel_flat[idx]
            if not allowed_fragment.any():
                continue
            idx = idx[allowed_fragment]
            d_frag = d_frag[allowed_fragment]
            c_frag = c_frag[allowed_fragment]
            w_final = w_final[allowed_fragment]
            if has_dyn:
                dyn_frag = dyn_frag[allowed_fragment]
            if splat_frag is not None:
                splat_frag = splat_frag[allowed_fragment]
            if id_frag is not None:
                id_frag = id_frag[allowed_fragment]

        order = torch.argsort(d_frag, descending=True)
        idx = idx[order]
        d_frag = d_frag[order]
        c_frag = c_frag[order]
        w_final = w_final[order]
        if has_dyn:
            dyn_frag = dyn_frag[order]
        if splat_frag is not None:
            splat_frag = splat_frag[order]
        if id_frag is not None:
            id_frag = id_frag[order]

        min_z_batch.fill_(float("inf"))
        min_z_batch[idx] = d_frag.to(torch.float32)
        min_d = min_z_batch[idx]

        in_batch = (
            torch.log1p(d_frag.to(torch.float32)) - torch.log1p(min_d)
        ).abs() <= z_tolerance
        if not in_batch.any():
            continue

        idx = idx[in_batch]
        d_frag = d_frag[in_batch]
        c_frag = c_frag[in_batch]
        w_final = w_final[in_batch]
        if has_dyn:
            dyn_frag = dyn_frag[in_batch]
        if splat_frag is not None:
            splat_frag = splat_frag[in_batch]
        if id_frag is not None:
            id_frag = id_frag[in_batch]

        z_curr = z_buffer[idx]
        rel = (
            torch.log1p(d_frag.to(torch.float32))
            - torch.log1p(z_curr.to(torch.float32))
        ).abs()
        reset = (d_frag < z_curr) & (rel > z_tolerance)

        if reset.any():
            reset_idx = idx[reset].to(torch.int64)
            accum_color.index_fill_(0, reset_idx, 0.0)
            accum_depth.index_fill_(0, reset_idx, 0.0)
            accum_weight.index_fill_(0, reset_idx, 0.0)
            if has_dyn:
                accum_dyn.index_fill_(0, reset_idx, 0.0)
            if return_correspondence:
                correspondence_id.index_fill_(0, reset_idx, -1)
                correspondence_score.index_fill_(0, reset_idx, 0.0)
            z_buffer[reset_idx] = d_frag[reset]

        blend = reset | (rel <= z_tolerance)
        if not blend.any():
            continue

        # All front-surface fragments, including dynamically occluded candidates,
        # participate in depth selection. Only allowed fragments contribute color.
        contribute = blend
        if splat_frag is not None:
            contribute = contribute & splat_frag
        if not contribute.any():
            continue

        idxb = idx[contribute]
        wb = w_final[contribute]

        accum_color.index_add_(
            0, idxb, c_frag[contribute].to(torch.float32) * wb[:, None]
        )
        accum_depth.index_add_(0, idxb, d_frag[contribute].to(torch.float32) * wb)
        accum_weight.index_add_(0, idxb, wb)

        if has_dyn:
            accum_dyn.index_add_(0, idxb, dyn_frag[contribute] * wb)

        if return_correspondence:
            idb = id_frag[contribute]
            local_max_correspondence_score.fill_(0.0)
            local_max_correspondence_score.scatter_reduce_(
                0, idxb, wb, reduce="amax", include_self=True
            )
            is_local_best = wb >= local_max_correspondence_score[idxb]

            local_correspondence_id.fill_(-1)
            local_correspondence_id.scatter_reduce_(
                0,
                idxb[is_local_best],
                idb[is_local_best],
                reduce="amax",
                include_self=True,
            )
            touched = local_correspondence_id >= 0
            touched_idx = torch.nonzero(touched, as_tuple=False).flatten()
            better = (
                local_max_correspondence_score[touched_idx]
                >= correspondence_score[touched_idx]
            )
            if better.any():
                best_idx = touched_idx[better]
                correspondence_score[best_idx] = (
                    local_max_correspondence_score[best_idx]
                )
                correspondence_id[best_idx] = local_correspondence_id[best_idx]

    valid = accum_weight > 1e-16

    rgb = torch.zeros((H * W, 3), dtype=out_dtype, device=device)
    rgb[valid] = (accum_color[valid] / accum_weight[valid, None]).to(dtype=out_dtype)

    depth = torch.zeros((H * W,), dtype=out_dtype, device=device)
    depth[valid] = (accum_depth[valid] / accum_weight[valid]).to(dtype=out_dtype)

    dyn_out = None
    if has_dyn:
        dyn_out = torch.zeros((H * W,), dtype=torch.bool, device=device)
        dyn_out[valid] = (accum_dyn[valid] / accum_weight[valid]) > 0.0
        dyn_out = dyn_out.reshape(H, W)

    result = (
        rgb.reshape(H, W, 3),
        depth.reshape(H, W),
        valid.reshape(H, W),
        dyn_out,
    )
    if return_blocked_mask:
        blocked = torch.isfinite(z_buffer) & ~valid
        result = (*result, blocked.reshape(H, W))
    if return_correspondence:
        result = (
            *result,
            correspondence_id.reshape(H, W),
            correspondence_score.reshape(H, W),
        )
    return result


# @torch.autocast(device_type="cuda", enabled=False)
# def render(
#     points_color,
#     points_pos,
#     visible,
#     cam_c2w,
#     K,
#     H,
#     W,
#     dynamic_mask=None,
#     verbose=False,
#     z_tolerance=0.02,
#     occlusion_sharpness=10.0,
#     batch_size=int(1e7),
# ):
#     _, F = visible.shape
#     dtype, device = points_pos.dtype, points_pos.device

#     video = torch.zeros(F, H, W, 3, device=device, dtype=dtype)
#     alpha = torch.zeros(F, H, W, device=device, dtype=torch.bool)
#     depth = torch.zeros(F, H, W, device=device, dtype=dtype)

#     if dynamic_mask == "visible":
#         dynamic_mask = visible.sum(dim=1) == 1

#     has_dyn = dynamic_mask is not None
#     dyn_video = (
#         torch.zeros(F, H, W, device=device, dtype=torch.bool) if has_dyn else None
#     )

#     iterator = trange(F, desc="Rendering point cloud") if verbose else range(F)

#     for i in iterator:
#         vis = visible[:, i]
#         dyn_i = dynamic_mask[vis] if has_dyn else None

#         video[i], depth[i], alpha[i], dyn_out = render_frame(
#             points_color[vis],
#             points_pos[vis],
#             cam_c2w[i],
#             K[i],
#             H,
#             W,
#             dynamic_mask=dyn_i,
#             z_tolerance=z_tolerance,
#             occlusion_sharpness=occlusion_sharpness,
#             batch_size=batch_size,
#         )

#         if has_dyn:
#             dyn_video[i] = dyn_out

#     return video, depth, alpha, dyn_video


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def render(
    points_color,
    points_pos,
    visible,
    indices,
    cam_c2w,
    K,
    H,
    W,
    dynamic_mask=None,
    verbose=False,
    z_tolerance=0.02,
    occlusion_sharpness=10.0,
    batch_size=int(1e7),
    source_cam_c2w=None,
    source_K=None,
    source_depth=None,
    source_dynamic_mask=None,
    global_fill_static_holes_only=True,
    dynamic_hole_depth_abs_tolerance=0.10,
    dynamic_hole_depth_rel_tolerance=0.02,
    dynamic_hole_mask_dilation_radius=0,
    invalid_dynamic_depth_is_occluded=True,
    return_hole_masks=False,
    return_local_render=False,
    return_local_correspondence=False,
    local_only=False,
):
    """
    Renders video frames using a two-pass local/global hybrid strategy:
    1. Passes single-frame (local) PCD to avoid temporal ghosting/misalignment.
    2. Locks locally covered target pixels.
    3. Classifies remaining holes using current-frame dynamic occlusion.
    4. Fills only static-caused holes with nonlocal static PCD points.

    Dynamic-caused holes remain black. A global static candidate that is hidden
    behind the current source-frame dynamic mask still participates in the
    second-pass z-buffer, but cannot add RGB/depth. Therefore a farther point
    cannot incorrectly fill through a moving foreground object.

    When ``return_local_render`` is true, the function also returns the RGB,
    depth, coverage, and dynamic mask produced by pass 1. This is the exact
    same-frame-only (no temporal persistence) target render needed for Vista4D-
    style double reprojection, so it does not need to be rendered a second time.
    When ``return_local_correspondence`` is true, it additionally returns the
    winning packed source-pixel ID and its splat score for every covered local
    target pixel. That sparse diagnostic map is not used for double
    reprojection: a target-pixel winner map is not a dense inverse.
    ``local_only`` stops after this first pass when explicitly requested.
    Double reprojection enables it for the virtual render, performs its dense
    source-coordinate return, and then directly splats the static global PCD
    into the still-uncovered source-camera pixels.
    """
    _, F = visible.shape
    device = points_pos.device
    dtype = points_color.dtype

    video_out = torch.zeros(F, H, W, 3, device=device, dtype=dtype)
    alpha_out = torch.zeros(F, H, W, device=device, dtype=torch.bool)
    depth_out = torch.zeros(F, H, W, device=device, dtype=dtype)

    has_dyn = dynamic_mask is not None
    dyn_video = torch.zeros(F, H, W, device=device, dtype=torch.bool) if has_dyn else None
    static_hole_video = torch.zeros(F, H, W, device=device, dtype=torch.bool)
    dynamic_hole_video = torch.zeros(F, H, W, device=device, dtype=torch.bool)
    if return_local_render:
        local_video_out = torch.zeros(F, H, W, 3, device=device, dtype=dtype)
        local_depth_out = torch.zeros(F, H, W, device=device, dtype=dtype)
        local_alpha_out = torch.zeros(F, H, W, device=device, dtype=torch.bool)
        local_dyn_video = (
            torch.zeros(F, H, W, device=device, dtype=torch.bool)
            if has_dyn
            else None
        )
    if return_local_correspondence:
        local_correspondence_id = torch.full(
            (F, H, W), -1, device=device, dtype=torch.int64
        )
        local_correspondence_score = torch.zeros(
            (F, H, W), device=device, dtype=torch.float32
        )

    if bool(global_fill_static_holes_only):
        required = {
            "dynamic_mask": dynamic_mask,
            "source_cam_c2w": source_cam_c2w,
            "source_K": source_K,
            "source_depth": source_depth,
            "source_dynamic_mask": source_dynamic_mask,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Static-only global hole filling requires: " + ", ".join(missing)
            )

        source_cam_c2w = source_cam_c2w.to(device=device, dtype=torch.float32)
        source_K = source_K.to(device=device, dtype=torch.float32)
        source_depth = source_depth.to(device=device, dtype=torch.float32)
        source_dynamic_mask = source_dynamic_mask.to(device=device, dtype=torch.bool)
        if tuple(source_cam_c2w.shape) != (F, 4, 4):
            raise ValueError(
                f"source_cam_c2w must have shape [F, 4, 4], got {tuple(source_cam_c2w.shape)}"
            )
        if tuple(source_K.shape) != (F, 3, 3):
            raise ValueError(
                f"source_K must have shape [F, 3, 3], got {tuple(source_K.shape)}"
            )
        if tuple(source_depth.shape) != (F, H, W):
            raise ValueError(
                f"source_depth must have shape [F, H, W], got {tuple(source_depth.shape)}"
            )
        if tuple(source_dynamic_mask.shape) != (F, H, W):
            raise ValueError(
                "source_dynamic_mask must have shape [F, H, W], got "
                f"{tuple(source_dynamic_mask.shape)}"
            )

    # Source frame index per point (index 0 in unproject indices tensor)
    source_frames = indices[:, 0]
    source_pixel_ids = (
        indices[:, 0].to(torch.int64) * (H * W)
        + indices[:, 1].to(torch.int64) * W
        + indices[:, 2].to(torch.int64)
    )

    iterator = trange(F, desc="Rendering Hybrid Local-Global PCD") if verbose else range(F)

    for i in iterator:
        # Global visible points for frame i
        vis_global = visible[:, i]

        # Local visible points (points that originated from source frame i)
        vis_local = vis_global & (source_frames == i)
        dyn_local = dynamic_mask[vis_local] if has_dyn else None

        # --- Pass 1: Current Frame Only (Local Single-Frame Z-Buffering) ---
        local_result = render_frame(
            points_color[vis_local],
            points_pos[vis_local],
            cam_c2w[i],
            K[i],
            H,
            W,
            dynamic_mask=dyn_local,
            z_tolerance=z_tolerance,
            occlusion_sharpness=occlusion_sharpness,
            batch_size=batch_size,
            point_ids=(
                source_pixel_ids[vis_local]
                if return_local_correspondence
                else None
            ),
            return_correspondence=return_local_correspondence,
        )
        rgb_loc, depth_loc, valid_loc, dyn_loc = local_result[:4]
        if return_local_correspondence:
            correspondence_loc, correspondence_score_loc = local_result[4:6]
            local_correspondence_id[i] = correspondence_loc
            local_correspondence_score[i] = correspondence_score_loc

        local_holes = ~valid_loc

        if return_local_render:
            local_video_out[i] = rgb_loc
            local_depth_out[i] = depth_loc
            local_alpha_out[i] = valid_loc
            if has_dyn:
                local_dyn_video[i] = dyn_loc

        if local_only:
            video_out[i] = rgb_loc
            depth_out[i] = depth_loc
            alpha_out[i] = valid_loc
            static_hole_video[i] = local_holes
            if has_dyn:
                dyn_video[i] = dyn_loc
            continue

        if bool(global_fill_static_holes_only):
            # The fill PCD contains only static points from other source frames.
            # Current-frame points were already committed by the local pass.
            vis_fill = (
                vis_global
                & ~dynamic_mask
                & (source_frames != i)
            )

            dynamic_occluded_points = (
                classify_points_occluded_by_current_dynamic_mask(
                    points_world=points_pos[vis_fill],
                    source_cam_c2w=source_cam_c2w[i],
                    source_K=source_K[i],
                    source_depth=source_depth[i],
                    source_dynamic_mask=source_dynamic_mask[i],
                    depth_abs_tolerance=dynamic_hole_depth_abs_tolerance,
                    depth_rel_tolerance=dynamic_hole_depth_rel_tolerance,
                    dynamic_mask_dilation_radius=dynamic_hole_mask_dilation_radius,
                    invalid_dynamic_depth_is_occluded=invalid_dynamic_depth_is_occluded,
                )
            )

            # Pixels covered locally are locked: no second-pass fragment is even
            # admitted to its z-buffer. Dynamic-occluded candidates set depth but
            # cannot contribute color, keeping those target holes black.
            rgb_glob, depth_glob, valid_glob, _, blocked_dynamic = render_frame(
                points_color[vis_fill],
                points_pos[vis_fill],
                cam_c2w[i],
                K[i],
                H,
                W,
                dynamic_mask=None,
                z_tolerance=z_tolerance,
                occlusion_sharpness=occlusion_sharpness,
                batch_size=batch_size,
                allowed_pixel_mask=local_holes,
                point_splat_mask=~dynamic_occluded_points,
                return_blocked_mask=True,
            )
            dynamic_holes = local_holes & blocked_dynamic
            static_holes = local_holes & ~dynamic_holes
            dyn_glob = None
        else:
            # Backward-compatible full global pass.
            dyn_global = dynamic_mask[vis_global] if has_dyn else None
            rgb_glob, depth_glob, valid_glob, dyn_glob = render_frame(
                points_color[vis_global],
                points_pos[vis_global],
                cam_c2w[i],
                K[i],
                H,
                W,
                dynamic_mask=dyn_global,
                z_tolerance=z_tolerance,
                occlusion_sharpness=occlusion_sharpness,
                batch_size=batch_size,
            )
            dynamic_holes = torch.zeros_like(local_holes)
            static_holes = local_holes

        # --- Pass 3: Composite (Local priority, Global fills black regions) ---
        valid_mask_3d = valid_loc.unsqueeze(-1)  # [H, W, 1]

        # Prioritize local frame render; fill unrendered (black) pixels using global render
        rgb_final = torch.where(valid_mask_3d, rgb_loc, rgb_glob)
        depth_final = torch.where(valid_loc, depth_loc, depth_glob)
        alpha_final = valid_loc | valid_glob

        video_out[i] = rgb_final
        depth_out[i] = depth_final
        alpha_out[i] = alpha_final
        static_hole_video[i] = static_holes
        dynamic_hole_video[i] = dynamic_holes

        if has_dyn:
            if bool(global_fill_static_holes_only):
                # The fill pass contains static points only.
                dyn_final = dyn_loc
            else:
                dyn_final = torch.where(valid_loc, dyn_loc, dyn_glob)
            dyn_video[i] = dyn_final

    result = (video_out, depth_out, alpha_out, dyn_video)
    if return_hole_masks:
        result = (*result, static_hole_video, dynamic_hole_video)
    if return_local_render:
        result = (
            *result,
            local_video_out,
            local_depth_out,
            local_alpha_out,
            local_dyn_video,
        )
    if return_local_correspondence:
        result = (
            *result,
            local_correspondence_id,
            local_correspondence_score,
        )
    return result


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def gather_virtual_render_to_source_grid(
    first_warped,
    first_depth,
    first_coverage,
    points_pos,
    indices,
    virtual_cam_c2w,
    virtual_K,
    source_depth,
    source_dynamic_mask=None,
    point_visibility=None,
    z_tolerance=0.02,
):
    """Return a virtual-camera render to its original source-pixel grid.

    This is an inverse-coordinate gather, not a second point-cloud splat. Every
    current-frame source point is projected into the virtual camera. A point is
    accepted when at least one of its four first-splat neighbours survived the
    virtual z-buffer at a compatible depth. The closest compatible virtual
    pixel is copied back to the point's original source pixel.

    Copying an actual first-pass pixel avoids a second RGB blend. Returned
    depth comes from the original source camera because the output is aligned
    with that camera, rather than incorrectly reusing virtual-camera depth.
    """
    F, H, W, channels = first_warped.shape
    if channels != 3:
        raise ValueError(
            f"first_warped must have shape [F, H, W, 3], got {tuple(first_warped.shape)}"
        )
    expected_shape = (F, H, W)
    for name, value in (
        ("first_depth", first_depth),
        ("first_coverage", first_coverage),
        ("source_depth", source_depth),
    ):
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
            )
    if (
        source_dynamic_mask is not None
        and tuple(source_dynamic_mask.shape) != expected_shape
    ):
        raise ValueError(
            "source_dynamic_mask must match the source image grid, got "
            f"{tuple(source_dynamic_mask.shape)}"
        )
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"indices must have shape [N, 3], got {tuple(indices.shape)}")
    if tuple(points_pos.shape) != (indices.shape[0], 3):
        raise ValueError(
            "points_pos and indices must contain the same number of 3D points"
        )
    if tuple(virtual_cam_c2w.shape) != (F, 4, 4):
        raise ValueError(
            f"virtual_cam_c2w must have shape [F, 4, 4], got {tuple(virtual_cam_c2w.shape)}"
        )
    if tuple(virtual_K.shape) != (F, 3, 3):
        raise ValueError(
            f"virtual_K must have shape [F, 3, 3], got {tuple(virtual_K.shape)}"
        )
    if point_visibility is not None and tuple(point_visibility.shape) != (
        indices.shape[0],
        F,
    ):
        raise ValueError(
            "point_visibility must have shape [N, F], got "
            f"{tuple(point_visibility.shape)}"
        )

    device = first_warped.device
    first_depth = first_depth.to(device=device, dtype=torch.float32)
    first_coverage = first_coverage.to(device=device, dtype=torch.bool)
    points_pos = points_pos.to(device=device, dtype=torch.float32)
    indices = indices.to(device=device, dtype=torch.int64)
    virtual_cam_c2w = virtual_cam_c2w.to(device=device, dtype=torch.float32)
    virtual_K = virtual_K.to(device=device, dtype=torch.float32)
    source_depth = source_depth.to(device=device, dtype=torch.float32)
    if source_dynamic_mask is not None:
        source_dynamic_mask = source_dynamic_mask.to(device=device, dtype=torch.bool)
    if point_visibility is not None:
        point_visibility = point_visibility.to(device=device, dtype=torch.bool)

    output_video = torch.zeros_like(first_warped)
    output_depth = torch.zeros_like(first_depth, dtype=first_warped.dtype)
    output_coverage = torch.zeros_like(first_coverage)
    output_dynamic = (
        torch.zeros_like(source_dynamic_mask)
        if source_dynamic_mask is not None
        else None
    )

    for frame_index in range(F):
        frame_points = indices[:, 0] == frame_index
        if point_visibility is not None:
            frame_points = frame_points & point_visibility[:, frame_index]
        if not frame_points.any():
            continue

        point_indices = indices[frame_points]
        pos = points_pos[frame_points]

        virtual_w2c = torch.linalg.inv(virtual_cam_c2w[frame_index])
        pc = pos @ virtual_w2c[:3, :3].T + virtual_w2c[:3, 3]
        point_z = pc[:, 2]
        positive_z = point_z > 1e-5

        uvz = pc @ virtual_K[frame_index].T
        u = uvz[:, 0] / point_z.clamp(min=1e-5)
        v = uvz[:, 1] / point_z.clamp(min=1e-5)
        u0 = torch.floor(u).to(torch.int64)
        v0 = torch.floor(v).to(torch.int64)
        du = u - u0
        dv = v - v0

        candidate_u = torch.stack((u0, u0, u0 + 1, u0 + 1), dim=1)
        candidate_v = torch.stack((v0, v0 + 1, v0, v0 + 1), dim=1)
        candidate_weight = torch.stack(
            (
                (1.0 - du) * (1.0 - dv),
                (1.0 - du) * dv,
                du * (1.0 - dv),
                du * dv,
            ),
            dim=1,
        ).to(torch.float32)

        inside = (
            positive_z[:, None]
            & (candidate_u >= 0)
            & (candidate_u < W)
            & (candidate_v >= 0)
            & (candidate_v < H)
            & (candidate_weight > 0.0)
        )
        safe_u = candidate_u.clamp(0, W - 1)
        safe_v = candidate_v.clamp(0, H - 1)
        candidate_pixel = safe_v * W + safe_u

        virtual_depth = first_depth[frame_index].reshape(-1)[candidate_pixel]
        virtual_covered = first_coverage[frame_index].reshape(-1)[candidate_pixel]
        depth_compatible = (
            torch.log1p(point_z.clamp(min=0.0))[:, None]
            - torch.log1p(virtual_depth.clamp(min=0.0))
        ).abs() <= float(z_tolerance)
        candidate_valid = (
            inside
            & virtual_covered
            & torch.isfinite(virtual_depth)
            & (virtual_depth > 0.0)
            & depth_compatible
        )

        # The largest bilinear footprint is the closest raster neighbour. Its
        # RGB is copied exactly, so this return step introduces no new colors.
        candidate_score = torch.where(
            candidate_valid,
            candidate_weight,
            torch.full_like(candidate_weight, -1.0),
        )
        best_score, best_slot = candidate_score.max(dim=1)
        visible_point = best_score > 0.0
        if not visible_point.any():
            continue

        row = torch.arange(pos.shape[0], device=device, dtype=torch.int64)
        selected_virtual_pixel = candidate_pixel[row, best_slot]
        selected_virtual_pixel = selected_virtual_pixel[visible_point]
        source_y = point_indices[visible_point, 1]
        source_x = point_indices[visible_point, 2]
        source_pixel = source_y * W + source_x

        output_video[frame_index].view(-1, 3)[source_pixel] = (
            first_warped[frame_index].view(-1, 3)[selected_virtual_pixel]
        )
        output_depth[frame_index].view(-1)[source_pixel] = source_depth[
            frame_index
        ].reshape(-1)[source_pixel].to(dtype=output_depth.dtype)
        output_coverage[frame_index].view(-1)[source_pixel] = True
        if output_dynamic is not None:
            output_dynamic[frame_index].view(-1)[source_pixel] = (
                source_dynamic_mask[frame_index].reshape(-1)[source_pixel]
            )

    return output_video, output_depth, output_coverage, output_dynamic


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def splat_global_static_points_to_source_holes(
    points_color,
    points_pos,
    visible,
    indices,
    point_dynamic,
    source_cam_c2w,
    source_K,
    source_depth,
    source_dynamic_mask,
    allowed_return_mask,
    z_tolerance=0.02,
    occlusion_sharpness=10.0,
    batch_size=int(1e7),
    dynamic_hole_depth_abs_tolerance=0.10,
    dynamic_hole_depth_rel_tolerance=0.02,
    dynamic_hole_mask_dilation_radius=0,
    invalid_dynamic_depth_is_occluded=True,
    return_source_frame_provenance=False,
):
    """Directly splat nonlocal static global PCD into source-camera holes.

    This is the same local-lock/global-fill policy used by monocular shift, but
    its locked mask is the completed local double-reprojection coverage. Only
    static points originating from other frames are candidates. Points hidden
    by the current dynamic mask participate in depth selection but cannot add
    RGB, so a farther static surface cannot fill through a moving object.

    When ``return_source_frame_provenance`` is true, the function also returns
    an ``[F, H, W]`` map containing the sampled source-frame index of the point
    that won each covered global-fill pixel. Uncovered and dynamically blocked
    pixels contain ``-1``. This optional map lets double reprojection rank
    reference images from the pixels that actually survive its final z-buffer.
    """
    if points_color.ndim != 2 or points_color.shape[1] != 3:
        raise ValueError(
            f"points_color must have shape [N, 3], got {tuple(points_color.shape)}"
        )
    if tuple(points_pos.shape) != tuple(points_color.shape):
        raise ValueError("points_pos and points_color must both have shape [N, 3]")
    N = points_pos.shape[0]
    if tuple(indices.shape) != (N, 3):
        raise ValueError(f"indices must have shape [N, 3], got {tuple(indices.shape)}")
    if tuple(point_dynamic.shape) != (N,):
        raise ValueError(
            f"point_dynamic must have shape [N], got {tuple(point_dynamic.shape)}"
        )
    if visible.ndim != 2 or visible.shape[0] != N:
        raise ValueError(f"visible must have shape [N, F], got {tuple(visible.shape)}")

    F = visible.shape[1]
    if allowed_return_mask.ndim != 3 or allowed_return_mask.shape[0] != F:
        raise ValueError(
            "allowed_return_mask must have shape [F, H, W], got "
            f"{tuple(allowed_return_mask.shape)}"
        )
    _, H, W = allowed_return_mask.shape
    expected_image_shape = (F, H, W)
    for name, value in (
        ("source_depth", source_depth),
        ("source_dynamic_mask", source_dynamic_mask),
    ):
        if tuple(value.shape) != expected_image_shape:
            raise ValueError(
                f"{name} must have shape {expected_image_shape}, "
                f"got {tuple(value.shape)}"
            )
    if tuple(source_cam_c2w.shape) != (F, 4, 4):
        raise ValueError(
            "source_cam_c2w must have shape [F, 4, 4], got "
            f"{tuple(source_cam_c2w.shape)}"
        )
    if tuple(source_K.shape) != (F, 3, 3):
        raise ValueError(
            f"source_K must have shape [F, 3, 3], got {tuple(source_K.shape)}"
        )

    device = points_pos.device
    out_dtype = points_color.dtype
    points_color = points_color.to(device=device, dtype=torch.float32)
    points_pos = points_pos.to(device=device, dtype=torch.float32)
    visible = visible.to(device=device, dtype=torch.bool)
    indices = indices.to(device=device, dtype=torch.int64)
    point_dynamic = point_dynamic.to(device=device, dtype=torch.bool)
    source_cam_c2w = source_cam_c2w.to(device=device, dtype=torch.float32)
    source_K = source_K.to(device=device, dtype=torch.float32)
    source_depth = source_depth.to(device=device, dtype=torch.float32)
    source_dynamic_mask = source_dynamic_mask.to(device=device, dtype=torch.bool)
    allowed_return_mask = allowed_return_mask.to(device=device, dtype=torch.bool)

    output_video = torch.zeros((F, H, W, 3), device=device, dtype=out_dtype)
    output_depth = torch.zeros((F, H, W), device=device, dtype=out_dtype)
    output_coverage = torch.zeros((F, H, W), device=device, dtype=torch.bool)
    blocked_dynamic = torch.zeros((F, H, W), device=device, dtype=torch.bool)
    if return_source_frame_provenance:
        output_source_frame = torch.full(
            (F, H, W), -1, device=device, dtype=torch.int64
        )

    source_frames = indices[:, 0]
    for frame_index in range(F):
        global_static_points = (
            visible[:, frame_index]
            & ~point_dynamic
            & (source_frames != frame_index)
        )
        if (
            not global_static_points.any()
            or not allowed_return_mask[frame_index].any()
        ):
            continue

        points_world = points_pos[global_static_points]
        point_colors = points_color[global_static_points]
        point_source_frames = source_frames[global_static_points]

        dynamic_occluded_points = (
            classify_points_occluded_by_current_dynamic_mask(
                points_world=points_world,
                source_cam_c2w=source_cam_c2w[frame_index],
                source_K=source_K[frame_index],
                source_depth=source_depth[frame_index],
                source_dynamic_mask=source_dynamic_mask[frame_index],
                depth_abs_tolerance=dynamic_hole_depth_abs_tolerance,
                depth_rel_tolerance=dynamic_hole_depth_rel_tolerance,
                dynamic_mask_dilation_radius=dynamic_hole_mask_dilation_radius,
                invalid_dynamic_depth_is_occluded=(
                    invalid_dynamic_depth_is_occluded
                ),
            )
        )

        render_result = render_frame(
            points_color=point_colors,
            points_pos=points_world,
            cam_c2w=source_cam_c2w[frame_index],
            K=source_K[frame_index],
            H=H,
            W=W,
            dynamic_mask=None,
            z_tolerance=z_tolerance,
            occlusion_sharpness=occlusion_sharpness,
            batch_size=batch_size,
            allowed_pixel_mask=allowed_return_mask[frame_index],
            point_splat_mask=~dynamic_occluded_points,
            return_blocked_mask=True,
            point_ids=(
                point_source_frames
                if return_source_frame_provenance
                else None
            ),
            return_correspondence=return_source_frame_provenance,
        )
        rgb_global, depth_global, valid_global, _, blocked = render_result[:5]
        output_video[frame_index] = rgb_global.to(dtype=out_dtype)
        output_depth[frame_index] = depth_global.to(dtype=out_dtype)
        output_coverage[frame_index] = valid_global
        blocked_dynamic[frame_index] = blocked
        if return_source_frame_provenance:
            source_frame_map = render_result[5]
            output_source_frame[frame_index] = torch.where(
                valid_global,
                source_frame_map,
                torch.full_like(source_frame_map, -1),
            )

    result = (output_video, output_depth, output_coverage, blocked_dynamic)
    if return_source_frame_provenance:
        result = (*result, output_source_frame)
    return result


def compute_double_reprojection_source_frame_visibility_counts(
    local_return_coverage,
    global_return_coverage,
    global_return_source_frame,
):
    """Count final double-reprojection pixels by their true source frame.

    Locally returned pixels originate from the matching timestep and therefore
    contribute to the diagonal. Global-fill pixels are counted from the source
    frame of the fragment that actually won the final source-camera z-buffer.
    The two coverage masks must be disjoint because the local result is locked
    before the global fill pass.
    """
    if local_return_coverage.ndim != 3:
        raise ValueError(
            "local_return_coverage must have shape [F, H, W], got "
            f"{tuple(local_return_coverage.shape)}"
        )
    expected_shape = tuple(local_return_coverage.shape)
    for name, value in (
        ("global_return_coverage", global_return_coverage),
        ("global_return_source_frame", global_return_source_frame),
    ):
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got "
                f"{tuple(value.shape)}"
            )

    local_return_coverage = local_return_coverage.to(dtype=torch.bool)
    global_return_coverage = global_return_coverage.to(
        device=local_return_coverage.device,
        dtype=torch.bool,
    )
    global_return_source_frame = global_return_source_frame.to(
        device=local_return_coverage.device,
        dtype=torch.int64,
    )
    if (local_return_coverage & global_return_coverage).any():
        raise ValueError(
            "Local and global double-reprojection coverage must be disjoint"
        )

    F = expected_shape[0]
    counts = torch.zeros((F, F), dtype=torch.int64, device="cpu")
    local_counts = local_return_coverage.reshape(F, -1).sum(dim=1)
    counts[torch.arange(F), torch.arange(F)] = (
        local_counts.detach().cpu().to(torch.int64)
    )

    for target_frame in range(F):
        source_frames = global_return_source_frame[target_frame][
            global_return_coverage[target_frame]
        ]
        if source_frames.numel() == 0:
            continue
        min_source_frame = int(source_frames.min().item())
        max_source_frame = int(source_frames.max().item())
        if min_source_frame < 0 or max_source_frame >= F:
            raise ValueError(
                "Every covered global-return pixel must have a valid source "
                "frame index: "
                f"min={min_source_frame}, max={max_source_frame}, F={F}"
            )
        counts[target_frame] += torch.bincount(
            source_frames,
            minlength=F,
        ).detach().cpu().to(torch.int64)

    expected_covered = (
        local_return_coverage | global_return_coverage
    ).reshape(F, -1).sum(dim=1).detach().cpu().to(torch.int64)
    if not torch.equal(counts.sum(dim=1), expected_covered):
        raise RuntimeError(
            "Double-reprojection reference counts do not match final coverage"
        )
    return counts


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def compute_target_visibility(points_world, target_cam_c2w, target_K, H, W):
    """
    points_world: [N, 3]
    target_cam_c2w: [F, 4, 4]
    target_K: [F, 3, 3]
    return visible: [N, F]
    """
    device = points_world.device
    N = points_world.shape[0]
    F = target_cam_c2w.shape[0]

    # Important: inverse/projection math must be fp32.
    points_world_f = points_world.to(device=device, dtype=torch.float32)
    target_cam_c2w_f = target_cam_c2w.to(device=device, dtype=torch.float32)
    target_K_f = target_K.to(device=device, dtype=torch.float32)

    visible = torch.zeros((N, F), dtype=torch.bool, device=device)

    for i in range(F):
        cam_w2c = torch.linalg.inv(target_cam_c2w_f[i])

        R = cam_w2c[:3, :3]
        t = cam_w2c[:3, 3]

        pts_cam = points_world_f @ R.T + t[None, :]
        z = pts_cam[:, 2]

        valid_z = z > 1e-5

        uv = pts_cam @ target_K_f[i].T
        u = uv[:, 0] / uv[:, 2]
        v = uv[:, 1] / uv[:, 2]

        valid_uv = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        visible[:, i] = valid_uv

    return visible


@torch.no_grad()
def build_pointcloud_visibility(
    source_visible,
    indices,
    point_dynamic,
    points_world,
    target_cam_c2w,
    target_K,
    H,
    W,
    mode="target_frustum_static_dynamic_same_frame",
):
    """
    Decide which source point-cloud points are allowed to contribute to each target frame.

    source_visible comes from unproject(): static points are visible for all source frames,
    dynamic points are one-hot at their source frame. That is useful, but not enough for
    novel target cameras because we also want target-frustum visibility.

    Modes:
      same_frame:
        Point from source frame i only renders into target frame i. This is stereo-style.
      target_frustum_all:
        Any point can render into any target frame where it projects inside the target image.
      target_frustum_static_dynamic_same_frame:
        Static points use global target-frustum visibility. Dynamic points stay same-frame only.
    """
    same_frame_visible = torch.zeros_like(source_visible)
    same_frame_visible.scatter_(1, indices[:, 0:1], True)

    mode = str(mode).lower()
    if mode == "same_frame":
        return same_frame_visible

    target_visible = compute_target_visibility(
        points_world=points_world,
        target_cam_c2w=target_cam_c2w,
        target_K=target_K,
        H=H,
        W=W,
    )

    if mode == "target_frustum_all":
        return target_visible

    if mode == "target_frustum_static_dynamic_same_frame":
        return torch.where(point_dynamic[:, None], same_frame_visible, target_visible)

    raise ValueError(
        f"Unknown render.visibility_mode={mode!r}. Expected one of: "
        "same_frame, target_frustum_all, target_frustum_static_dynamic_same_frame."
    )




def build_temporal_instance_visibility(
    source_visible,
    indices,
    points_world,
    target_cam_c2w,
    target_K,
    H,
    W,
    mode="dynamic_temporal",
    max_past_frames=None,
):
    """Build visibility for warped dynamic instance IDs.

    This is intentionally separate from RGB rendering visibility.
    RGB can keep dynamic points same-frame only to avoid photometric ghosting,
    while seginstance_warped can reuse dynamic-instance observations from
    other sampled timesteps when they project into the target stereo view.

    Modes:
      same_frame:
        Only source frame i can render into target frame i. This matches the
        original seginstance_warped behavior.
      dynamic_temporal / target_frustum_all:
        Any source-frame instance point can render into any target frame where it
        projects inside the target camera frustum. This includes both past and
        future sampled frames.
      past_to_current / target_frustum_past_to_current:
        A dynamic-instance point from source frame j can render into target
        frame i when j <= i and it projects into the target camera frustum.
        If max_past_frames is set, require i - j <= max_past_frames.
    """
    mode = str(mode).strip().lower()
    if mode in {"same_as_warped", "same_as_rgb", "warped", "rgb"}:
        return source_visible

    same_frame_visible = torch.zeros_like(source_visible)
    same_frame_visible.scatter_(1, indices[:, 0:1], True)

    if mode in {"same_frame", "sameframe"}:
        return same_frame_visible

    target_visible = compute_target_visibility(
        points_world=points_world,
        target_cam_c2w=target_cam_c2w,
        target_K=target_K,
        H=H,
        W=W,
    )

    if mode in {"dynamic_temporal", "target_frustum_all", "all", "target_all"}:
        return target_visible

    if mode in {
        "target_frustum_past_to_current",
        "past_to_current",
        "past_current",
        "past",
    }:
        F = source_visible.shape[1]
        device = source_visible.device
        src_frame = indices[:, 0].to(device=device, dtype=torch.long)
        tgt_frame = torch.arange(F, device=device, dtype=torch.long)

        temporal = src_frame[:, None] <= tgt_frame[None, :]
        if max_past_frames is not None:
            max_past_frames = int(max_past_frames)
            if max_past_frames >= 0:
                temporal = temporal & ((tgt_frame[None, :] - src_frame[:, None]) <= max_past_frames)

        return target_visible & temporal

    raise ValueError(
        f"Unknown seginstance_warped.visibility_mode={mode!r}. Expected one of: "
        "same_as_warped, same_frame, dynamic_temporal, past_to_current, target_frustum_all."
    )


def _as_bool_tensor_like(mask, ref, name):
    """Convert an optional mask to a bool tensor on the same device as ref."""
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        out = mask.detach().to(dtype=torch.bool, device=ref.device)
    else:
        out = torch.from_numpy(np.asarray(mask)).to(dtype=torch.bool, device=ref.device)
    if out.shape != ref.shape:
        raise ValueError(f"{name} has shape {tuple(out.shape)}, expected {tuple(ref.shape)}")
    return out


def build_seginstance_id_maps(
    instance_id_maps,
    dynamic_mask=None,
    static_mask=None,
    sky_mask=None,
    object_mode="same_as_warped",
):
    """Return ID-consistent source-frame instance maps for seginstance_warped.

    instance_id_maps: [F, H, W], SAM3 track IDs where 0 is background.
    dynamic_mask: [F, H, W], refined dynamic mask after static-vehicle and sky removal.
    static_mask: [F, H, W], static/background mask used for RGB warping.
    sky_mask: [F, H, W], optional sky mask to force to 0.

    object_mode controls which SAM3 instance IDs are kept:
      same_as_warped / all_instances / all_sam3_instances:
        Keep every non-sky SAM3 instance ID. Its temporal behavior is then controlled
        by seginstance_visibility_mode. With seginstance_visibility_mode=same_as_warped,
        static instances use past/current/future visibility while dynamic instances stay
        same-frame, exactly like RGB front_warped.
      dynamic_only:
        Keep only IDs on the refined dynamic mask. This is the old behavior.
      static_only:
        Keep only IDs on the static mask. Since background has ID 0, this effectively
        keeps static SAM3 instances such as parked/static vehicles.
    """
    if isinstance(instance_id_maps, torch.Tensor):
        inst = instance_id_maps.detach().clone().to(dtype=torch.int64)
    else:
        inst = torch.from_numpy(np.asarray(instance_id_maps)).to(dtype=torch.int64)

    mode = str(object_mode).strip().lower()
    keep = inst > 0

    dyn = _as_bool_tensor_like(dynamic_mask, inst, "dynamic_mask")
    sta = _as_bool_tensor_like(static_mask, inst, "static_mask")
    sky = _as_bool_tensor_like(sky_mask, inst, "sky_mask")

    if sky is not None:
        keep = keep & ~sky

    if mode in {
        "same_as_warped",
        "same_as_rgb",
        "warped",
        "rgb",
        "all_instances",
        "all_instance",
        "all_sam3_instances",
        "all_sam3",
        "all",
    }:
        pass
    elif mode in {"dynamic_only", "dynamic", "moving_only", "moving"}:
        if dyn is None:
            raise ValueError("seginstance object_mode='dynamic_only' requires dynamic_mask")
        keep = keep & dyn
    elif mode in {"static_only", "static"}:
        if sta is None:
            if dyn is None:
                raise ValueError(
                    "seginstance object_mode='static_only' requires static_mask or dynamic_mask"
                )
            keep = keep & ~dyn
        else:
            keep = keep & sta
    else:
        raise ValueError(
            f"Unknown seginstance_warped.object_mode={object_mode!r}. Expected one of: "
            "same_as_warped, all_instances, dynamic_only, static_only."
        )

    inst[~keep] = 0
    return inst


def build_dynamic_instance_id_maps(instance_id_maps, dynamic_mask):
    """Return ID-consistent source-frame dynamic-instance maps.

    This preserves the previous seginstance_warped behavior: static vehicles, sky,
    and background are set to 0. New code should usually use
    build_seginstance_id_maps(..., object_mode='same_as_warped') so static vehicle
    IDs can be warped with the same temporal rule as RGB front_warped.
    """
    return build_seginstance_id_maps(
        instance_id_maps=instance_id_maps,
        dynamic_mask=dynamic_mask,
        object_mode="dynamic_only",
    )


@torch.autocast(device_type="cuda", enabled=False)
def render_instance_id_frame(
    points_instance_id,
    points_pos,
    cam_c2w,
    K,
    H,
    W,
    z_tolerance=0.02,
    occlusion_sharpness=10.0,
    batch_size=int(1e7),
):
    """Splat categorical instance IDs into one target camera frame.

    This uses the same projection, 4-neighbor splat footprint, depth weighting, and
    log-depth z tolerance as render_frame(), but writes an integer instance ID
    instead of blended RGB. Only points with instance_id > 0 are splatted.
    """
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError(
            "render_instance_id_frame requires torch.Tensor.scatter_reduce_. "
            "Please use PyTorch >= 1.12."
        )

    device = points_pos.device
    N = points_pos.shape[0]
    batch_size = int(batch_size)

    points_instance_id = points_instance_id.to(device=device, dtype=torch.int64)
    points_pos = points_pos.to(device=device, dtype=torch.float32)
    cam_c2w = cam_c2w.to(device=device, dtype=torch.float32)
    K = K.to(device=device, dtype=torch.float32)

    out_id = torch.zeros((H * W,), dtype=torch.int64, device=device)
    z_buffer = torch.full((H * W,), float("inf"), dtype=torch.float32, device=device)
    score_buffer = torch.zeros((H * W,), dtype=torch.float32, device=device)

    local_min_depth = torch.empty((H * W,), dtype=torch.float32, device=device)
    local_max_score = torch.empty((H * W,), dtype=torch.float32, device=device)
    local_label = torch.empty((H * W,), dtype=torch.int64, device=device)

    cam_w2c = torch.linalg.inv(cam_c2w)
    R = cam_w2c[:3, :3]
    T = cam_w2c[:3, 3]

    for start in range(0, N, batch_size):
        pos = points_pos[start : start + batch_size]
        ids = points_instance_id[start : start + batch_size]

        id_valid = ids > 0
        if not id_valid.any():
            continue

        pos = pos[id_valid]
        ids = ids[id_valid]

        pc = (pos @ R.T) + T
        d = pc[:, 2]
        valid = d > 1e-5
        if not valid.any():
            continue

        pc = pc[valid]
        ids = ids[valid]
        d = d[valid]

        uvz = (K @ pc.T).T
        u = uvz[:, 0] / uvz[:, 2]
        v = uvz[:, 1] / uvz[:, 2]

        u0 = torch.floor(u).to(torch.int64)
        v0 = torch.floor(v).to(torch.int64)
        du = u - u0
        dv = v - v0

        u_frag = torch.cat([u0, u0, u0 + 1, u0 + 1])
        v_frag = torch.cat([v0, v0 + 1, v0, v0 + 1])
        w_geom = torch.cat(
            [
                (1 - du) * (1 - dv),
                (1 - du) * dv,
                du * (1 - dv),
                du * dv,
            ]
        )

        d_frag = d.repeat(4)
        id_frag = ids.repeat(4)

        w_depth = torch.pow(
            d_frag.to(torch.float32).clamp(min=1e-1),
            -occlusion_sharpness,
        ).clamp(min=1e-10)
        score_frag = w_geom.to(torch.float32) * w_depth

        valid_uv = (
            (u_frag >= 0)
            & (u_frag < W)
            & (v_frag >= 0)
            & (v_frag < H)
            & (score_frag > 0)
        )
        if not valid_uv.any():
            continue

        idx = v_frag[valid_uv] * W + u_frag[valid_uv]
        d_frag = d_frag[valid_uv].to(torch.float32)
        id_frag = id_frag[valid_uv]
        score_frag = score_frag[valid_uv]

        # Within this chunk, keep only the front surface per pixel, matching the
        # log-depth tolerance used for RGB splatting.
        local_min_depth.fill_(float("inf"))
        local_min_depth.scatter_reduce_(0, idx, d_frag, reduce="amin", include_self=True)
        min_d = local_min_depth[idx]
        front = (torch.log1p(d_frag) - torch.log1p(min_d)).abs() <= z_tolerance
        if not front.any():
            continue

        idx_front = idx[front]
        d_front = d_frag[front]
        id_front = id_frag[front]
        score_front = score_frag[front]

        local_max_score.fill_(0.0)
        local_max_score.scatter_reduce_(
            0, idx_front, score_front, reduce="amax", include_self=True
        )
        is_best_score = score_front >= local_max_score[idx_front]

        local_label.fill_(0)
        local_label.scatter_reduce_(
            0,
            idx_front[is_best_score],
            id_front[is_best_score].to(torch.int64),
            reduce="amax",
            include_self=True,
        )

        touched = torch.isfinite(local_min_depth) & (local_label > 0)
        if not touched.any():
            continue

        candidate_idx = torch.nonzero(touched, as_tuple=False).flatten()
        candidate_depth = local_min_depth[candidate_idx]
        candidate_score = local_max_score[candidate_idx]
        candidate_label = local_label[candidate_idx]

        z_curr = z_buffer[candidate_idx]
        rel = (torch.log1p(candidate_depth) - torch.log1p(z_curr)).abs()
        reset = (candidate_depth < z_curr) & (rel > z_tolerance)
        blend = reset | (rel <= z_tolerance)
        if not blend.any():
            continue

        update_idx = candidate_idx[blend]
        update_depth = candidate_depth[blend]
        update_score = candidate_score[blend]
        update_label = candidate_label[blend]
        update_reset = reset[blend]

        if update_reset.any():
            reset_idx = update_idx[update_reset]
            z_buffer[reset_idx] = update_depth[update_reset]
            score_buffer[reset_idx] = 0.0
            out_id[reset_idx] = 0

        better = update_score >= score_buffer[update_idx]
        if better.any():
            best_idx = update_idx[better]
            score_buffer[best_idx] = update_score[better]
            out_id[best_idx] = update_label[better]

    return out_id.reshape(H, W)


@torch.autocast(device_type="cuda", enabled=False)
def render_instance_ids(
    points_instance_id,
    points_pos,
    visible,
    cam_c2w,
    K,
    H,
    W,
    verbose=False,
    z_tolerance=0.02,
    occlusion_sharpness=10.0,
    batch_size=int(1e7),
):
    """Render a target-view video of warped dynamic-instance IDs."""
    _, F = visible.shape
    device = points_pos.device
    instance_video = torch.zeros(F, H, W, device=device, dtype=torch.int64)
    points_instance_id = points_instance_id.to(device=device, dtype=torch.int64)

    iterator = trange(F, desc="Rendering warped instance IDs") if verbose else range(F)
    for i in iterator:
        vis = visible[:, i]
        if not vis.any():
            continue
        instance_video[i] = render_instance_id_frame(
            points_instance_id=points_instance_id[vis],
            points_pos=points_pos[vis],
            cam_c2w=cam_c2w[i],
            K=K[i],
            H=H,
            W=W,
            z_tolerance=z_tolerance,
            occlusion_sharpness=occlusion_sharpness,
            batch_size=batch_size,
        )

    return instance_video


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def compute_source_frame_visibility_counts(
    points_source_frame,
    points_pos,
    visible,
    cam_c2w,
    K,
    H,
    W,
    verbose=False,
    z_tolerance=0.02,
    occlusion_sharpness=10.0,
    batch_size=int(1e7),
):
    """Count the dominant source-frame pixels for every rendered target frame.

    Each source point carries its zero-based sampled source-frame index. For each
    target frame, those indices are rendered as categorical labels through the
    same projection, 4-neighbor splatting, log-depth z tolerance, and depth
    weighting used by the RGB and instance renderers. Therefore, points that are
    outside the configured visibility mask or hidden behind a closer surface do
    not contribute to the score.

    A covered target pixel is assigned to one dominant source frame. The returned
    matrix stores the number of dominated target pixels:

        counts[target_frame, source_frame]

    The matrix stays on CPU and has shape [F, F], so this avoids retaining a full
    [F, H, W] source-provenance video in GPU memory.
    """
    if points_source_frame.ndim != 1:
        raise ValueError(
            "points_source_frame must be a 1D tensor, got "
            f"shape={tuple(points_source_frame.shape)}"
        )
    if points_source_frame.shape[0] != points_pos.shape[0]:
        raise ValueError(
            "points_source_frame and points_pos must contain the same number of points: "
            f"{points_source_frame.shape[0]} != {points_pos.shape[0]}"
        )

    _, F = visible.shape
    source_frame = points_source_frame.to(
        device=points_pos.device,
        dtype=torch.int64,
    )

    if source_frame.numel() > 0:
        min_source_frame = int(source_frame.min().item())
        max_source_frame = int(source_frame.max().item())
        if min_source_frame < 0 or max_source_frame >= F:
            raise ValueError(
                "points_source_frame contains an index outside the sampled sequence: "
                f"min={min_source_frame}, max={max_source_frame}, F={F}"
            )

    # render_instance_id_frame reserves label 0 for uncovered pixels.
    source_labels = source_frame + 1
    counts = torch.zeros((F, F), dtype=torch.int64, device="cpu")

    iterator = (
        trange(F, desc="Ranking source reference frames") if verbose else range(F)
    )
    for target_frame in iterator:
        vis = visible[:, target_frame]
        if not vis.any():
            continue

        source_label_map = render_instance_id_frame(
            points_instance_id=source_labels[vis],
            points_pos=points_pos[vis],
            cam_c2w=cam_c2w[target_frame],
            K=K[target_frame],
            H=H,
            W=W,
            z_tolerance=z_tolerance,
            occlusion_sharpness=occlusion_sharpness,
            batch_size=batch_size,
        )
        frame_counts = torch.bincount(
            source_label_map.reshape(-1),
            minlength=F + 1,
        )[1 : F + 1]
        counts[target_frame] = frame_counts.detach().cpu()

    return counts
