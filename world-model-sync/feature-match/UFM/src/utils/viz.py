from pathlib import Path

import cv2
import numpy as np
import torch
import os


def color_from_id(obj_id):
    # deterministic color
    x = (int(obj_id) * 2654435761) & 0xFFFFFFFF
    r = 64 + (x & 0xFF) // 2
    g = 64 + ((x >> 8) & 0xFF) // 2
    b = 64 + ((x >> 16) & 0xFF) // 2
    return int(r), int(g), int(b)



def make_instance_id_overlay(rgb, instance_id_map, alpha=0.50, draw_ids=True):
    """Return an RGB uint8 visualization of an instance-id map over an RGB image.

    Args:
        rgb: HxWx3 image. Accepts float [0, 1] or uint8 [0, 255], RGB order.
        instance_id_map: HxW integer map, where 0 is background.
        alpha: overlay strength for instance colors.
        draw_ids: draw text labels at each instance centroid.
    """
    rgb_np = np.asarray(rgb)
    if rgb_np.dtype != np.uint8:
        rgb_np = (np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        rgb_np = rgb_np.copy()

    inst_np = np.asarray(instance_id_map).astype(np.int64)
    overlay = rgb_np.copy()

    ids = np.unique(inst_np)
    ids = ids[ids > 0]

    for obj_id in ids:
        mask = inst_np == obj_id
        if not np.any(mask):
            continue

        color = np.array(color_from_id(int(obj_id)), dtype=np.float32)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1.0 - float(alpha))
            + color * float(alpha)
        ).clip(0, 255).astype(np.uint8)

        if draw_ids:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx = int(xs.mean())
                cy = int(ys.mean())
                cv2.putText(
                    overlay,
                    f"id={int(obj_id)}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"id={int(obj_id)}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    tuple(int(x) for x in color.tolist()),
                    1,
                    cv2.LINE_AA,
                )

    return overlay


def _frame_to_numpy(frame):
    """Move one tensor-like frame to CPU and return it as a NumPy array."""
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    return np.asarray(frame)


def _rgb_frame_to_uint8(frame, channel_first=False):
    """Convert one RGB frame to HxWx3 uint8 without resizing it."""
    frame_np = _frame_to_numpy(frame)
    if channel_first:
        if frame_np.ndim != 3 or frame_np.shape[0] != 3:
            raise ValueError(
                "Expected a channel-first RGB frame with shape [3, H, W], got "
                f"{tuple(frame_np.shape)}"
            )
        frame_np = np.transpose(frame_np, (1, 2, 0))

    if frame_np.ndim != 3 or frame_np.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB frame with shape [H, W, 3], got {tuple(frame_np.shape)}"
        )

    if frame_np.dtype == np.uint8:
        return frame_np.copy()
    return (np.clip(frame_np, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_visualization_grid_video(
    path,
    source_video,
    warped_video,
    coverage_video,
    dynamic_instance_id_maps,
    seginstance_warped,
    fps=10,
    overlay_alpha=0.50,
    draw_ids=True,
):
    """Save the pipeline visualization as one native-tile-resolution 2x3 video.

    The grid is::

        source | warped | confident coverage
        dynamic-instance overlay on source | warped-instance overlay | blank

    No tile is resized. If the input frames are HxW, the encoded video is
    therefore (2H)x(3W).
    """
    path = Path(path)
    if float(fps) <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    source_shape = tuple(source_video.shape)
    warped_shape = tuple(warped_video.shape)
    coverage_shape = tuple(coverage_video.shape)
    dynamic_shape = tuple(dynamic_instance_id_maps.shape)
    seginstance_shape = tuple(seginstance_warped.shape)

    if len(source_shape) != 4 or source_shape[1] != 3:
        raise ValueError(
            f"Expected source_video shape [F, 3, H, W], got {source_shape}"
        )
    frame_count, _, height, width = source_shape
    expected_rgb_shape = (frame_count, height, width, 3)
    expected_map_shape = (frame_count, height, width)
    if warped_shape != expected_rgb_shape:
        raise ValueError(
            f"Expected warped_video shape {expected_rgb_shape}, got {warped_shape}"
        )
    for name, shape in (
        ("coverage_video", coverage_shape),
        ("dynamic_instance_id_maps", dynamic_shape),
        ("seginstance_warped", seginstance_shape),
    ):
        if shape != expected_map_shape:
            raise ValueError(f"Expected {name} shape {expected_map_shape}, got {shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    grid_size = (3 * width, 2 * height)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        grid_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Could not open video writer for {path} at exact size {grid_size}"
        )

    try:
        for frame_index in range(frame_count):
            source_rgb = _rgb_frame_to_uint8(
                source_video[frame_index], channel_first=True
            )
            warped_rgb = _rgb_frame_to_uint8(warped_video[frame_index])

            coverage_np = _frame_to_numpy(coverage_video[frame_index])
            if coverage_np.dtype == np.uint8:
                coverage_u8 = coverage_np.copy()
            else:
                coverage_u8 = (
                    np.clip(coverage_np.astype(np.float32), 0.0, 1.0) * 255.0
                ).astype(np.uint8)
            coverage_rgb = np.repeat(coverage_u8[:, :, None], 3, axis=2)

            dynamic_ids = _frame_to_numpy(
                dynamic_instance_id_maps[frame_index]
            ).astype(np.int64)
            warped_ids = _frame_to_numpy(seginstance_warped[frame_index]).astype(
                np.int64
            )
            dynamic_overlay = make_instance_id_overlay(
                rgb=source_rgb,
                instance_id_map=dynamic_ids,
                alpha=overlay_alpha,
                draw_ids=draw_ids,
            )
            warped_instance_overlay = make_instance_id_overlay(
                rgb=warped_rgb,
                instance_id_map=warped_ids,
                alpha=overlay_alpha,
                draw_ids=draw_ids,
            )
            blank = np.zeros_like(source_rgb)

            top_row = np.concatenate(
                [source_rgb, warped_rgb, coverage_rgb], axis=1
            )
            bottom_row = np.concatenate(
                [dynamic_overlay, warped_instance_overlay, blank], axis=1
            )
            grid_rgb = np.concatenate([top_row, bottom_row], axis=0)
            writer.write(cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return path


def save_instance_id_overlay_image(path, rgb, instance_id_map, alpha=0.50, draw_ids=True):
    """Save one RGB overlay image for an instance-id map."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = make_instance_id_overlay(
        rgb=rgb,
        instance_id_map=instance_id_map,
        alpha=alpha,
        draw_ids=draw_ids,
    )
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_instance_id_overlay_frames(
    left_video,  # torch [F, 3, H, W], RGB [0, 1]
    instance_id_maps,  # torch/int [F, H, W], 0 = background
    out_dir,
    alpha=0.50,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_np = left_video.detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()

    inst_np = instance_id_maps.detach().cpu().numpy().astype(np.int32)

    for i in range(left_np.shape[0]):
        rgb = (left_np[i] * 255).astype(np.uint8)
        overlay = rgb.copy()

        ids = np.unique(inst_np[i])
        ids = ids[ids > 0]

        for obj_id in ids:
            mask = inst_np[i] == obj_id
            color = np.array(color_from_id(obj_id), dtype=np.float32)

            overlay[mask] = (
                (overlay[mask].astype(np.float32) * (1.0 - alpha) + color * alpha)
                .clip(0, 255)
                .astype(np.uint8)
            )

            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx = int(xs.mean())
                cy = int(ys.mean())

                cv2.putText(
                    overlay,
                    f"id={int(obj_id)}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    overlay,
                    f"id={int(obj_id)}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    tuple(int(x) for x in color.tolist()),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imwrite(
            str(out_dir / f"{i:04d}_instance_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        # Also save raw instance id map as uint16 PNG
        cv2.imwrite(
            str(out_dir / f"{i:04d}_instance_ids.png"),
            inst_np[i].astype(np.uint16),
        )


def save_rgb(path, rgb):
    x = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
