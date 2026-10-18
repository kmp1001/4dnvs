"""Audit Top20-global then eight-reference pixel-union selection.

This is deliberately opt-in. The original UFM ranking remains unchanged.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.pointcloud import (
    classify_points_occluded_by_current_dynamic_mask,
    render_frame,
)
from utils.utils import tensor_to_rgb_numpy
from utils.viz import save_rgb


def _write_mask(path, mask):
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def _tiles(mask):
    height, width = mask.shape
    return [
        int(mask[: height // 2, : width // 2].sum()),
        int(mask[: height // 2, width // 2 :].sum()),
        int(mask[height // 2 :, : width // 2].sum()),
        int(mask[height // 2 :, width // 2 :].sum()),
    ]


@torch.no_grad()
def save_union_selection_trace(
    *, output_root, scene_output_name, target_index, source_video,
    points_pos, indices, point_dynamic, target_visibility,
    target_cam_c2w, target_K, source_depth, source_dynamic_mask,
    local_return_coverage, local_return_depth,
    global_return_depth, final_coverage, global_counts,
    top_k=20, select_k=8, z_tolerance=0.10,
    occlusion_sharpness=10.0, batch_size=10000000,
):
    """Save all candidate masks and greedy union gains for one target frame.

    Candidate j>target is not assumed disjoint from another candidate. Its
    source-j static points are rendered independently into the target camera.
    Eligibility uses the same frustum, dynamic-object exclusion, splat, and
    depth tolerance as the reconstruction. Pixel support is then restricted
    to the final reconstructed target and front-surface depth. The matching
    target frame uses its actual local double-reprojection coverage.
    """
    target_index = int(target_index)
    frames, height, width = local_return_coverage.shape
    if not 0 <= target_index < frames:
        raise IndexError(target_index)
    counts = np.asarray(global_counts[target_index], dtype=np.int64)
    top20 = sorted(
        range(frames),
        key=lambda j: (-int(counts[j]), abs(j - target_index), j),
    )[: min(int(top_k), frames)]
    trace_dir = (Path(output_root) / "reference_union_trace"
                 / scene_output_name / f"{target_index:05d}")
    trace_dir.mkdir(parents=True, exist_ok=True)

    local = local_return_coverage[target_index].bool()
    final = final_coverage[target_index].bool()
    target_dynamic = source_dynamic_mask[target_index].bool()
    final_depth = torch.where(
        local,
        local_return_depth[target_index].float(),
        global_return_depth[target_index].float(),
    )
    source_frames = indices[:, 0].long()
    masks = {}
    candidate_stats = []
    for j in top20:
        if j == target_index:
            accepted = local
            raw_count = int(local.sum().item())
            depth_rejected = 0
        else:
            point_selection = (
                (source_frames == j) & (~point_dynamic.bool())
                & target_visibility[:, target_index].bool()
            )
            subset = points_pos[point_selection]
            if subset.numel() == 0:
                accepted = torch.zeros_like(final)
                raw_count = 0
                depth_rejected = 0
            else:
                occluded = classify_points_occluded_by_current_dynamic_mask(
                    points_world=subset,
                    source_cam_c2w=target_cam_c2w[target_index],
                    source_K=target_K[target_index],
                    source_depth=source_depth[target_index],
                    source_dynamic_mask=target_dynamic,
                )
                rendered = render_frame(
                    points_color=torch.zeros_like(subset),
                    points_pos=subset,
                    cam_c2w=target_cam_c2w[target_index],
                    K=target_K[target_index],
                    H=height, W=width,
                    z_tolerance=float(z_tolerance),
                    occlusion_sharpness=float(occlusion_sharpness),
                    batch_size=int(batch_size),
                    allowed_pixel_mask=final & ~target_dynamic,
                    point_splat_mask=~occluded,
                )
                candidate_depth, candidate_coverage = rendered[1:3]
                raw_count = int(candidate_coverage.sum().item())
                matched_depth = (
                    torch.log1p(candidate_depth.float().clamp_min(0))
                    - torch.log1p(final_depth.clamp_min(0))
                ).abs() <= float(z_tolerance)
                accepted = candidate_coverage & matched_depth
                depth_rejected = raw_count - int(accepted.sum().item())
        mask = accepted.detach().cpu().numpy().astype(bool)
        masks[j] = mask
        _write_mask(trace_dir / f"candidate_{j:05d}_support.png", mask)
        image = tensor_to_rgb_numpy(source_video[j]).astype(np.float32) / 255.0
        save_rgb(trace_dir / f"candidate_{j:05d}_source.png", image)
        candidate_stats.append(dict(
            source_frame=int(j), baseline_global_pixels=int(counts[j]),
            raw_independent_pixels=int(raw_count),
            depth_rejected_pixels=int(depth_rejected),
            accepted_pixels=int(mask.sum()),
            accepted_tiles=_tiles(mask),
        ))

    selected = [top20[0]]
    union = masks[selected[0]].copy()
    steps = [dict(
        rank=1, chosen=int(selected[0]),
        chosen_new_tiles=_tiles(union), chosen_new_pixels=int(union.sum()),
        union_pixels_after=int(union.sum()),
        candidates=[],
    )]
    _write_mask(trace_dir / "rank_01_new_pixels.png", union)
    _write_mask(trace_dir / "rank_01_union.png", union)
    rank_in_top20 = {j: rank for rank, j in enumerate(top20)}
    for rank in range(2, min(int(select_k), len(top20)) + 1):
        options = []
        for j in top20:
            if j in selected:
                continue
            gain = masks[j] & ~union
            tiles = _tiles(gain)
            options.append(dict(
                source_frame=int(j), marginal_tiles=tiles,
                marginal_pixels=int(sum(tiles)),
                overlapping_pixels=int((masks[j] & union).sum()),
                baseline_global_pixels=int(counts[j]),
            ))
        options.sort(key=lambda item: (
            -item["marginal_pixels"],
            rank_in_top20[item["source_frame"]],
        ))
        chosen = options[0]["source_frame"]
        new = masks[chosen] & ~union
        union |= masks[chosen]
        selected.append(chosen)
        steps.append(dict(
            rank=rank, chosen=int(chosen),
            chosen_new_tiles=_tiles(new),
            chosen_new_pixels=int(new.sum()),
            union_pixels_after=int(union.sum()),
            candidates=options,
        ))
        _write_mask(trace_dir / f"rank_{rank:02d}_new_pixels.png", new)
        _write_mask(trace_dir / f"rank_{rank:02d}_union.png", union)

    result = dict(
        scene=scene_output_name, target_frame=target_index,
        frame_size=[height, width],
        metric=("Top20 by final pixel provenance; then maximize newly "
                "covered target pixels across four tiles, using independent "
                "source-frame front-surface masks and union subtraction"),
        target_domain_pixels=int(final.sum().item()),
        baseline_top20=[int(j) for j in top20],
        baseline_top8=[int(j) for j in top20[:8]],
        union_top8=[int(j) for j in selected],
        candidate_stats=candidate_stats,
        steps=steps,
    )
    with (trace_dir / "selection.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    with (trace_dir / "candidate_steps.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "selection_rank", "candidate_source_frame", "chosen",
            "baseline_global_pixels", "independent_support_pixels",
            "overlap_with_selected_pixels", "marginal_pixels",
            "marginal_TL", "marginal_TR", "marginal_BL", "marginal_BR",
        ])
        for step in steps[1:]:
            for option in step["candidates"]:
                writer.writerow([
                    step["rank"], option["source_frame"],
                    int(option["source_frame"] == step["chosen"]),
                    option["baseline_global_pixels"],
                    next(x["accepted_pixels"] for x in candidate_stats
                         if x["source_frame"] == option["source_frame"]),
                    option["overlapping_pixels"], option["marginal_pixels"],
                    *option["marginal_tiles"],
                ])
    print(f"Saved union-gain reference trace: {trace_dir}", flush=True)
    return result
