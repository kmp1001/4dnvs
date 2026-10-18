"""Detailed DeDup audit for one double-reprojection target frame."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from utils.utils import tensor_to_rgb_numpy
from utils.viz import save_rgb


def _rgb(path, frame):
    image = frame.detach().float().cpu().numpy()
    save_rgb(path, image)


def _mask(path, frame):
    values = frame.detach().cpu().numpy().astype(np.uint8) * 255
    cv2.imwrite(str(path), values)


def save_reference_trace(
    output_root,
    scene_output_name,
    target_index,
    source_video,
    virtual_rgb,
    virtual_coverage,
    local_rgb,
    local_coverage,
    global_rgb,
    global_coverage,
    final_rgb,
    final_coverage,
    source_dynamic_mask,
    global_source_frame,
    global_counts,
    tile_counts,
    top_k,
):
    """Save images, all-source visibility, and every greedy candidate score."""
    target_index = int(target_index)
    frames, height, width = local_coverage.shape
    if not 0 <= target_index < frames:
        raise IndexError(f"Trace frame {target_index} outside 0..{frames - 1}")
    trace_dir = (
        Path(output_root) / "reference_trace" / scene_output_name
        / f"{target_index:05d}"
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    source = tensor_to_rgb_numpy(source_video[target_index]).astype("float32") / 255.0
    save_rgb(trace_dir / "00_source.png", source)
    for filename, video in (
        ("01_virtual_render.png", virtual_rgb),
        ("02_local_return.png", local_rgb),
        ("03_global_fill_only.png", global_rgb),
        ("04_final_double_reprojected.png", final_rgb),
    ):
        _rgb(trace_dir / filename, video[target_index])
    for filename, masks in (
        ("01_virtual_coverage.png", virtual_coverage),
        ("02_local_coverage.png", local_coverage),
        ("03_global_fill_coverage.png", global_coverage),
        ("04_final_coverage.png", final_coverage),
        ("00_source_dynamic_mask.png", source_dynamic_mask),
    ):
        _mask(trace_dir / filename, masks[target_index])
    local = local_coverage[target_index].detach().cpu().numpy().astype(bool)
    global_mask = global_coverage[target_index].detach().cpu().numpy().astype(bool)
    global_ids = global_source_frame[target_index].detach().cpu().numpy()
    provenance = np.full((height, width), -1, dtype=np.int16)
    provenance[local] = target_index
    provenance[global_mask] = global_ids[global_mask]
    np.save(trace_dir / "04_final_source_frame_ids.npy", provenance)
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    valid = provenance >= 0
    hsv[..., 0][valid] = (provenance[valid].astype(np.int32) * 137) % 180
    hsv[..., 1][valid] = 220
    hsv[..., 2][valid] = 255
    cv2.imwrite(
        str(trace_dir / "04_final_source_frame_ids_color.png"),
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
    )

    global_row = np.asarray(global_counts[target_index], dtype=np.int64)
    tile_row = np.asarray(tile_counts[target_index], dtype=np.int64)
    if tile_row.shape != (4, frames):
        raise ValueError(f"Unexpected tile-vote shape: {tile_row.shape}")
    if not np.array_equal(tile_row.sum(axis=0), global_row):
        raise ValueError("Tile votes do not equal global visibility")
    baseline = sorted(
        range(frames),
        key=lambda candidate: (
            -int(global_row[candidate]),
            abs(candidate - target_index),
            candidate,
        ),
    )[: min(int(top_k), frames)]
    with (trace_dir / "all_source_visibility.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["source_frame", "global_pixels", "tile_0", "tile_1", "tile_2", "tile_3"]
        )
        for candidate in range(frames):
            writer.writerow(
                [candidate, int(global_row[candidate])]
                + [int(tile_row[tile, candidate]) for tile in range(4)]
            )

    traces = {}
    for lambda_tenths in (3, 5, 7):
        selected = []
        covered = np.zeros(4, dtype=np.int64)
        steps = []
        csv_path = trace_dir / f"dedup_lambda_0p{lambda_tenths}_steps.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["step", "candidate_source_frame", "selected_before",
                 "global_pixels", "tile_0", "tile_1", "tile_2", "tile_3",
                 "C_0", "C_1", "C_2", "C_3",
                 "delta_0", "delta_1", "delta_2", "delta_3",
                 "delta_sum", "lambda_tile_bonus_pixels", "total_score_pixels",
                 "chosen"]
            )
            for step_index in range(min(int(top_k), frames)):
                before = covered.copy()
                candidates = []
                best_key = None
                chosen = None
                for candidate in range(frames):
                    tile_values = tile_row[:, candidate]
                    delta = np.maximum(tile_values - before, 0)
                    delta_sum = int(delta.sum())
                    global_pixels = int(global_row[candidate])
                    score_tenths = (
                        10 * global_pixels + lambda_tenths * delta_sum
                    )
                    eligible = candidate not in selected
                    key = (
                        score_tenths,
                        -abs(candidate - target_index),
                        -candidate,
                    )
                    if eligible and (best_key is None or key > best_key):
                        best_key = key
                        chosen = candidate
                    candidates.append(
                        {
                            "source_frame": candidate,
                            "selected_before": not eligible,
                            "global_pixels": global_pixels,
                            "tile_pixels": [int(v) for v in tile_values],
                            "delta_tiles": [int(v) for v in delta],
                            "delta_sum": delta_sum,
                            "weighted_tile_bonus": lambda_tenths * delta_sum / 10.0,
                            "total_score": score_tenths / 10.0,
                        }
                    )
                for row in candidates:
                    writer.writerow(
                        [step_index + 1, row["source_frame"], int(row["selected_before"]),
                         row["global_pixels"]]
                        + row["tile_pixels"]
                        + [int(v) for v in before]
                        + row["delta_tiles"]
                        + [row["delta_sum"], row["weighted_tile_bonus"],
                           row["total_score"], int(row["source_frame"] == chosen)]
                    )
                selected.append(chosen)
                covered = np.maximum(covered, tile_row[:, chosen])
                steps.append(
                    {
                        "step": step_index + 1,
                        "selected_before": selected[:-1],
                        "C_before": [int(v) for v in before],
                        "chosen_source_frame": chosen,
                        "C_after": [int(v) for v in covered],
                        "candidates": candidates,
                    }
                )
        traces[f"0.{lambda_tenths}"] = {
            "ranking": selected,
            "steps": steps,
        }
    payload = {
        "scene_output_name": scene_output_name,
        "target_frame": target_index,
        "image_size": [height, width],
        "tile_grid": [2, 2],
        "tile_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
        "lambda_global": 1.0,
        "global_visibility": [int(v) for v in global_row],
        "tile_visibility": [[int(v) for v in row] for row in tile_row],
        "baseline_ranking": baseline,
        "dedup": traces,
        "local_covered_pixels": int(local.sum()),
        "global_fill_pixels": int(global_mask.sum()),
        "final_covered_pixels": int((local | global_mask).sum()),
    }
    (trace_dir / "dedup_trace.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved DeDup step trace: {trace_dir}", flush=True)
