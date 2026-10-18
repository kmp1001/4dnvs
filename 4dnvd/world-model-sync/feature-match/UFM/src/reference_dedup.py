"""DeDup ranking adapted to UFM double-reprojection pixel provenance."""

import numpy as np
import torch


def tile_vote_counts(local_coverage, global_coverage, global_source_frame):
    """Return [target_frame, tile, source_frame] pixel vote counts."""
    frames, height, width = local_coverage.shape
    if global_coverage.shape != local_coverage.shape:
        raise ValueError("global coverage shape mismatch")
    if global_source_frame.shape != local_coverage.shape:
        raise ValueError("source provenance shape mismatch")
    result = np.zeros((frames, 4, frames), dtype=np.int64)
    row_ids = np.arange(frames)
    tiles = (
        (slice(0, height // 2), slice(0, width // 2)),
        (slice(0, height // 2), slice(width // 2, width)),
        (slice(height // 2, height), slice(0, width // 2)),
        (slice(height // 2, height), slice(width // 2, width)),
    )
    for tile_index, (ys, xs) in enumerate(tiles):
        local = local_coverage[:, ys, xs].reshape(frames, -1).sum(dim=1)
        result[row_ids, tile_index, row_ids] = local.cpu().numpy()
        for target_index in range(frames):
            ids = global_source_frame[target_index, ys, xs][
                global_coverage[target_index, ys, xs]
            ].to(dtype=torch.int64)
            if ids.numel():
                if int(ids.min()) < 0 or int(ids.max()) >= frames:
                    raise ValueError("invalid source frame index")
                result[target_index, tile_index] += (
                    torch.bincount(ids, minlength=frames).cpu().numpy()
                )
    return result


def dedup_ranks(global_counts, tile_counts, lambda_tile_tenths, top_k):
    """Greedy DeDup; lambda_global=1; temporal-distance/index tie breaks."""
    global_counts = np.asarray(global_counts, dtype=np.int64)
    tile_counts = np.asarray(tile_counts, dtype=np.int64)
    frames = global_counts.shape[0]
    if global_counts.shape != (frames, frames):
        raise ValueError("global counts must be [F,F]")
    if tile_counts.shape != (frames, 4, frames):
        raise ValueError("tile counts must be [F,4,F]")
    if not np.array_equal(tile_counts.sum(axis=1), global_counts):
        raise ValueError("tile votes do not sum to global votes")
    if lambda_tile_tenths not in (3, 5, 7):
        raise ValueError("supported lambda_tile values are 0.3, 0.5, 0.7")
    selected = np.zeros((frames, frames), dtype=bool)
    covered_tiles = np.zeros((frames, 4), dtype=np.int64)
    ranks = np.empty((frames, min(int(top_k), frames)), dtype=np.int64)
    rows = np.arange(frames)
    distance = np.abs(rows[:, None] - rows[None, :])
    for position in range(ranks.shape[1]):
        marginal = np.maximum(tile_counts - covered_tiles[:, :, None], 0)
        score = 10 * global_counts + lambda_tile_tenths * marginal.sum(axis=1)
        score[selected] = -1
        best_score = score.max(axis=1)
        tie_distance = np.where(score == best_score[:, None], distance, frames + 1)
        chosen = tie_distance.argmin(axis=1)
        ranks[:, position] = chosen
        selected[rows, chosen] = True
        covered_tiles = np.maximum(covered_tiles, tile_counts[rows, :, chosen])
    return ranks
