#!/usr/bin/env python3
"""Create a DriveNVS manifest from visibility-ranked source references.

For every warped frame, the script reads the matching ranking file:

    top20_reference/<scene>/<frame>.json

and selects the highest-ranked source frames whose visibility exceeds the
configured threshold. A sample is emitted only when enough qualifying source
frames are available. The current source image is excluded from
``reference_images`` by default because it is already stored in
``source_image``. Use ``--include-current-reference`` to include it.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    # Keep dataset creation usable in lightweight preprocessing environments.
    def tqdm(iterable, **_kwargs):
        return iterable


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_sort_key(path: Path) -> Tuple[object, ...]:
    """Sort names containing numbers in frame order (2 before 10)."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.as_posix()))


def list_images_recursive(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {folder}")

    images = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMG_EXTS]
    return sorted(
        images,
        key=lambda path: natural_sort_key(path.relative_to(folder)),
    )


def get_scene_name(relative_path: Path) -> str:
    """Return the first directory below ``front_warped`` as the scene ID."""
    if len(relative_path.parts) < 2:
        raise ValueError(f"Expected every image to be inside a scene directory, got: {relative_path}")
    return relative_path.parts[0]


def group_frames_by_scene(
    warped_images: Iterable[Path],
    front_warped_dir: Path,
) -> Dict[str, List[Path]]:
    scenes: Dict[str, List[Path]] = {}
    for warped_path in warped_images:
        relative_path = warped_path.relative_to(front_warped_dir)
        scene = get_scene_name(relative_path)
        scenes.setdefault(scene, []).append(warped_path)

    for frames in scenes.values():
        frames.sort(key=lambda path: natural_sort_key(path.relative_to(front_warped_dir)))
    return dict(sorted(scenes.items()))


def require_file(path: Path, field_name: str, warped_path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {field_name}.\nWarped image: {warped_path}\nExpected file: {path}")
    return path


def add_optional_file(
    sample: dict,
    field_name: str,
    path: Path,
    warped_path: Path,
    strict: bool,
) -> None:
    if path.is_file():
        sample[field_name] = str(path.resolve())
    elif strict:
        raise FileNotFoundError(
            f"Missing optional {field_name} requested with --strict.\n"
            f"Warped image: {warped_path}\n"
            f"Expected file: {path}"
        )


def resolve_reference_source_path(root: Path, value: str) -> Path:
    """Resolve a source path stored in top20_reference JSON."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def select_ranked_reference_paths(
    root: Path,
    top_reference_dir: Path,
    relative_path: Path,
    current_source_path: Path,
    warped_path: Path,
    num_reference_frames: int,
    min_reference_visibility: float,
    include_current_reference: bool,
) -> Optional[Tuple[List[str], Path]]:
    """Select the highest-ranked unique source paths for one warped frame."""
    ranking_path = top_reference_dir / relative_path.with_suffix(".json")
    require_file(ranking_path, "top-reference ranking JSON", warped_path)

    with ranking_path.open("r", encoding="utf-8") as file:
        ranking = json.load(file)

    entries = ranking.get("reference_frames")
    if not isinstance(entries, list):
        raise ValueError(f"Expected 'reference_frames' to be a list in: {ranking_path}")

    current_source_resolved = current_source_path.resolve()
    selected: List[str] = []
    selected_set: Set[str] = set()

    # The preprocessing pipeline stores reference_frames in highest-first order.
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("source_path"):
            raise ValueError(f"Invalid reference entry at index {entry_index} in: {ranking_path}")

        visibility = entry.get("visible_pixel_count")
        if not isinstance(visibility, (int, float)) or isinstance(visibility, bool):
            raise ValueError(f"Missing or non-numeric 'visibility' at reference entry {entry_index} in: {ranking_path}")

        # The condition is intentionally strict: visibility must be > the
        # threshold, so a visibility exactly equal to 1000 does not qualify.
        if visibility <= min_reference_visibility:
            continue

        candidate = resolve_reference_source_path(root, entry["source_path"])
        require_file(
            candidate,
            f"ranked reference source at ranking index {entry_index}",
            warped_path,
        )

        if not include_current_reference and candidate == current_source_resolved:
            continue

        candidate_str = str(candidate)
        if candidate_str in selected_set:
            continue

        selected.append(candidate_str)
        selected_set.add(candidate_str)
        if len(selected) == num_reference_frames:
            break

    if len(selected) < num_reference_frames:
        return None

    return selected, ranking_path.resolve()


def build_sample(
    root: Path,
    front_warped_dir: Path,
    top_reference_dir: Path,
    warped_path: Path,
    frame_index: int,
    prompt: str,
    num_reference_frames: int,
    min_reference_visibility: float,
    include_current_reference: bool,
    include_seginstance: bool,
    include_confident_coverage: bool,
    strict: bool,
) -> Optional[dict]:
    """Build one sample using visibility-ranked source references."""
    relative_path = warped_path.relative_to(front_warped_dir)

    source_path = require_file(
        root / "front_source" / relative_path,
        "source_image",
        warped_path,
    )
    try:
        target_path = require_file(
            root / "front_target" / relative_path,
            "target_image",
            warped_path,
        )
    except:
        target_path = require_file(
            root / "front_source" / relative_path,
            "source_image",
            warped_path,
        )

    reference_selection = select_ranked_reference_paths(
        root=root,
        top_reference_dir=top_reference_dir,
        relative_path=relative_path,
        current_source_path=source_path,
        warped_path=warped_path,
        num_reference_frames=num_reference_frames,
        min_reference_visibility=min_reference_visibility,
        include_current_reference=include_current_reference,
    )
    if reference_selection is None:
        return None
    reference_paths, ranking_path = reference_selection

    sample = {
        "image": str(warped_path.resolve()),
        "source_image": str(source_path.resolve()),
        "target_image": str(target_path.resolve()),
        "reference_images": reference_paths,
        "reference_ranking": str(ranking_path),
        "prompt": prompt,
        "scene": get_scene_name(relative_path),
        "frame_name": relative_path.name,
        "relative_path": relative_path.as_posix(),
        "frame_index": frame_index,
    }

    if include_seginstance:
        add_optional_file(
            sample,
            "seginstance_warped",
            root / "seginstance_warped" / relative_path,
            warped_path,
            strict,
        )

    if include_confident_coverage:
        add_optional_file(
            sample,
            "confident_coverage",
            root / "confident_coverage" / relative_path,
            warped_path,
            strict,
        )

    return sample


def add_sample(split_data: Dict[str, dict], sample: dict) -> None:
    split_data[f"{len(split_data):06d}"] = sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train_drivenvs.py manifest using the highest-ranked source frames from top20_reference JSON files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/project/data/dh_train/trainval_nuscenes_dr2m_final"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to ROOT/dataset.json.",
    )
    parser.add_argument("--prompt", default="remove degradation")
    parser.add_argument(
        "--num-reference-frames",
        type=int,
        default=8,
        help="Number of highest-ranked source frames selected per sample.",
    )
    parser.add_argument(
        "--min-reference-visibility",
        type=float,
        default=1000,
        help=(
            "Select only references with visibility strictly greater than "
            "this value. Samples without enough qualifying references are "
            "skipped."
        ),
    )
    parser.add_argument(
        "--top-reference-subdir",
        default="top20_reference",
        help="Ranking directory relative to ROOT.",
    )
    parser.add_argument(
        "--include-current-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow the current source_image to also appear in reference_images. "
            "By default it is skipped and the next-ranked frame is selected."
        ),
    )
    parser.add_argument(
        "--test-first-k-scenes",
        type=int,
        default=10,
        help="Use the first k naturally sorted scenes for validation.",
    )
    parser.add_argument(
        "--train-use-all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include validation scenes in train as well. Use --no-train-use-all for scene-disjoint train/test splits."
        ),
    )
    parser.add_argument(
        "--include-seginstance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-confident-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if a requested optional file is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = args.root.expanduser().resolve()
    output_json = args.output.expanduser().resolve() if args.output is not None else root / "dataset.json"
    front_warped_dir = root / "front_warped"
    top_reference_dir = root / args.top_reference_subdir

    if args.num_reference_frames <= 0:
        raise ValueError("--num-reference-frames must be > 0")
    if args.test_first_k_scenes < 0:
        raise ValueError("--test-first-k-scenes must be >= 0")
    if not root.is_dir():
        raise NotADirectoryError(f"Root does not exist: {root}")
    if not front_warped_dir.is_dir():
        raise NotADirectoryError(f"Missing required directory: {front_warped_dir}")
    if not top_reference_dir.is_dir():
        raise NotADirectoryError(f"Missing required directory: {top_reference_dir}")

    warped_images = list_images_recursive(front_warped_dir)
    if not warped_images:
        raise RuntimeError(f"No warped images found under: {front_warped_dir}")

    scenes = group_frames_by_scene(warped_images, front_warped_dir)
    all_scene_names = list(scenes)
    k = min(args.test_first_k_scenes, len(all_scene_names))
    test_scenes: Set[str] = set(all_scene_names[:k])

    train_data: Dict[str, dict] = {}
    test_data: Dict[str, dict] = {}
    skipped_samples = 0

    for scene, scene_frames in tqdm(scenes.items(), desc="Scenes"):
        # Unlike fixed past/future offsets, visibility-ranked references allow
        # boundary frames to be emitted as long as their ranking has enough paths.
        for frame_index, warped_path in enumerate(scene_frames):
            sample = build_sample(
                root=root,
                front_warped_dir=front_warped_dir,
                top_reference_dir=top_reference_dir,
                warped_path=warped_path,
                frame_index=frame_index,
                prompt=args.prompt,
                num_reference_frames=args.num_reference_frames,
                min_reference_visibility=args.min_reference_visibility,
                include_current_reference=args.include_current_reference,
                include_seginstance=args.include_seginstance,
                include_confident_coverage=args.include_confident_coverage,
                strict=args.strict,
            )
            if sample is None:
                skipped_samples += 1
                continue

            if scene in test_scenes:
                add_sample(test_data, sample)
            if args.train_use_all or scene not in test_scenes:
                add_sample(train_data, sample)

    dataset = {"train": train_data, "test": test_data}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_json}")
    print(f"Total scenes: {len(all_scene_names)}")
    print(f"Reference frames per sample: {args.num_reference_frames}")
    print(f"Minimum reference visibility (exclusive): {args.min_reference_visibility}")
    print(f"Current source included in references: {args.include_current_reference}")
    print(f"Test scenes ({k}):")
    for scene in all_scene_names[:k]:
        print(f"  - {scene}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Skipped samples with insufficient valid references: {skipped_samples}")


if __name__ == "__main__":
    main()
