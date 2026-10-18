#!/usr/bin/env python3
"""Select physical scenes with all four render directions and four held-out validation scenes."""

import json
import random
from collections import Counter
from pathlib import Path

import click
from create_fixed_meanflow_subsets import build_training_from_folders, build_validation_from_scene, nuscenes_scene

DIRECTIONS = ["D2M", "L2M", "R2M", "U2M"]


@click.command()
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--train-scenes", type=click.IntRange(min=1), default=100)
@click.option(
    "--val-frames", type=click.IntRange(min=4), default=80, help="Total targets, divided equally among four scenes."
)
@click.option("--seed", type=int, default=42)
def main(manifest, train_scenes, val_frames, seed):
    if val_frames % 4:
        raise ValueError("Validation target count must be divisible by four")
    manifest = manifest.resolve()
    root = manifest.parent
    training_path = root / f"dataset_fixed_train{train_scenes}_4dir_seed{seed}.json"
    validation_path = root / f"dataset_fixed_val{val_frames}_4scene_4dir_seed{seed}.json"
    if training_path.exists() or validation_path.exists():
        raise FileExistsError("The selected output manifests already exist")
    source = json.loads(manifest.read_text())
    references = len(next(iter(source["train"].values()))["reference_images"])
    held_out = {nuscenes_scene(entry) for entry in source["train"].values()}
    rng = random.Random(seed)
    validation = {}
    for direction in DIRECTIONS:
        scenes = sorted(path.name for path in (root / "front_warped").glob(f"{direction}_scene-*") if path.is_dir())
        rng.shuffle(scenes)
        for scene in scenes:
            physical_scene = nuscenes_scene({"scene": scene})
            if physical_scene in held_out:
                continue
            selected = build_validation_from_scene(
                root, scene, references=references, val_frames=val_frames // 4, seed=seed
            )
            if selected is not None:
                validation.update(selected)
                held_out.add(physical_scene)
                print(f"Validation {scene}: {len(selected)} consecutive targets", flush=True)
                break
        else:
            raise ValueError(f"No held-out {direction} scene has a qualified validation window")
    training, selected_scenes = build_training_from_folders(
        root,
        source,
        validation,
        train_scenes=train_scenes,
        seed=seed,
        variants=DIRECTIONS,
        match_direction_frames=True,
    )
    metadata = {
        "source_manifest": str(manifest),
        "seed": seed,
        "directions": DIRECTIONS,
        "reference_images": references,
        "min_reference_visibility_exclusive": 1000,
        "include_current_reference": False,
        "match_direction_frames": True,
    }
    # The original baseline reads both splits from one manifest. MeanFlow uses
    # the identical test records from the separate validation manifest.
    training_path.write_text(
        json.dumps(
            {
                "train": training,
                "test": validation,
                "selection": {**metadata, "scene_ids": selected_scenes},
            },
            indent=2,
        )
    )
    validation_path.write_text(
        json.dumps(
            {
                "test": validation,
                "selection": {
                    **metadata,
                    "scene_ids": sorted({nuscenes_scene(e) for e in validation.values()}),
                    "sample_keys": list(validation),
                },
            },
            indent=2,
        )
    )
    counts = Counter(entry["scene"] for entry in training.values())
    print(f"Training: {len(selected_scenes)} scenes, {len(training)} samples: {training_path}")
    print(f"Qualified frames per scene/direction: min={min(counts.values())}, max={max(counts.values())}")
    print(f"Validation: {len(validation)} samples from four distinct scenes: {validation_path}")
    print(f"16 GPUs, batch size 1: {(len(training) + 15) // 16} steps per epoch")


if __name__ == "__main__":
    main()
