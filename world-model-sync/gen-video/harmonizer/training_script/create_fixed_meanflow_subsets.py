#!/usr/bin/env python3
"""Save fixed scene-level training and chronological frame-level validation subsets."""

import json
import random
import re
import runpy
from collections import defaultdict
from pathlib import Path

import click


def nuscenes_scene(entry):
    # Keep R2M/U2M variants of the same physical scene in the same partition.
    return re.search(r"scene-\d+", entry["scene"]).group()


def select_subsets(manifest, *, train_scenes, val_frames, seed):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from utils.temporal import TEMPORAL_HISTORY_FRAMES, temporal_frame_index, temporal_sequence_key

    rng = random.Random(seed)
    scenes = sorted({nuscenes_scene(entry) for entry in manifest["train"].values()})
    if len(scenes) < train_scenes:
        raise ValueError(f"Requested {train_scenes} training scenes, but the manifest has only {len(scenes)}")
    selected_scenes = sorted(rng.sample(scenes, train_scenes))
    training = {key: entry for key, entry in manifest["train"].items() if nuscenes_scene(entry) in selected_scenes}
    sequences = defaultdict(dict)
    training_variants = {entry["scene"].split("_scene-")[0] for entry in training.values()}
    for key, entry in manifest["test"].items():
        frame = temporal_frame_index(entry, key)
        if (
            nuscenes_scene(entry) not in selected_scenes
            and entry["scene"].split("_scene-")[0] in training_variants
            and frame is not None
        ):
            sequence = temporal_sequence_key(entry, key)
            if frame in sequences[sequence]:
                raise ValueError(f"Duplicate frame {frame} in validation sequence {sequence}")
            sequences[sequence][frame] = (key, entry)
    sequence_names = sorted(sequences)
    rng.shuffle(sequence_names)
    for sequence in sequence_names:
        frames = sequences[sequence]
        starts = [
            start
            for start in sorted(frames)
            if all(frame in frames for frame in range(start - TEMPORAL_HISTORY_FRAMES, start + val_frames))
        ]
        if not starts:
            continue
        start = rng.choice(starts)
        validation = {}
        for frame in range(start, start + val_frames):
            key, original = frames[frame]
            entry = dict(original)
            entry["temporal_history_images"] = [
                frames[frame - offset][1]["target_image"] for offset in range(1, TEMPORAL_HISTORY_FRAMES + 1)
            ]
            validation[key] = entry
        return training, validation, selected_scenes
    raise ValueError(f"No held-out sequence has {val_frames} targets plus four consecutive preceding GT frames")


def build_validation_from_scene(root, scene, *, references, val_frames, seed):
    """Select fixed qualified targets and real GT predecessors from one render sequence."""
    build_sample = runpy.run_path(str(Path(__file__).with_name("create_dataset.py")))["build_sample"]
    samples = {}
    for image in sorted((root / "front_warped" / scene).glob("*.png")):
        frame = int(image.stem)
        sample = build_sample(
            root,
            root / "front_warped",
            root / "top20_reference",
            image,
            frame,
            "remove degradation",
            references,
            1000,
            False,
            False,
            True,
            True,
        )
        if sample is not None:
            samples[frame] = sample
    if not samples:
        return None
    target_dir = Path(next(iter(samples.values()))["target_image"]).parent
    starts = [
        start
        for start in sorted(samples)
        if start >= 4
        and all(frame in samples for frame in range(start, start + val_frames))
        and all((target_dir / f"{frame:05d}.png").is_file() for frame in range(start - 4, start))
    ]
    if not starts:
        return None
    start = random.Random(seed).choice(starts)
    validation = {}
    for frame in range(start, start + val_frames):
        entry = dict(samples[frame])
        entry["temporal_history_images"] = [str(target_dir / f"{frame - offset:05d}.png") for offset in range(1, 5)]
        validation[f"{scene}/{frame:05d}"] = entry
    return validation


def build_training_from_folders(
    root, source, validation, *, train_scenes, seed, variants=None, match_direction_frames=False
):
    """Use the existing visibility-ranked dataset builder on fixed scene folders."""
    build_sample = runpy.run_path(str(Path(__file__).with_name("create_dataset.py")))["build_sample"]
    if variants is None:
        variants = sorted({entry["scene"].split("_scene-")[0] for entry in source["train"].values()})
    held_out = {nuscenes_scene(entry) for entry in validation.values()}
    first_variant = variants[0]
    scenes = sorted(
        {
            nuscenes_scene({"scene": path.name})
            for path in (root / "front_warped").glob(f"{first_variant}_scene-*")
            if path.is_dir() and nuscenes_scene({"scene": path.name}) not in held_out
        }
    )
    if len(scenes) < train_scenes:
        raise ValueError(f"Requested {train_scenes} training scenes, but the folders have only {len(scenes)}")
    random.Random(seed).shuffle(scenes)
    selected_scenes = []
    references = len(next(iter(source["train"].values()))["reference_images"])
    training = {}
    for scene_id in scenes:
        scene_samples = {}
        direction_frames = []
        eligible = True
        for variant in variants:
            scene = f"{variant}_{scene_id}"
            accepted_frames = set()
            for image in sorted((root / "front_warped" / scene).glob("*.png")):
                frame = int(image.stem)
                sample = build_sample(
                    root,
                    root / "front_warped",
                    root / "top20_reference",
                    image,
                    frame,
                    "remove degradation",
                    references,
                    1000,
                    False,
                    False,
                    True,
                    True,
                )
                if sample is not None:
                    scene_samples[f"{scene}/{image.stem}"] = sample
                    accepted_frames.add(frame)
            if not any(all(frame - offset in accepted_frames for offset in range(5)) for frame in accepted_frames):
                print(f"Excluded {scene}: no five-frame window under the reference-selection rule", flush=True)
                eligible = False
                break
            direction_frames.append(accepted_frames)
        if eligible and match_direction_frames:
            common_frames = set.intersection(*direction_frames)
            if not any(all(frame - offset in common_frames for offset in range(5)) for frame in common_frames):
                print(f"Excluded {scene_id}: no five-frame window shared by all directions", flush=True)
                eligible = False
            scene_samples = {
                key: sample for key, sample in scene_samples.items() if sample["frame_index"] in common_frames
            }
        if eligible:
            selected_scenes.append(scene_id)
            training.update(scene_samples)
            if len(selected_scenes) % 20 == 0:
                print(f"Built {len(selected_scenes)}/{train_scenes} scenes, {len(training)} frame records", flush=True)
            if len(selected_scenes) == train_scenes:
                break
    if len(selected_scenes) != train_scenes:
        raise ValueError(f"Only {len(selected_scenes)} scenes have five-frame windows; requested {train_scenes}")
    return training, sorted(selected_scenes)


@click.command()
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--train-scenes", type=click.IntRange(min=1), default=200)
@click.option("--val-frames", type=click.IntRange(min=1), default=20)
@click.option("--seed", type=int, default=42)
@click.option("--from-scene-folders", is_flag=True, help="Build training records from all available scene folders.")
def main(manifest, train_scenes, val_frames, seed, from_scene_folders):
    manifest = manifest.resolve()
    training_path = manifest.with_name(f"dataset_fixed_train{train_scenes}_seed{seed}.json")
    validation_path = manifest.with_name(f"dataset_fixed_val{val_frames}_seed{seed}.json")
    if training_path.exists():
        raise FileExistsError(f"Fixed training subset already exists; reuse it: {training_path}")
    source = json.loads(manifest.read_text())
    if from_scene_folders:
        if validation_path.exists():
            validation = json.loads(validation_path.read_text())["test"]
        else:
            variant = next(iter(source["train"].values()))["scene"].split("_scene-")[0]
            original_train_scenes = {nuscenes_scene(e) for e in source["train"].values()}
            scenes = sorted(
                path.name for path in (manifest.parent / "front_warped").glob(f"{variant}_scene-*") if path.is_dir()
            )
            random.Random(seed).shuffle(scenes)
            validation = None
            for scene in scenes:
                if nuscenes_scene({"scene": scene}) in original_train_scenes:
                    continue
                validation = build_validation_from_scene(
                    manifest.parent,
                    scene,
                    references=len(next(iter(source["train"].values()))["reference_images"]),
                    val_frames=val_frames,
                    seed=seed,
                )
                if validation:
                    break
            if validation is None:
                raise ValueError(f"No held-out {variant} scene has {val_frames} consecutive qualified targets")
        training, selected_scenes = build_training_from_folders(
            manifest.parent,
            source,
            validation,
            train_scenes=train_scenes,
            seed=seed,
        )
    else:
        if validation_path.exists():
            raise FileExistsError(f"Fixed validation subset already exists; reuse it: {validation_path}")
        training, validation, selected_scenes = select_subsets(
            source,
            train_scenes=train_scenes,
            val_frames=val_frames,
            seed=seed,
        )
    # Write beside the source manifest so relative media and calibration paths retain their meaning.
    metadata = {"source_manifest": str(manifest), "seed": seed}
    training_path.write_text(
        json.dumps(
            {
                "train": training,
                "selection": {**metadata, "scene_ids": selected_scenes},
            },
            indent=2,
        )
    )
    if not validation_path.exists():
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
    print(f"Training: {train_scenes} nuScenes scene IDs, {len(training)} frame records: {training_path}")
    print(f"Validation: {len(validation)} consecutive current frames: {validation_path}")


if __name__ == "__main__":
    main()
