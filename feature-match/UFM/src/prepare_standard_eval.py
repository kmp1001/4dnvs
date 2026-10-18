"""Prepare the fixed ten-scene, four-direction monocular evaluation dataset."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import click
import yaml

UFM = Path(__file__).resolve().parents[1]
ROOT = UFM.parents[1]
OUTPUT = ROOT / "data/dh_train/nuscenes_standard_eval10_4dir_512_v1"
DIRECTIONS = ("D2M", "L2M", "R2M", "U2M")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def runs(indices):
    result = []
    for i in sorted(indices):
        if not result or i != result[-1][-1] + 1:
            result.append([])
        result[-1].append(i)
    return result


def eligible(root, scene, include_current, directions=DIRECTIONS):
    per_direction = []
    for direction in directions:
        frames = set()
        for path in (root / "top20_reference" / f"{direction}_{scene}").glob("*.json"):
            index = int(path.stem)
            ranking = json.loads(path.read_text())
            references = {
                item["source_path"]
                for item in ranking["reference_frames"]
                if item["visible_pixel_count"] > 1000
                and (include_current or int(Path(item["source_path"]).stem) != index)
            }
            if len(references) >= 4:
                frames.add(index)
        per_direction.append(frames)
    return set.intersection(*per_direction)


@click.group()
def main():
    pass


@main.command()
@click.option("--candidates", default=16, type=int)
def prepare(candidates):
    """Write explicit per-scene 1 m configs, preserving existing render settings."""
    plan = json.loads((OUTPUT / "double_candidates.json").read_text())
    mapping = json.loads((OUTPUT / "scene_indices.json").read_text())
    tasks = []
    for row in plan["candidates"][:candidates]:
        scene = row["scene"]
        configs = {}
        for mode, source in [
            ("shift_1m", "monocular_shift/nuscene.yaml"),
            ("double_reprojection", "double_reprojection/nuscene_smoke.yaml"),
        ]:
            config = yaml.safe_load((UFM / "src/config" / source).read_text())
            config["paths"]["temp_path"] = str(ROOT)
            config["paths"]["nusc_path"] = "/fs/gamma-datasets/nuscenes"
            config["paths"]["output_dir"] = str(OUTPUT / mode)
            config["models"]["da3_path"] = str(
                ROOT / "weights/DA3NESTED-GIANT-LARGE-1.1"
            )
            config["models"]["sam3_checkpoint_path"] = str(
                ROOT / "weights/sam3/hub/models--facebook--sam3/snapshots/"
                "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
            )
            config["models"]["sam3_bpe_path"] = str(
                UFM / "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            )
            config["runtime"]["gpu"] = 1
            config["dataset"].update(
                split_types=["train", "val"],
                shuffle_filter=False,
                shuffle_dataloader=False,
                target_steps=100,
                frame_sampling=1,
                image_size=[512, 288],
                render_directions=list(DIRECTIONS),
                batch_indices=[mapping[scene]],
            )
            config["monocular_shift"]["distance_m"] = 1.0 if mode == "shift_1m" else 2.0
            config["double_reprojection"] = {"enabled": mode == "double_reprojection"}
            path = OUTPUT / "configs" / f"{scene}_{mode}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            configs[mode] = str(path)
        tasks.append({**row, "configs": configs})
    write_json(OUTPUT / "preprocess_tasks.json", tasks)
    print(f"Prepared {len(tasks)} candidate scenes", flush=True)


@main.command()
@click.option("--rank", required=True, type=int)
@click.option("--workers", default=16, type=int)
def worker(rank, workers):
    """Generate 1 m RGB products and camera-matched controls on one GPU."""
    from preprocess import run_pipeline

    tasks = json.loads((OUTPUT / "preprocess_tasks.json").read_text())
    for task in tasks[rank::workers]:
        scene = task["scene"]
        done = OUTPUT / "progress" / f"{scene}.json"
        if done.exists():
            continue
        print(f"START {scene}", flush=True)
        config = yaml.safe_load(Path(task["configs"]["shift_1m"]).read_text())
        run_pipeline(config)
        for mode in ("shift_1m", "double_reprojection"):
            subprocess.run(
                [
                    sys.executable,
                    str(UFM / "src/preprocess_cosmos_render.py"),
                    "--config",
                    task["configs"][mode],
                    "--log-names",
                    scene,
                    "--scene-index",
                    "0",
                    "--frames",
                    "100",
                    "--frames-only",
                    "--no-rgb-boxes",
                    "--save-boxes",
                    "--output-dir",
                    str(OUTPUT / mode / "conditions"),
                ],
                cwd=UFM,
                check=True,
            )
        common = eligible(ROOT / "data/dh_train/smoke_nuscenes_dr2m", scene, False)
        common &= eligible(OUTPUT / "shift_1m", scene, True)
        longest = max(runs(common), key=len, default=[])
        write_json(
            done, {"scene": scene, "common_count": len(common), "longest_run": longest}
        )
        print(f"COMPLETE {scene}: joint consecutive frames={len(longest)}", flush=True)


@main.command()
def finalize():
    """Select ten scenes and verify 2000 camera-matched samples in each mode."""
    from PIL import Image
    import numpy as np

    sys.path.insert(0, str(ROOT / "gen-video/harmonizer/training_script"))
    from create_dataset import build_sample

    candidates = json.loads((OUTPUT / "double_candidates.json").read_text())
    excluded = set(candidates["excluded_training_scenes"])
    selected = []
    tasks = json.loads((OUTPUT / "preprocess_tasks.json").read_text())
    for task in tasks:
        scene = task["scene"]
        common = eligible(ROOT / "data/dh_train/smoke_nuscenes_dr2m", scene, False)
        common &= eligible(OUTPUT / "shift_1m", scene, True)
        longest = max(runs(common), key=len, default=[])
        if len(longest) >= 50:
            assert (OUTPUT / "progress" / f"{scene}.json").is_file(), (
                f"Controls incomplete: {scene}"
            )
            selected.append({"scene": scene, "frames": longest[:50]})
        if len(selected) == 10:
            break
    assert len(selected) == 10, (
        f"Need ten jointly eligible scenes, found {len(selected)}"
    )
    assert not ({row["scene"] for row in selected} & excluded)
    for name in (
        "dataset_fixed_train100_4dir_seed42.json",
        "dataset_map_box_train100_4dir_ordered.json",
    ):
        training = json.loads(
            (ROOT / "data/dh_train/smoke_nuscenes_dr2m" / name).read_text()
        )["train"]
        training_scenes = {entry["scene"].split("_")[-1] for entry in training.values()}
        assert not ({row["scene"] for row in selected} & training_scenes), name
    for row in selected:
        for index in row["frames"]:
            relative = Path(f"D2M_{row['scene']}") / f"{index:05d}.png"
            source = ROOT / "data/dh_train/smoke_nuscenes_dr2m/front_source" / relative
            shifted_source = OUTPUT / "shift_1m/front_source" / relative
            with Image.open(source) as first, Image.open(shifted_source) as second:
                assert np.array_equal(np.asarray(first), np.asarray(second)), (
                    row["scene"],
                    index,
                )
    sizes = {}
    for mode in ("double_reprojection", "shift_1m"):
        data_root = (
            ROOT / "data/dh_train/smoke_nuscenes_dr2m"
            if mode == "double_reprojection"
            else OUTPUT / mode
        )
        controls = OUTPUT / mode / "conditions"
        source_controls = OUTPUT / "double_reprojection/conditions"
        records = {}
        for row in selected:
            scene = row["scene"]
            boxes_folder = next((controls / "videos").glob(f"*/*{scene}_boxes"))
            metadata = json.loads(
                (controls / "videos" / scene / "metadata.json").read_text()
            )
            offsets = (
                {"original": [0, 0, 0]}
                if mode == "double_reprojection"
                else {
                    "D2M": [0, 1, 0],
                    "L2M": [-1, 0, 0],
                    "R2M": [1, 0, 0],
                    "U2M": [0, -1, 0],
                }
            )
            frames = [
                frame for chunk in metadata["chunks"] for frame in chunk["frames"]
            ]
            for index in row["frames"]:
                source_pose = np.asarray(frames[index]["source_camera_to_world"])
                for view, offset in offsets.items():
                    target_pose = np.asarray(
                        frames[index]["target_camera_to_world"][view]
                    )
                    delta = source_pose[:3, :3].T @ (
                        target_pose[:3, 3] - source_pose[:3, 3]
                    )
                    assert np.allclose(delta, offset, atol=1e-5), (scene, index, view)
            for direction in DIRECTIONS:
                sequence = f"{direction}_{scene}"
                view = (
                    f"original_{scene}" if mode == "double_reprojection" else sequence
                )
                for index in row["frames"]:
                    relative = Path(sequence) / f"{index:05d}.png"
                    entry = build_sample(
                        root=data_root,
                        front_warped_dir=data_root / "front_warped",
                        warped_path=data_root / "front_warped" / relative,
                        frame_index=index,
                        top_reference_dir=data_root / "top20_reference",
                        num_reference_frames=4,
                        min_reference_visibility=1000,
                        include_current_reference=mode == "shift_1m",
                        prompt="remove degradation",
                        include_seginstance=False,
                        include_confident_coverage=True,
                        strict=True,
                    )
                    assert entry is not None
                    entry.update(
                        source_calibration=str(
                            data_root
                            / "calibration"
                            / sequence
                            / "source_calibration.txt"
                        ),
                        target_calibration=str(
                            data_root
                            / "calibration"
                            / sequence
                            / "target_calibration.txt"
                        ),
                        hdmap_image=str(controls / "hdmap" / view / relative.name),
                        world_scenario_image=str(
                            controls / "world_scenario" / view / relative.name
                        ),
                        map_box_metadata=str(
                            controls / "videos" / scene / "metadata.json"
                        ),
                        boxes_json=str(boxes_folder / f"{index:06d}.json"),
                        reference_world_scenario_images=[
                            str(
                                source_controls
                                / "world_scenario"
                                / f"original_{scene}"
                                / Path(ref).name
                            )
                            for ref in entry["reference_images"]
                        ],
                    )
                    for field in (
                        "image",
                        "target_image",
                        "confident_coverage",
                        "world_scenario_image",
                        "hdmap_image",
                    ):
                        path = entry[field]
                        if path not in sizes:
                            with Image.open(path) as image:
                                sizes[path] = image.size
                        assert sizes[path] == (512, 288), (path, sizes[path])
                    assert Path(entry["boxes_json"]).is_file()
                    for ref in [
                        *entry["reference_images"],
                        *entry["reference_world_scenario_images"],
                    ]:
                        if ref not in sizes:
                            with Image.open(ref) as image:
                                sizes[ref] = image.size
                        assert sizes[ref] == (512, 288), (ref, sizes[ref])
                    records[f"{sequence}/{index:05d}"] = entry
        assert len(records) == 2000
        write_json(
            OUTPUT / mode / "dataset_test.json",
            {
                "test": records,
                "selection": {
                    "benchmark": OUTPUT.name,
                    "mode": mode,
                    "scenes": selected,
                    "directions": list(DIRECTIONS),
                    "references": 4,
                    "min_reference_visibility_exclusive": 1000,
                    "include_current_reference": mode == "shift_1m",
                    "consecutive_frames": 50,
                    "image_size_wh": [512, 288],
                    "reference_protocol": "paired_original_view_gt"
                    if mode == "double_reprojection"
                    else "real_source_distribution_not_paired_novel_view_gt",
                },
            },
        )
    write_json(
        OUTPUT / "selection.json",
        {
            "scenes": selected,
            "excluded_training_scenes": sorted(excluded),
            "training_manifests": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in [
                    ROOT
                    / "data/dh_train/smoke_nuscenes_dr2m/dataset_fixed_train100_4dir_seed42.json",
                    ROOT
                    / "data/dh_train/smoke_nuscenes_dr2m/dataset_map_box_train100_4dir_ordered.json",
                ]
            ],
            "renderer_sha256": hashlib.sha256(
                (UFM / "src/preprocess_cosmos_render.py").read_bytes()
            ).hexdigest(),
            "modes": ["double_reprojection", "shift_1m"],
        },
    )
    print(
        "FINALIZED: same ten scenes × four directions × fifty consecutive frames in both modes",
        flush=True,
    )


if __name__ == "__main__":
    main()
