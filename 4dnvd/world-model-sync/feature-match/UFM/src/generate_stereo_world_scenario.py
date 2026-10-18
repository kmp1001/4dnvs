"""Generate paired nuScenes stereo RGB and camera-matched World Scenario data.

Uses the four existing stereo YAMLs and the existing RGB/map renderers.
One worker owns both directions and all three camera maps for each scene.
"""

import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import yaml
from PIL import Image

UFM = Path(__file__).resolve().parents[1]
ROOT = UFM.parents[1]
DEFAULT_OUTPUT = ROOT / "data/dh_train/nuscenes_stereo_world_scenario_train100_test20"
CAMERAS = ("pcam_f0", "pcam_l0", "pcam_r0")
TARGET_CAMERA = {"LR": "pcam_r0", "RL": "pcam_l0"}
STANDARD_EVAL = ROOT / "data/dh_train/nuscenes_stereo_standard_eval10_2dir_512_v1"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def control_directory(output, split, camera, scene):
    return output / split / "conditions" / camera / "world_scenario" / f"original_{scene}"


@click.group()
def main():
    pass


@main.command()
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT)
@click.option("--train-scenes", type=int, default=100)
@click.option("--test-scenes", type=int, default=20)
@click.option("--seed", type=int, default=42)
def prepare(output, train_scenes, test_scenes, seed):
    """Select disjoint official train/val scenes and save effective configs."""
    from py123d.api import SceneFilter, get_filtered_scenes

    output = output.resolve()
    configs = {}
    provenance = {}
    for split in ("train", "test"):
        for direction in ("LR", "RL"):
            path = UFM / f"src/config/stereo/{split}_nuscene_{direction}.yaml"
            config = yaml.safe_load(path.read_text())
            provenance[f"{split}_{direction}"] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            config["paths"]["temp_path"] = str(ROOT)
            config["paths"]["nusc_path"] = "/fs/gamma-datasets/nuscenes"
            config["paths"]["output_dir"] = str(output / split)
            config["models"]["da3_path"] = str(ROOT / "weights/DA3NESTED-GIANT-LARGE-1.1")
            config["models"]["sam3_checkpoint_path"] = str(
                ROOT / "weights/sam3/hub/models--facebook--sam3/snapshots/"
                "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
            )
            config["models"]["sam3_bpe_path"] = str(UFM / "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
            config["dataset"]["split_types"] = ["train" if split == "train" else "val"]
            config["dataset"]["required_scene_modalities"] = [
                "ego_state_se3",
                *[f"camera.{c}" for c in CAMERAS],
            ]
            config["dataset"]["shuffle_filter"] = False
            config["dataset"]["shuffle_dataloader"] = False
            config["dataset"]["target_steps"] = 100
            config["runtime"]["gpu"] = 1
            configs[f"{split}_{direction}"] = config
    selected = {}
    for split, count in (("train", train_scenes), ("test", test_scenes)):
        native_split = "train" if split == "train" else "val"
        scenes = get_filtered_scenes(
            SceneFilter(
                datasets=["nuscenes-interpolated"],
                split_types=[native_split],
                required_scene_modalities=[
                    "ego_state_se3",
                    *[f"camera.{c}" for c in CAMERAS],
                ],
                shuffle=False,
            ),
            data_root=ROOT,
        )
        eligible = [
            i
            for i, scene in enumerate(scenes)
            if scene.number_of_iterations + scene.number_of_history_iterations >= 100
        ]
        indices = sorted(random.Random(seed).sample(eligible, count))
        selected[split] = [
            {
                "index": index,
                "scene": scenes[index].log_name,
                "native_split": native_split,
                "native_frames": scenes[index].number_of_iterations + scenes[index].number_of_history_iterations,
                "frame_limit": configs[f"{split}_LR"]["dataset"]["target_steps"],
            }
            for index in indices
        ]
        for direction in ("LR", "RL"):
            config = configs[f"{split}_{direction}"]
            config["dataset"]["batch_indices"] = indices
            path = output / "configs" / f"{split}_{direction}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
    assert not ({r["scene"] for r in selected["train"]} & {r["scene"] for r in selected["test"]})
    write_json(
        output / "selection.json",
        {
            "seed": seed,
            "source_configs": provenance,
            "splits": selected,
            "source_camera": "pcam_f0",
            "target_cameras": TARGET_CAMERA,
            "references": 4,
            "min_reference_visibility": 1000,
            "include_current_reference": False,
        },
    )
    print(f"Prepared {train_scenes} train and {test_scenes} held-out val scenes at {output}")


@main.command()
@click.option("--output", type=click.Path(exists=True, path_type=Path), default=DEFAULT_OUTPUT)
@click.option("--rank", type=int, default=lambda: int(os.environ["SLURM_PROCID"]))
@click.option("--workers", type=int, default=8)
@click.option(
    "--maps-only",
    is_flag=True,
    help="Reuse existing RGB products and generate their maps.",
)
@click.option(
    "--limit-scenes",
    type=int,
    default=0,
    help="Smoke check only; 0 processes the complete shard.",
)
def worker(output, rank, workers, maps_only, limit_scenes):
    """Generate both stereo directions and three-camera controls on one GPU."""
    if not maps_only:
        from preprocess import run_pipeline

    output = output.resolve()
    plan = json.loads((output / "selection.json").read_text())
    tasks = [(split, scene) for split, scenes in plan["splits"].items() for scene in scenes][rank::workers]
    if limit_scenes:
        tasks = tasks[:limit_scenes]
    progress_path = output / "progress" / f"worker_{rank}.json"
    completed = json.loads(progress_path.read_text()) if progress_path.exists() else []
    completed_names = {(item["split"], item["scene"]) for item in completed}
    for split, scene in tasks:
        if (split, scene["scene"]) in completed_names:
            continue
        # Do not invent extra frames after the real sequence ends.
        frame_count = min(scene["frame_limit"], scene["native_frames"])
        name = scene["scene"]
        for direction in ("LR", "RL"):
            config = yaml.safe_load((output / "configs" / f"{split}_{direction}.yaml").read_text())
            config["dataset"]["batch_indices"] = [scene["index"]]
            config["dataset"]["target_steps"] = frame_count
            effective = output / "configs" / f"worker{rank}_{split}_{direction}_{name}.yaml"
            effective.write_text(yaml.safe_dump(config, sort_keys=False))
            print(
                f"WORKER {rank}: {split} {direction} {name}, {frame_count} frames",
                flush=True,
            )
            if not maps_only:
                run_pipeline(config)
            sequence = f"{direction}_{name}"
            for category in (
                "front_source",
                "front_target",
                "front_warped",
                "confident_coverage",
                "top20_reference",
            ):
                extension = ".json" if category == "top20_reference" else ".png"
                count = len(list((output / split / category / sequence).glob(f"*{extension}")))
                assert count == frame_count, (
                    split,
                    sequence,
                    category,
                    count,
                    frame_count,
                )
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = []
            for camera in CAMERAS:
                futures.append(
                    pool.submit(
                        subprocess.run,
                        [
                            sys.executable,
                            str(UFM / "src/preprocess_cosmos_render.py"),
                            "--config",
                            str(output / "configs" / f"{split}_LR.yaml"),
                            "--splits",
                            scene["native_split"],
                            "--log-names",
                            name,
                            "--scene-index",
                            "0",
                            "--camera-id",
                            f"camera.{camera}",
                            "--frames",
                            str(frame_count),
                            "--frames-only",
                            "--no-rgb-boxes",
                            "--save-boxes",
                            "--output-dir",
                            str(output / split / "conditions" / camera),
                        ],
                        cwd=UFM,
                        check=True,
                    )
                )
            for future in futures:
                future.result()
        completed.append({**scene, "split": split, "frames": frame_count})
        write_json(progress_path, completed)
        print(f"SCENE_COMPLETE {split} {name}", flush=True)


@main.command()
@click.option("--output", type=click.Path(exists=True, path_type=Path), default=DEFAULT_OUTPUT)
@click.option(
    "--include-current-reference",
    is_flag=True,
    help="Allow the current source frame among the four visibility-ranked stereo references.",
)
def finalize(output, include_current_reference):
    """Verify all products and build paired, camera-matched train/test manifests."""
    output = output.resolve()
    plan = json.loads((output / "selection.json").read_text())
    helper_path = ROOT / "gen-video/harmonizer/training_script/create_dataset.py"
    spec = importlib.util.spec_from_file_location("stereo_manifest_builder", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    report = {}
    checked_images = set()
    for split, scenes in plan["splits"].items():
        data = {}
        counts = defaultdict(int)
        for record in scenes:
            name = record["scene"]
            frames = min(record["frame_limit"], record["native_frames"])
            for camera in CAMERAS:
                camera_root = output / split / "conditions" / camera
                metadata = json.loads((camera_root / "videos" / name / "metadata.json").read_text())
                assert metadata["source_camera"].lower() == camera
                assert len(metadata["chunks"][0]["frames"]) == frames
                assert all(frame["boxes_available"] for frame in metadata["chunks"][0]["frames"])
                for index in range(frames):
                    for category in ("world_scenario", "hdmap"):
                        image = camera_root / category / f"original_{name}" / f"{index:05d}.png"
                        with Image.open(image) as png:
                            assert png.size == (800, 450), image
                            png.verify()
                        checked_images.add(str(image))
            for direction, target_camera in TARGET_CAMERA.items():
                sequence = f"{direction}_{name}"
                for index in range(frames):
                    filename = f"{index:05d}.png"
                    base = output / split
                    for category in (
                        "front_source",
                        "front_target",
                        "front_warped",
                        "confident_coverage",
                    ):
                        assert (base / category / sequence / filename).is_file(), (
                            category,
                            sequence,
                            filename,
                        )
                    sample = helper.build_sample(
                        root=base,
                        front_warped_dir=base / "front_warped",
                        top_reference_dir=base / "top20_reference",
                        warped_path=base / "front_warped" / sequence / filename,
                        frame_index=index,
                        prompt="",
                        num_reference_frames=4,
                        min_reference_visibility=1000,
                        include_current_reference=include_current_reference,
                        include_seginstance=False,
                        include_confident_coverage=True,
                        strict=True,
                    )
                    if sample is None:
                        continue
                    source_controls = control_directory(output, split, "pcam_f0", name)
                    target_controls = control_directory(output, split, target_camera, name)
                    sample.update(
                        world_scenario_image=str(target_controls / filename),
                        source_world_scenario_image=str(source_controls / filename),
                        reference_world_scenario_images=[
                            str(source_controls / Path(ref).name) for ref in sample["reference_images"]
                        ],
                        hdmap_image=str(target_controls.parent.parent / "hdmap" / f"original_{name}" / filename),
                        source_camera="pcam_f0",
                        target_camera=target_camera,
                        boxes_json=str(
                            base
                            / "conditions"
                            / target_camera
                            / "videos"
                            / f"pinhole_{target_camera}"
                            / f"000000_{name}_boxes"
                            / f"{index:06d}.json"
                        ),
                        source_calibration=str(base / "calibration" / sequence / "source_calibration.txt"),
                        target_calibration=str(base / "calibration" / sequence / "target_calibration.txt"),
                        map_box_metadata=str(base / "conditions" / target_camera / "videos" / name / "metadata.json"),
                    )
                    sample["reference_visibility_scores"] = helper_reference_scores(sample)
                    assert sample["target_image"] == str((base / "front_target" / sequence / filename).resolve())
                    for field in (
                        "source_calibration",
                        "target_calibration",
                        "boxes_json",
                    ):
                        assert Path(sample[field]).is_file(), sample[field]
                    data[f"{len(data):06d}"] = sample
                    counts[sequence] += 1
            print(
                f"{split}: {name} processed; {len(data)} eligible frames so far",
                flush=True,
            )
        omitted_sequences = [
            f"{direction}_{record['scene']}"
            for record in scenes
            for direction in TARGET_CAMERA
            if f"{direction}_{record['scene']}" not in counts
        ]
        manifest = {
            "dataset_setup": "stereo",
            "ground_truth": "real_target_camera",
            split: data,
            "selection": {
                "scene_ids": [s["scene"] for s in scenes],
                "seed": plan["seed"],
                "include_current_reference": include_current_reference,
                "references": 4,
                "min_reference_visibility": 1000,
            },
        }
        write_json(output / f"dataset_{split}.json", manifest)
        report[split] = {
            "physical_scenes": len(scenes),
            "records": len(data),
            "per_sequence": dict(counts),
            "omitted_sequences": omitted_sequences,
        }
    report["verified_control_images"] = len(checked_images)
    write_json(output / "DATASET_AUDIT.json", report)
    print(json.dumps(report, indent=2))


def helper_reference_scores(sample):
    ranking = json.loads(Path(sample["reference_ranking"]).read_text())
    assert ranking["source_cam_id"] == sample["source_camera"]
    assert ranking["target_cam_id"] == sample["target_camera"]
    by_path = {Path(item["source_path"]).name: item["visible_pixel_count"] for item in ranking["reference_frames"]}
    return [by_path[Path(path).name] for path in sample["reference_images"]]


@main.command("regenerate-conditions")
@click.option("--output", type=click.Path(exists=True, path_type=Path), default=DEFAULT_OUTPUT)
@click.option("--rank", type=int, default=lambda: int(os.environ["SLURM_PROCID"]))
@click.option("--workers", type=int, default=4, help="Total node shards.")
@click.option("--parallel", type=int, default=16, help="CPU renderer processes on this node.")
def regenerate_conditions(output, rank, workers, parallel):
    """Overwrite ALL selected camera conditions with the current renderer."""
    output = output.resolve()
    renderer = UFM / "src/preprocess_cosmos_render.py"
    renderer_hash = hashlib.sha256(renderer.read_bytes()).hexdigest()
    plan = json.loads((output / "selection.json").read_text())
    tasks = [
        (split, scene, camera) for split, scenes in plan["splits"].items() for scene in scenes for camera in CAMERAS
    ][rank::workers]
    logs = output / "logs/conditions_regeneration"
    progress = output / "progress/conditions_regeneration"
    logs.mkdir(parents=True, exist_ok=True)
    progress.mkdir(parents=True, exist_ok=True)

    def render(task):
        split, scene, camera = task
        name = scene["scene"]
        frames = min(scene["frame_limit"], scene["native_frames"])
        destination = output / split / "conditions" / camera
        command = [
            sys.executable,
            str(renderer),
            "--config",
            str(output / "configs" / f"{split}_LR.yaml"),
            "--splits",
            scene["native_split"],
            "--log-names",
            name,
            "--scene-index",
            "0",
            "--camera-id",
            f"camera.{camera}",
            "--frames",
            str(frames),
            "--frames-only",
            "--no-rgb-boxes",
            "--save-boxes",
            "--no-scene-data",
            "--output-dir",
            str(destination),
        ]
        log_path = logs / f"{split}_{camera}_{name}.log"
        print(f"START {split} {camera} {name}", flush=True)
        with log_path.open("w") as log:
            subprocess.run(command, cwd=UFM, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert hashlib.sha256(renderer.read_bytes()).hexdigest() == renderer_hash, (
            "Renderer changed during regeneration"
        )
        metadata = json.loads((destination / "videos" / name / "metadata.json").read_text())
        assert metadata["source_camera"].lower() == camera
        assert metadata["render_views"][0]["name"] == "original"
        assert len(metadata["chunks"][0]["frames"]) == frames
        for category in ("hdmap", "world_scenario"):
            folder = destination / category / f"original_{name}"
            assert len(list(folder.glob("*.png"))) == frames, folder
        record = {
            "split": split,
            "scene": name,
            "camera": camera,
            "frames": frames,
            "renderer_sha256": renderer_hash,
            "log": str(log_path),
        }
        write_json(progress / f"{split}_{camera}_{name}.json", record)
        print(f"DONE {split} {camera} {name}: {frames} frames", flush=True)

    print(
        f"Shard {rank}/{workers}: regenerating {len(tasks)} camera-scenes using {parallel} CPU processes; renderer={renderer_hash}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(render, tasks))
    print(f"SHARD_COMPLETE {rank}: {len(tasks)} camera-scenes", flush=True)


@main.command("audit-conditions")
@click.option("--output", type=click.Path(exists=True, path_type=Path), default=DEFAULT_OUTPUT)
def audit_conditions(output):
    """Verify every regenerated condition PNG and recorded renderer version."""
    output = output.resolve()
    run = json.loads((output / "checks/conditions_regeneration_run.json").read_text())
    renderer_hash = hashlib.sha256((UFM / "src/preprocess_cosmos_render.py").read_bytes()).hexdigest()
    assert renderer_hash == run["renderer_sha256"]
    for path, expected in run["preserved_inputs"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    plan = json.loads((output / "selection.json").read_text())
    tasks = [
        (split, scene, camera) for split, scenes in plan["splits"].items() for scene in scenes for camera in CAMERAS
    ]

    def check(task):
        split, scene, camera = task
        name = scene["scene"]
        frames = min(scene["frame_limit"], scene["native_frames"])
        record = json.loads((output / "progress/conditions_regeneration" / f"{split}_{camera}_{name}.json").read_text())
        assert record["renderer_sha256"] == renderer_hash
        assert record["frames"] == frames
        root = output / split / "conditions" / camera
        metadata = json.loads((root / "videos" / name / "metadata.json").read_text())
        assert metadata["source_camera"].lower() == camera
        assert metadata["image_size_wh"] == [800, 450]
        assert metadata["render_views"][0]["name"] == "original"
        assert len(metadata["chunks"][0]["frames"]) == frames
        assert all(frame["boxes_available"] for frame in metadata["chunks"][0]["frames"])
        for category in ("hdmap", "world_scenario"):
            folder = root / category / f"original_{name}"
            assert {p.name for p in folder.glob("*.png")} == {f"{index:05d}.png" for index in range(frames)}
            for index in range(frames):
                with Image.open(folder / f"{index:05d}.png") as image:
                    assert image.size == (800, 450)
                    image.verify()
        return frames * 2

    with ThreadPoolExecutor(max_workers=16) as pool:
        counts = list(pool.map(check, tasks))
    run.update(
        status="complete",
        camera_scenes=len(tasks),
        verified_condition_pngs=sum(counts),
        preserved_inputs_verified=True,
        cameras_verified=True,
    )
    write_json(output / "checks/conditions_regeneration_audit.json", run)
    write_json(output / "checks/conditions_regeneration_run.json", run)
    print(
        f"Verified all {len(tasks)} camera-scenes and {sum(counts)} condition PNGs; scene selection and configs unchanged."
    )


@main.command("prepare-eval")
@click.option("--candidates", type=int, default=16)
def prepare_eval(candidates):
    """Configure native-512 stereo reconstruction on held-out candidate scenes."""
    snapshot = STANDARD_EVAL / "checks/preprocess_cosmos_render_used.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes((UFM / "src/preprocess_cosmos_render.py").read_bytes())
    inventory_path = STANDARD_EVAL / "candidate_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    tasks = inventory["candidates"][:candidates]
    for direction in TARGET_CAMERA:
        config = yaml.safe_load((DEFAULT_OUTPUT / "configs" / f"test_{direction}.yaml").read_text())
        config["paths"]["output_dir"] = str(STANDARD_EVAL / "stereo")
        config["runtime"]["gpu"] = 1
        config["dataset"].update(
            image_size=[512, 288],
            target_steps=100,
            frame_sampling=1,
            batch_indices=[row["index"] for row in tasks],
        )
        path = STANDARD_EVAL / "configs" / f"{direction}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(STANDARD_EVAL / "preprocess_tasks.json", tasks)
    print(
        f"Prepared {len(tasks)} stereo candidates; two directions at 512x288",
        flush=True,
    )


@main.command("eval-worker")
@click.option("--rank", required=True, type=int)
@click.option("--workers", default=4, type=int)
def eval_worker(rank, workers):
    """Reconstruct both stereo directions, then render eligible camera controls."""
    from preprocess import run_pipeline
    from prepare_standard_eval import eligible, runs

    tasks = json.loads((STANDARD_EVAL / "preprocess_tasks.json").read_text())[rank::workers]
    base = STANDARD_EVAL / "stereo"
    for direction in TARGET_CAMERA:
        pending = [
            row
            for row in tasks
            if not (base / "top20_reference" / f"{direction}_{row['scene']}" / "00099.json").exists()
        ]
        if pending:
            config = yaml.safe_load((STANDARD_EVAL / "configs" / f"{direction}.yaml").read_text())
            config["dataset"]["batch_indices"] = [row["index"] for row in pending]
            print(
                f"START RGB {direction}: {[row['scene'] for row in pending]}",
                flush=True,
            )
            run_pipeline(config)
    for row in tasks:
        scene = row["scene"]
        done = STANDARD_EVAL / "progress" / f"{scene}.json"
        if done.exists():
            continue
        common = eligible(base, scene, True, directions=tuple(TARGET_CAMERA))
        longest = max(runs(common), key=len, default=[])
        print(f"{scene}: jointly eligible consecutive frames={len(longest)}", flush=True)
        if len(longest) >= 50:
            for camera in CAMERAS:
                subprocess.run(
                    [
                        sys.executable,
                        str(UFM / "src/preprocess_cosmos_render.py"),
                        "--config",
                        str(STANDARD_EVAL / "configs/LR.yaml"),
                        "--splits",
                        "val",
                        "--log-names",
                        scene,
                        "--scene-index",
                        "0",
                        "--camera-id",
                        f"camera.{camera}",
                        "--frames",
                        "100",
                        "--frames-only",
                        "--no-rgb-boxes",
                        "--save-boxes",
                        "--no-scene-data",
                        "--output-dir",
                        str(base / "conditions" / camera),
                    ],
                    cwd=UFM,
                    check=True,
                )
        write_json(done, {"scene": scene, "common_count": len(common), "longest_run": longest})
        print(f"COMPLETE candidate {scene}", flush=True)


@main.command("finalize-eval")
def finalize_eval():
    """Freeze ten jointly eligible scenes, each with two consecutive 50-frame clips."""
    from prepare_standard_eval import eligible, runs

    if (STANDARD_EVAL / "selection.json").exists():
        print("Fixed stereo evaluation dataset already finalized", flush=True)
        return
    assert (STANDARD_EVAL / "checks/preprocess_cosmos_render_used.py").read_bytes() == (
        UFM / "src/preprocess_cosmos_render.py"
    ).read_bytes(), "Renderer changed during dataset construction"
    inventory = json.loads((STANDARD_EVAL / "candidate_inventory.json").read_text())
    excluded = set(inventory["excluded_training_scenes"])
    for item in inventory["training_manifests"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    tasks = json.loads((STANDARD_EVAL / "preprocess_tasks.json").read_text())
    base = STANDARD_EVAL / "stereo"
    selected = []
    for row in tasks:
        scene = row["scene"]
        common = eligible(base, scene, True, directions=tuple(TARGET_CAMERA))
        longest = max(runs(common), key=len, default=[])
        if len(longest) >= 50:
            assert (STANDARD_EVAL / "progress" / f"{scene}.json").is_file(), f"Controls incomplete: {scene}"
            selected.append({"scene": scene, "frames": longest[:50]})
        if len(selected) == 10:
            break
    assert len(selected) == 10, f"Only {len(selected)} eligible scenes; preprocess more candidates"
    assert not ({row["scene"] for row in selected} & excluded)
    helper_path = ROOT / "gen-video/harmonizer/training_script/create_dataset.py"
    spec = importlib.util.spec_from_file_location("stereo_eval_manifest", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    records = {}
    checked = set()
    for row in selected:
        scene = row["scene"]
        for direction, target_camera in TARGET_CAMERA.items():
            sequence = f"{direction}_{scene}"
            target_root = base / "conditions" / target_camera
            source_root = base / "conditions/pcam_f0"
            metadata = target_root / "videos" / scene / "metadata.json"
            camera_metadata = json.loads(metadata.read_text())
            assert camera_metadata["source_camera"].lower() == target_camera
            assert camera_metadata["render_views"][0]["name"] == "original"
            for role, camera in (("source", "pcam_f0"), ("target", target_camera)):
                header = (base / "calibration" / sequence / f"{role}_calibration.txt").read_text().split("[frame 0]")[0]
                assert f"camera_id: {camera}\n" in header
                assert "image_width: 512\n" in header and "image_height: 288\n" in header
                assert "is_virtual_camera: false\n" in header
            for index in row["frames"]:
                filename = f"{index:05d}.png"
                entry = helper.build_sample(
                    root=base,
                    front_warped_dir=base / "front_warped",
                    top_reference_dir=base / "top20_reference",
                    warped_path=base / "front_warped" / sequence / filename,
                    frame_index=index,
                    prompt="remove degradation",
                    num_reference_frames=4,
                    min_reference_visibility=1000,
                    include_current_reference=True,
                    include_seginstance=False,
                    include_confident_coverage=True,
                    strict=True,
                )
                assert entry is not None
                entry.update(
                    source_camera="pcam_f0",
                    target_camera=target_camera,
                    source_world_scenario_image=str(source_root / "world_scenario" / f"original_{scene}" / filename),
                    world_scenario_image=str(target_root / "world_scenario" / f"original_{scene}" / filename),
                    hdmap_image=str(target_root / "hdmap" / f"original_{scene}" / filename),
                    reference_world_scenario_images=[
                        str(source_root / "world_scenario" / f"original_{scene}" / Path(ref).name)
                        for ref in entry["reference_images"]
                    ],
                    source_calibration=str(base / "calibration" / sequence / "source_calibration.txt"),
                    target_calibration=str(base / "calibration" / sequence / "target_calibration.txt"),
                    map_box_metadata=str(metadata),
                    boxes_json=str(
                        target_root
                        / "videos"
                        / f"pinhole_{target_camera}"
                        / f"000000_{scene}_boxes"
                        / f"{index:06d}.json"
                    ),
                )
                entry["reference_visibility_scores"] = helper_reference_scores(entry)
                assert len(set(entry["reference_images"])) == 4
                assert min(entry["reference_visibility_scores"]) > 1000
                assert entry["target_image"] == str(base / "front_target" / sequence / filename)
                for field in ("source_calibration", "target_calibration", "boxes_json"):
                    assert Path(entry[field]).is_file(), entry[field]
                paths = [
                    entry[field]
                    for field in (
                        "image",
                        "target_image",
                        "confident_coverage",
                        "world_scenario_image",
                        "source_world_scenario_image",
                        "hdmap_image",
                    )
                ]
                for path in [
                    *paths,
                    *entry["reference_images"],
                    *entry["reference_world_scenario_images"],
                ]:
                    if path not in checked:
                        with Image.open(path) as image:
                            assert image.size == (512, 288), path
                            image.verify()
                        checked.add(path)
                records[f"{sequence}/{index:05d}"] = entry
    assert len(records) == 1000
    selection = dict(
        scenes=selected,
        directions=list(TARGET_CAMERA),
        target_cameras=TARGET_CAMERA,
        source_camera="pcam_f0",
        references=4,
        include_current_reference=True,
        min_reference_visibility_exclusive=1000,
        consecutive_frames=50,
        image_size_wh=[512, 288],
        ground_truth="real_target_camera",
        training_manifests=inventory["training_manifests"],
        excluded_training_scenes=sorted(excluded),
        renderer_sha256=hashlib.sha256((UFM / "src/preprocess_cosmos_render.py").read_bytes()).hexdigest(),
    )
    write_json(base / "dataset_test.json", {"test": records, "selection": selection})
    write_json(STANDARD_EVAL / "selection.json", selection)
    print(
        "FINALIZED: ten scenes x two directions x fifty consecutive frames = 1000 targets / 20 videos",
        flush=True,
    )


if __name__ == "__main__":
    main()
