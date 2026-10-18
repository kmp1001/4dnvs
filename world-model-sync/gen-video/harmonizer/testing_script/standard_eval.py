"""Evaluate the fixed ten-scene benchmark, one model/mode task per GPU."""

import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import click
import torch
import yaml

HARMONIZER = Path(__file__).resolve().parents[1]
ROOT = HARMONIZER.parents[1]
DATA = ROOT / "data/dh_train/nuscenes_standard_eval10_4dir_512_v1"
OUTPUT = ROOT / "out_standard_eval10_4dir_512_v1"
STEREO_DATA = ROOT / "data/dh_train/nuscenes_stereo_standard_eval10_2dir_512_v1"
STEREO_OUTPUT = ROOT / "out_stereo_standard_eval10_2dir_512_v1"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


@click.group()
def main():
    pass


@main.command()
def inventory():
    """Freeze the highest available stage/step checkpoint for every requested run."""
    roots = [(ROOT / "train_baseline/nuscenes_100scene_4dir", "baseline", 0)]
    for path in sorted((ROOT / "train_meanflow").glob("*/*")):
        if path.is_dir() and list(path.glob("stage*/checkpoints/model_*.pkl")):
            roots.append((path, "ours", 0))
    for method in ("baseline", "ours"):
        for mode in (0, 1):
            roots.append((ROOT / f"train_world_scenario/{method}_mode{mode}", method, mode))
    models = []
    for path, method, mode in roots:
        checkpoints = list(path.glob("stage*/checkpoints/model_*.pkl"))
        checkpoint = max(
            checkpoints,
            key=lambda p: (int(p.parent.parent.name.removeprefix("stage")), int(p.stem.removeprefix("model_"))),
        )
        saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        assert "state_dict_unet" in saved, checkpoint
        config_path = checkpoint.parent.parent / "training_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        relative = path.relative_to(ROOT)
        row = {
            "name": "__".join(relative.parts),
            "experiment": str(path),
            "checkpoint": str(checkpoint),
            "method": method,
            "world_scenario_mode": mode,
            "steps": 4 if method == "baseline" else 1,
            "weights": "raw",
            "checkpoint_step": int(checkpoint.stem.removeprefix("model_")),
            "configured_max_steps": config.get("max_steps"),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        }
        if method == "ours":
            assert saved["conditioning_config"]["meanflow_time_conditioning"] == "cosmos_t_plus_meanflow_t_minus_r_v2"
        models.append(row)
        del saved
        print(row["name"], checkpoint.name, method, row["steps"], "raw", flush=True)
    write_json(OUTPUT / "models.json", models)
    tasks = [{**model, "test_mode": mode} for model in models for mode in ("double_reprojection", "shift_1m")]
    write_json(OUTPUT / "tasks.json", tasks)
    print(f"Frozen {len(models)} models / {len(tasks)} tasks", flush=True)


@main.command("stereo-inventory")
def stereo_inventory():
    """Use each stereo experiment's final-stage endpoint, with raw weights."""
    models = []
    for method in ("baseline", "ours"):
        for mode in (0, 1):
            experiment = ROOT / f"train_world_scenario_stereo/{method}_mode{mode}"
            stage, step = (2, 12000) if method == "baseline" else (3, 7200)
            checkpoint = experiment / f"stage{stage}/checkpoints/model_{step}.pkl"
            models.append(
                dict(
                    name=f"{method}_mode{mode}",
                    experiment=str(experiment),
                    checkpoint=str(checkpoint),
                    method=method,
                    world_scenario_mode=mode,
                    steps=4 if method == "baseline" else 1,
                    weights="raw",
                    checkpoint_step=step,
                    configured_max_steps=step,
                )
            )
    write_json(STEREO_OUTPUT / "models.json", models)
    write_json(STEREO_OUTPUT / "tasks.json", [{**model, "test_mode": "stereo"} for model in models])
    print("Prepared four final-stage stereo model tasks; wait for unfinished Stage III checkpoints", flush=True)


def run(command, log):
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("a") as handle:
        subprocess.run(command, cwd=HARMONIZER, stdout=handle, stderr=subprocess.STDOUT, check=True)


def export_videos(output, manifest, dataset):
    entries = json.loads(dataset.read_text())["test"]
    first_frames = {}
    for record in manifest["samples"]:
        first_frames.setdefault(record["scene"], record)
    destination = output / "videos"
    destination.mkdir(exist_ok=True)
    for scene, record in first_frames.items():
        entry = entries[record["sample_key"]]
        fps = json.loads(Path(entry["map_box_metadata"]).read_text())["fps"]
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-start_number",
                str(record["frame_index"]),
                "-i",
                str(Path(record["prediction_path"]).parent / "%05d.png"),
                "-frames:v",
                "50",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(destination / f"{scene}.mp4"),
            ],
            output / "videos.log",
        )


def evaluate_tasks(tasks, output_root, dataset_root=DATA):
    for task in tasks:
        if task["test_mode"] == "stereo":
            output = output_root / task["name"] / task["test_mode"]
            output.mkdir(parents=True, exist_ok=True)
            with (output / "evaluation.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                evaluate_task(task, output_root, dataset_root)
        else:
            evaluate_task(task, output_root, dataset_root)


def evaluate_task(task, output_root, dataset_root):
    output = output_root / task["name"] / task["test_mode"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        print("SKIP completed", output, flush=True)
        return
    dataset = dataset_root / task["test_mode"] / "dataset_test.json"
    entries = json.loads(dataset.read_text())["test"]
    sequence_counts = Counter(entry["scene"] for entry in entries.values())
    assert set(sequence_counts.values()) == {50}
    directional_images = Counter(entry["scene"][0] for entry in entries.values())
    directional_clips = Counter(scene[0] for scene in sequence_counts)
    manifest = output / "evaluation_manifest.json"
    write_json(
        output / "task.json",
        {
            **task,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        },
    )
    env = os.environ
    env["COSMOS_MODEL_SIZE"] = "0.6B"
    env["COSMOS_CHECKPOINT_DIR"] = str(ROOT / "weights/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image")
    inference = [
        sys.executable,
        "src/evaluate_test_dataset.py",
        "--dataset_json",
        str(dataset),
        "--split",
        "test",
        "--model_path",
        task["checkpoint"],
        "--output_dir",
        str(output),
        "--num_reference_images",
        "4",
        "--world_scenario_mode",
        str(task["world_scenario_mode"]),
        "--num-gpus",
        "1",
        "--camera_embed",
        "--time_embed",
        "--num_inference_steps",
        str(task["steps"]),
        "--resolution",
        "resize_288x512",
        "--seed",
        "42",
    ]
    if task["method"] == "ours":
        inference.append("--meanflow")
        if task["weights"] == "ema":
            inference.append("--meanflow_use_ema")
    if not manifest.exists():
        run(inference, output / "inference.log")
    generated = json.loads(manifest.read_text())
    assert generated["complete"] and generated["num_samples"] == len(entries)
    assert generated["config"]["meanflow_use_ema"] == (task["weights"] == "ema")
    from PIL import Image

    for record in generated["samples"]:
        with Image.open(record["prediction_path"]) as image:
            assert image.size == (512, 288), record["prediction_path"]
    generative = output / "generative_metrics.yaml"
    if not generative.exists():
        run(
            [
                sys.executable,
                "testing_script/calculate_generative_metrics.py",
                "--prediction_dir",
                str(output),
                "--prediction_manifest",
                str(manifest),
                "--metrics_output",
                str(generative),
                "--device",
                "cuda:0",
                "--group-by-prefix",
                "--max-frames-per-scene",
                "0",
                "--fid-mode",
                "clean",
                "--fid-batch-size",
                "32",
                "--fvd-method",
                "videogpt",
                "--fvd-clip-length",
                "0",
                "--fvd-batch-size",
                "1",
                "--seed",
                "42",
            ],
            output / "metrics.log",
        )
    measured = yaml.safe_load(generative.read_text())
    assert measured["overall"]["total_images"] == len(entries)
    assert measured["overall"]["fvd_num_clips"] == len(sequence_counts)
    for direction in directional_images:
        assert measured["pools"][direction]["num_images"] == directional_images[direction]
        assert measured["pools"][direction]["fvd_num_clips"] == directional_clips[direction]
    if task["test_mode"] in ("double_reprojection", "stereo"):
        reconstruction = output / "reconstruction_metrics.yaml"
        if not reconstruction.exists():
            run(
                [
                    sys.executable,
                    "testing_script/calculate_reconstruction_metrics.py",
                    "--prediction_dir",
                    str(output),
                    "--prediction_manifest",
                    str(manifest),
                    "--metrics_output",
                    str(reconstruction),
                    "--device",
                    "cuda:0",
                    "--group-by-prefix",
                    "--seed",
                    "42",
                ],
                output / "reconstruction.log",
            )
    export_videos(output, generated, dataset)
    write_json(output / "complete.json", task)
    print("TASK COMPLETE", output, flush=True)


@main.command()
@click.option("--rank", required=True, type=int)
@click.option("--workers", default=16, type=int)
@click.option("--setting", type=click.Choice(["monocular", "stereo"]), default="monocular")
@click.option("--wait-for-checkpoint", is_flag=True)
def worker(rank, workers, setting, wait_for_checkpoint):
    """Inference, metrics, then the next assigned task on this same GPU."""
    output = STEREO_OUTPUT if setting == "stereo" else OUTPUT
    data = STEREO_DATA if setting == "stereo" else DATA
    tasks = json.loads((output / "tasks.json").read_text())[rank::workers]
    for task in tasks:
        checkpoint = Path(task["checkpoint"])
        ready = checkpoint.with_suffix(".ready.json")
        if (
            wait_for_checkpoint
            and task["method"] == "ours"
            and not (output / task["name"] / task["test_mode"] / "complete.json").exists()
        ):
            print(f"WAIT for final raw checkpoint: {checkpoint}", flush=True)
            while not ready.exists():
                time.sleep(30)
        evaluate_tasks([task], output, data)


@main.command("single")
@click.option("--method", type=click.Choice(["baseline", "ours"]), required=True)
@click.option("--mode", type=click.IntRange(0, 2), required=True)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--setting", type=click.Choice(["monocular", "stereo"]), default="monocular")
@click.option("--wait-for-dataset", is_flag=True)
def single(method, mode, checkpoint, output_root, setting, wait_for_dataset):
    """Evaluate raw weights on the fixed monocular or paired-stereo benchmark."""
    model = dict(
        name=f"{method}_mode{mode}",
        method=method,
        world_scenario_mode=mode,
        checkpoint=str(checkpoint.resolve()),
        weights="raw",
        steps=4 if method == "baseline" else 1,
    )
    if setting == "stereo":
        saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        assert "state_dict_unet" in saved, checkpoint
        del saved
        write_json(checkpoint.with_suffix(".ready.json"), {"checkpoint": str(checkpoint.resolve())})
        if wait_for_dataset:
            print(f"WAIT for fixed stereo dataset: {STEREO_DATA / 'selection.json'}", flush=True)
            while not (STEREO_DATA / "selection.json").exists():
                time.sleep(30)
        evaluate_tasks([{**model, "test_mode": "stereo"}], output_root.resolve(), STEREO_DATA)
    else:
        tasks = [{**model, "test_mode": test_mode} for test_mode in ("double_reprojection", "shift_1m")]
        evaluate_tasks(tasks, output_root.resolve())


@main.command()
@click.option("--setting", type=click.Choice(["monocular", "stereo"]), default="monocular")
def summary(setting):
    """Collect overall and per-direction results without changing any metrics."""
    output = STEREO_OUTPUT if setting == "stereo" else OUTPUT
    directions = ("L", "R") if setting == "stereo" else ("D", "L", "R", "U")
    rows = []
    for task in json.loads((output / "tasks.json").read_text()):
        path = output / task["name"] / task["test_mode"]
        if not (path / "complete.json").exists():
            continue
        metrics = yaml.safe_load((path / "generative_metrics.yaml").read_text())
        row = {
            k: task[k]
            for k in ("name", "test_mode", "method", "world_scenario_mode", "checkpoint_step", "steps", "weights")
        }
        row.update(
            FID=metrics["overall"]["fid"], FVD=metrics["overall"]["fvd"], FVD_clips=metrics["overall"]["fvd_num_clips"]
        )
        for direction in directions:
            label = {"L": "LR", "R": "RL"}[direction] if setting == "stereo" else direction
            row[f"FID_{label}"] = metrics["pools"][direction]["fid"]
            row[f"FVD_{label}"] = metrics["pools"][direction]["fvd"]
        row["PSNR"] = row["SSIM"] = None
        for direction in directions:
            label = {"L": "LR", "R": "RL"}[direction] if setting == "stereo" else direction
            row[f"PSNR_{label}"] = row[f"SSIM_{label}"] = None
        if task["test_mode"] in ("double_reprojection", "stereo"):
            rec = yaml.safe_load((path / "reconstruction_metrics.yaml").read_text())
            row.update(PSNR=rec["overall"]["psnr_mean"], SSIM=rec["overall"]["ssim_mean"])
            for direction in directions:
                label = {"L": "LR", "R": "RL"}[direction] if setting == "stereo" else direction
                row[f"PSNR_{label}"] = rec["pools"][direction]["psnr_mean"]
                row[f"SSIM_{label}"] = rec["pools"][direction]["ssim_mean"]
        rows.append(row)
    if rows:
        with (output / "metrics_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Completed {len(rows)} / {len(json.loads((output / 'tasks.json').read_text()))} tasks")


if __name__ == "__main__":
    main()
