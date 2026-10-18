"""Recompute stereo metrics from original prediction PNGs and 512x288 real images."""

import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import yaml
from PIL import Image

HARMONIZER = Path(__file__).resolve().parents[1]
ROOT = HARMONIZER.parents[1]
DATA = ROOT / "data/dh_train/nuscenes_stereo_world_scenario_train100_test20"
RUNS = ROOT / "train_world_scenario_stereo"
CACHE = DATA / "metrics_512"
EXPERIMENTS = ("baseline_mode0", "baseline_mode1", "ours_mode0", "ours_mode1")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@click.group()
def main():
    pass


@main.command()
def prepare():
    """Share one GT cache across simultaneous training callbacks."""
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = CACHE / "real_images.json"
        if metadata.exists():
            saved = json.loads(metadata.read_text())
            assert saved["dataset_sha256"] == hashlib.sha256((DATA / "dataset_test.json").read_bytes()).hexdigest()
            print("Reusing verified 512x288 real/input cache", flush=True)
            return
        build_cache()


def build_cache():
    """Downsample only GT and warped input; preserve reconstruction and rankings."""
    source = DATA / "dataset_test.json"
    manifest = json.loads(source.read_text())

    def resize_entry(entry):
        paths = {}
        for field, category in (("target_image", "target"), ("image", "input")):
            original = entry[field]
            destination = CACHE / category / entry["scene"] / Path(original).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with Image.open(original) as image:
                    image.convert("RGB").resize((512, 288), Image.Resampling.LANCZOS).save(destination)
            with Image.open(destination) as image:
                assert image.size == (512, 288), destination
            paths[original] = str(destination)
        return paths

    mapping = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, paths in enumerate(pool.map(resize_entry, manifest["test"].values()), 1):
            mapping.update(paths)
            if index % 500 == 0:
                print(f"Prepared real/input images for {index}/{len(manifest['test'])} samples", flush=True)
    write_json(
        CACHE / "real_images.json",
        {
            "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "image_size_wh": [512, 288],
            "resampling": "PIL LANCZOS, matching training GT resize",
            "paths": mapping,
        },
    )
    print(f"Prepared {len(mapping)} real/input PNGs at 512x288", flush=True)


def prepare_manifest(source, destination, cache):
    manifest = json.loads(source.read_text())
    assert manifest["complete"] and manifest["num_samples"] == len(manifest["samples"])
    predictions = destination.parent / "predictions"
    if not predictions.exists():
        predictions.symlink_to(source.parent, target_is_directory=True)
    for record in manifest["samples"]:
        path = source.parent / record["prediction_relative_path"]
        with Image.open(path) as image:
            assert image.size == (512, 288), f"Prediction is not native 512x288: {path}"
        record["prediction_relative_path"] = str(Path("predictions") / record["prediction_relative_path"])
        record["target_path"] = cache["paths"][record["target_path"]]
        record["input_path"] = cache["paths"][record["input_path"]]
    manifest["metrics_preprocessing"] = {
        "source_manifest": str(source),
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset_sha256": cache["dataset_sha256"],
        "image_size_wh": [512, 288],
        "real_resampling": cache["resampling"],
        "predictions": "original lossless PNGs, no resizing or MP4 decoding",
    }
    write_json(destination, manifest)


@main.command()
@click.option("--experiment", type=click.Choice(EXPERIMENTS), required=True)
@click.option("--wait", "wait_for_predictions", is_flag=True)
@click.option("--run-dir", type=click.Path(path_type=Path))
@click.option("--gpus", type=click.IntRange(min=1), default=1)
def worker(experiment, wait_for_predictions, run_dir, gpus):
    """Wait for complete inference if requested, then compute metrics on one GPU."""
    source = (run_dir if run_dir is not None else RUNS / experiment) / "test/evaluation_manifest.json"
    if wait_for_predictions and not source.exists():
        print(f"WAIT for complete inference: {source}", flush=True)
        while not source.exists():
            time.sleep(30)
    output = source.parent / "metrics_512"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        calculate_metrics(source, output, experiment, gpus)


def calculate_metrics(source, output, experiment, gpus):
    if (output / "complete.json").exists():
        completed = json.loads((output / "complete.json").read_text())
        if completed["source_manifest_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest():
            print(f"SKIP completed: {experiment}", flush=True)
            return
    cache = json.loads((CACHE / "real_images.json").read_text())
    assert cache["dataset_sha256"] == hashlib.sha256((DATA / "dataset_test.json").read_bytes()).hexdigest()
    manifest = output / "evaluation_manifest.json"
    prepare_manifest(source, manifest, cache)
    print(f"START {experiment}: original predictions and real images are all 512x288", flush=True)
    subprocess.run(
        [sys.executable, "testing_script/world_scenario_outputs.py", "metrics-cache"],
        cwd=HARMONIZER,
        check=True,
    )
    launcher = [sys.executable, "-u"]
    if gpus > 1:
        launcher += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={gpus}"]
    subprocess.run(
        launcher
        + [
            "testing_script/calculate_generative_metrics.py",
            "--prediction_dir",
            str(output),
            "--prediction_manifest",
            str(manifest),
            "--metrics_output",
            str(output / "generative_metrics.yaml"),
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
        cwd=HARMONIZER,
        check=True,
    )
    metrics = yaml.safe_load((output / "generative_metrics.yaml").read_text())
    assert metrics["overall"]["total_images"] == json.loads(manifest.read_text())["num_samples"]
    write_json(output / "complete.json", json.loads(manifest.read_text())["metrics_preprocessing"])
    print(f"COMPLETE {experiment}: {output / 'generative_metrics.yaml'}", flush=True)


@main.command()
def summary():
    rows = []
    for experiment in EXPERIMENTS:
        output = RUNS / experiment / "test/metrics_512"
        if not (output / "complete.json").exists():
            continue
        metrics = yaml.safe_load((output / "generative_metrics.yaml").read_text())
        row = {"experiment": experiment, "image_size": "512x288"}
        for label, values in [
            ("overall", metrics["overall"]),
            ("LR", metrics["pools"]["L"]),
            ("RL", metrics["pools"]["R"]),
        ]:
            for key in ("fid", "kid_mean", "fvd", "fvd_num_clips"):
                row[f"{label}_{key}"] = values[key]
        rows.append(row)
    if rows:
        path = RUNS / "metrics_summary_512.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)}/4 experiments: {path}", flush=True)


if __name__ == "__main__":
    main()
