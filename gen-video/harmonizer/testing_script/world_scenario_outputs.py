"""Record experiment identity, export retained-frame videos, and collect metrics.

FFmpeg concat demuxer: https://ffmpeg.org/ffmpeg-formats.html#concat-1
Video frames follow the evaluation manifest order, including frame-index gaps.
"""

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import click
import yaml


@click.group()
def main():
    pass


@main.command("record-config")
@click.option("--method", type=click.Choice(["baseline", "ours"]), required=True)
@click.option("--mode", type=click.IntRange(0, 2), required=True)
@click.option("--nodes", type=int, required=True)
@click.option("--gpus-per-node", type=int, required=True)
@click.option("--output", type=click.Path(), required=True)
def record_config(method, mode, nodes, gpus_per_node, output):
    gpu_count = nodes * gpus_per_node
    manifests = {
        name: Path(os.environ[field])
        for name, field in [("train", "DATASET_FOLDER"), ("validation", "VALIDATION_DATASET"), ("test", "TEST_DATASET")]
    }
    budgets = [48000, 48000] if method == "baseline" else [48000, 19200, 28800]
    config = dict(
        method=method,
        dataset_setup=os.environ.get("WORLD_SCENARIO_SETTING", "monocular"),
        world_scenario_mode=mode,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        stage_steps=[count // gpu_count for count in budgets],
        stage_current_target_samples=budgets,
        controls="world_scenario_image" if mode else None,
        control_blocks=3 if mode == 2 else 0,
        flow_estimator="raft_small" if method == "baseline" else "waft",
        history_generator=None if method == "baseline" else "stage3_ema",
        ema_stages=[] if method == "baseline" else [3],
        video_protocol="retained_frame_order_including_gaps",
        fvd_backend="videogpt",
        manifests={
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in manifests.items()
        },
    )
    Path(output).write_text(json.dumps(config, indent=2) + "\n")


@main.command("metrics-cache")
def metrics_cache():
    import torch

    source = Path(__file__).resolve().parents[3] / "weights/metrics/cleanfid/inception-2015-12-05.pt"
    cache = Path("/tmp/inception-2015-12-05.pt")
    if not cache.exists():
        temporary = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(cache)
    torch.jit.load(str(cache), map_location="cpu").eval()


@main.command("videos")
@click.option("--manifest", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
def videos(manifest, output_dir):
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    assert data["complete"]
    original = json.loads(Path(data["dataset_json"]).read_text())[data["split"]]
    groups = defaultdict(list)
    for record in data["samples"]:
        groups[record["scene"]].append(record)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    video_records = []
    for scene, records in sorted(groups.items()):
        records.sort(key=lambda record: record["frame_index"])
        entry = original[records[0]["sample_key"]]
        metadata = json.loads(Path(entry["map_box_metadata"]).read_text())
        fps = metadata["fps"]
        indices = [record["frame_index"] for record in records]
        for kind, field in [("prediction", "prediction_path"), ("target", "target_path"), ("input", "input_path")]:
            concat = output / f"{scene}_{kind}.ffconcat"
            lines = ["ffconcat version 1.0"]
            for record in records:
                escaped = str(Path(record[field]).resolve()).replace("'", "'\\''")
                lines.extend([f"file '{escaped}'", f"duration {1 / fps:.12f}"])
            concat.write_text("\n".join(lines) + "\n")
            destination = output / f"{scene}_{kind}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat),
                    "-vf",
                    "scale=512:288",
                    "-r",
                    str(fps),
                    "-frames:v",
                    str(len(records)),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(destination),
                ],
                check=True,
            )
            concat.unlink()
        video_records.append(
            {
                "scene": scene,
                "frame_indices": indices,
                "fps": fps,
                "contains_gaps": any(b != a + 1 for a, b in pairwise(indices)),
            }
        )
    (output / "video_manifest.json").write_text(
        json.dumps({"protocol": "retained_frame_order_including_gaps", "videos": video_records}, indent=2) + "\n"
    )
    print(f"Exported prediction/target/input MP4s for {len(groups)} direction-scene sequences")


@main.command("summary")
@click.option("--output-root", type=click.Path(exists=True), required=True)
def summary(output_root):
    parent = Path(output_root).parent
    rows = []
    for config_path in sorted(parent.glob("*/run_config.json")):
        run = config_path.parent
        metric_path = run / "test/generative_metrics.yaml"
        if not metric_path.exists():
            continue
        config = json.loads(config_path.read_text())
        metrics = yaml.safe_load(metric_path.read_text())
        overall = metrics["overall"]
        row = dict(
            experiment=run.name,
            method=config["method"],
            mode=config["world_scenario_mode"],
            stage_steps="/".join(map(str, config["stage_steps"])),
            FID=overall["fid"],
            KID_mean=overall["kid_mean"],
            FVD_videogpt=overall["fvd"],
        )
        for prefix in ("D", "L", "R", "U"):
            pool = metrics["pools"][prefix]
            row[f"FID_{prefix}2M"] = pool["fid"]
            row[f"FVD_{prefix}2M"] = pool["fvd"]
        rows.append(row)
    destination = parent / "metrics_summary.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {destination}")


@main.command("stereo-summary")
@click.option("--output-root", type=click.Path(exists=True), required=True)
def stereo_summary(output_root):
    parent = Path(output_root).parent
    rows = []
    for config_path in sorted(parent.glob("*/run_config.json")):
        run = config_path.parent
        metric_path = run / "test/generative_metrics.yaml"
        if not metric_path.exists():
            continue
        config = json.loads(config_path.read_text())
        metrics = yaml.safe_load(metric_path.read_text())
        overall = metrics["overall"]
        row = dict(
            experiment=run.name,
            method=config["method"],
            mode=config["world_scenario_mode"],
            stage_steps="/".join(map(str, config["stage_steps"])),
            FID=overall["fid"],
            KID_mean=overall["kid_mean"],
            FVD_videogpt=overall["fvd"],
        )
        for direction, prefix in [("LR", "L"), ("RL", "R")]:
            pool = metrics["pools"][prefix]
            row[f"FID_{direction}"] = pool["fid"]
            row[f"KID_{direction}"] = pool["kid_mean"]
            row[f"FVD_{direction}"] = pool["fvd"]
        rows.append(row)
    destination = parent / "metrics_summary.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
