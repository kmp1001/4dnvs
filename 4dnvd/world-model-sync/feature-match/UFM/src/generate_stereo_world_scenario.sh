#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:?Slurm allocation job ID}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
ROOT="$(cd ../.. && pwd)"
OUT="${2:-$ROOT/data/dh_train/nuscenes_stereo_world_scenario_train100_test20}"
PIXI=/nfshomes/jlin1219/.pixi/bin/pixi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT/logs"
srun --jobid="$JOB_ID" --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    --gpus-per-task=1 --cpus-per-task=4 --exact --cpu-bind=none --kill-on-bad-exit=1 \
    --open-mode=append --output="$OUT/logs/worker-%t.log" --error="$OUT/logs/worker-%t.log" \
    "$PIXI" run --locked python src/generate_stereo_world_scenario.py worker --output "$OUT" --workers 8
"$PIXI" run --locked python src/generate_stereo_world_scenario.py finalize --output "$OUT"
