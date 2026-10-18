#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
ROOT="$(cd ../.. && pwd)"
PIXI=/nfshomes/jlin1219/.pixi/bin/pixi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${1:-}" == --worker ]]; then
    experiments=(baseline_mode0 baseline_mode1 ours_mode0 ours_mode1)
    exec "$PIXI" run --locked python -u testing_script/reevaluate_stereo_512.py worker \
        --experiment "${experiments[$SLURM_PROCID]}" --wait
fi
JOB="${1:?Pass the allocation job ID}" NODE="${2:?Pass one idle node with four GPUs}"
LOGS="$ROOT/train_world_scenario_stereo/metrics_512_logs"
mkdir -p "$LOGS"
"$PIXI" run --locked python -u testing_script/reevaluate_stereo_512.py prepare
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=4 --ntasks-per-node=4 --nodelist="$NODE" \
    --gpus-per-task=1 --cpus-per-task=4 --exact --cpu-bind=none --kill-on-bad-exit=1 \
    --output="$LOGS/experiment-%t.log" --error="$LOGS/experiment-%t.log" \
    bash "$SCRIPT_DIR/reevaluate_stereo_512.sh" --worker
"$PIXI" run --locked python testing_script/reevaluate_stereo_512.py summary
