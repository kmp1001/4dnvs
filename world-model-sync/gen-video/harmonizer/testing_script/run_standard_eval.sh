#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARMONIZER="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd -- "$HARMONIZER/../.." && pwd)"
DATA="$ROOT/data/dh_train/nuscenes_standard_eval10_4dir_512_v1"
OUTPUT="$ROOT/out_standard_eval10_4dir_512_v1"
PIXI=/nfshomes/jlin1219/.pixi/bin/pixi
cd "$HARMONIZER"

if [[ "${1:-}" == --stereo-worker ]]; then
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if [[ "$2" == prepare ]]; then
        cd "$ROOT/feature-match/UFM"
        exec "$PIXI" run --locked python -u src/generate_stereo_world_scenario.py eval-worker --rank "$SLURM_PROCID" --workers 4
    else
        "$PIXI" run --locked python testing_script/world_scenario_outputs.py metrics-cache
        exec "$PIXI" run --locked python -u testing_script/standard_eval.py worker --setting stereo --rank "$SLURM_PROCID" --workers 4 --wait-for-checkpoint
    fi
fi

if [[ "${1:-}" == --node-worker ]]; then
    PHASE="$2"
    echo "$(hostname): $PHASE workers $((SLURM_PROCID * 4))..$((SLURM_PROCID * 4 + 3)); one worker per GPU"
    PIDS=()
    for local_gpu in 0 1 2 3; do
        rank=$((SLURM_PROCID * 4 + local_gpu))
        if [[ "$PHASE" == prepare && "${STANDARD_SKIP_PREPARE_RANK0:-0}" == 1 && "$rank" == 0 ]]; then continue; fi
        (
            export CUDA_VISIBLE_DEVICES="$local_gpu"
            export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            if [[ "$PHASE" == prepare ]]; then
                cd "$ROOT/feature-match/UFM"
                exec "$PIXI" run --locked python -u src/prepare_standard_eval.py worker --rank "$rank" --workers 16
            else
                cd "$HARMONIZER"
                exec "$PIXI" run --locked python -u testing_script/standard_eval.py worker --rank "$rank" --workers 16
            fi
        ) > "$OUTPUT/launcher_logs/${PHASE}-worker-${rank}.log" 2>&1 &
        PIDS+=("$!")
    done
    status=0
    for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
    exit "$status"
fi

JOB="" PHASE=all SETTING=monocular NODE=""
while (($#)); do
    case "$1" in
        --jobid) JOB="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --setting) SETTING="$2"; shift 2 ;;
        --node) NODE="$2"; shift 2 ;;
        *) echo "Usage: bash testing_script/run_standard_eval.sh --jobid JOB --phase all|prepare|evaluate" >&2; exit 2 ;;
    esac
done
: "${JOB:?Pass --jobid}"
if [[ "$SETTING" == stereo ]]; then
    : "${NODE:?Pass --node with one idle four-GPU node}"
    DATA="$ROOT/data/dh_train/nuscenes_stereo_standard_eval10_2dir_512_v1"
    OUTPUT="$ROOT/out_stereo_standard_eval10_2dir_512_v1"
    mkdir -p "$OUTPUT/launcher_logs"
    echo "Stereo standard evaluation: phase=$PHASE job=$JOB node=$NODE; logs: $OUTPUT/launcher_logs"
    stereo_launch() {
        srun --jobid="$JOB" --overlap --nodes=1 --ntasks=4 --ntasks-per-node=4 --nodelist="$NODE" \
            --gpus-per-task=1 --cpus-per-task=4 --exact --cpu-bind=none --kill-on-bad-exit=1 \
            --output="$OUTPUT/launcher_logs/$1-worker-%t.log" --error="$OUTPUT/launcher_logs/$1-worker-%t.log" \
            bash "$SCRIPT_DIR/run_standard_eval.sh" --stereo-worker "$1"
    }
    if [[ "$PHASE" == all || "$PHASE" == prepare ]]; then
        stereo_launch prepare
        (cd "$ROOT/feature-match/UFM"; "$PIXI" run --locked python src/generate_stereo_world_scenario.py finalize-eval)
    fi
    if [[ "$PHASE" == all || "$PHASE" == evaluate ]]; then
        test -f "$DATA/selection.json"
        "$PIXI" run --locked python testing_script/standard_eval.py stereo-inventory
        stereo_launch evaluate
        "$PIXI" run --locked python testing_script/standard_eval.py summary --setting stereo
    fi
    exit
fi
mkdir -p "$OUTPUT/launcher_logs"
cd "$HARMONIZER"
echo "Standard evaluation: phase=$PHASE, job=$JOB, 16 GPU workers"
echo "Results: $OUTPUT"
echo "Live logs: $OUTPUT/launcher_logs/"

launch() {
    srun --jobid="$JOB" --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 \
        --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none \
        bash "$SCRIPT_DIR/run_standard_eval.sh" --node-worker "$1"
}

if [[ "$PHASE" == all || "$PHASE" == prepare ]]; then
    launch prepare
    cd "$ROOT/feature-match/UFM"
    "$PIXI" run --locked python -u src/prepare_standard_eval.py finalize
    cd "$HARMONIZER"
fi
if [[ "$PHASE" == all || "$PHASE" == evaluate ]]; then
    test -f "$DATA/selection.json"
    srun --jobid="$JOB" --overlap -N4 -n4 --ntasks-per-node=1 --cpus-per-task=1 \
        bash -c 'cd "$1"; "$2" run --locked python testing_script/world_scenario_outputs.py metrics-cache' _ "$HARMONIZER" "$PIXI"
    launch evaluate
    "$PIXI" run --locked python testing_script/standard_eval.py summary
fi
