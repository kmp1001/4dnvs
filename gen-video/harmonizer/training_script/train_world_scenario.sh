#!/usr/bin/env bash
set -euo pipefail

# Example: bash training_script/train_world_scenario.sh --method ours --mode 2 --nodes 2 --gpus-per-node 4
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARMONIZER="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd -- "$HARMONIZER/../.." && pwd)"
export COSMOS_MODEL_SIZE=0.6B
export COSMOS_CHECKPOINT_DIR="${COSMOS_CHECKPOINT_DIR:-$ROOT/weights/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image}"
SETTING=monocular METHOD="" MODE="" NODES=1 GPUS_PER_NODE=4 WORKER=0
while (($#)); do
    case "$1" in
        --setting) SETTING="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --nodes) NODES="$2"; shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --dry-run) export DRY_RUN=1; shift ;;
        --resume) export RESUME_RUN=1; shift ;;
        --worker) WORKER=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$SETTING" == monocular || "$SETTING" == stereo ]] || { echo '--setting monocular|stereo' >&2; exit 2; }
[[ "$METHOD" == baseline || "$METHOD" == ours ]] || { echo '--method baseline|ours is required' >&2; exit 2; }
[[ "$MODE" == 0 || "$MODE" == 1 || "$MODE" == 2 ]] || { echo '--mode 0|1|2 is required' >&2; exit 2; }
[[ "$NODES" =~ ^[1-9][0-9]*$ && "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || exit 2
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
for samples in 24000 9600 14400; do
    (( samples % TOTAL_GPUS == 0 )) || { echo "GPU count $TOTAL_GPUS cannot exactly preserve stage sample budgets" >&2; exit 2; }
done
export NUM_NODES="$NODES" NUM_GPUS="$TOTAL_GPUS" WORLD_SCENARIO_MODE="$MODE"
export STAGE1_STEPS=$((48000 / TOTAL_GPUS)) STAGE2_STEPS=$((48000 / TOTAL_GPUS))
export WARMUP_STEPS=$((19200 / TOTAL_GPUS)) STAGE3_STEPS=$((28800 / TOTAL_GPUS))
export WORLD_SCENARIO_SETTING="$SETTING"
if [[ "$SETTING" == stereo ]]; then
    [[ "$NODES" == 1 && "$GPUS_PER_NODE" == 4 ]] || { echo 'Stereo experiments use one node with four GPUs each' >&2; exit 2; }
    STEREO_DATA="$ROOT/data/dh_train/nuscenes_stereo_world_scenario_train100_test20"
    export DATASET_FOLDER="$STEREO_DATA/dataset_train.json"
    export VALIDATION_DATASET="$STEREO_DATA/dataset_validation_test80.json"
    export TEST_DATASET="$ROOT/data/dh_train/nuscenes_stereo_standard_eval10_2dir_512_v1/stereo/dataset_test.json"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/train_world_scenario_stereo/${METHOD}_mode${MODE}}"
    export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-0}" VIZ_FREQ="${VIZ_FREQ:-300}" EVAL_FREQ="${EVAL_FREQ:-600}"
else
    export DATASET_FOLDER="$ROOT/data/dh_train/smoke_nuscenes_dr2m/dataset_map_box_train100_4dir_ordered.json"
    export VALIDATION_DATASET="$ROOT/data/dh_train/smoke_nuscenes_dr2m/dataset_map_box_val80_4scene_4dir_ordered.json"
    export TEST_DATASET="$ROOT/data/dh_train/smoke_nuscenes_dr2m/dataset_map_box_test20_4dir_ordered.json"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/train_world_scenario/${METHOD}_mode${MODE}}"
fi
export IMAGE_PREP=resize_288x512 NUM_REFERENCE_IMAGES=4 CAMERA_EMBED=1 TIME_EMBED=1
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-0}" VIZ_FREQ="${VIZ_FREQ:-150}" EVAL_FREQ="${EVAL_FREQ:-300}"
export NUM_SAMPLES_EVAL=80 WANDB_MODE="${WANDB_MODE:-disabled}" REPORT_TO="${REPORT_TO:-none}"
METHOD_PORT=0
[[ "$METHOD" != ours ]] || METHOD_PORT=40
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((29600 + MODE * 10 + METHOD_PORT))}"
PIXI="${PIXI:-/nfshomes/jlin1219/.pixi/bin/pixi}"
export PIXI
cd "$HARMONIZER"

if (( WORKER == 0 )) && [[ -n "${SLURM_JOB_ID:-}" || "$NODES" -gt 1 ]] && [[ "${DRY_RUN:-0}" != 1 ]]; then
    : "${SLURM_JOB_ID:?Multi-node entrypoint must run inside a Slurm allocation}"
    exec srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
        --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="${CPUS_PER_NODE:-16}" \
        --cpu-bind=none --kill-on-bad-exit=1 --label \
        bash "$SCRIPT_DIR/train_world_scenario.sh" --setting "$SETTING" --method "$METHOD" --mode "$MODE" \
        --nodes "$NODES" --gpus-per-node "$GPUS_PER_NODE" --worker
fi
if [[ "${DRY_RUN:-0}" == 1 && "$NODES" -gt 1 ]]; then
    export MACHINE_RANK=0 MAIN_PROCESS_IP=127.0.0.1
fi
printf 'Experiment %s mode %s: %s nodes x %s GPUs; 96000 current-target samples total\n' "$SETTING/$METHOD" "$MODE" "$NODES" "$GPUS_PER_NODE"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "$OUTPUT_ROOT"
    if [[ "$NODES" == 1 || "${SLURM_NODEID:-0}" == 0 ]]; then
        "$PIXI" run --locked python testing_script/world_scenario_outputs.py record-config \
            --method "$METHOD" --mode "$MODE" --nodes "$NODES" --gpus-per-node "$GPUS_PER_NODE" --output "$OUTPUT_ROOT/run_config.json"
    fi
fi
if [[ "$METHOD" == baseline ]]; then
    "$PIXI" run --locked bash "$SCRIPT_DIR/train_nuscenes_smoke_100scene_4dir.sh"
    FINAL_CHECKPOINT="$OUTPUT_ROOT/stage2/checkpoints/model_${STAGE2_STEPS}.pkl"
else
    export FLOW_ESTIMATOR=waft HISTORY_GENERATOR=stage3_ema
    export STAGE1_EMA=0 STAGE2_EMA=0 STAGE3_EMA=1 STAGE2_LORA=0 STAGE3_LORA=0
    "$PIXI" run --locked bash "$SCRIPT_DIR/train_nuscenes_meanflow_three_stage.sh"
    FINAL_CHECKPOINT="$OUTPUT_ROOT/stage3/checkpoints/model_${STAGE3_STEPS}.pkl"
fi
# All distributed training workers have exited; node zero evaluates full scenes on this node's GPUs.
if [[ "$NODES" == 1 || "${SLURM_NODEID:-0}" == 0 ]]; then
    EVAL_SCRIPT=eval_world_scenario.sh
    if [[ "$SETTING" == stereo ]]; then EVAL_SCRIPT=eval_stereo_world_scenario.sh; fi
    bash "$HARMONIZER/testing_script/$EVAL_SCRIPT" "$METHOD" "$MODE" "$FINAL_CHECKPOINT" "$GPUS_PER_NODE"
fi
