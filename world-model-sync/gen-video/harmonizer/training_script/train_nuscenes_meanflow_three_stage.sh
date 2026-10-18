#!/usr/bin/env bash
set -euo pipefail

# Run from harmonizer: pixi run bash training_script/train_nuscenes_meanflow_three_stage.sh
# These are optimizer-step budgets, matching train_nuscenes_smoke.sh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARMONIZER_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd -- "${HARMONIZER_DIR}/../.." && pwd)"
cd "${HARMONIZER_DIR}"
export COSMOS_MODEL_SIZE=0.6B
export COSMOS_CHECKPOINT_DIR="${COSMOS_CHECKPOINT_DIR:-${ROOT_DIR}/weights/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_FOLDER="${DATASET_FOLDER:-/project/data/dh_train/trainval_nuscenes_dr2m_final/dataset.json}"
FLOW_ESTIMATOR="${FLOW_ESTIMATOR:-waft}"
[[ "${FLOW_ESTIMATOR}" == raft || "${FLOW_ESTIMATOR}" == waft ]] || {
    echo "FLOW_ESTIMATOR must be raft or waft" >&2; exit 1;
}
FLOW_ARGS=(--flow-estimator "${FLOW_ESTIMATOR}")
if [[ "${FLOW_ESTIMATOR}" == waft ]]; then
    WAFT_REPO="${WAFT_REPO:-${ROOT_DIR}/WAFT}"
    WAFT_CONFIG="${WAFT_CONFIG:-${WAFT_REPO}/config/a1/tar-c-t.json}"
    WAFT_CHECKPOINT="${WAFT_CHECKPOINT:-${ROOT_DIR}/tar-c-t.pth}"
    FLOW_ARGS+=(--waft-repo "${WAFT_REPO}" --waft-config "${WAFT_CONFIG}" --waft-checkpoint "${WAFT_CHECKPOINT}")
    if [[ -n "${WAFT_ITERATIONS:-}" ]]; then FLOW_ARGS+=(--waft-iterations "${WAFT_ITERATIONS}"); fi
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        for waft_file in "${WAFT_REPO}/pixi.toml" "${WAFT_REPO}/pixi.lock" "${WAFT_CONFIG}" "${WAFT_CHECKPOINT}"; do
            [[ -f "${waft_file}" ]] || { echo "Missing WAFT file: ${waft_file}" >&2; exit 1; }
        done
    fi
fi
# LoRA is opt-in for each temporal stage. Full-training defaults stay unchanged.
STAGE2_LORA="${STAGE2_LORA:-0}"
STAGE3_LORA="${STAGE3_LORA:-0}"
for value in "${STAGE2_LORA}" "${STAGE3_LORA}"; do
    [[ "${value}" == 0 || "${value}" == 1 ]] || { echo "LoRA switches must be 0 or 1" >&2; exit 1; }
done
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
OUTPUT_VARIANT=""
if [[ "${STAGE2_LORA}" == 1 || "${STAGE3_LORA}" == 1 ]]; then
    OUTPUT_VARIANT="_lora_s2${STAGE2_LORA}_s3${STAGE3_LORA}"
fi
if [[ "${FLOW_ESTIMATOR}" == waft ]]; then OUTPUT_VARIANT+="_waft"; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/train_meanflow/nuscenes_three_stage${OUTPUT_VARIANT}}"
STAGE1_STEPS="${STAGE1_STEPS:-200000}"
STAGE2_STEPS="${STAGE2_STEPS:-100000}" # Original Stage-II total, now split across II+III.
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
for value in "${STAGE1_STEPS}" "${STAGE2_STEPS}" "${WARMUP_STEPS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Step budgets must be positive integers" >&2; exit 1; }
done
STAGE3_STEPS="${STAGE3_STEPS:-$((STAGE2_STEPS - WARMUP_STEPS))}"
[[ "${STAGE3_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "Stage III needs a positive budget" >&2; exit 1; }
(( WARMUP_STEPS + STAGE3_STEPS == STAGE2_STEPS )) || {
    echo "WARMUP_STEPS + STAGE3_STEPS must equal original STAGE2_STEPS" >&2; exit 1;
}

# Three independent EMA switches (1=enabled, 0=disabled).
STAGE1_EMA="${STAGE1_EMA:-1}"
STAGE2_EMA="${STAGE2_EMA:-1}"
STAGE3_EMA="${STAGE3_EMA:-1}"
for value in "${STAGE1_EMA}" "${STAGE2_EMA}" "${STAGE3_EMA}" "${HISTORY_CHECKPOINT_EMA:-0}"; do
    [[ "${value}" == 0 || "${value}" == 1 ]] || { echo "EMA switches must be 0 or 1" >&2; exit 1; }
done
HISTORY_GENERATOR="${HISTORY_GENERATOR:-frozen_stage2}" # Or stage3_ema.
[[ "${HISTORY_GENERATOR}" == frozen_stage2 || "${HISTORY_GENERATOR}" == stage3_ema ]] || {
    echo "HISTORY_GENERATOR must be frozen_stage2 or stage3_ema" >&2; exit 1;
}
[[ "${HISTORY_GENERATOR}" != stage3_ema || "${STAGE3_EMA}" == 1 ]] || {
    echo "stage3_ema generation requires STAGE3_EMA=1" >&2; exit 1;
}
if [[ "${HISTORY_GENERATOR}" == frozen_stage2 && "${HISTORY_CHECKPOINT_EMA:-0}" == 1 && "${STAGE2_EMA}" != 1 ]]; then
    echo "Stage-II EMA generator requires STAGE2_EMA=1" >&2; exit 1
fi
NUM_GPUS="${NUM_GPUS:-4}" # Total GPU workers across all nodes.
NUM_NODES="${NUM_NODES:-1}"
[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ && "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || exit 1
(( NUM_GPUS % NUM_NODES == 0 )) || {
    echo "NUM_GPUS must be divisible by NUM_NODES" >&2; exit 1;
}
MULTINODE_ARGS=()
if (( NUM_NODES > 1 )); then
    # srun starts one launcher per node; Accelerate starts the GPU workers.
    MACHINE_RANK="${MACHINE_RANK:-${SLURM_NODEID:-}}"
    [[ -n "${MACHINE_RANK}" ]] || {
        echo "Multi-node training requires srun or an explicit MACHINE_RANK" >&2; exit 1;
    }
    if [[ -z "${MAIN_PROCESS_IP:-}" ]]; then
        [[ -n "${SLURM_JOB_NODELIST:-}" ]] || {
            echo "Set MAIN_PROCESS_IP to the master node address" >&2; exit 1;
        }
        mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
        MAIN_PROCESS_IP="${ALLOCATED_NODES[0]}"
    fi
    MULTINODE_ARGS=(--machine_rank "${MACHINE_RANK}" --main_process_ip "${MAIN_PROCESS_IP}" --same_network)
    # NCCL RoCE registers pinned host memory; Slurm can default to an 8 MiB soft limit.
    if [[ "${DRY_RUN:-0}" != 1 ]]; then ulimit -l unlimited; fi
    printf 'Machine rank %s/%s: %s GPU workers; master=%s\n' \
        "${MACHINE_RANK}" "${NUM_NODES}" "$((NUM_GPUS / NUM_NODES))" "${MAIN_PROCESS_IP}"
fi
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_ROOT}/stage1}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_ROOT}/stage2}"
STAGE3_OUTPUT_DIR="${STAGE3_OUTPUT_DIR:-${OUTPUT_ROOT}/stage3}"
STAGE1_CHECKPOINT="${STAGE1_OUTPUT_DIR}/checkpoints/model_${STAGE1_STEPS}.pkl"
STAGE2_CHECKPOINT="${STAGE2_OUTPUT_DIR}/checkpoints/model_${WARMUP_STEPS}.pkl"

COMMON=(
    "${FLOW_ARGS[@]}"
    --dataset "${DATASET_FOLDER}"
    --image-prep "${IMAGE_PREP:-resize_288x512}"
    --references "${NUM_REFERENCE_IMAGES:-8}"
    --workers "${DATALOADER_NUM_WORKERS:-4}"
    --mixed-precision bf16
    --checkpointing-steps "${CHECKPOINTING_STEPS:-00}"
    --viz-freq "${VIZ_FREQ:-1000}"
    --eval-freq "${EVAL_FREQ:-1000}"
    --num-samples-eval "${NUM_SAMPLES_EVAL:-20}"
    --validation-steps "${VALIDATION_STEPS:-1}"
    --validation-split "${VALIDATION_SPLIT:-test}"
    --temporal-probability "${TEMPORAL_SAMPLE_PROBABILITY:-0.75}"
    --lambda-latent 1.0
    --small-raft
    --ema-decay "${EMA_DECAY:-0.9999}"
    --lora-rank "${LORA_RANK}" --lora-alpha "${LORA_ALPHA}" --lora-dropout "${LORA_DROPOUT}"
    --time-sampler "${MEANFLOW_TIME_SAMPLER:-logit_normal}"
    --time-mu "${MEANFLOW_TIME_MU:--0.4}"
    --time-sigma "${MEANFLOW_TIME_SIGMA:-1.0}"
    --interval-probability "${MEANFLOW_INTERVAL_PROBABILITY:-0.25}"
    --adaptive-p "${MEANFLOW_ADAPTIVE_P:-1.0}"
    --jvp-backend "${MEANFLOW_JVP_BACKEND:-func}"
    --report-to "${REPORT_TO:-none}"
    --seed "${SEED:-42}"
)
if [[ -n "${VALIDATION_DATASET:-}" ]]; then COMMON+=(--validation-dataset "${VALIDATION_DATASET}"); fi
if [[ -n "${WORLD_SCENARIO_MODE:-}" ]]; then COMMON+=(--world-scenario-mode "${WORLD_SCENARIO_MODE}"); fi
if [[ "${CAMERA_EMBED:-1}" == 1 ]]; then COMMON+=(--camera); else COMMON+=(--no-camera); fi
if [[ "${TIME_EMBED:-1}" == 1 ]]; then COMMON+=(--time); else COMMON+=(--no-time); fi

run_stage() {
    local stage="$1" steps="$2" directory="$3" rate="$4" ema_switch="$5"
    shift 5
    [[ "${ema_switch}" == 0 || "${ema_switch}" == 1 ]] || { echo "EMA switches must be 0 or 1" >&2; exit 1; }
    local extra=("$@") saved latest saved_step
    if [[ "${RESUME_RUN:-0}" == 1 ]]; then
        if [[ -f "${directory}/checkpoints/model_${steps}.pkl" ]]; then
            echo "Stage ${stage} already completed (${steps} steps); skipping training"
            return
        fi
        saved=("${directory}"/checkpoints/model_*.pkl)
        if [[ -f "${saved[0]}" ]]; then
            latest="$(printf '%s\n' "${saved[@]}" | sort -V | tail -n 1)"
            saved_step="${latest##*/model_}"; saved_step="${saved_step%.pkl}"
            (( saved_step < steps )) || { echo "Checkpoint step exceeds stage target: $latest" >&2; return 2; }
            extra=()
            while (($#)); do
                if [[ "$1" == --initial-checkpoint ]]; then shift 2; else extra+=("$1"); shift; fi
            done
            extra+=(--resume "$latest")
            echo "Resume stage ${stage}: ${saved_step} -> ${steps}, checkpoint ${latest}"
        fi
    fi
    local ema_arg=--ema
    [[ "${ema_switch}" == 1 ]] || ema_arg=--no-ema
    local lora_arg=--no-lora
    if [[ "${stage}" == 2 && "${STAGE2_LORA}" == 1 ]] || [[ "${stage}" == 3 && "${STAGE3_LORA}" == 1 ]]; then
        lora_arg=--lora
    fi
    local launch=(accelerate launch --mixed_precision=bf16
        --main_process_port "$(( ${MAIN_PROCESS_PORT:-29521} + stage - 1 ))"
        --num_machines "${NUM_NODES}" --num_processes "${NUM_GPUS}" "${MULTINODE_ARGS[@]}")
    if (( NUM_GPUS > 1 )); then launch+=(--multi_gpu); fi
    local command=("${launch[@]}" src/train_meanflow_curriculum.py "${COMMON[@]}"
        --stage "${stage}" --max-steps "${steps}" --output-dir "${directory}"
        --learning-rate "${rate}" "${ema_arg}" "${lora_arg}" --run-name "meanflow_stage${stage}" "${extra[@]}")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -f "${DATASET_FOLDER}" ]] || { echo "Missing dataset: ${DATASET_FOLDER}" >&2; exit 1; }
    for model_file in model.pt tokenizer/tokenizer.pth; do
        [[ -f "${COSMOS_CHECKPOINT_DIR}/${model_file}" ]] || { echo "Missing Cosmos weights: ${model_file}" >&2; exit 1; }
    done
fi
printf 'Optimizer steps: I=%s II=%s III=%s; II+III=%s\n' "${STAGE1_STEPS}" "${WARMUP_STEPS}" "${STAGE3_STEPS}" "${STAGE2_STEPS}"
run_stage 1 "${STAGE1_STEPS}" "${STAGE1_OUTPUT_DIR}" "${STAGE1_LEARNING_RATE:-2e-5}" "${STAGE1_EMA}" --lambda-temporal 0.0
run_stage 2 "${WARMUP_STEPS}" "${STAGE2_OUTPUT_DIR}" "${STAGE2_LEARNING_RATE:-5e-6}" "${STAGE2_EMA}" \
    --initial-checkpoint "${STAGE1_CHECKPOINT}" --lambda-temporal 0.5
HISTORY_ARGS=(--history-generator "${HISTORY_GENERATOR}" --resampling-shift "${RESAMPLING_SHIFT:-0.6}")
if [[ "${HISTORY_CHECKPOINT_EMA:-0}" == 1 ]]; then
    [[ "${STAGE2_EMA}" == 1 ]] || { echo "Stage-II EMA generator requires STAGE2_EMA=1" >&2; exit 1; }
    HISTORY_ARGS+=(--history-checkpoint-ema)
else
    HISTORY_ARGS+=(--no-history-checkpoint-ema)
fi
run_stage 3 "${STAGE3_STEPS}" "${STAGE3_OUTPUT_DIR}" "${STAGE3_LEARNING_RATE:-5e-6}" "${STAGE3_EMA}" \
    --initial-checkpoint "${STAGE2_CHECKPOINT}" --history-checkpoint "${STAGE2_CHECKPOINT}" \
    --lambda-temporal 0.5 "${HISTORY_ARGS[@]}"
