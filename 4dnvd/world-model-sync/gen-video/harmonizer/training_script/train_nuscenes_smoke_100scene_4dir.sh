#!/usr/bin/env bash
set -euo pipefail

# Original train_nuscenes_smoke.sh rectified-flow objective and two-stage curriculum.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARMONIZER_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd -- "${HARMONIZER_DIR}/../.." && pwd)"
cd "${HARMONIZER_DIR}"
export COSMOS_MODEL_SIZE=0.6B
export COSMOS_CHECKPOINT_DIR="${COSMOS_CHECKPOINT_DIR:-${ROOT_DIR}/weights/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_FOLDER="${DATASET_FOLDER:-${ROOT_DIR}/data/dh_train/smoke_nuscenes_dr2m/dataset_fixed_train100_4dir_seed42.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/train_baseline/nuscenes_100scene_4dir}"
IMAGE_PREP="${IMAGE_PREP:-resize_288x512}"
NUM_GPUS="${NUM_GPUS:-16}"
NUM_NODES="${NUM_NODES:-4}"
STAGE1_STEPS="${STAGE1_STEPS:-3000}"
STAGE2_STEPS="${STAGE2_STEPS:-3000}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_ROOT}/stage1}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_ROOT}/stage2}"
STAGE1_CHECKPOINT="${STAGE1_OUTPUT_DIR}/checkpoints/model_${STAGE1_STEPS}.pkl"

MULTINODE_ARGS=()
if (( NUM_NODES > 1 )); then
    MACHINE_RANK="${MACHINE_RANK:-${SLURM_NODEID:?Run one launcher per node through srun}}"
    if [[ -z "${MAIN_PROCESS_IP:-}" ]]; then
        mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
        MAIN_PROCESS_IP="${ALLOCATED_NODES[0]}"
    fi
    MULTINODE_ARGS=(--machine_rank "${MACHINE_RANK}" --main_process_ip "${MAIN_PROCESS_IP}" --same_network)
    if [[ "${DRY_RUN:-0}" != 1 ]]; then ulimit -l unlimited; fi
fi

COMMON_ARGS=(
    --dataset_folder "${DATASET_FOLDER}"
    --training_objective rectified_flow
    --train_batch_size 1 --gradient_accumulation_steps 1
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}"
    --checkpointing_steps "${CHECKPOINTING_STEPS:-0}"
    --eval_freq "${EVAL_FREQ:-300}" --viz_freq "${VIZ_FREQ:-150}"
    --train_image_prep "${IMAGE_PREP}" --test_image_prep "${IMAGE_PREP}"
    --lambda_lpips 0.0 --lambda_l2 0.0 --lambda_latent 1.0
    --train_full_unet --freeze_vae
    --bridge_timestep_shift 5.0 --rf_num_train_timesteps 1000
    --validation_steps "${VALIDATION_STEPS:-10}"
    --num_reference_images "${NUM_REFERENCE_IMAGES:-4}"
    --camera_translation_scale 1.0 --reference_time_embedding_dim 32 --reference_time_scale 1.0
    --num_samples_eval "${NUM_SAMPLES_EVAL:-80}"
    --report_to "${REPORT_TO:-wandb}" --tracker_project_name cosmos_harmonizer
    --mixed_precision bf16 --seed 42
)
if [[ -n "${VALIDATION_DATASET:-}" ]]; then COMMON_ARGS+=(--validation_dataset "${VALIDATION_DATASET}"); fi
if [[ -n "${WORLD_SCENARIO_MODE:-}" ]]; then COMMON_ARGS+=(--world_scenario_mode "${WORLD_SCENARIO_MODE}"); fi
if [[ "${CAMERA_EMBED:-1}" == 1 ]]; then COMMON_ARGS+=(--camera_embed); fi
if [[ "${TIME_EMBED:-1}" == 1 ]]; then COMMON_ARGS+=(--time_embed); fi

run_stage() {
    local stage="$1" steps="$2" directory="$3" rate="$4"
    shift 4
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
                if [[ "$1" == --pretrained_path ]]; then shift 2; else extra+=("$1"); shift; fi
            done
            extra+=(--resume "$latest")
            echo "Resume stage ${stage}: ${saved_step} -> ${steps}, checkpoint ${latest}"
        fi
    fi
    local command=(accelerate launch --mixed_precision=bf16
        --main_process_port "$(( ${MAIN_PROCESS_PORT:-29501} + stage - 1 ))"
        --num_machines "${NUM_NODES}" --num_processes "${NUM_GPUS}" "${MULTINODE_ARGS[@]}")
    if (( NUM_GPUS > 1 )); then command+=(--multi_gpu); fi
    command+=(src/train_drivenvs.py "${COMMON_ARGS[@]}"
        --training_stage "stage${stage}" --max_train_steps "${steps}"
        --output_dir "${directory}" --learning_rate "${rate}"
        --tracker_run_name "baseline_stage${stage}_rectified_flow" "${extra[@]}")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

run_stage 1 "${STAGE1_STEPS}" "${STAGE1_OUTPUT_DIR}" "${STAGE1_LEARNING_RATE:-2e-5}" --lambda_temporal 0.0
if [[ "${DRY_RUN:-0}" != 1 && ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "Stage-1 checkpoint was not created: ${STAGE1_CHECKPOINT}" >&2
    exit 1
fi
run_stage 2 "${STAGE2_STEPS}" "${STAGE2_OUTPUT_DIR}" "${STAGE2_LEARNING_RATE:-5e-6}" \
    --pretrained_path "${STAGE1_CHECKPOINT}" \
    --temporal_sample_probability "${TEMPORAL_SAMPLE_PROBABILITY:-0.75}" \
    --lambda_temporal 0.5 --flow_estimator raft \
    --temporal_loss_use_small_raft --temporal_loss_warmup_steps 0
