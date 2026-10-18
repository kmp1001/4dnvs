#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
LOGS="$ROOT/train_world_scenario_stereo/launcher_logs"
if [[ "${1:-}" == --resume ]]; then export RESUME_RUN=1; shift; fi
export CHECKPOINTING_STEPS=0

if [[ "${1:-}" == --worker ]]; then
    INDEX="${STEREO_EXPERIMENT_INDEX:-${SLURM_PROCID}}"
    case "$INDEX" in
        0) METHOD=baseline; MODE=0 ;;
        1) METHOD=baseline; MODE=1 ;;
        2) METHOD=ours; MODE=0 ;;
        3) METHOD=ours; MODE=1 ;;
        *) echo "Experiment index must be 0..3" >&2; exit 2 ;;
    esac
    export OUTPUT_ROOT="$ROOT/train_world_scenario_stereo/${METHOD}_mode${MODE}"
    echo "$(hostname): $METHOD mode $MODE, one node x four GPUs"
    exec bash "$SCRIPT_DIR/train_world_scenario.sh" --setting stereo \
        --method "$METHOD" --mode "$MODE" --nodes 1 --gpus-per-node 4 --worker
fi
if [[ "${RESUME_RUN:-0}" == 1 ]]; then LOGS="$ROOT/train_world_scenario_stereo/resume_logs"; fi
mkdir -p "$LOGS"
if [[ "${1:-}" == --jobs ]]; then
    JOBS=("${2:?first two-node job ID}" "${3:?second two-node job ID}")
    HOSTS=()
    for job in "${JOBS[@]}"; do
        mapfile -t JOB_HOSTS < <(scontrol show hostnames "$(squeue -h -j "$job" -o '%N')")
        (( ${#JOB_HOSTS[@]} == 2 )) || { echo "Job $job must have exactly two allocated nodes" >&2; exit 2; }
        HOSTS+=("${JOB_HOSTS[@]}")
    done
    PIDS=()
    for index in 0 1 2 3; do
        echo "Experiment $index -> job ${JOBS[$((index / 2))]}, node ${HOSTS[$index]}; log: $LOGS/experiment-$index.log"
        STEREO_EXPERIMENT_INDEX="$index" srun --jobid="${JOBS[$((index / 2))]}" \
            --nodes=1 --ntasks=1 --nodelist="${HOSTS[$index]}" --gpus-per-node=4 \
            --cpus-per-task=16 --exact --cpu-bind=none --kill-on-bad-exit=1 \
            --output="$LOGS/experiment-$index.log" --error="$LOGS/experiment-$index.log" \
            bash "$SCRIPT_DIR/train_stereo_four_experiments.sh" --worker &
        PIDS+=("$!")
    done
    STATUS=0
    for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
    exit "$STATUS"
fi
: "${SLURM_JOB_ID:?Run inside a four-node allocation, or pass --jobs JOB_A JOB_B for two two-node allocations}"
echo "Starting four experiments, one per node. Logs: $LOGS/experiment-0.log through experiment-3.log"
exec srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-node=4 \
    --cpus-per-task=16 --exact --cpu-bind=none --kill-on-bad-exit=1 \
    --output="$LOGS/experiment-%t.log" --error="$LOGS/experiment-%t.log" \
    bash "$SCRIPT_DIR/train_stereo_four_experiments.sh" --worker
