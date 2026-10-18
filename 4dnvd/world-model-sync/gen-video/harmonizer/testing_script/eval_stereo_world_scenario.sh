#!/usr/bin/env bash
set -euo pipefail
METHOD="${1:?baseline|ours}" MODE="${2:?0|1|2}" CHECKPOINT="${3:?checkpoint}" GPUS="${4:?GPUs on evaluation node}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
ROOT="$(cd ../.. && pwd)"
export COSMOS_MODEL_SIZE=0.6B
export COSMOS_CHECKPOINT_DIR="${COSMOS_CHECKPOINT_DIR:-$ROOT/weights/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image}"
PIXI="${PIXI:-/nfshomes/jlin1219/.pixi/bin/pixi}"
OUT="$ROOT/out_stereo_standard_eval10_2dir_512_v1"
run() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
# The same entry point as monocular evaluation, with paired stereo data and raw weights.
# One GPU owns the complete model evaluation; concurrent launchers share its output lock.
run "$PIXI" run --locked python testing_script/world_scenario_outputs.py metrics-cache
run "$PIXI" run --locked python testing_script/standard_eval.py single \
    --setting stereo --method "$METHOD" --mode "$MODE" --checkpoint "$CHECKPOINT" \
    --output-root "$OUT" --wait-for-dataset
run "$PIXI" run --locked python testing_script/standard_eval.py summary --setting stereo
