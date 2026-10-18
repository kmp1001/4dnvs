#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
ROOT="$(cd ../.. && pwd)"
OUT="$ROOT/data/dh_train/nuscenes_stereo_world_scenario_train100_test20"
PIXI=/nfshomes/jlin1219/.pixi/bin/pixi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${1:-}" == --worker ]]; then
    OFFSET="${2:?shard offset}"
    exec "$PIXI" run --locked python src/generate_stereo_world_scenario.py regenerate-conditions \
        --output "$OUT" --rank "$((SLURM_PROCID + OFFSET))" --workers 4 --parallel 16
fi
"$PIXI" run --locked python - "$OUT" "$SCRIPT_DIR/preprocess_cosmos_render.py" <<'PYINIT'
import hashlib
import json
import shutil
import sys
from pathlib import Path
out, renderer = Path(sys.argv[1]), Path(sys.argv[2])
plan = json.loads((out / "selection.json").read_text())
checks = out / "checks"
checks.mkdir(exist_ok=True)
progress = out / "progress/conditions_regeneration"
progress.mkdir(parents=True, exist_ok=True)
for path in progress.glob("*.json"):
    path.unlink()
(checks / "conditions_regeneration_audit.json").unlink(missing_ok=True)
inputs = [out / "selection.json", *[out / "configs" / f"{split}_{direction}.yaml" for split in ("train", "test") for direction in ("LR", "RL")]]
record = {
    "renderer_path": str(renderer),
    "renderer_sha256": hashlib.sha256(renderer.read_bytes()).hexdigest(),
    "scenes": {split: len(scenes) for split, scenes in plan["splits"].items()},
    "cameras": ["pcam_f0", "pcam_l0", "pcam_r0"],
    "frame_count_per_camera": {split: sum(min(s["frame_limit"], s["native_frames"]) for s in scenes) for split, scenes in plan["splits"].items()},
    "preserved_inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    "outputs": ["hdmap", "world_scenario"],
    "status": "running",
}
(checks / "conditions_regeneration_run.json").write_text(json.dumps(record, indent=2) + "\n")
shutil.copyfile(renderer, checks / "conditions_renderer_used.py")
PYINIT
mkdir -p "$OUT/logs/conditions_regeneration"
srun --jobid=7865814 --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=16 \
    --cpu-bind=none --kill-on-bad-exit=1 --label \
    bash "$SCRIPT_DIR/regenerate_stereo_conditions.sh" --worker 0 > "$OUT/logs/conditions_regeneration/nodes_13_14.log" 2>&1 &
PID_A=$!
srun --jobid=7865797 --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=16 \
    --cpu-bind=none --kill-on-bad-exit=1 --label \
    bash "$SCRIPT_DIR/regenerate_stereo_conditions.sh" --worker 2 > "$OUT/logs/conditions_regeneration/nodes_18_19.log" 2>&1 &
PID_B=$!
STATUS=0
wait "$PID_A" || STATUS=1
wait "$PID_B" || STATUS=1
if (( STATUS != 0 )); then exit "$STATUS"; fi
exec "$PIXI" run --locked python src/generate_stereo_world_scenario.py audit-conditions --output "$OUT"
