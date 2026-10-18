#!/usr/bin/env bash
set -euo pipefail

ROOT=/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm
COLLEAGUE=/fs/nexus-projects/AD_dashrecon/phucnda
DATASET="$ROOT/data/dh_train/smoke_nuscenes_dr2m"
PIXI=/nfshomes/jlin1219/.pixi/bin/pixi

cp "$COLLEAGUE/drivenvs/feature-match/UFM/src/preprocess_cosmos_render.py" \
   "$ROOT/feature-match/UFM/src/"
cp "$COLLEAGUE/drivenvs/feature-match/UFM/src/README_HDMap.MD" \
   "$ROOT/feature-match/UFM/src/"

mkdir -p "$ROOT/maps"
cp -a "$COLLEAGUE/maps/nuscenes" "$ROOT/maps/"

cd "$ROOT/feature-match/UFM"

mapfile -t SCENES < <(
    python - "$DATASET" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1]) / "front_source"
names = sorted({
    directory.name.split("_", 1)[1]
    for directory in source.iterdir()
    if directory.is_dir()
})
print("\n".join(names))
PY
)

INDICES=()
for ((i = 0; i < ${#SCENES[@]}; i++)); do
    INDICES+=("$i")
done

"$PIXI" run --locked python src/preprocess_cosmos_render.py \
    --config src/config/double_reprojection/nuscene_smoke.yaml \
    --data-root "$ROOT" \
    --splits train val \
    --log-names "${SCENES[@]}" \
    --scene-indices "${INDICES[@]}" \
    --frames 100 \
    --stride 1 \
    --image-size 512 288 \
    --output-dir "$DATASET" \
    --frames-only \
    --save-boxes \
    --no-rgb-boxes

python - "$DATASET" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for source in sorted((root / "front_source").iterdir()):
    if not source.is_dir():
        continue
    scene = source.name.split("_", 1)[1]
    for category in ("hdmap", "world_scenario"):
        parent = root / category
        target = parent / f"original_{scene}"
        assert target.is_dir(), target
        assert len(list(target.glob("*.png"))) == 100, target
        link = parent / source.name
        link.symlink_to(target.name, target_is_directory=True)

print("Finished: map/box control images linked to all training directions.")
PY