# Stereo RGB and World Scenario generation

User-confirmed settings: 100 physical training scenes from official nuScenes
train, 20 disjoint physical test scenes from official val (annotations available),
100 real consecutive frames per scene, both LR and RL, seed 42, 800×450.

The four source configs remain unchanged:

- `config/stereo/train_nuscene_LR.yaml`
- `config/stereo/train_nuscene_RL.yaml`
- `config/stereo/test_nuscene_LR.yaml`
- `config/stereo/test_nuscene_RL.yaml`

`generate_stereo_world_scenario.py prepare` records their SHA256s, selects scenes
with all three cameras and at least 100 frames, and writes effective config copies.
Those copies set the actual shared filesystem roots, the requested scene/split/
frame selection, and one GPU per worker. Rendering, SAM3, DA3,
motion classification, reference ranking and camera geometry use existing code.
No model weights, losses, or original YAML files are modified.

RGB preprocessing uses `preprocess.py` and requires CUDA for DA3/SAM3 and point
rendering. Native map/box rendering uses `preprocess_cosmos_render.py` on CPU.

| Direction | RGB source / reference camera | Target / warped / history camera |
| --- | --- | --- |
| LR | PCAM_F0 | PCAM_R0 |
| RL | PCAM_F0 | PCAM_L0 |

The original RL config places PCAM_L0 on the left side and PCAM_F0 on the right
side; its RL operation therefore renders **front → front-left**, not left → front.
World Scenario images must follow the camera that produced the corresponding
RGB, not just its frame filename. Each scene has 100 maps for all three cameras.

This is paired stereo data: `target_image` is always a real image saved under
`front_target`, never a source-view proxy. The manifest labels its ground-truth
role explicitly and audits source/target camera IDs against reference-ranking
metadata. Source/target calibration files and target-frame box JSONs are required.

Pre-run checks on scene-0008 verified exact pixel equality with resized native
target-camera RGB at frames 0/50/99 in both directions (six checks). All 100 frames
of all three map cameras have annotations. Both directions retain 100/100 records
with the existing four-reference filter. Actual Stage-II training-loader checks
also passed for five target/history frames, four source references, the matching
condition tensors, and parsed camera intrinsics/extrinsics and signed time offsets.

Output root (relative to the repository root):

`data/dh_train/nuscenes_stereo_world_scenario_train100_test20/`

```text
selection.json
configs/{train,test}_{LR,RL}.yaml
train/{front_source,front_target,front_warped,confident_coverage,top20_reference,calibration,metadata}/
test/{front_source,front_target,front_warped,confident_coverage,top20_reference,calibration,metadata}/
{train,test}/conditions/{pcam_f0,pcam_l0,pcam_r0}/world_scenario/original_scene-XXXX/00000.png
{train,test}/conditions/{pcam_f0,pcam_l0,pcam_r0}/hdmap/original_scene-XXXX/00000.png
{train,test}/conditions/{pcam_f0,pcam_l0,pcam_r0}/videos/scene-XXXX/metadata.json
dataset_train.json
dataset_test.json
DATASET_AUDIT.json
logs/worker-N.log
progress/worker_N.json
```

The condition renderer also saves clean RGB previews and native global box JSONs.
World Scenario uses map + filled depth-tested 3D cuboids; HDMap uses wireframe boxes.
No images are padded from the last frame. Original `target_steps=200` is explicitly
overridden to the user-requested 100 frames in the generated configs.

The existing manifest builder selects four visibility-ranked source references,
excluding the matching source frame and requiring >1000 visible pixels. Frames
without four qualifying references are omitted from the manifests, while their
rendered RGB/map files remain present. Final audit requires both directions of
all 100/20 selected scenes to have retained samples.

Each record contains `world_scenario_image` for the target camera,
`source_world_scenario_image` for the front source camera, and the ordered
`reference_world_scenario_images` list for all four reference RGB paths. Current
and historical target maps share the target-camera directory. Training and
inference data loaders now honor the explicit reference list; old same-view
double-reprojection manifests keep their existing mapping behavior.

From `feature-match/UFM`:

```bash
PIXI=/nfshomes/jlin1219/.pixi/bin/pixi
$PIXI run --locked python src/generate_stereo_world_scenario.py prepare
bash src/generate_stereo_world_scenario.sh 7865797
```

The shell launcher uses two nodes, four independent GPU workers per node, one
scene shard per GPU. A worker processes both directions and all three camera
maps for each physical scene; the three CPU map calls run concurrently within
its four-CPU allocation. It records a completed scene only after all RGB
products exist and every map-rendering command succeeds. Rerunning the same
worker layout skips these completed scenes. Finalization validates control
PNGs, annotated frame metadata, scene coverage, RGB paths and map associations,
then emits the two manifests and audit. There is no training or evaluation launch.

`worker --maps-only` reuses already generated RGB products and runs only the map
stage. The same RGB file-count checks still apply before map rendering.
