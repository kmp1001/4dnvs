# Ordered scene splits with map/box

Created 2026-10-15 from `smoke_nuscenes_dr2m`. Existing dataset manifests and training/evaluation code were not changed.

Physical scene IDs are sorted by scene number across all 685 existing `front_source` scenes. Rank means position in this sorted list, not the numerical suffix of the scene ID. All four render directions of a physical scene stay in the same partition.

| File | JSON split key | Scene ranks (1-based) | Physical scenes | Target records |
| --- | --- | --- | ---: | ---: |
| `dataset_map_box_train100_4dir_ordered.json` | `train` | 1-100 | 100 | 24,256 |
| `dataset_map_box_test20_4dir_ordered.json` | `test` | 101-120 | 20 | 4,844 |
| `dataset_map_box_val80_4scene_4dir_ordered.json` | `test` | 121-124 | 4 | 80 |

Train range: scene-0001 through scene-0171. Test range: scene-0172 through scene-0193. There are gaps in the native scene IDs. Actual scene IDs and ranks are stored in each file's `selection` object.

Train has 6,064 records per direction; test has 1,211 per direction. They retain exactly the target frame indices shared by all four directions under the original visibility-based selection rule: four distinct source reference images, exclude the current frame, visible pixel count strictly greater than 1,000. The existing `training_script/create_dataset.py:build_sample` implementation was reused. Therefore a scene may contribute fewer than its 100 original frames, and retained target frames need not be consecutive. The test JSON is not a guarantee of 20 complete 100-frame videos.

The exact first 100 scenes were retained. Two of them, scene-0021 and scene-0072, have qualified single-frame samples but no qualified five-consecutive-frame window shared by all four directions. Their records remain in train as requested. Existing curriculum code uses eligible windows from other scenes for temporal sampling when a requested record lacks a window. No training logic was changed.

## Validation selection and original validation cost

The original configured validation manifest contains four independent physical scenes, one selected direction per scene, with 20 consecutive target frames each: 80 targets total. Every target has four real preceding GT frames and four source references. Stage II's current monitoring runs both GT-history validation and autoregressive validation on the same 80 targets, making 160 predictions per validation event; Stage I and Stage III each run one 80-target pass. This count is controlled by `--num-samples-eval`; the trainer's generic default is 20 but the experiment launchers explicitly select 80. The loop uses at most the number of available validation records.

The new validation set preserves this size and sampling method, choosing four distinct scenes from rank 121 onward. Within each direction, the first unused scene with a valid 20-consecutive-target window is selected; `random.Random(42)` chooses the window start, as in the original builder.

| Physical scene | Rank | Direction | Target frames | Preceding frames for first target |
| --- | ---: | --- | --- | --- |
| scene-0194 | 121 | D2M | 44-63 | 40-43 |
| scene-0196 | 123 | L2M | 80-99 | 76-79 |
| scene-0195 | 122 | R2M | 7-26 | 3-6 |
| scene-0199 | 124 | U2M | 18-37 | 14-17 |

Validation uses the JSON key `test` to match the current trainer's `--validation-split test`. This is a separate physical scene partition from the new test JSON despite the same internal key. Pass the validation JSON via `--validation-dataset` and use `--num-samples-eval 80` to cover all four windows.

## Map/box fields and checks

All target records in all three JSONs include absolute paths under these fields:

- `hdmap_image`: original-camera HD map plus wireframe box PNG.
- `world_scenario_image`: original-camera map plus filled box PNG.
- `boxes_json`: per-frame native global-coordinate 3D annotations.
- `map_box_metadata`: per-scene frame times and camera calibration/poses.

For the selected 124 scenes, every original frame's map/box paths were checked, box annotation availability and frame/timestamp consistency were verified, and renderer metadata confirmed the original camera and double-reprojection mode. Existing sample construction checked the RGB, coverage, ranking and reference paths. All written target-record paths were checked again. Train/test/validation physical scene sets are disjoint; all references remain in their own target scene; validation windows contain 20 consecutive targets and four valid real predecessors.

These three files all describe double-reprojection data. The new test partition is not the earlier monocular-shift/virtual-camera evaluation setting. Paired target-view reconstruction metrics are meaningful for this double-reprojection data; source RGB still must not be treated as novel-view GT in monocular-shift evaluation.

Adding these path fields does not activate map/box conditioning: the existing training loader/model does not yet consume these controls. No model, training configuration, or evaluator was modified.

Read-only validation-loader checks passed for Stage I, Stage II GT history, Stage II autoregressive, and Stage III. Each dataset exposed all 80 targets. One sample per validation scene was decoded with four references and camera/time embeddings enabled; Stage II GT-history inputs contained five frames, and the other modes contained one frame before generated-history rollout. No model inference or GPU training was launched.
