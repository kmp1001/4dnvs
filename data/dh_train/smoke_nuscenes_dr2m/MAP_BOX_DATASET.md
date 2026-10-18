# Map and box outputs / 地图与 3D 框数据

Checked on 2026-10-15. Renderer: `feature-match/UFM/src/preprocess_cosmos_render.py`.

## Output contents

| Path | Contents |
| --- | --- |
| `hdmap/original_scene-XXXX/00000.png` | Original front-camera projection of native Arrow HD map geometry plus class-colored wireframe 3D boxes. This image already contains both map and boxes. |
| `world_scenario/original_scene-XXXX/00000.png` | Same camera and frame, with styled map markings and filled 3D boxes. Road/box annotation geometry shares a depth buffer. This is an alternative control rendering, not RGB or BEV. |
| `videos/scene-XXXX/00000.png` | Native source front-camera RGB resized to 512x288. Export metadata confirms `source_rgb_box_overlay=false` for the verified scenes. |
| `videos/scene-XXXX/metadata.json` | Scene, native camera calibration, output image size, frame iteration/timestamp, source/target camera-to-world poses, render view, and frame-directory information. Calibration width/height describe the native sensor; output size is in `image_size_wh`. |
| `videos/pinhole_pcam_f0/000197_scene-0300_boxes/000002.json` | Example per-frame global 3D annotations. `pinhole` is the camera projection model; `PCAM_F0` is the front camera. The six-digit prefix is the renderer scene index, not a timestamp or frame number. The JSON filename uses six digits, corresponding to five-digit PNG `00002.png`. |

The box JSON includes `boxes_available`, `box_count`, `iteration`, `timestamp_us`, camera poses and `boxes`. Each box has a track token, semantic label, center, dimensions in length/width/height order, quaternion in w/x/y/z order, and eight global 3D corners. These are native Arrow annotations, not SAM3 predictions or 2D pixel bounding boxes. An annotated box need not be visible in the front camera.

Verified metadata specifies double reprojection enabled, the original camera view, and zero target-camera offset. The four training directions therefore share `original_scene-XXXX` control images. No direction symlinks are needed for the new manifest: it points directly to these files. Images are perspective camera views, not top-down maps. The world-scenario depth buffer uses annotation geometry, not measured image depth.

PNG-only export explains why the folder is named `videos` while these outputs contain images/JSON rather than MP4 files. The original preprocessing folders `front_source`, `front_warped`, `confident_coverage`, `top20_reference`, and `calibration` remain the RGB targets, degraded inputs, coverage, reference rankings and camera data respectively.

## Verified manifest

Primary manifest, updated in place with ALL original records retained:
`dataset_fixed_train100_4dir_seed42.json`

Per user preference, this primary manifest keeps all 24,188 train records and 80 test records, along with unchanged selection metadata. Only 7,960 train records (34 scenes) and 40 test records (2 scenes) have the four new map/box path fields. The remaining 16,228 train and 40 test records are unchanged and do not have these fields. There are no empty or fabricated paths. Original bytes are backed up as `dataset_fixed_train100_4dir_seed42.json.before_map_box`.

Optional subset created earlier with complete map/box paths:
`dataset_fixed_train34_4dir_seed42_map_box.json`

| Split | Original records | Retained records | Retained physical scenes |
| --- | ---: | ---: | ---: |
| train | 24,188 | 7,960 | 34 |
| test | 80 | 40 | 2 |

Each train direction D2M/L2M/R2M/U2M has 1,990 records. Test records retain the original split assignment: 20 D2M and 20 U2M records, from scene-0010 and scene-0130. Filtering did not resample scenes, references, frames, or validation targets. Every retained original record field is unchanged. `selection.scene_ids` lists the 34 retained training scenes; `selection.map_box` records provenance.

New per-record absolute-path fields:

```json
{
  "hdmap_image": ".../hdmap/original_scene-0300/00002.png",
  "world_scenario_image": ".../world_scenario/original_scene-0300/00002.png",
  "boxes_json": ".../videos/pinhole_pcam_f0/000197_scene-0300_boxes/000002.json",
  "map_box_metadata": ".../videos/scene-0300/metadata.json"
}
```

Coverage details, including all 66 missing training scenes and their record counts:
`dataset_fixed_train100_4dir_seed42_map_box_coverage.json`.

At inspection, map directories covered 224 physical scenes through scene-0359: 223 had 100 PNGs in each control category; scene-0359 had 96 PNGs per category and no final metadata. Scene numbers are not counts. This incomplete scene is not part of the retained training subset.

For all 36 retained train/test scenes, all 100 frames were checked: 3,600 HD map PNGs, 3,600 world-scenario PNGs and 3,600 box JSONs. PNGs decoded/verified at 512x288. Box availability, box counts, frame filenames/iterations, timestamps and source/target camera poses matched renderer metadata. All retained path fields point to existing files. Nine RGB comparisons across three training scenes also supported same-frame alignment; mean RGB differences were approximately 1.2-1.8/255 because the RGB resizing implementations differ.

The primary manifest retains all 100 original training scenes. Map/box path fields are absent for the remaining 66 training scenes; generating those is required for complete map/box coverage. The optional 34-scene subset contains only complete records.

## Training integration status

Only data/manifest files were added. `gen-video/harmonizer/src/utils/training_utils.py:DrivenVSDataset.__getitem__` currently loads warped RGB, target RGB, coverage, references, and optional camera/time data. It does not load or send the new map/box fields to the model. The new manifest is indexed data, not an enabled map-conditioning training implementation. No training, model, configuration or evaluation code was changed.
