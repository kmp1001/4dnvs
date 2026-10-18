# Refactored preprocessing pipeline

This package refactors the original dataset and preprocessing scripts into a cleaner module layout.

## Run

```bash
# NuScenes front <-> front-right
python preprocess.py --config config/stereo/nuscene.yaml

# Argoverse 2 stereo left <-> stereo right
python preprocess.py --config config/stereo/av2.yaml

# Shifted monocular front camera -> virtual cameras 2 metres left/right/up
python preprocess.py --config config/monocular_shift/nuscene.yaml
python preprocess.py --config config/monocular_shift/av2.yaml
python preprocess.py --config config/monocular_shift/kitti360.yaml

# Correspondence-inverted round trip: source -> left/right/up/down 2 m -> source
python preprocess.py --config config/double_reprojection/nuscene.yaml
python preprocess.py --config config/double_reprojection/av2.yaml
python preprocess.py --config config/double_reprojection/kitti360.yaml
```

You can also override config values directly from the command line:

```bash
python preprocess.py --config config/stereo/av2.yaml dataset.target_steps=80 dataset.frame_sampling=2 dataset.render_directions='[LR, RL]' render.render_moving=false
```

## Frame sampling

Temporal frame sampling is controlled by:

```yaml
dataset:
  target_steps: 40
  frame_sampling: 2
```

`frame_sampling` is the native-frame stride. For example, `frame_sampling: 2` reads native frames `0, 2, 4, ...`; `frame_sampling: 1` keeps every native frame. The default configs use:

- NuScenes: `frame_sampling: 1`
- AV2: `frame_sampling: 2`
- KITTI-360: `frame_sampling: 1`

Sampling affects only which native frames are read. After loading, every saved
artifact uses a contiguous zero-based timeline. For example, native frames
`0, 2, 4, 6` are stored as `00000.png`, `00001.png`, `00002.png`, and
`00003.png`. Calibration, metadata, coverage, masks, and reference JSON files
use the same `0, 1, 2, 3` indices. The recorded `frame_sampling` value remains
`2`, so the acquisition stride is still known.

## Source preprocessing reuse

DA3 depth estimation, SAM3 object/sky segmentation, static-versus-dynamic
classification, sky handling, and source-point-cloud construction are cached by
physical source stream within each scene sample. Target-camera rendering remains
independent.

Consequently, `L2M`, `R2M`, and `U2M` share one source preprocessing run in both
`monocular_shift` and `double_reprojection`; `D2M` shares that same run when
enabled by `double_reprojection`. For stereo, repeated renders from the same
physical source camera also share preprocessing. `LR` and `RL` require separate
preprocessing because they reverse which physical camera is the source.

All shipped configs use:

```yaml
render:
  max_depth: 300.0
```

The configured sky depth remains `1000.0`, so sky points are still excluded by
the 300-metre point-cloud cutoff.

## Monocular shifted-camera generation

The shifted-monocular setup uses only one physical camera stream:

```yaml
dataset:
  setup: monocular_shifted
  source_cam_id: camera.pcam_f0
  render_directions:
  - L2M
  - R2M
  - U2M

monocular_shift:
  distance_m: 2.0
```

There is no physical `front_target` video and the pipeline does not create a
`front_target` output directory. Each direction retains the source camera's
orientation and intrinsics but translates its virtual target pose in local
OpenCV camera coordinates (`+x` right, `+y` down, `+z` forward):

- `L2M`: `(-2, 0, 0)` metres, two metres left.
- `R2M`: `(+2, 0, 0)` metres, two metres right.
- `U2M`: `(0, -2, 0)` metres, two metres up.

The generated folders follow the same direction-prefixed layout as stereo:

```text
front_source/L2M_<scene.log_name>/00000.png
front_warped/L2M_<scene.log_name>/00000.png
confident_coverage/L2M_<scene.log_name>/00000.png

front_source/R2M_<scene.log_name>/00000.png
front_warped/R2M_<scene.log_name>/00000.png

front_source/U2M_<scene.log_name>/00000.png
front_warped/U2M_<scene.log_name>/00000.png
```

Source and virtual-target calibration files are still saved. The virtual target
calibration records its parent camera and local translation explicitly.

Saved frame filenames always use the contiguous output index. Therefore AV2
with `frame_sampling: 2` still writes `00000.png`, `00001.png`,
`00002.png`, ... . For compatibility, metadata fields named
`native_frame_index` mirror this same stored output index rather than exposing
the skipped native numbering.

## Double reprojection

The configs in `config/double_reprojection/` render all four virtual shifts
(`L2M`, `R2M`, `U2M`, and `D2M`). `D2M` translates the virtual camera two
metres down with local OpenCV-camera offset `(0, +2, 0)`. The virtual view is
first rendered only from the same-frame point cloud. That local render returns
to the source grid with dense source-to-virtual coordinates. Its returned pixels
are then locked, and the nonlocal static global PCD is directly splatted into
the remaining source-camera holes, matching the local-first/global-second policy
used by monocular shift.

For source frame `t`, let `P_t(x)` be the point at source pixel `x`, `C'_t` the
shifted virtual camera, and `G_t` the nonlocal static global PCD. The local
virtual render is:

```text
(I'_L, Z'_L, A'_L) = Splat(P_t, C'_t)
```

The local portion returns with the dense source-coordinate gather. Let `Z_t(x)`
be the original source-camera depth and `q_t(x)` the continuous projection of
`P_t(x)` into the virtual camera:

```text
q_t(x), z'_t(x) = Project(P_t(x), C'_t)
N_t(x) = the four bilinear raster neighbours of q_t(x)
V_t(x) = {v in N_t(x) | A'_L(v)=1 and z'_t(x) matches Z'_L(v)}
v*(x) = the member of V_t(x) with the largest bilinear footprint
I''_L(x) = I'_L(v*(x)),  Z''_L(x)=Z_t(x),  A''_L(x)=1
```

If `V_t(x)` is empty, the local return stays black and `A''_L(x)=0`. The RGB is
copied from an actual first-pass pixel, so the return step performs no second
color blend. The returned depth is the source-camera depth `Z_t(x)`, not the
virtual-camera depth. Testing every source point avoids the range loss caused
by retaining only one winning source ID per virtual pixel while preserving the
first z-buffer's occlusion decision. After that local return, the global PCD is
directly splatted into the original source camera:

```text
(I''_G, Z''_G, A''_G) = Splat(
    G_t,
    source_camera,
    allowed = not A''_L and not source_dynamic_mask,
)
```

Only static points originating from other frames enter this global z-buffer.
Candidates hidden by a current-frame dynamic object can block depth but cannot
add RGB, preventing farther static surfaces from filling through moving
objects. Final coverage is `A'' = A''_L or A''_G`; local pixels always win and
dynamic-caused holes remain black. This path is enabled only by the
`double_reprojection.enabled` branch, so regular monocular-shift and stereo
behavior is unchanged.

Enable the mode with:

```yaml
double_reprojection:
  enabled: true
```

For each direction, the main products are:

```text
front_warped/L2M_<scene>/00000.png
confident_coverage/L2M_<scene>/00000.png

front_warped/R2M_<scene>/00000.png
front_warped/U2M_<scene>/00000.png
front_warped/D2M_<scene>/00000.png
```

The virtual view is retained only long enough for the local gather and static
global return and is not saved. In double-reprojection mode, `front_warped` and
`confident_coverage` are both aligned with the original source-camera pixel grid.

Run the focused CPU smoke test in an environment with PyTorch installed:

```bash
python tests/test_double_reprojection.py
```

## Stereo direction generation

The source/target camera direction is controlled by:

```yaml
dataset:
  render_directions:
  - LR
  - RL
```

Supported values are `LR`, `RL`, `both`, or a YAML list such as `[LR, RL]`.

- `LR` renders left/source camera into right/target camera.
- `RL` renders right/source camera into left/target camera.

For NuScenes, `source_cam_id` is treated as the left/source side and `target_cam_id` is treated as the right/target side. For AV2 and KITTI-360, `left_cam_id` and `right_cam_id` are used directly.

Saved scene folders are direction-prefixed and use the human-readable `scene.log_name` instead of the py123d UUID:

```text
front_source/LR_<scene.log_name>/00000.png
front_target/LR_<scene.log_name>/00000.png
front_warped/LR_<scene.log_name>/00000.png
confident_coverage/LR_<scene.log_name>/00000.png
seginstance_warped/LR_<scene.log_name>/00000.png

front_source/RL_<scene.log_name>/00000.png
front_target/RL_<scene.log_name>/00000.png
front_warped/RL_<scene.log_name>/00000.png
confident_coverage/RL_<scene.log_name>/00000.png
seginstance_warped/RL_<scene.log_name>/00000.png
```

If a log name contains path separators or other filesystem-unfriendly characters, the output folder uses a sanitized version. The metadata still records the original `scene.log_name`.

`seginstance_warped` stores a raw `uint16` PNG where `0` means background/no SAM3 instance and positive values are SAM3 track IDs. It is produced by splatting source instance IDs through the same projection, z-buffer, and splatting code path used by `front_warped`.

By default, `seginstance_warped` now follows the same visibility rule as the RGB warped image:

```yaml
render:
  visibility_mode: target_frustum_static_dynamic_same_frame
  seginstance_object_mode: same_as_warped
  seginstance_visibility_mode: same_as_warped
```

This means:

- Static SAM3 instances, such as parked/static vehicles, can use past/current/future source points if they project into the target camera frustum.
- Dynamic SAM3 instances use only the matching source timestep for the target timestep.

Supported `render.seginstance_object_mode` values:

- `same_as_warped` / `all_instances`: keep all non-sky SAM3 instance IDs and let `seginstance_visibility_mode` decide how they render. This is the default.
- `dynamic_only`: keep only IDs on the refined dynamic mask. This matches the older behavior.
- `static_only`: keep only IDs on the static mask.

Supported `render.seginstance_visibility_mode` values:

- `same_as_warped`: use exactly the same point visibility as `front_warped`; with the default RGB visibility mode, static masks are temporal and dynamic masks are same-frame.
- `same_frame`: use only source frame `i` for target frame `i`.
- `dynamic_temporal`: use all sampled source frames that project into the target frame, including past and future frames.
- `past_to_current`: use source frames `j <= i` if they project into target frame `i`.
- `target_frustum_all`: alias for `dynamic_temporal`.
- `target_frustum_past_to_current`: alias for `past_to_current`.

`seginstance_max_past_frames: null` means all previous sampled frames are allowed when using `past_to_current`. For example, `seginstance_max_past_frames: 5` only carries instance points from the last five sampled frames.

For backward compatibility, older overrides using `seginstance_warped.visibility_mode` still work, but the preferred config field is now `render.seginstance_visibility_mode`.

Optional outputs such as `dynamic_masks`, `sky_masks`, and `metadata` use the same `LR_...` / `RL_...` scene-folder convention when enabled.

## Static-only global hole filling

RGB rendering now commits the current-frame point cloud first. Pixels covered by
that local render are locked and are not admitted to the second z-buffer. The
second pass considers only static points originating from other source frames.

For every nonlocal static candidate, the renderer reprojects the point into the
current source camera. A candidate is classified as dynamically occluded when it
lands on the current dynamic mask and lies behind the current observed depth. A
dynamically occluded candidate participates in target-view depth selection but
cannot contribute RGB or depth. This prevents a farther background point from
filling through a hole caused by a moving foreground object.

```yaml
render:
  global_fill_static_holes_only: true
  dynamic_hole_depth_abs_tolerance: 0.10
  dynamic_hole_depth_rel_tolerance: 0.02
  dynamic_hole_mask_dilation_radius: 0
  invalid_dynamic_depth_is_occluded: true

save:
  hole_masks: true
```

The diagnostic masks use the same scene/frame layout as the RGB outputs:

```text
static_holes/LR_<scene.log_name>/00000.png
dynamic_holes/LR_<scene.log_name>/00000.png
```

- `dynamic_holes`: locally uncovered pixels whose front global candidate was
  hidden by a current-frame dynamic object. These pixels remain black.
- `static_holes`: all remaining locally uncovered pixels. Nonlocal static PCD
  points may fill these pixels when geometry is available.

Set `render.global_fill_static_holes_only: false` to restore the previous full
global second pass.

Run the CPU smoke test in an environment with PyTorch installed:

```bash
python tests/test_static_hole_fill.py
```

## Top-20 source reference ranking

Every warped frame can store its highest-ranked source reference frames under:

```text
top20_reference/LR_<scene.log_name>/00000.json
top20_reference/RL_<scene.log_name>/00000.json
```

Enable the output and choose the number of references with:

```yaml
reference:
  top_k: 20

save:
  top20_reference: true
```

The score is an occlusion-aware dominant-pixel count. Each source point keeps its sampled source-frame index. That index is projected into the target frame through the same configured visibility, 4-neighbor splatting, depth weighting, and z-buffer test used by the renderer. Every covered target pixel is attributed to its dominant surviving source frame, and source frames are ranked by the number of attributed pixels from highest to lowest.

The matching source timestep is included. Equal scores are resolved deterministically by preferring the source timestep closest to the warped timestep and then the lower sampled frame index. If the sequence has fewer than 20 sampled frames, all available frames are returned.

For double reprojection, ranking uses the final source-coordinate composite:
locally returned pixels are credited to the matching source timestep, while
global static-fill pixels are credited to the source frame that actually wins
the final source-camera z-buffer. The same `top_k` behavior is then used for
double reprojection, monocular shift, and stereo; zero-contribution frames are
used only when fewer than `top_k` frames have positive coverage.

Each JSON contains the warped frame path and a `reference_frames` list. Source paths are relative to the dataset output root, for example:

```json
{
  "warped_path": "front_warped/LR_scene_name/00020.png",
  "source_calibration_path": "calibration/LR_scene_name/source_calibration.txt",
  "target_calibration_path": "calibration/LR_scene_name/target_calibration.txt",
  "ranking_metric": "occlusion_aware_dominant_pixel_count",
  "ranking_order": "highest_first",
  "top_k_requested": 20,
  "reference_frames": [
    {
      "rank": 1,
      "source_output_frame_index": 20,
      "source_native_frame_index": 20,
      "source_path": "front_source/LR_scene_name/00020.png",
      "visible_pixel_count": 428103,
      "visibility_score": 0.539
    }
  ]
}
```

`visibility_score` is the frame's dominant-pixel count divided by the total dominant-pixel count over all sampled source frames for that warped frame.

## Source and target calibration files

Each scene and render direction saves one source calibration file and one target calibration file:

```text
calibration/LR_<scene.log_name>/source_calibration.txt
calibration/LR_<scene.log_name>/target_calibration.txt

calibration/RL_<scene.log_name>/source_calibration.txt
calibration/RL_<scene.log_name>/target_calibration.txt
```

Enable or disable them with:

```yaml
paths:
  calibration_subdir: calibration

save:
  calibration: true
```

Each text file contains every sampled frame in the scene-direction output. For each frame it records:

- `output_frame_index`
- `native_frame_index` (legacy compatibility key; mirrors `output_frame_index`)
- resized pixel-space intrinsics `K` as a 3x3 row-major matrix
- camera-to-world extrinsics `T_world_cam` as a 4x4 row-major matrix
- world-to-camera extrinsics `T_cam_world` as the explicit inverse 4x4 matrix

Both frame-index fields follow the contiguous stored timeline regardless of the
native sampling stride. The saved poses are the dataset loader's camera poses
before the optional global `render.coord_flip_xy` convention transform. The
header records both the actual `frame_sampling` stride and the contiguous index
policy. Each top-20 reference JSON also contains `source_calibration_path` and
`target_calibration_path` relative to the dataset output root.


## Scene names

Each dataset item now includes both the original UUID and the easier-to-read log name:

```python
item["scene"] = {
    "uuid": "...",
    "log_name": "...",
}
item["scene_uuid"] = item["scene"]["uuid"]
item["scene_log_name"] = item["scene"]["log_name"]
```

`preprocess.py` prints `scene.log_name` in logs and uses it for output scene folders. The UUID is kept only in metadata for traceability.

## Multi-GPU scene sharding

Multi-GPU processing is controlled by:

```yaml
runtime:
  gpu: 1
```

`gpu: 1` keeps the original single-process behavior. When `gpu >= 2`, `preprocess.py` launches one process per GPU and assigns each process a contiguous shard of scene indices. For example, if the dataset has `n` scenes and `runtime.gpu: 2`:

```text
cuda:0 -> scene indices [0, ..., n//2 - 1]
cuda:1 -> scene indices [n//2, ..., n - 1]
```

For more than two GPUs, the split uses floor boundaries:

```python
start = floor(rank * n / gpu)
end = floor((rank + 1) * n / gpu)
```

You can enable it from the CLI as well:

```bash
python preprocess.py --config config/stereo/av2.yaml runtime.gpu=2
```

If `dataset.batch_indices` is set, the same contiguous sharding is applied over that selected index list instead of the full dataset. In multi-GPU mode, `dataset.shuffle_dataloader` is ignored so the shards remain deterministic.

Outputs are safe to share across workers because scene folders are already unique and direction-prefixed, e.g. `LR_<scene.log_name>` and `RL_<scene.log_name>`. Metadata includes `worker_id`, `num_workers`, and `global_scene_index`.

## Layout

```text
refactored_preprocess/
├── preprocess.py
├── dataset.py
├── config/
│   ├── stereo/
│   │   ├── default.yaml
│   │   ├── nuscene.yaml
│   │   ├── av2.yaml
│   │   └── kitti360.yaml
│   └── monocular_shift/
│       ├── nuscene.yaml
│       ├── av2.yaml
│       └── kitti360.yaml
├── foundation/
│   ├── da3.py
│   └── sam3.py
└── utils/
    ├── calib.py
    ├── pointcloud.py
    ├── utils.py
    └── viz.py
```

## Notes

- Configs are YAML instead of JSON, so they are easier to edit and can include comments.
- `preprocess.py` accepts both NuScenes-style camera keys, `source_cam_id` / `target_cam_id`, and stereo/AV2-style keys, `left_cam_id` / `right_cam_id`.
- Inpainted outputs and the legacy `use_virtual_target` flag remain removed;
  shifted virtual cameras are selected explicitly with
  `dataset.setup: monocular_shifted`.
- Stereo target poses come from the selected physical target camera. In
  `monocular_shifted`, the target poses are translated virtual copies of the
  physical source-camera poses.
- Metadata files are still written as `.json` because they are machine-readable outputs, not user-edited configs.

## Dependency

The YAML loader uses PyYAML:

```bash
pip install pyyaml
```


## Output naming

The canonical image-output folder keys are now:

```yaml
paths:
  front_source: front_source
  front_target: front_target  # stereo only; omitted for monocular_shifted
  front_warped: front_warped
  confident_coverage_subdir: confident_coverage
  static_hole_subdir: static_holes
  dynamic_hole_subdir: dynamic_holes
  seginstance_warped_subdir: seginstance_warped
  top20_reference_subdir: top20_reference
  calibration_subdir: calibration
```

Optional debug outputs can be turned off without changing the subdir names:

```yaml
save:
  hole_masks: false
  seginstance_warped: false
  top20_reference: false
  calibration: false
  vehicle_masks: false
  sky_masks: false
  metadata: false
```

When a save flag is `false`, that directory is not created and no files for that output type are written.


## Composite visualization video

Visualization is a separate option from the image and metadata `save` settings. It
is disabled by default in every stereo and monocular-shift config. Enable it with:

```yaml
visualize:
  enabled: true
  output_subdir: visualization
  fps: 10
  overlay_alpha: 0.50
  draw_ids: true
```

This writes one MP4 for each processed scene and render direction:

```text
visualization/<direction_scene_log_name>.mp4
```

Each video frame uses this 2x3 layout:

```text
front source                       | front warped                 | confident coverage
dynamic-instance overlay on source | seginstance-warped overlay   | blank
```

The six tiles are never resized. For input frames of `H x W`, the composite video
is encoded at exactly `2H x 3W` and 10 FPS. Turning `visualize.enabled` off prevents
the visualization video from being written; it does not change any existing
`save.*` behavior. A root-level `visualize: true` override is also accepted and
uses the defaults shown above.


## Warped instance overlay in metadata

The raw warped instance ID maps are saved as `uint16` PNG files under:

```text
seginstance_warped/<direction_scene_log_name>/<frame>.png
```

A visual overlay can also be saved inside each metadata scene folder:

```yaml
save:
  seginstance_warped_overlay: true

metadata:
  seginstance_warped_overlay_background: warped  # warped or target
  seginstance_warped_overlay_alpha: 0.50
  seginstance_warped_overlay_draw_ids: true
```

This writes:

```text
metadata/<direction_scene_log_name>/seginstance_warped_overlay/<frame>.png
```

Each per-frame metadata JSON records `seginstance_warped_overlay_enabled`,
`seginstance_warped_overlay_background`, and `seginstance_warped_overlay_file`.
