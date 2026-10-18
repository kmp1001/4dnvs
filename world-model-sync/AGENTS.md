The repo is a mono-repo with multiple sub projects. Each sub project is a sub-folder of the main repo.

This project uses `pixi` for dependency management.

To run python code, cd into the project directory (ex: `cd hello-world`) and then run `pixi run python file.py`

Prefer `pypi` deps when possible and add them using `pixi add --pypi <package>`.
Only use `pixi add` without the `--pypi` flag if the package is not available in pypi and only available in conda.

Use `dltype` for runtime shape checking of tensors. Here is a minimal example of how to annotate a function with tensor shapes:
```python
import torch
    import numpy as np
    from dltype import FloatTensor, dltyped

    @dltyped()
    def transform_tensors(
        points: Annotated[np.ndarray, FloatTensor["N 3"]],
        transform: Annotated[torch.Tensor, FloatTensor["3 3"]] | None, # for an optional tensor
    ) -> Annotated[torch.Tensor, FloatTensor["N 3"]]:
        return torch.from_numpy(points) @ transform
```

Format code with `ruff format file.py`.

Use `viser` for 3D interactive visualizations.

Use `click` for CLIs.

<!-- For monocular_shift evaluation, compute generative FID/KID/FVD only; do not compute PSNR, SSIM, or LPIPS against source frames because matching novel-view GT is unavailable. Use calculate_generative_metrics.py; the requested FVD backend is videogpt. Label front_source images as source-view real references, never novel-view GT. Paired reconstruction metrics require a separate double-reprojection or actual target-camera dataset.

The default monocular evaluation standard is now data/dh_train/nuscenes_standard_eval10_4dir_512_v1, using the same ten held-out physical scenes and fifty consecutive qualified frames in each of D2M/L2M/R2M/U2M for both double_reprojection and shift_1m. Exclude the union of the original two 100-scene training sets. Require four visibility-ranked references, each with strictly more than 1000 visible pixels; exclude the current source only for double reprojection. Controls must match the final output camera; reference controls use the source camera. Infer and save at 512×288, without upscaling. Baseline uses DrivenVSModel with four steps; ours uses MeanFlowDrivenVSModel with one step and raw weights (EMA only if raw is unavailable). Report VideoGPT FVD/FID for both modes and PSNR/SSIM only for double reprojection. Use gen-video/harmonizer/testing_script/run_standard_eval.sh or standard_eval.py single; preserve the intentionally shared camera-conditioning convention.

The default stereo evaluation standard is data/dh_train/nuscenes_stereo_standard_eval10_2dir_512_v1: ten held-out scenes, two real camera directions (LR front→right, RL front→left), the same 50 consecutive jointly eligible frames per scene, 1000 targets / 20 videos. Reconstruct/infer/save/evaluate at 512×288, four references each strictly >1000 visible pixels, allow same-time source references. Exclude original monocular plus stereo training scenes. Use target-camera controls for target/history and source-camera controls for references. Use standard_eval.py --setting stereo, baseline final Stage II raw 4 steps, ours final Stage III raw 1 step. Compute FID/FVD/PSNR/SSIM against real target-camera GT. The user corrected an initial four-direction request to two stereo directions. -->

# Experimental Setup, Evaluation Protocol, and Implementation Requirements

## 1. Resolution and Evaluation Protocol

### 1.1 Current Resolution Inconsistency

There is currently a resolution inconsistency between our model and the original dataset:

- **Model input resolution:** 512 × 288
- **Original/final dataset resolution:** 800 × 450

We should standardize the evaluation protocol as follows:

1. **Model input:** 512 × 288.
2. **Model prediction:** Save the original generated PNG at 512 × 288, without upscaling.
3. **Ground truth (GT):** Resize all GT images to 512 × 288.
4. **Metric computation:** Apply the standard preprocessing associated with each metric's feature extraction network (FID/KID/FVD).

The reconstruction/rendering resolution should be distinguished from the model input, prediction, and evaluation resolutions.

---

## 2. Training Implementation

### 2.1 Core Training Scripts

| Method | Training implementation |
|---|---|
| Baseline | `src/train_drivenvs.py` |
| Ours | `src/train_meanflow_curriculum.py` |

### 2.2 Monocular Setting

**Training datasets (including validation):**

Training:
```text
/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/data/dh_train/smoke_nuscenes_dr2m/dataset_map_box_train100_4dir_ordered.json
```

Validation:
```text
data/dh_train/smoke_nuscenes_dr2m/dataset_map_box_val80_4scene_4dir_ordered.json
```

**Reference training command:**

```bash
bash training_script/train_world_scenario.sh \
  --method ours --mode 2 --nodes 2 --gpus-per-node 4
```

This script supports the complete experimental configuration, including:

- Baseline and our method.
- Experiments with and without world-scenario conditioning.
- Different experimental modes.

### 2.3 Existing Monocular Test Dataset

```text
data/dh_train/nuscenes_standard_eval10_4dir_512_v1
```

This dataset supports two evaluation modes:

1. Double-reprojection.
2. Monocular shift (1.0 m).

For evaluating a single model, refer to:

```bash
pixi run --locked python testing_script/standard_eval.py single \
  --method ours \
  --mode 0 \
  --checkpoint /absolute/path/model.pkl \
  --output-root /absolute/path/results
```

---

## 3. Stereo Setting

### 3.1 Dataset Construction

All conditioning inputs were generated using our existing renderer:

```text
/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/feature-match/UFM/src/preprocess_cosmos_render.py
```

Dataset specifications:

- 100 training scenes + 20 testing scenes.
- Three cameras: `pcam_f0`, `pcam_l0`, and `pcam_r0`.
- 72,000 HDMap and world-scenario PNGs at 800 × 450.

**Important clarification:** All four stereo experiments use 512 × 288 for training, training-time previews, and final inference. The 800 × 450 resolution refers to the rendered conditioning images stored on disk after preprocessing.

**Generated dataset:**

```text
/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/data/dh_train/nuscenes_stereo_world_scenario_train100_test20
```

### 3.2 Stereo Training and Evaluation

The main reference script is:

```text
/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/gen-video/harmonizer/training_script/train_stereo_four_experiments.sh
```

**Script hierarchy:**

```text
training_script/train_stereo_four_experiments.sh
└── training_script/train_world_scenario.sh --setting stereo
    ├── baseline
    │   └── train_nuscenes_smoke_100scene_4dir.sh
    │       └── src/train_drivenvs.py
    │
    └── ours
        └── train_nuscenes_meanflow_three_stage.sh
            └── src/train_meanflow_curriculum.py
```

After training, both methods use:

```text
testing_script/eval_stereo_world_scenario.sh
```

The current evaluation pipeline performs three steps:

1. `src/evaluate_test_dataset.py`: Run inference at 512 × 288 and save the predicted PNGs.
2. Export visualization videos.
3. `reevaluate_stereo_512.py`: Evaluate the original predicted PNGs against 512 × 288 GT images using the existing metric implementations.

---

# 4. New Test Dataset Requirements

I want to construct a new test dataset satisfying the following requirements.

## 4.1 Conditioning Information

The new test dataset must include **HDMap information**, so that it can be used for future comparisons involving map-conditioned models.

If GT 3D bounding boxes are available, we should preserve them as well.

## 4.2 Strict Data Independence

The new test dataset must have **no overlap with either of the two existing training datasets** (monocular and stereo).

This requirement should be enforced at the scene level, not merely at the frame level.

## 4.3 Two Evaluation Modes

Both evaluation modes must be supported and tested.

| Evaluation mode | Metrics |
|---|---|
| Double-reprojection | FID, FVD, PSNR, SSIM |
| Monocular shift (1.0 m) | FID, FVD |

## 4.4 Four-Direction Reference Selection

The evaluation must strictly cover all four directions.

Requirements:

1. Each target frame must have **four valid reference images**.
2. For double-reprojection, exclude the current source frame from the reference candidates.
3. Each selected reference must have a `visible_pixel_count` strictly greater than 1,000.
4. Compute the intersection of valid frames across all four directions.
5. Only select videos whose four-direction valid-frame intersection contains at least **50 consecutive frames**.
6. Evaluate exactly 50 consecutive frames per selected video.
7. Select 10 videos/scenes, each evaluated in four directions.

**Expected evaluation size:**

- 10 videos/scenes.
- 4 directions per video.
- 40 direction-specific video sequences in total.
- 50 consecutive frames per sequence.
- 2,000 evaluated frames per evaluation mode.

The four directions should use the same selected consecutive 50-frame interval within each scene.

**Important issue: Variable-length sequences**

Because reference validity depends on visibility and four-direction intersection, the valid sequence length may differ between scenes.

Previously, some resulting videos contained only approximately six frames, which creates difficulties for FVD and other video-level metrics.

The new dataset construction procedure must explicitly enforce the 50-consecutive-frame requirement rather than allowing arbitrary short sequences.

## 4.5 Unified Evaluation Resolution

The complete evaluation pipeline must follow:

```text
Model inference: 512 × 288
        ↓
Save raw predictions: 512 × 288 PNG
        ↓
Resize GT to 512 × 288
        ↓
Metric-specific preprocessing
        ↓
Compute FID / FVD / PSNR / SSIM
```

**Do not upscale the generated predictions to 800 × 450 before saving or evaluation.**

## 4.6 Checkpoint Selection and Inference Steps

Use **raw model weights** for all evaluations, unless only EMA weights are available.

Inference-step requirements:

| Method | Inference steps |
|---|---:|
| Baseline | 4 |
| Ours (MeanFlow) | 1 |

For our method, evaluate only the one-step variant. Do not evaluate the two-step variant.

## 4.7 Map Conditioning and Camera View Alignment

Special attention is required when evaluating experiments that use HDMap or world-scenario conditioning.

**Double-reprojection:**

The testing setup should be consistent with the training setup.

**Monocular shift:**

The map corresponding to the target frame and the history frames should be generated or projected using the **target camera viewpoint**.

This is important because the conditioning information must remain geometrically aligned with the image being generated.

## 4.8 Camera Pose Convention

During training, we intentionally use a **2 m camera displacement** for the history frames and the current target frame in the monocular-shift-related double-reprojection configuration.

Specifically:

- The source frame viewpoint equals the final target frame viewpoint in double-reprojection.
- An intermediate camera viewpoint is displaced by 2 m.
- All other reference frames use their original source camera poses.

This is an intentional design choice and should not be accidentally modified during test dataset construction.

---

# 5. Proposed Dataset Construction Configuration

The following table compares the current official YAML configuration, the earlier draft, and my proposed configuration.

**These are proposed settings that still require confirmation.**

| Configuration | Current official YAML | Earlier draft | Proposed configuration |
|---|---|---|---|
| RGB reconstruction resolution | 800 × 450 | 512 × 288 | Restore the official 800 × 450 reconstruction resolution |
| Model input / prediction / evaluation GT | Separate from reconstruction resolution | 512 × 288 | Standardize all to 512 × 288; never upscale predictions |
| `visible_pixel_count > 1000` | Counted on the reconstruction canvas | Counted at 512 × 288 | Use the official 800 × 450 reconstruction for filtering and explicitly record the counting resolution |
| Dataset split | Official `test` | Exclude previously trained scenes from the training pool | Prefer official `val`, for the reasons explained below |
| Maximum preprocessing frames | DR: 200; shift: 100 | Both: 100 | Preserve 200/100, but evaluate the same consecutive 50 frames |
| Depth parameters | YAML: 3000/6000; README: 300/1000 | 3000/6000 | Preserve the current YAML values of 3000/6000 |

## 5.1 Why Use the Official Validation Split?

The official nuScenes test split does not provide public GT 3D bounding-box annotations.

I checked several test scenes and found that:

- Map information is available.
- The corresponding GT box lists are empty.
- The official nuScenes dataset only releases annotation labels for the train/val splits.

Therefore, if we want to preserve both HDMap and GT bounding-box conditioning, I suggest constructing the new evaluation dataset from the **official nuScenes validation split**.

I have also verified that its 150 validation scenes do not overlap with either of our two existing training datasets.

## 5.2 Clarification: Double-Reprojection with 0 m Final Displacement

"Double-reprojection 0 m" refers to the **final output camera viewpoint**, not the intermediate camera displacement.

The existing process is:

```text
Original source camera viewpoint
              ↓
Intermediate viewpoint (+2 m)
              ↓
Original source camera viewpoint
              ↓
Final output displacement = 0 m
```

Therefore, we should **not change `distance_m` to 0**.

The intentional 2 m intermediate displacement must remain unchanged.

## 5.3 Different Reference-Pool Lengths

Keeping the current YAML configuration means:

- Double-reprojection preprocessing uses up to 200 frames.
- Monocular-shift preprocessing uses up to 100 frames.

These settings produce reference candidate pools of different lengths.

However, the final evaluation should still use the same scenes and the same 50 consecutive target frames.

If we also want identical reference-pool lengths, we must explicitly choose either:

- 100 frames for both modes; or
- 200 frames for both modes.

My current preference is to preserve the official 200/100 configuration unless there is a strong reason to change it.

## 5.4 Other Settings to Preserve

All other important preprocessing and selection settings should remain consistent:

- Same DA3 and SAM3 weights.
- Same prompts.
- Static threshold: 2 m.
- `z_tolerance = 0.3`.
- `occlusion_sharpness = 3.0`.
- `frame_sampling = 1`.
- Reference candidate ranking: `top_k = 20`.
- Select four valid references from the ranked candidates.
- Preserve the four-direction evaluation protocol.
- Ensure map/camera-view alignment.
- Prefer raw checkpoints.
- Baseline: four inference steps.
- Ours: one inference step.

---

# 6. Common Implementation and Engineering Issues

## 6.1 Checkpoint Saving and Disk-Space Management

Storage space is limited, so checkpoint management is important.

Requirements:

1. During normal training, save checkpoints at the end of each training stage, rather than saving unnecessary intermediate checkpoints.
2. If training unexpectedly stops or exits, the system should also attempt to save a recoverable checkpoint. **This behavior needs to be tested.**
3. Periodically monitor the available disk space.
4. Clean up unnecessary visualization-video folders and model-weight files.
5. Pay particular attention to large checkpoint directories, which can occupy approximately 4–6 GB each.

The cleanup process should not interfere with active experiments or remove checkpoints needed for recovery or reproduction.

## 6.2 Poor ControlNet Training Results

The current ControlNet-based implementation for introducing map conditioning performs very poorly.

This strongly suggests that there may be a significant issue in either:

- The implementation; or
- The conditioning/training design.

We need to investigate the root cause rather than assuming the poor performance is expected.

## 6.3 Checkpoint and Intermediate-File Cleanup

I would also like to organize and clean up unimportant files, especially those occupying substantial disk space.

Examples include:

- Intermediate checkpoints (`.ckpt`, `.pkl`) from completed experiments.
- Redundant model-weight files.
- Unnecessary visualization videos.
- Other large temporary or intermediate files that are no longer useful.

Please identify files that are safe to remove while preserving the final checkpoints, necessary experiment results, and reproducibility.

## 6.4 Visualizing World-Scenario Overlays

Existing reference script:

```text
testing_script/visualize_world_scenario.py
```

I want to visualize the world scenario, including HDMap and 3D bounding boxes, overlaid on the generated video.

My understanding is that the map and bounding boxes for each frame should first be represented in **world coordinates**, and then projected into the camera view corresponding to that specific frame.

Conceptually:

```text
HDMap / 3D bounding boxes in world coordinates
                    ↓
        Frame-specific camera pose
                    ↓
       World-to-camera transformation
                    ↓
           Camera projection
                    ↓
     2D map / bounding-box visualization
                    ↓
       Overlay on generated video
```

The visualization must use the correct camera pose and calibration for each frame so that the world-scenario elements align with the generated content.

---

# 7. Key Requirements Summary

The main goals are to:

1. **Build a clean, independent evaluation dataset** with HDMap and, preferably, GT 3D bounding-box annotations, without overlapping either existing training dataset.
2. **Evaluate both double-reprojection and 1 m monocular shift**, using their corresponding metrics.
3. **Strictly enforce the four-direction and 50-consecutive-frame requirements**, selecting 10 scenes and producing 40 direction-specific sequences per evaluation mode.
4. **Unify model inference, prediction storage, GT images, and evaluation at 512 × 288**, while retaining 800 × 450 for rendering/reconstruction where appropriate.
5. **Preserve the intentional camera-pose conventions**, including the intermediate 2 m displacement in double-reprojection and target-view map alignment.
6. **Use raw checkpoints whenever possible**, with four-step baseline inference and one-step inference for our method.
7. **Maintain training reliability and storage efficiency**, including stage-end checkpoint saving, interruption recovery, and safe cleanup.
8. **Investigate the poor ControlNet results** and ensure that world-scenario overlays use geometrically correct camera projections.

The implementation should follow the existing training and evaluation infrastructure as closely as possible, avoiding unnecessary changes to established behavior.
