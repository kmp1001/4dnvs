# Stereo data (nuScenes, paired real cameras)

Root: `data/dh_train/` (`H = gen-video/harmonizer`, `U = feature-match/UFM`). Directions:

| Direction | Source / reference camera | Target (real GT) camera |
|---|---|---|
| `LR` | `pcam_f0` | `pcam_r0` |
| `RL` | `pcam_f0` | `pcam_l0` |

## 1. Train / val: `nuscenes_stereo_world_scenario_train100_test20/` (~55 GB)

Built by `U/src/generate_stereo_world_scenario.py` (`prepare` → `generate_stereo_world_scenario.sh` → `finalize`). That script calls `preprocess.py` for RGB and `preprocess_cosmos_render.py` for maps. Images on disk are **800×450**. Training, previews and inference resize them to 512×288. Scenes: 100 train scenes (official train) and 20 test scenes (official val), 100 frames each.

```text
nuscenes_stereo_world_scenario_train100_test20/
├── selection.json                                   # synced; scene list per split
├── configs/{train,test}_{LR,RL}.yaml                # effective configs
├── dataset_train.json                               # 18,505 records (key "train")
├── dataset_validation_test80.json                   # 80 records, from H/training_script/prepare_stereo_validation.py
├── dataset_test.json                                # 3,599 records (old test, replaced by section 2)
└── {train,test}/
    ├── front_source/LR_scene-0008/00000.png         # pcam_f0 RGB (references come from here)
    ├── front_target/LR_scene-0008/00000.png         # real target-camera GT
    ├── front_warped/  confident_coverage/  top20_reference/  calibration/   (LR_scene-XXXX/...)
    └── conditions/{pcam_f0,pcam_l0,pcam_r0}/
        ├── hdmap/original_scene-0008/00000.png
        ├── world_scenario/original_scene-0008/00000.png
        └── videos/scene-0008/metadata.json, videos/pinhole_pcam_XX/..._boxes/*.json
```

Example record. It has the same fields as the monocular records, plus these camera-specific controls:

```json
{
  "image": ".../train/front_warped/LR_scene-0008/00000.png",
  "source_image": ".../train/front_source/LR_scene-0008/00000.png",
  "target_image": ".../train/front_target/LR_scene-0008/00000.png",
  "reference_images": [".../train/front_source/LR_scene-0008/00000.png", "... x4"],
  "world_scenario_image":        ".../conditions/pcam_r0/world_scenario/original_scene-0008/00000.png",
  "source_world_scenario_image": ".../conditions/pcam_f0/world_scenario/original_scene-0008/00000.png",
  "reference_world_scenario_images": [".../conditions/pcam_f0/world_scenario/... x4"],
  "hdmap_image": ".../conditions/pcam_r0/hdmap/original_scene-0008/00000.png",
  "boxes_json":  ".../conditions/pcam_r0/videos/pinhole_pcam_r0/000000_scene-0008_boxes/000000.json",
  "source_camera": "pcam_f0", "target_camera": "pcam_r0",
  "source_calibration": "...", "target_calibration": "..."
}
```

Top-level manifest keys: `dataset_setup: "stereo"`, `ground_truth: "real_target_camera"`.

## 2. Test: `nuscenes_stereo_standard_eval10_2dir_512_v1/` (~3.3 GB)

Built by `generate_stereo_world_scenario.py prepare-eval / eval-worker / finalize-eval`, launched via `H/testing_script/run_standard_eval.sh --setting stereo`. It has 10 scenes × 2 directions × 50 consecutive frames, 1,000 targets in total, all at **512×288**. Metrics: FID/FVD/PSNR/SSIM. The 10 scenes are 0003, 0012, 0013, 0015, 0016, 0017, 0018, 0035, 0036 and 0038. 252 training scenes from both training sets are excluded.

```text
nuscenes_stereo_standard_eval10_2dir_512_v1/
├── README.md, selection.json                 # synced
├── configs/{LR,RL}.yaml
└── stereo/
    ├── dataset_test.json                     # 1,000 records
    ├── front_source/ front_target/ front_warped/ confident_coverage/ top20_reference/ calibration/  (LR_scene-0003/...)
    └── conditions/{pcam_f0,pcam_l0,pcam_r0}/{hdmap,world_scenario,videos}/
```

Protocol: `out_stereo_standard_eval10_2dir_512_v1/PROTOCOL.md` (synced).
