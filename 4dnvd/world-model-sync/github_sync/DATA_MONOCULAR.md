# Monocular data (nuScenes, front camera `pcam_f0`)

Root: `data/dh_train/` (`H = gen-video/harmonizer`, `U = feature-match/UFM`). Images are stored at **512×288**. Directions: `D2M L2M R2M U2M` (2 m virtual shift: down/left/right/up, then back to the source view).

## 1. Training data: `smoke_nuscenes_dr2m/` (double reprojection)

Built by `U/src/preprocess.py` + `config/double_reprojection/nuscene_smoke.yaml` (100 frames/scene, 512×288), then `H/training_script/extract_training_map.sh` for map/box.

```text
smoke_nuscenes_dr2m/
├── front_source/D2M_scene-0001/00000.png ...      # GT / source RGB (2740 = 685 scenes x 4 dirs)
├── front_warped/D2M_scene-0001/00000.png          # degraded input after double reprojection
├── confident_coverage/D2M_scene-0001/00000.png
├── top20_reference/D2M_scene-0001/00000.json      # ranked reference candidates + visible_pixel_count
├── calibration/D2M_scene-0001/{source,target}_calibration.txt
├── hdmap/original_scene-0001/00000.png            # map + wireframe boxes (shared by 4 dirs)
├── world_scenario/original_scene-0001/00000.png   # map + filled boxes (shared by 4 dirs)
├── videos/scene-0001/metadata.json                # fps, calibration, per-frame poses
├── videos/pinhole_pcam_f0/000000_scene-0001_boxes/000000.json   # GT 3D boxes, global frame
├── dataset_map_box_train100_4dir_ordered.json     # TRAIN: scene ranks 1-100, 24,256 records
├── dataset_map_box_val80_4scene_4dir_ordered.json # VAL:   4 scenes x 20 consecutive frames
├── dataset_map_box_test20_4dir_ordered.json       # old test (ranks 101-120), replaced by section 2
└── MAP_BOX_DATASET.md, MAP_BOX_ORDERED_SPLITS.md  # synced; full details
```

Example record (`train` key, keyed by id):

```json
{
  "image":          ".../front_warped/D2M_scene-0001/00003.png",
  "source_image":   ".../front_source/D2M_scene-0001/00003.png",
  "target_image":   ".../front_source/D2M_scene-0001/00003.png",
  "reference_images": [".../front_source/D2M_scene-0001/00000.png", "... x4"],
  "reference_ranking": ".../top20_reference/D2M_scene-0001/00003.json",
  "confident_coverage": ".../confident_coverage/D2M_scene-0001/00003.png",
  "scene": "D2M_scene-0001", "frame_index": 3, "prompt": "remove degradation",
  "hdmap_image": ".../hdmap/original_scene-0001/00003.png",
  "world_scenario_image": ".../world_scenario/original_scene-0001/00003.png",
  "boxes_json": ".../videos/pinhole_pcam_f0/000000_scene-0001_boxes/000003.json",
  "map_box_metadata": ".../videos/scene-0001/metadata.json"
}
```

## 2. Test data: `nuscenes_standard_eval10_4dir_512_v1/`

Built by `U/src/prepare_standard_eval.py`. It has 10 held-out scenes × 4 directions × 50 consecutive frames, so each mode has 2,000 targets. Each target has 4 references, each with >1000 visible pixels. The current frame is excluded only in DR. The 10 scenes are 0172, 0176, 0177, 0184, 0200, 0203, 0224, 0226, 0229 and 0233. 186 training scenes are excluded.

```text
nuscenes_standard_eval10_4dir_512_v1/
├── README.md, selection.json                 # synced; exact frames, excluded scenes, hashes
├── configs/scene-0172_{double_reprojection,shift_1m}.yaml
├── double_reprojection/
│   ├── dataset_test.json                     # RGB paths point into smoke_nuscenes_dr2m/
│   └── conditions/{hdmap,world_scenario}/original_scene-0172/00002.png, conditions/videos/...
└── shift_1m/                                  # 1 m monocular shift, metrics: FID/FVD only
    ├── dataset_test.json
    ├── front_source/ front_warped/ confident_coverage/ top20_reference/ calibration/   (D2M_scene-0172/...)
    └── conditions/{hdmap,world_scenario}/D2M_scene-0172/00002.png   # rendered in the SHIFTED target camera
```

Protocol and results: `out_standard_eval10_4dir_512_v1/PROTOCOL.md` (synced).
