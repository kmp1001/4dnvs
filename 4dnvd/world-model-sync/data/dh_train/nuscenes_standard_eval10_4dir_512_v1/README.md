Approved original test protocol, 2026-10-17

Both dataset_test.json manifests are finalized: the same ten held-out scenes, four directions, fifty consecutive frames per direction, 2000 targets per mode. Reconstruction, controls and model evaluation resolution: 512×288. Each mode preprocesses 100 source frames. Four distinct visibility-ranked references each strictly exceed 1000 pixels. Exclude current source only in double_reprojection.

selection.json records exact frames, the 186 excluded training scenes, training manifest hashes and renderer hash. The map and box renderer is the latest preprocess_cosmos_render.py used during this run. Shifted target and historical controls use the shifted camera; source-reference controls use the original camera.

Evaluation protocol and model inventory: /fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/out_standard_eval10_4dir_512_v1/PROTOCOL.md
