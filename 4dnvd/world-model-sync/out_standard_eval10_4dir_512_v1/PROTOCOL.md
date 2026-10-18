APPROVED — the user confirmed the original draft on 2026-10-17. Reconstruction remains 512×288 with 100 source frames in both modes, using scenes excluded from both actual training sets. The alternative README-alignment proposal was not adopted.

Fixed monocular evaluation standard, 2026-10-17

This benchmark is explicitly requested by the user and supersedes the earlier monocular test splits for future comparisons.

- Dataset: data/dh_train/nuscenes_standard_eval10_4dir_512_v1.
- The exact same ten physical scenes and fifty consecutive target frames are used in double_reprojection and shift_1m, across D2M/L2M/R2M/U2M: forty videos and 2000 target frames per mode.
- Exclude the union of dataset_fixed_train100_4dir_seed42.json and dataset_map_box_train100_4dir_ordered.json: 186 unique training scenes.
- Four highest visibility-ranked distinct references, each visible_pixel_count >1000. Exclude the same-time source only for double reprojection; permit it for 1 m shift.
- Select scenes by ascending scene ID among preprocessed candidates that have a jointly eligible consecutive run of at least fifty frames. Take the first ten and the first fifty frames of each longest run. Selection uses only input/reference eligibility, never model metrics.
- Double reprojection uses the existing 2 m round-trip RGB products at 512×288, with original-camera GT. Fresh controls are rendered in the original camera.
- Shift mode freshly preprocesses at 1 m and 512×288, with controls rendered in the shifted camera. Real references for generative metrics are original-camera source images; there is no paired novel-view GT.
- The current target and each historical prediction use their own timestamp's target-view control, including shifted target views in shift_1m. Reference controls always use the original source camera and each reference's own timestamp. Save both wireframe HD maps and world-scenario maps/filled boxes, plus global box JSONs and renderer metadata.
- Preserve the intentionally shared camera-conditioning convention. No camera-pose correction is introduced.
- Use the corresponding baseline DrivenVSModel with four inference steps, or MeanFlowDrivenVSModel with one inference step. Use raw for every present checkpoint, as explicitly clarified by the user; none of the selected checkpoints is EMA-only. No two-step MeanFlow evaluation.
- Each task owns one GPU, runs inference and then its metrics. Sixteen workers process twenty model/mode tasks.
- Infer/save at 512×288. No original-size upscaling. Metrics use PNGs, not compressed MP4s.
- Both modes: Clean-FID and VideoGPT FVD, overall and each direction. FVD uses one full fifty-frame clip per direction-scene, forty clips total.
- Double reprojection only: PSNR and SSIM against corresponding original-camera RGB, overall and each direction. Existing evaluators also retain their KID/LPIPS fields; these do not replace the requested metrics.
- Export forty generated 512×288 MP4s per task after computing metrics, for viewing only.

Frozen model inventory

| Model | Selected stage/checkpoint | Backbone / steps | Weights |
| --- | --- | --- | --- |
| train_baseline__nuscenes_100scene_4dir | stage2/model_3000.pkl | baseline / 4 | raw |
| train_meanflow__nuscenes_100scene_4dir_extensions_4gpu__exp01_stage2_10k | stage2/model_10500.pkl | ours / 1 | raw |
| train_meanflow__nuscenes_100scene_4dir_extensions_4gpu__exp02_stage2_2800_stage3_7200 | stage3/model_5000.pkl (configured endpoint 7200; latest saved) | ours / 1 | raw |
| train_meanflow__nuscenes_100scene_4dir_extensions_4gpu__exp06_no_flow | stage3/model_7200.pkl | ours / 1 | raw |
| train_meanflow__nuscenes_100scene_4dir_extensions_4gpu_workers4_raft_from_stage1_20261015_100351__exp02_stage2_2800_stage3_7200 | stage3/model_3000.pkl (configured endpoint 3600; latest saved) | ours / 1 | raw |
| train_meanflow__nuscenes_100scene_4dir_extensions_4gpu_workers4_v2__exp02_stage2_2800_stage3_7200 | stage3/model_6000.pkl (configured endpoint 7200; latest saved) | ours / 1 | raw |
| train_world_scenario__baseline_mode0 | stage2/model_6000.pkl | baseline / 4 | raw |
| train_world_scenario__baseline_mode1 | stage2/model_6000.pkl | baseline / 4 | raw |
| train_world_scenario__ours_mode0 | stage3/model_3600.pkl | ours / 1 | raw |
| train_world_scenario__ours_mode1 | stage3/model_3600.pkl | ours / 1 | raw |

The original train_meanflow/nuscenes_100scene_4dir checkpoint was deleted before this request and cannot be evaluated. The model list includes every experiment with a remaining checkpoint under the requested train_meanflow root. Interrupted runs are labeled above and use the highest available stage and largest saved step, not the configured endpoint. No checkpoint is copied, changed or deleted.

Commands, run from gen-video/harmonizer

Dataset generation and final validation are complete: both dataset_test.json files contain 2000 samples in forty 50-frame sequences, with the common selection recorded in selection.json. Eleven related tests passed. Model evaluations have not been launched by the agent. Dataset preparation is performed separately. Run all twenty model/mode evaluations with:

    bash testing_script/run_standard_eval.sh --jobid 7869449 --phase evaluate

Completed candidate preprocessing and completed evaluation tasks are skipped on rerun. A partial task without a complete inference manifest is rerun from the beginning. The same commands accept a replacement four-node job ID.

Future single-model evaluation (raw, both test modes)

    pixi run --locked python testing_script/standard_eval.py single --method ours --mode 0 --checkpoint /absolute/path/model.pkl --output-root /absolute/path/results

The monocular eval_world_scenario.sh training callback now calls this command. The stereo callback keeps its stereo dataset.

Outputs

- models.json / tasks.json: frozen checkpoint identities and task assignments.
- launcher_logs/evaluate-worker-N.log: one live worker log per GPU.
- <model>/<mode>/inference.log: inference progress and errors.
- <model>/<mode>/evaluation_manifest.json: exact generated frame paths and inference configuration.
- <model>/<mode>/generative_metrics.yaml: FID/FVD and directional results.
- <model>/double_reprojection/reconstruction_metrics.yaml: PSNR/SSIM and directional results.
- <model>/<mode>/videos/: forty generated MP4s at 512×288.
- <model>/<mode>/complete.json: written only after inference, metrics and video export succeed.
- metrics_summary.csv: one row per completed model/mode evaluation.

Dataset selection.json records exact scenes, frames, training-manifest hashes and renderer hash. Both dataset_test.json files record target controls, reference controls and explicit calibration paths. Effective per-scene preprocessing YAMLs and candidate eligibility progress are preserved. Existing evaluation outputs are not overwritten.
