Stereo standard evaluation, 2026-10-17

Status: dataset finalized and strict validation passed: 1000 samples, twenty 50-frame sequences, ten held-out scenes, both directions sharing the same frame indices. Twenty related tests passed. Model evaluations use final-stage checkpoints; unfinished training checkpoints are waited for. metrics_summary.csv lists completed evaluations, and each complete.json confirms successful inference, all metrics and video export.

The user explicitly corrected four directions to TWO existing stereo directions. LR = pcam_f0 → pcam_r0; RL = pcam_f0 → pcam_l0. Keep four distinct visibility-ranked source references per target, each visible_pixel_count strictly greater than 1000; same-time source references are allowed for stereo. Visibility is recomputed at native reconstruction resolution 512×288, using the existing stereo reconstruction implementation and its other settings. Preprocess the first 100 native frames with frame_sampling=1.

Exclude the union of the two original monocular training manifests and the stereo training manifest: 252 physical scenes. Candidates come from official native val, with all three cameras and at least 100 frames. Select the first ten jointly eligible scenes by ascending scene ID; take the first 50 frames of each longest consecutive run in the LR/RL eligibility intersection. Thus both directions use the exact same frames: 1000 targets / 20 clips total. Eligibility uses only inputs and visibility, never model metrics. Increase the candidate count if fewer than ten qualify; do not relax criteria or invent frames.

Save world_scenario and wireframe hdmap PNGs, native boxes JSON and render metadata for all three cameras. Each target and historical frame uses its own timestamp and real target-camera map; reference maps come from pcam_f0 at each reference timestamp. GT is the real target-camera RGB.

Common inference and metric entry point: gen-video/harmonizer/testing_script/standard_eval.py with --setting stereo. DrivenVSModel baseline: four steps, final Stage II raw model_12000.pkl. MeanFlowDrivenVSModel ours: one step, final Stage III raw model_7200.pkl; wait for final training to finish. Each task owns one GPU and complete temporal sequences. Raw weights are present in the checkpoint format used by all four runs; EMA is not selected when raw exists.

Infer and save lossless PNGs at 512×288. Compute Clean-FID / VideoGPT FVD / PSNR / SSIM overall and separately for LR and RL from 512×288 PNGs. Preserve normal metric-network internal preprocessing. FVD uses the full 50-frame clip, twenty clips overall and ten per direction. Export generated MP4s after metric computation, for viewing only. Do not evaluate decoded MP4s.

Dataset root: data/dh_train/nuscenes_stereo_standard_eval10_2dir_512_v1. Its stereo/dataset_test.json records all samples and camera-specific control paths; selection.json records exact selected scenes and frame indices, excluded training scenes, training manifest hashes and renderer hash. Original stereo datasets and results are retained.

Commands from gen-video/harmonizer:

bash testing_script/run_standard_eval.sh --setting stereo --jobid 7875414 --node gammagpu19 --phase prepare
bash testing_script/run_standard_eval.sh --setting stereo --jobid 7875414 --node gammagpu18 --phase evaluate

--node must identify an idle four-GPU node in the supplied allocation. Evaluation is already scheduled after preparation in this session. Future training callbacks also invoke the same raw-weight stereo evaluation through standard_eval.py single. Output locks prevent duplicate evaluations from callback/launcher overlap.

Logs: launcher_logs/prepare-worker-{0,1,2,3}.log, launcher_logs/evaluate-worker-{0,1,2,3}.log and <method>_mode<0|1>/stereo/{inference,metrics,reconstruction,videos}.log. Final scores: <method>_mode<0|1>/stereo/{generative_metrics,reconstruction_metrics}.yaml. metrics_summary.csv contains overall and directional scores. complete.json is written only after inference, metrics and video export succeed.
