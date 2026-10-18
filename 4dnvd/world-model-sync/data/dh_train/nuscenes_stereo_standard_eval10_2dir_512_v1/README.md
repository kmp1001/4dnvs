Fixed paired-stereo benchmark

Finalized: 10 scenes × 2 directions × 50 consecutive frames = 1000 targets / 20 videos. Both directions share identical frame indices per scene. LR: pcam_f0→pcam_r0; RL: pcam_f0→pcam_l0. Four visibility-ranked distinct source references each strictly >1000 visible pixels, measured at native512×288; current source is allowed. Excludes252 physical training scenes across the original monocular and stereo train manifests.

Native reconstruction, source/GT/warp/coverage and controls are512×288. Target/history controls use the real target camera; reference controls use pcam_f0 at each reference timestamp. Map/boxes are exported for all three cameras.

selection.json records exact frames, exclusion list and input hashes. stereo/dataset_test.json is the fixed test manifest. joint_eligibility_audit.json records all16 candidate eligibility results,14 qualifying.

Evaluation uses the shared standard_eval.py --setting stereo: baseline4 steps, ours1 step, raw. FID/FVD/PSNR/SSIM overall and separately forLR/RL.

Results: /fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/out_stereo_standard_eval10_2dir_512_v1
Protocol: /fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/out_stereo_standard_eval10_2dir_512_v1/PROTOCOL.md
