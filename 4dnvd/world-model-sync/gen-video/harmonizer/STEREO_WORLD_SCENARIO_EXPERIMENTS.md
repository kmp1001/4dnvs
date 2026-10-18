# Four stereo experiments, one node per experiment

Use the regenerated `data/dh_train/nuscenes_stereo_world_scenario_train100_test20`
conditions. The prepared manifests are `dataset_train.json`, `dataset_test.json`,
and `dataset_validation_test80.json` in that dataset directory.

The task changes launch scripts and manifest preparation. Model/loss code,
renderer settings, preprocessing YAMLs and the metric implementations are unchanged.

## Data and reference policy

This is real stereo camera transformation without double reprojection. Input is
`front_warped`; supervision is `front_target`. The source camera is PCAM_F0;
LR targets PCAM_R0 and RL targets PCAM_L0. The native source and target
calibration files are used by the existing camera-conditioning implementation.

Each sample uses four unique source-camera references in the stored visibility
ranking order, requiring visibility >1000 pixels. The current source frame is
allowed to qualify; it is not forced to be a reference. Target-camera RGB is
never used as a source reference. The source/target cameras differ even when
source and target frame indices match.

Mode 0 has no world-scenario conditioning. Mode 1 uses target-camera maps for
the current target and its target-view history, and PCAM_F0 maps for source
references. Both maps and all reference paths use the updated regenerated data.
Temporal histories and inference use the existing five-view/four-past-frame rule.

Training retains 18,505 frames from 93 physical scenes (186 direction-scenes).
Final inference retains 3,599 frames from 18 physical test scenes (36 direction-scenes).
As approved, samples without four qualifying references are omitted; the threshold
and reference count are unchanged. No frames qualify in either direction for train
scenes 0060, 0208, 0232, 0355, 0733, 0882, 1096, and test scenes 0553, 0554.
`DATASET_AUDIT.json` records counts and omitted direction-scenes;
`checks/stereo_reference_eligibility.json` records the per-direction ranking audit.
As requested, periodic previews use a fixed 80-frame subset of the test split:
the first four selected physical test scenes, ten consecutive retained frames
per direction. Baseline and Stage I/III rollout previews use up to 80 frames;
Stage II GT-history validation has 48 eligible targets after four context frames
per sequence. This follows the existing stage-specific validation behavior.

Manifest preparation was run with:

```bash
cd ../../feature-match/UFM
/nfshomes/jlin1219/.pixi/bin/pixi run --locked python \
  src/generate_stereo_world_scenario.py finalize --include-current-reference
cd ../../gen-video/harmonizer
/nfshomes/jlin1219/.pixi/bin/pixi run --locked python \
  training_script/prepare_stereo_validation.py
```

The commands above assume the initial working directory is `gen-video/harmonizer`.
`finalize` reads existing images and writes manifests; it does not render RGB again.
The original `selection.json` remains provenance for data generation; each new
manifest records `include_current_reference=true` for this experiment policy.

## Training schedule

Batch size is 1 per GPU, gradient accumulation is 1, and each experiment uses
one node × four GPUs (global batch 4). Preserve the previous 96,000 current-target
samples per experiment by doubling optimizer steps relative to an eight-GPU run:

| Experiment | Stage I | Stage II | Stage III | Current-target samples |
| --- | ---: | ---: | ---: | ---: |
| baseline mode 0 | 12,000 | 12,000 | — | 96,000 |
| baseline mode 1 | 12,000 | 12,000 | — | 96,000 |
| ours mode 0 | 12,000 | 4,800 | 7,200 | 96,000 |
| ours mode 1 | 12,000 | 4,800 | 7,200 | 96,000 |

Baseline remains rectified flow with the existing two stages and RAFT-small
temporal loss. Ours remains the existing three-stage MeanFlow method with WAFT,
Stage III EMA history generation, Stage III EMA enabled, and no temporal-stage
LoRA. Learning rates, losses, temporal probability and EMA decay stay at the
existing method settings. The baseline scheduler remains constant.

Checkpoints: save at the end of each stage and on a handled interruption. Periodic checkpointing defaults to disabled (`CHECKPOINTING_STEPS=0`).
Preview interval: 300 steps (1,200 samples).
Periodic validation interval: 600 steps (2,400 samples).
These preserve the sample intervals used by the earlier eight-GPU experiments.
Image preparation remains 512×288, with four references, camera/time embeddings,
bf16 precision, and seed 42.

## Resume the existing four experiments

```bash
bash training_script/train_stereo_four_experiments.sh --resume --jobs 7875414 7875416
```

Resume uses the highest `model_STEP.pkl` within each stage directory. Completed
stages are skipped. An incomplete stage uses `--resume` to restore training
state; later stages keep their original handoff and schedule. The four-experiment
launcher forces `CHECKPOINTING_STEPS=0`: save only at stage completion or a
handled stop. Resume logs are under `train_world_scenario_stereo/resume_logs/`.

Do not launch another copy while these steps are already running.

## Launch all four experiments

From `gen-video/harmonizer`, inside a four-node Slurm allocation:

```bash
bash training_script/train_stereo_four_experiments.sh
```

For two existing two-node allocations (the currently available allocation IDs):

```bash
bash training_script/train_stereo_four_experiments.sh --jobs 7875414 7875416
```

The helper reads the two node names in each allocation and pins exactly one
experiment to each. It runs the four jobs concurrently and waits for all four.
It does not launch training until you run the command. The job IDs must still
be active. On the current allocations, this assigns:

| Index | Experiment | Node |
| --- | --- | --- |
| 0 | baseline mode 0 | gammagpu18 |
| 1 | baseline mode 1 | gammagpu19 |
| 2 | ours mode 0 | gammagpu20 |
| 3 | ours mode 1 | gammagpu21 |

Logs: `<project>/train_world_scenario_stereo/launcher_logs/experiment-0.log`
through `experiment-3.log`. Outputs are separate per experiment under
`<project>/train_world_scenario_stereo/{baseline,ours}_mode{0,1}/`.

For a single experiment inside its own one-node allocation:

```bash
bash training_script/train_world_scenario.sh \
  --setting stereo --method ours --mode 0 --nodes 1 --gpus-per-node 4
```

Append `--dry-run` to inspect the full commands. The launcher defaults to the
original monocular setting unless `--setting stereo` is specified.

## Inference and results

Each experiment automatically evaluates its final checkpoint on all test samples,
exports prediction/real-target/input videos, and computes the existing
Clean-FID, KID and VideoGPT FVD metrics. Baseline inference uses four steps;
ours uses the Stage III EMA checkpoint with one step. LR and RL are summarized
separately, alongside the aggregate results. Real target images are taken from
`front_target`, not `front_source`.

Each run has `run_config.json`, stage directories, `test/evaluation_manifest.json`,
`test/videos/`, and `test/generative_metrics.yaml`. The stereo parent directory
has `metrics_summary.csv`. The metric implementation labels its first-letter
pools `L` and `R`; the stereo summary names them LR and RL.

To rerun inference explicitly, set `OUTPUT_ROOT` and `TEST_DATASET` and call
`testing_script/eval_stereo_world_scenario.sh METHOD MODE CHECKPOINT 4`.
The training launcher supplies these automatically.
