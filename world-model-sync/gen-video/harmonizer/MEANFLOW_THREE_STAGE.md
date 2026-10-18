# Three-stage MeanFlow mode-B training

Run from `gen-video/harmonizer` using the existing Pixi environment. This
curriculum starts from Cosmos-Predict2-0.6B and keeps the original source-reference
fusion, camera/time conditions, frozen VAE, AdamW settings, and batch size one
per process. No training is launched by creating or checking these scripts.
The defaults below refer to the shell launchers; direct Python CLI defaults
are not identical in every case.

## Budgets and losses

The original `training_script/train_nuscenes_smoke.sh` uses 200,000 Stage-I
optimizer steps and 100,000 Stage-II steps; these are not epochs. The new
`training_script/train_nuscenes_meanflow_three_stage.sh` uses:

| Stage | Steps | Previous-frame conditioning | Loss |
| --- | ---: | --- | --- |
| I | 200,000 | None (V=1) | MeanFlow, weight 1 |
| II | 10,000 | Four GT frames (teacher forcing) | MeanFlow + 0.5 optical-flow temporal |
| III | 90,000 | Four rendered-bridge resampled predictions | MeanFlow + 0.5 optical-flow temporal |

`STAGE1_STEPS` and `STAGE2_STEPS` retain their original meanings. Set
`WARMUP_STEPS` to change Stage II; Stage III defaults to
`STAGE2_STEPS - WARMUP_STEPS`. The launcher rejects any explicit Stage-III
budget that breaks the equality. L2 and LPIPS remain disabled. Stages II and
III retain the original 75% V=5 / 25% V=1 mixture; set
`TEMPORAL_SAMPLE_PROBABILITY=1` for temporal context on every example.
The optical-flow loss defaults to small RAFT and has no loss-weight warm-up,
matching the smoke script. Both stages call the existing
`temporal_loss.get_dh_temporal_loss` against real current/past GT frames.
Generated histories are never substituted for GT optical-flow supervision.
The shared Charbonnier loss is normalized by valid pixel/channel values;
out-of-bounds pixels contribute zero, including the Charbonnier epsilon term.
For each of four previous GT frames, the estimator computes previous-to-current
flow from the GT pair. This flow samples the predicted current RGB image in
previous-frame coordinates. The loss averages RGB Charbonnier penalties over
valid pixels separately for each pair, then averages the four pair losses.
It supervises only predicted current RGB, not decoded history RGB. The mask
checks image bounds; there is no occlusion or forward/backward consistency mask.
For this auxiliary loss, the model uses `u(z_t, r=0, t)` and decodes
`z_t - t*u`; when the sampled MeanFlow `r` is nonzero this requires another
DiT call. The frozen VAE decoder passes gradients to the predicted latent.
The primary objective changes from the smoke script's weighted FM MSE to
MeanFlow adaptive L2 in every stage.

Stage I uses current-frame GT as its supervised target and omits previous-frame
context. In temporal samples, `build_mode_b_paths` constructs the following
RGB sequences, all ordered current first and then newest-to-oldest history:

```text
source:           [current_render, H1, H2, H3, H4]
MeanFlow target:  [current_GT,     H1, H2, H3, H4]
flow supervision: [current_GT, GT_H1, GT_H2, GT_H3, GT_H4]
```

`Hi` is GT in Stage II and detached generated RGB in Stage III. History opacity
is one. Source and MeanFlow target sequences are VAE-encoded separately;
temporal VAE coupling means identical RGB histories need not yield identical
history latent cells. MeanFlow loss selects latent time cell 0; its JVP is
computed on the full latent sequence. The optical-flow loss always uses the
separate real-GT sequence.

Epoch counts depend on dataset size and the sharded loader length. With one
sample per GPU, steps per epoch are approximately
ceil(number of training samples / number of GPUs). Temporal examples require
five consecutive frame indices within one sequence. If the selected current
sample has no complete window, the dataset samples another eligible window;
the 75/25 probability is not an exact per-epoch quota.

## Optional WAFT optical-flow estimator

RAFT remains the default. To use the existing `world-model-phucnda-wm/tar-c-t.pth`
checkpoint with WAFT a1 in both temporal stages:

```bash
# Separate environment, matching WAFT's README: Python 3.12,
# PyTorch 2.7.0 / torchvision 0.22.0 CUDA 12.8, xFormers 0.0.30.
cd ../../WAFT
pixi install
pixi run check-env
cd ../gen-video/harmonizer
FLOW_ESTIMATOR=waft pixi run bash training_script/train_nuscenes_meanflow_three_stage.sh
```

The launcher defaults to `WAFT/config/a1/tar-c-t.json`, five refinement iterations,
and `../../tar-c-t.pth`. Override `WAFT_REPO`, `WAFT_CONFIG`, `WAFT_CHECKPOINT`,
or `WAFT_ITERATIONS` if needed. Only a1 is supported by this adapter. A WAFT run
gets an `_waft` output-directory suffix by default. The original
`train_nuscenes_smoke.sh` also accepts `FLOW_ESTIMATOR=waft`; its default output
paths retain the original behavior, so set `OUTPUT_ROOT` for a separate run.

The direct three-stage CLI accepts `--flow-estimator waft`, `--waft-repo`,
`--waft-config`, `--waft-checkpoint`, and `--waft-iterations`. The original trainer
uses corresponding underscore options (`--flow_estimator`, etc.). The estimator
settings are recorded in the three-stage training configuration/checkpoint;
resume rejects estimator-setting changes, while initialization allows a new run.

Each training rank starts one persistent WAFT worker through WAFT's own
`pixi run`, on that rank's GPU. Harmonizer dependencies are not changed.
Only GT image pairs and detached flow arrays cross the process boundary;
RGB is converted from harmonizer's `[-1,1]` to WAFT's `[0,255]`. The worker
uses the final flow prediction, in original-image pixel units, with WAFT's
native padding/unpadding and config iteration count. It loads the full checkpoint
strictly, including all backbones, without separate pretrained-backbone downloads.
Standalone WAFT training still initializes pretrained backbones by default.

Both temporal stages retaiqn identical GT supervision, differentiable image
warping, masked Charbonnier loss and weight 0.5. WAFT is frozen and runs in float32.
Stage I and inference do not start a flow worker. The separate worker adds GT/flow
CPU transfers and its own model/activation memory on each GPU.

To check the real checkpoint without training, from harmonizer:

```bash
pixi run python -c 'import torch; from sys import path; path.insert(0,"src"); from waft_estimator import WAFTOpticalFlow; device="cuda:0" if torch.cuda.is_available() else "cpu"; estimator=WAFTOpticalFlow(device); image=torch.zeros(1,3,64,96,device=device); print(estimator.backward(image,image).shape); estimator.close()'
```

Validation: the real `tar-c-t.pth` loads strictly and produces finite flow in
WAFT's installed environment. A CPU check through the persistent worker and
five-frame temporal loss produces gradients for the predicted current frame,
zero gradients for history outputs, and closes the worker cleanly. GPU training
and memory use remain unverified because this session has no CUDA device.

## Reference MeanFlow audit

`src/meanflow_conditioning.py` reproduces the additional interval embedder in
`MeanFlow/sit.py`: sinusoidal features of **t-r**, 256 features (128 frequency pairs),
Linear -> SiLU -> Linear, normal(0, 0.02) weights, zero biases. It replaces the
previous direct-r, zero-projection path. Cosmos's pretrained t-conditioning,
normalization, and AdaLN machinery are retained; this is not a replacement of
Cosmos with the reference SiT backbone.

The loss follows the local `MeanFlow/README.md` ImageNet-256 examples with
`SILoss(weighting="adaptive")`: logit-normal(-0.4,1.0)
ordered r/t, P(r<t)=0.25 (75% r=t), endpoint override off, JVP tangents (v,0,1), detached
v-(t-r)*JVP target, per-sample squared-error **sum**, detached adaptive weight
(E+0.001)^(-1). Default JVP is torch.func; autograd JVP is an explicit
compatibility option. Sampling uses uniform descending intervals and
z_r = z_t - (t-r)*u. The `SILoss` constructor itself defaults to uniform
weighting; the reference training CLI selects adaptive weighting. Its unequal-time
CLI default is 0.75, but the README's ImageNet examples explicitly select 0.25,
which the three-stage launcher, curriculum CLI, and shared sampler now use.
This probability is independent of the unchanged 75% temporal V=5 mixture.
The harmonizer
skips the JVP when all samples have r=t, where its correction is exactly zero.
For JVP computation only, `meanflow_loss.py` selects PyTorch's math SDPA kernel:
automatic fused-kernel selection in Cosmos attention rejects forward AD or
double backward in the installed PyTorch. This retains the attention equation,
not an approximation, and leaves ordinary inference kernel selection unchanged.
The actual Cosmos attention function's MeanFlow target and parameter gradient
are checked against centered finite differences for both JVP backends. Math
attention may require more GPU memory than fused attention; full GPU memory
has not been measured here.
Class-label dropout and class CFG are inapplicable to this
unconditional-text, image-conditioned model. Guided class examples in the
reference README are not reproduced here; their omega/kappa and class dropout
settings require a class-conditioned model.

The intentional harmonizer adaptations are the opacity-guided rendered/noise
source endpoint, Cosmos tokenizer/backbone, current-cell-only supervision, and the original
harmonizer optimizer settings (betas 0.9/0.999, weight decay 0.01). The
reference optimizer defaults are betas 0.9/0.95 and weight decay 0; those
optimization hyperparameters are not part of the loss-parity claim.
Mathematical loss/gradient parity and interval-embedding parity are checked
against the actual local reference code on small CPU models. These checks do
not establish full Cosmos training parity or GPU compatibility.

The reference maintains EMA at decay 0.9999. Three independent switches,
`STAGE1_EMA`, `STAGE2_EMA`, and `STAGE3_EMA`, default to 1; use 0 to disable
any stage's EMA. Shadows are float32 on CPU, updated once per optimizer step
on every training rank. Online weights initialize the next stage. EMA is carried
across handoffs when available, or initialized from online parameters after an
EMA-disabled stage. Saved checkpoints contain online weights and optional EMA.
Set `USE_EMA=0` for inference when the final stage has no EMA.

New checkpoints record `cosmos_t_plus_meanflow_t_minus_r_v2` in conditioning
metadata. Old direct-r MeanFlow checkpoints are rejected before loading. They
cannot be silently reinterpreted as t-r; restart this curriculum from Cosmos.
Ordinary rectified-flow inference still uses its original model path.

## Stage-III history generation

For each history frame separately, under no_grad:

```python
x0 = encode(history_GT)
render = encode(history_render)
x1 = opacity_latent * render + (1 - opacity_latent) * gaussian_noise
z = (1 - t_s) * x0 + t_s * x1
history = decode(z - t_s * u(z, r=0, t=t_s)).detach()
```

The endpoint matches the current-frame training bridge. Resampling times use
LogitNormal(0,1) followed by shift 0.6. Mode B resamples independently without
previous-frame context, then inserts the four predictions newest-first into
the original V=5 slots. One shifted `t_s` is shared by the four history frames
in a training window; each frame gets its own Gaussian noise. The resampling
calls still use that history frame's reference images and camera/time conditions.
GT-based corruption occurs only during training. The diffusion sampler does
not consume target/history GT at inference; the current manifest interface
still requires the `target_image` field for metadata/calibration and output records.

Choose `HISTORY_GENERATOR=frozen_stage2` (default) to generate with a fixed
Stage-II checkpoint. By default it uses Stage-II online weights;
`HISTORY_CHECKPOINT_EMA=1` selects that checkpoint's saved EMA instead.
The checkpoint is never updated.

Choose `HISTORY_GENERATOR=stage3_ema` to regenerate histories from the current
Stage-III EMA before each update. This requires `STAGE3_EMA=1`. Weights are
swapped only during no-grad history generation and restored before the
training graph is built, avoiding a second Cosmos model on GPU.

The Resampling Forcing authors use **current online weights**, stopped
gradients, autoregressive history resampling, and teacher-forcing warm-up:
[Resampling Forcing, Section 3.2](https://arxiv.org/html/2512.15702v2#S3.SS2).
Their paper treats a generated chunk
as a frame. EMA and fixed-checkpoint generation here are requested variants;
independent mode-B resampling does not reproduce their causal autoregressive
history construction or sparse routing. EMA provides smoother but lagged errors,
so it may differ from the online model's current inference errors.

## Launch
If STAGE1_EMA=0, validation uses Stage-I online weights instead.
At the handoff, Stage II starts from Stage-I online weights. Its EMA carries over when compatible.

If STAGE2_EMA=0, validation uses Stage-II online weights.
At the handoff, Stage III starts from Stage-II online weights. The Stage-II checkpoint also supplies the default training-history generator.

This is why STAGE3_EMA=1 does not automatically select EMA training histories: it enables EMA maintenance and EMA validation. To also use that EMA for training-history generation, select HISTORY_GENERATOR=stage3_ema.

```bash
cd /fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/gen-video/harmonizer

DATASET_FOLDER=../../data/dh_train/smoke_nuscenes_dr2m/dataset_fixed_train200_seed42.json \
VALIDATION_DATASET=../../data/dh_train/smoke_nuscenes_dr2m/dataset_fixed_val20_seed42.json \
VALIDATION_SPLIT=test \
IMAGE_PREP=resize_256x256 \
OUTPUT_ROOT=../../train_meanflow/nuscenes_three_stage_256 \
NUM_GPUS=8 NUM_REFERENCE_IMAGES=4 \
STAGE1_STEPS=2000 STAGE2_STEPS=1000 WARMUP_STEPS=100 \
HISTORY_GENERATOR=stage3_ema \
STAGE1_EMA=0 STAGE2_EMA=0 STAGE3_EMA=1 \
WANDB_MODE=disabled REPORT_TO=none \
VIZ_FREQ=50 EVAL_FREQ=100 \
NUM_SAMPLES_EVAL=20 VALIDATION_STEPS=1 \
pixi run bash training_script/train_nuscenes_meanflow_three_stage.sh
```

For two Slurm nodes with eight GPUs each, start **one launcher per node**.
`NUM_GPUS=16` is the total worker count; Accelerate starts eight workers on
each node. The launcher reads the machine rank from `SLURM_NODEID` and uses
the first allocated node as the shared master address. The three stages use
master ports 29521, 29522, and 29523 (override the base with `MAIN_PROCESS_PORT`).
This follows the [Accelerate multi-node launch instructions](https://huggingface.co/docs/accelerate/main/en/basic_tutorials/launch#multi-node-training).

Before launching multi-node training, the shell raises the locked-memory limit
with `ulimit -l unlimited`. On allocation 7814456, both nodes defaulted to an
8 MiB soft limit (with an unlimited hard limit), causing NCCL's RoCE transport
to fail at VAE synchronization with `ibv_reg_mr_iova2 failed ... Cannot allocate memory`.
A 16-worker parameter-shape check and all-reduce reproduced the failure and
passed on all workers after raising this limit. This follows
[NVIDIA's guidance on pinned-memory registration limits](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html#troubleshooting-nccl-errors).
The change affects the launcher and its child processes; it does not edit
system configuration or replace RoCE with TCP. The launcher fails if the shell
cannot raise the limit. Dry runs leave the limit untouched.

Run this from the allocation shell. If you entered a compute node through
`srun --pty bash`, first `exit` that interactive step back to the allocation
shell so it releases the resources needed by the new step. With allocation 7814456:

```bash
cd /fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/gen-video/harmonizer

DATASET_FOLDER=../../data/dh_train/smoke_nuscenes_dr2m/dataset_fixed_train200_seed42.json \
VALIDATION_DATASET=../../data/dh_train/smoke_nuscenes_dr2m/dataset_fixed_val20_seed42.json \
VALIDATION_SPLIT=test \
IMAGE_PREP=resize_256x256 \
OUTPUT_ROOT=../../train_meanflow/nuscenes_three_stage_256 \
NUM_NODES=2 NUM_GPUS=16 NUM_REFERENCE_IMAGES=4 \
STAGE1_STEPS=2000 STAGE2_STEPS=1000 WARMUP_STEPS=100 \
HISTORY_GENERATOR=stage3_ema \
STAGE1_EMA=0 STAGE2_EMA=0 STAGE3_EMA=1 \
WANDB_MODE=disabled REPORT_TO=none \
VIZ_FREQ=50 EVAL_FREQ=100 \
NUM_SAMPLES_EVAL=20 VALIDATION_STEPS=1 \
srun --jobid=7814859 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --cpus-per-task=32 --cpu-bind=none \
  --gres=gpu:rtxa5000:8 --kill-on-bad-exit=1 --label \
  pixi run bash training_script/train_nuscenes_meanflow_three_stage.sh
```

The environment assignments apply to `srun`, which exports them to both
node launchers. `--kill-on-bad-exit=1` terminates the other launcher if one
fails. Use `DRY_RUN=1` before `srun` to print all three stage commands on both
nodes without starting training. Code, data, weights, output directories, and
the Pixi environment are shared through `/fs`. Outside Slurm, explicitly set
`MACHINE_RANK` separately on each node and the same `MAIN_PROCESS_IP` on both.

The original allocation 7814456 reserved only one CPU per node. For the next
allocation, request 32 CPUs per launcher for training and data loading:

```bash
salloc --account=gamma --partition=gamma --qos=gamma-huge-long \
  --nodes=2 --ntasks-per-node=1 --cpus-per-task=32 \
  --gres=gpu:rtxa5000:8 --mem=96G --time=12:00:00
```

Then replace the example `--jobid=7814456` with `--jobid="${SLURM_JOB_ID}"`
and add `--cpus-per-task=32 --cpu-bind=none` to `srun`. CPU binding is disabled
so the eight GPU workers can share the node's allocated CPUs rather than all
being pinned to the same core. `salloc` reserves resources; `srun` executes
the launchers on those resources ([Slurm documentation](https://slurm.schedmd.com/srun.html)).

This local command uses two saved selections beside `dataset_smoke_r1.json`:
`dataset_fixed_train200_seed42.json` contains qualified frames from 200 fixed
R2M nuScenes scene IDs, selected from the available scene folders rather than
the small smoke manifest's seven training scenes. Scene selection uses seed 42
and requires a five-frame window after reference filtering. It uses four highest-ranked
references with visibility >1000, excluding the current source image, via the
existing `training_script/create_dataset.py` builder. Training shuffles frame
records each epoch and retains the 75% temporal / 25% single-frame mixture;
it does not resample scene membership on subsequent runs.

`dataset_fixed_val20_seed42.json` fixes current frames 7..26 of
`R2M_scene-0138`, outside the 200 training scenes and matching their render direction.
Every target has four consecutive GT-history paths, including
external preceding frames for the first targets. All stages and both Stage-II
validation modes therefore evaluate the same 20 current targets. Autoregressive
modes still generate their first four selected targets without history, then
reuse generated predictions. Scene-0138 is excluded from all training variants.
GT context and reference images are conditioning and do not count toward the
20 evaluated targets.

The selected scene IDs and validation keys are recorded in each JSON's
`selection` metadata. Reuse these files for subsequent training runs. The creation
command was:

```bash
pixi run python training_script/create_fixed_meanflow_subsets.py \
  --manifest ../../data/dh_train/smoke_nuscenes_dr2m/dataset_smoke_r1.json \
  --train-scenes 200 --val-frames 20 --seed 42 --from-scene-folders
```

The script refuses to overwrite an existing training selection and preserves
an existing fixed validation JSON. Its `--from-scene-folders` option uses the
training render directions and reference count from the source smoke manifest.

```bash
# Full curriculum; defaults preserve 200k + 100k total steps.
pixi run bash training_script/train_nuscenes_meanflow_three_stage.sh

# Alternative history generator.
HISTORY_GENERATOR=stage3_ema STAGE3_EMA=1 \
  pixi run bash training_script/train_nuscenes_meanflow_three_stage.sh

# Preview all commands without loading data/models or launching training.
DRY_RUN=1 pixi run bash training_script/train_nuscenes_meanflow_three_stage.sh

# Small one-GPU smoke run (GPU allocation required; data has four references).
DATASET_FOLDER=../../data/dh_train/smoke_nuscenes_dr2m/dataset_smoke_r1.json \
NUM_GPUS=1 NUM_REFERENCE_IMAGES=4 STAGE1_STEPS=20 STAGE2_STEPS=20 WARMUP_STEPS=5 \
CHECKPOINTING_STEPS=5 OUTPUT_ROOT=/tmp/harmonizer-three-stage-smoke \
  pixi run bash training_script/train_nuscenes_meanflow_three_stage.sh
```

Defaults retain the original /project dataset path. Override `DATASET_FOLDER`
and `OUTPUT_ROOT` for your allocation. `NUM_GPUS=1` supports a single process.
The launcher sets `COSMOS_CHECKPOINT_DIR` to the local 0.6B directory under
`../../weights/checkpoints/nvidia/`. For direct Python calls, set the model size
and original checkpoint directory explicitly:

```bash
export COSMOS_MODEL_SIZE=0.6B
export COSMOS_CHECKPOINT_DIR=/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/weights/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image
```

`get_pipeline_config` otherwise defaults to model size 2B and its checkpoint
directory. The curriculum trainer overrides model size to 0.6B but does not
override that directory; direct inference CLIs override neither variable.
Stage I direct CLI calls also require `--lambda-temporal 0`.
The launchers use bf16, matching the original smoke script. The exposed direct
CLI `--mixed-precision no` has a dtype mismatch in the shared model path:
weights move to float32 but `_predict_velocity` still casts inputs using the
pipeline's stored bf16 `tensor_kwargs`. This also exists in the original trainer;
the current audit does not change that precision/configuration path.
For resuming an individual stage, use `src/train_meanflow_curriculum.py --resume`
with the same stage/options. A frozen-generator Stage-III resume must also
supply `--history-checkpoint` pointing to the original Stage-II checkpoint.
Checkpoint optimizer/scheduler and EMA progress are restored; data-loader/RNG
position is not restored, so resume is not bit-for-bit replay.

## Inference
```bash
DATASET_JSON=/project/data/dh_train/test_nuscenes_monocular_shifted_2m_final/dataset.json \
MODEL_PATH=/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/train_meanflow/nuscenes_three_stage/stage3/checkpoints/model_90000.pkl \
OUTPUT_DIR=/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/train_meanflow/nuscenes_three_stage/inference_step1 \
NUM_GPUS=4 \
NUM_REFERENCE_IMAGES=8 \
IMAGE_PREP=resize_288x512 \
CAMERA_EMBED=1 TIME_EMBED=1 \
USE_EMA=1 \
inference_step=1 \
pixi run bash testing_script/infer_nuscenes_meanflow_three_stage.sh
```

```bash
DATASET_JSON=/path/to/test/dataset.json \
MODEL_PATH=/path/to/stage3/checkpoints/model_90000.pkl \
  pixi run bash testing_script/infer_nuscenes_meanflow_three_stage.sh
```

The launcher follows `testing_script/eval_nuscenes.sh`'s scene-parallel
inference, adding `--meanflow`, one-step interval sampling, and optional EMA.
It loads final-stage weights and uses V=5 only when four consecutive previous
predictions exist in the same sequence. The first four consecutive frames use
V=1; gaps or missing frame indices also make a sample use V=1. It does not
resample GT at inference.
Use matching `IMAGE_PREP`, `NUM_REFERENCE_IMAGES`, `CAMERA_EMBED`, and `TIME_EMBED`.
Set `USE_EMA=0` for online weights. The single-GPU `src/inference_drivenvs.py`
also now supports `--meanflow --meanflow_use_ema --num_inference_steps 1` and
string resolution choices such as `--resolution resize_288x512`.
The shell launcher defaults to one step and EMA enabled; direct inference
Python CLIs default to four steps and online weights unless flags are supplied.
MeanFlow ignores the legacy `timestep_shift` argument and uses uniform intervals.

The inference launcher does not append training's `_waft` output suffix.
For a WAFT-trained run, set `MODEL_PATH` and `OUTPUT_DIR` explicitly, or set
`OUTPUT_ROOT` to the exact training directory. LoRA stage switches alone only
select the `_lora_s2..._s3...` suffix, not `_waft`.

For the configured tokenizer, V=1 encodes to one latent time cell and V=5 to
two. The DiT uses a shared patch projection and RoPE; its learned parameter
shapes do not depend on sequence length. Stage-II GT context warm-up learns
how to use the extra temporal context without a parameter-size mismatch.

## Training previews and validation

The three-stage launcher enables `VIZ_FREQ=1000` and `EVAL_FREQ=1000` by default.
Set either to 0 to disable it. `NUM_SAMPLES_EVAL=20` limits validation to a fixed
chronological subset; `VALIDATION_STEPS=1` uses one MeanFlow sampling interval.
Set `VALIDATION_STEPS=2` for two-step validation. The default validation dataset
is the training manifest's `test` split; use `VALIDATION_DATASET=/path/to/dataset.json`
and `VALIDATION_SPLIT=test` to select a separate held-out manifest. Direct Python
options are `--viz-freq`, `--eval-freq`, `--num-samples-eval`, `--validation-steps`,
`--validation-dataset`, and `--validation-split`.

At each preview step, images are written under
`stageN/visualizations/step_XXXXXXX/train/`: current input, target, prediction,
references, coverage, and a separate newest-first history strip when V=5.
Stage I has no history; Stage II shows the actual GT histories; Stage III shows
the actual detached resampled histories from the selected training generator.
The 25% V=1 samples in Stages II/III also have no history strip. These are online
training-forward predictions from a GT-interpolated latent at sampled t, using
the r=0 clean prediction; they are not source-endpoint inference previews.

Validation generates the current frame from its render/opacity/noise source
endpoint, never from a current-GT-interpolated latent. Stage I uses no history.
Stage II runs both modes at each validation interval: the existing GT-history
mode evaluates up to 20 eligible current frames with four GT histories;
the autoregressive mode evaluates up to 20 chronological current frames and
uses previous predictions once four consecutive predictions exist. Each mode
uses its own fixed subset (eligible GT-history samples versus all chronological
samples), so their current-frame sets need not match. `NUM_SAMPLES_EVAL` applies
separately to each mode, for up to 40 current predictions per Stage-II validation.
Stage III rolls out chronologically with the current stage's predictions: the
first four consecutive frames use V=1, then V=5 uses the previous four predictions.
Sequence changes and frame gaps prevent invalid history reuse. This follows
inference rather than training's GT-bridge history resampling; the frozen Stage-II
training generator is not used for Stage-III validation.

Validation uses current-stage EMA when that stage enables EMA, otherwise online
weights, for both current frames and generated histories. Online parameters,
training mode, frozen-VAE mode, and training RNG state are restored afterward.
Only the main rank validates; other ranks wait before the next training step.
Images and sample metadata are saved under
`stageN/visualizations/step_XXXXXXX/validation/sample_XXX/`.
Stage-II autoregressive previews use
`stage2/visualizations/step_XXXXXXX/validation_autoregressive/sample_XXX/`.
Both Stage-II modes use the same current-stage EMA/online selection.
Current-frame `val/l2` and `val/lpips`, the number of samples using temporal
context, and the EMA selection are logged to the tracker and
`stageN/validation.jsonl`.
Stage-II autoregressive metrics have a separate `val_autoregressive/` prefix
in the same tracker and JSONL file; existing GT-history metrics retain `val/`.
Coverage previews preserve [0,1]; RGB previews map [-1,1] to [0,1]. These metrics
do not directly measure temporal flicker.

## Validation checks

```bash
pixi run python -m pytest tests -q
pixi run python src/train_meanflow_curriculum.py --stage 2 \
  --dataset ../../data/dh_train/smoke_nuscenes_dr2m/dataset_smoke_r1.json \
  --references 4 --max-steps 1 --output-dir /tmp/meanflow-check-data \
  --check-data --report-to none
```

CPU tests cover reference losses and parameter gradients on both JVP backends,
interval embedding, generated-vs-GT history separation, bridge endpoints,
EMA updates, and safe generator-weight restoration. Dataset checks load both
V=1 samples and five-frame windows. The documented data check currently finds
350 samples and 322 temporal windows. The CPU suite passes 44 tests, including
LoRA, masked temporal-loss normalization, and WAFT transport tests. Real WAFT
checkpoint/flow and temporal-loss gradients were also checked on CPU as described
above. Actual Cosmos JVP/backward, GPU memory, training convergence, and image
inference require a GPU allocation and have not been validated here. RAFT
checkpoint loading is not exercised by these tests. Activation checkpointing
is left off because
MeanFlow's runtime conditioning/JVP path has not been GPU-validated with it.
Direct comparisons on three real smoke-data windows confirm Stage-II source,
reference, target, opacity and camera/time tensors exactly match the original
two-stage loader. Both loaders find 322 eligible windows in that manifest.
The new curriculum requires five consecutive records in the manifest; unlike
the original loader, it does not construct windows solely from external
`temporal_history_images` records.

## Optional LoRA in Stages II and III

Full training remains the default. Use independent `STAGE2_LORA=1` and
`STAGE3_LORA=1` switches to enable adapters in either temporal stage:

```bash
STAGE2_LORA=1 STAGE3_LORA=1 LORA_RANK=16 LORA_ALPHA=16 \
  pixi run bash training_script/train_nuscenes_meanflow_three_stage.sh
```

LoRA runs default to a separate output directory with an `_lora_s2..._s3...`
suffix. An explicit `OUTPUT_ROOT` overrides this. Stage I still trains the full
model. To start an individual LoRA stage from an already trained preceding
stage, call `src/train_meanflow_curriculum.py --stage 2 --lora` (or stage 3),
with `--initial-checkpoint`, the ordinary dataset/model options, and its budget.
The trainer's `--help` lists all arguments.

The [official Cosmos text-to-image guide](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_text2image.md)
supports `model.config.train_architecture=lora`.
That flag is interpreted by its official training wrapper, which this custom
MeanFlow trainer does not invoke. We use the same installed PEFT
`LoraConfig` + `inject_adapter_in_model` mechanism instead. Trainable adapters
are float32, including after bf16 model moves, as in the official implementation.

The targets strictly follow the local SimpleTuner Cosmos default:
`SimpleTuner/simpletuner/helpers/models/cosmos/model.py` defines
`[to_k, to_q, to_v, to_out.0]`. These map to native Cosmos
`[k_proj, q_proj, v_proj, output_proj]` in both self- and cross-attention.
The launcher default is rank 16, alpha 16, dropout 0, standard PEFT initialization
with zero B weights, and no trained biases or extra modules. Set `LORA_RANK`,
`LORA_ALPHA`, and `LORA_DROPOUT` to change those adapter hyperparameters.
In the launcher, alpha defaults to the selected rank; the direct Python CLI's
`--lora-alpha` default remains 16 independently of `--lora-rank`.
MLPs, the pretrained time path, the learned interval embedder, reference
projections, camera conditioning, and the output head are frozen in LoRA mode.
Enabling LoRA does not change the MeanFlow or selected optical-flow objectives.

Full -> LoRA handoffs load preceding weights before injecting zero-residual
adapters. LoRA -> LoRA handoffs retain adapters; rank/alpha/dropout must match
for continuation. LoRA -> full handoffs merge the learned deltas and enable
full training. Resuming an individual run cannot change its training mode.
EMA is carried across matching trainable parameter sets; it starts from the
new parameter set when switching between full and LoRA modes. The frozen
Stage-II history generator supports either source/student combination;
Stage-III EMA history generation swaps only trainable adapter parameters in
LoRA mode. Full-model behavior is unchanged when both switches are zero.

Checkpoints store the complete UNet/VAE state, including frozen/base weights,
adapter tensors and `lora_config`; they are not standalone
Diffusers/SimpleTuner adapter exports. The current loader still initializes a
Cosmos pipeline from the original `model.pt` and tokenizer checkpoint before
loading the harmonizer state, so those Cosmos files remain required.
Inference reconstructs the PEFT layers
before strict loading and applies saved EMA only to its trained parameters.
No separate adapter file or LoRA enable flag is required when `MODEL_PATH` is
specified. To infer the automatically named LoRA run, pass the same stage switches.

```bash
STAGE2_LORA=1 STAGE3_LORA=1 inference_step=2 \
  pixi run bash testing_script/infer_nuscenes_meanflow_three_stage.sh
```

## Time embeddings and two-step inference

In the reference SiT, t and t-r use the same embedder architecture with separate
weights. The harmonizer deliberately retains Cosmos's native t path (including
its AdaLN modulation output) and the reference SiT-style interval embedder.
The Cosmos path receives `t * t_scaling_factor`; the interval embedder receives
unscaled `t-r` in [0,1].
Those architectures differ. MeanFlow's identity requires dependence on both
interval endpoints and the correct JVP; it does not require weight sharing or
identical time-embedding architectures. Matching the native Cosmos embedder
would be another architecture choice and would require a new checkpoint version;
this LoRA change preserves the previously approved reference-style choice.
Cosmos's `use_adaln_lora` is a built-in modulation architecture, distinct from
optional PEFT LoRA training adapters.

`inference_step=2` (also `INFERENCE_STEP=2` or the existing
`NUM_INFERENCE_STEPS=2`) selects two intervals, with no training change:

```python
z_half = z_one - 0.5 * u(z_one, r=0.5, t=1.0)
z_zero = z_half - 0.5 * u(z_half, r=0.0, t=0.5)
```

One step remains the default. Multi-step output quality must be measured; two
calls are supported by the existing MeanFlow objective and sampler, but are not
guaranteed to improve a particular checkpoint. Tests cover adapter-only gradient
updates on both JVP backends, official injection parity, all full/LoRA frozen
history combinations, checkpoint inference/EMA loading, fp32 adapter preservation,
and the actual two-step sampler. Full Cosmos GPU training remains unverified in
the current CPU-only environment.
