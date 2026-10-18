# Code sync: world-scenario monocular + stereo experiments

Code, configs and small docs only. No datasets, no model weights.
`files.txt` lists every synced path (relative to the repo root, 117 files, about 2.9 MB):

```bash
rsync -a --files-from=github_sync/files.txt <repo_root>/ <git_checkout>/
```

Data layouts: [DATA_MONOCULAR.md](DATA_MONOCULAR.md), [DATA_STEREO.md](DATA_STEREO.md).

Paths below are relative to the repo root. `H = gen-video/harmonizer`, `U = feature-match/UFM`.

## 1. Environments

| Files | Role |
|---|---|
| `H/pyproject.toml`, `H/pixi.lock`, `H/setup_env_{a100,gb200}.py`, `H/setup_env_stack.sh`, `H/tokenizer.patch`, `H/text2image_dit.patch`, `H/ruff.toml` | Training/eval env. Setup installs `cosmos-predict2==1.0.9` and applies the two patches. |
| `U/pyproject.toml`, `U/pixi.lock`, `U/setup_env_stack.sh` | Data-creation env (DA3, SAM3, py123d). |
| `WAFT/pixi.toml`, `WAFT/pixi.lock`, `WAFT/config/a1/tar-c-t.json`, `WAFT/model/backbone/vit.py`, `WAFT/model/waft_a1.py` | Our local changes on top of `princeton-vl/WAFT@b152ff1`. WAFT is the optical-flow estimator for Stage II/III of our method. |

## 2. Data creation

All renderers run from `U/` (`pixi run python src/...`).

| File | Used for |
|---|---|
| `U/src/preprocess.py` (+ `dataset.py`, `reference_dedup*.py`, `reference_greedy_union.py`, `foundation/{da3,sam3}.py`, `utils/*.py`) | RGB pipeline: DA3 depth, SAM3 masks, warp, coverage, top-20 reference ranking, calibration. |
| `U/src/preprocess_cosmos_render.py` | HDMap / world-scenario / 3D box rendering. Used by every dataset below. |
| `U/src/config/double_reprojection/nuscene_smoke.yaml` | Monocular training data (`smoke_nuscenes_dr2m`) and monocular DR test. |
| `U/src/config/double_reprojection/nuscene.yaml` | Original official DR config (800×450, 200 frames), kept for reference. |
| `U/src/config/monocular_shift/nuscene.yaml` | Monocular 1 m shift test. |
| `U/src/config/stereo/{train,test}_nuscene_{LR,RL}.yaml` | Stereo train/test data. |
| `H/training_script/create_dataset.py` | Manifest builder (`build_sample`: 4 references, >1000 visible pixels). Also imported by the two UFM scripts below. |
| `H/training_script/create_nuscenes_four_direction_subsets.py`, `create_fixed_meanflow_subsets.py` | Monocular: 100-scene 4-direction train and validation manifests. |
| `H/training_script/extract_training_map.sh` | Monocular: renders map/box for `smoke_nuscenes_dr2m`. |
| `U/src/prepare_standard_eval.py` | Monocular test set `nuscenes_standard_eval10_4dir_512_v1` (DR + shift_1m). |
| `U/src/generate_stereo_world_scenario.py` + `.sh`, `U/src/regenerate_stereo_conditions.sh` | Stereo train100/test20 dataset (`prepare`/`worker`/`finalize`) and stereo test set (`prepare-eval`/`eval-worker`/`finalize-eval`). |
| `H/training_script/prepare_stereo_validation.py` | Stereo 80-frame validation manifest. |
| `U/src/README.md`, `README_HDMap.MD`, `STEREO_WORLD_SCENARIO.md` | Data-creation notes. |

**Missing script:** the monocular `dataset_map_box_{train100,test20,val80}_4dir_ordered.json` manifests were built ad hoc, and no script for them is in the repo. `data/.../MAP_BOX_ORDERED_SPLITS.md` documents how they were built (it reuses `create_dataset.py:build_sample`).

## 3. Training: `train_world_scenario.sh` and `train_stereo_four_experiments.sh`

```text
H/training_script/train_stereo_four_experiments.sh      (4 stereo runs: {baseline,ours} x mode {0,1})
└── H/training_script/train_world_scenario.sh --setting monocular|stereo --method baseline|ours --mode 0|1|2
    ├── baseline: train_nuscenes_smoke_100scene_4dir.sh -> src/train_drivenvs.py
    ├── ours:     train_nuscenes_meanflow_three_stage.sh -> src/train_meanflow_curriculum.py
    └── after training: testing_script/eval_world_scenario.sh (mono) | eval_stereo_world_scenario.sh (stereo)
```

Python dependencies, traced from imports:

- `H/src/train_drivenvs.py`, `train_meanflow_curriculum.py`: entry points.
- `H/src/drivenvs_model.py`, `meanflow_{model,loss,curriculum,conditioning,ema,lora,monitoring}.py`, `temporal_loss.py`, `resampling_forcing_model.py`, `self_forcing_harmonizer.py`: these are imported.
- `H/src/model/{__init__,text2image_dit,world_scenario,camera_conditioning,prope}.py`: DiT, world-scenario control branch, camera conditioning.
- `H/src/utils/{training_utils,batch_preparation,temporal,flow_match,inference_utils,camera_calibration,training_stop,profiling}.py`: dataset loader, batching, checkpoint-on-stop.
- `H/src/waft_{estimator,protocol,worker}.py`: run WAFT in its own pixi env.
- `H/WORLD_SCENARIO_EXPERIMENTS.md`, `STEREO_WORLD_SCENARIO_EXPERIMENTS.md`, `MEANFLOW_THREE_STAGE.md`: experiment notes.

## 4. Evaluation: `standard_eval.py` and the current pipelines

| File | Role |
|---|---|
| `H/testing_script/standard_eval.py` | Shared entry point (`single`, `worker`, `summary`, `--setting stereo`). It runs `src/evaluate_test_dataset.py` (inference at 512×288), exports MP4s with ffmpeg, then computes the metrics. |
| `H/src/evaluate_test_dataset.py` | Inference. Baseline uses 4 steps; ours uses 1 step with raw weights. |
| `H/testing_script/calculate_generative_metrics.py` | FID/KID (clean-fid) and FVD. |
| `H/testing_script/calculate_reconstruction_metrics.py` | PSNR/SSIM (DR and stereo only). |
| `H/testing_script/fvd/**.py` | VideoGPT / StyleGAN-V FVD implementations. |
| `H/testing_script/world_scenario_outputs.py` | `metrics-cache` (copies the Inception weight), `record-config`, video helpers. |
| `H/testing_script/eval_world_scenario.sh`, `eval_stereo_world_scenario.sh` | Post-training eval for monocular and stereo. |
| `H/testing_script/run_standard_eval.sh` | Multi-GPU Slurm launcher (prepare + evaluate, mono or stereo). |
| `H/testing_script/reevaluate_stereo_512.py` + `.sh` | Older stereo re-evaluation of saved PNGs against 512×288 GT. |
| `out_standard_eval10_4dir_512_v1/PROTOCOL.md`, `out_stereo_standard_eval10_2dir_512_v1/PROTOCOL.md` | Frozen evaluation protocols. |

## 5. Not synced (get separately)

| Item | Where it is expected |
|---|---|
| Cosmos-Predict2 0.6B Text2Image weights | `weights/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/{model.pt,tokenizer/tokenizer.pth}` |
| DA3 (`DA3NESTED-GIANT-LARGE-1.1`), SAM3 `sam3.pt` | `weights/...`, paths in the UFM YAMLs |
| WAFT checkpoint `tar-c-t.pth` | repo root |
| Inception (clean-fid) | `weights/metrics/cleanfid/inception-2015-12-05.pt` |
| I3D for FVD (`i3d_pretrained_400.pt`, `i3d_torchscript.pt`, ~50 MB each) | `H/testing_script/fvd/{videogpt,styleganv}/` |
| Vendored repos `U/Depth-Anything-3`, `U/sam3`, `U/UniCeption` | Clone upstream. They are editable path deps in `U/pyproject.toml`. |
| py123d-converted nuScenes logs + maps | `nuscenes-interpolated_{train,val}/scene-XXXX/*.arrow`, `maps/nuscenes/*.arrow` |
| Datasets / manifests (training manifests are 10–55 MB each) | see `DATA_*.md` |

The scripts hard-code `/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm` and `PIXI=/nfshomes/jlin1219/.pixi/bin/pixi` in several places. Edit them for a new machine.
