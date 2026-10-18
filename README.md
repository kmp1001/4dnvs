I've picked files (about 2.9 MB): code, configs and small docs only, with no datasets or weights. Everything is in [github_sync/](github_sync/):

- **[files.txt](github_sync/files.txt)**: every selected path, relative to the repo root. Copy them into a git checkout with `rsync -a --files-from=github_sync/files.txt <repo_root>/ <checkout>/`. All paths exist, and a scan found no credentials.
- **[README.md](github_sync/README.md)**: the files grouped into categories, with one line each on what they do:
  - **Data creation:** `preprocess.py` and `preprocess_cosmos_render.py` with their helpers and YAML configs, `create_dataset.py`, `prepare_standard_eval.py` (monocular test set) and `generate_stereo_world_scenario.py` (stereo train and test sets).
  - **Training:** the call tree from `train_stereo_four_experiments.sh` → `train_world_scenario.sh` → the baseline and ours stage scripts → `train_drivenvs.py` / `train_meanflow_curriculum.py`, plus every `src/` module they import. I found these by tracing imports, including scripts launched as subprocesses.
  - **Evaluation:** `standard_eval.py` → `evaluate_test_dataset.py`, the FID/FVD and PSNR/SSIM metric scripts, the `fvd/` code, both `eval_*_world_scenario.sh` scripts, `run_standard_eval.sh`, `reevaluate_stereo_512.*`, and the two `PROTOCOL.md` files.
- **[DATA_MONOCULAR.md](github_sync/DATA_MONOCULAR.md)** and **[DATA_STEREO.md](github_sync/DATA_STEREO.md)**: the folder trees with a few example file names, one example manifest record, image sizes, scene lists and which script builds each dataset.

Some choices you should check:

- **Left out because of size:** the training manifests (10–55 MB each), the two I3D `.pt` files for FVD (about 50 MB each), and the vendored `Depth-Anything-3`, `sam3` and `UniCeption` folders. The README says where each one has to be placed.
- **WAFT:** the copy has local edits on top of upstream commit `b152ff1`. I included only the changed files (`vit.py`, `waft_a1.py`, `pixi.toml`, `pixi.lock`) and the `tar-c-t.json` config, not the whole repo.
- **Missing script:** nothing in the repo builds `dataset_map_box_{train100,val80,test20}_4dir_ordered.json`; they were made by hand. Only `MAP_BOX_ORDERED_SPLITS.md` documents how. Training uses these files, so the README flags this.
- **Machine-specific paths:** many scripts hard-code `/fs/nexus-projects/...` and pixi path (`/nfshomes/jlin1219/.pixi/bin/pixi`). Collaborators will need to edit them.
- **Image sizes:** the monocular data on disk is already 512×288, because the `nuscene_smoke.yaml` config renders at that size. Only the stereo train100/test20 set is stored at 800×450.
