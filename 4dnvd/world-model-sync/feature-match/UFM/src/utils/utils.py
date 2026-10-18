import copy
import gc
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from py123d.api import SceneFilter


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge updates into base without mutating either input."""
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """
    Load a YAML config file.

    JSON is still accepted for backward compatibility, but the package configs are YAML:
        config/default.yaml
        config/nuscene.yaml
        config/av2.yaml
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping/dict: {path}")
    return config


def cfg_get(config: dict, dotted_key: str, default: Any = None) -> Any:
    """Read nested config values using dot notation, e.g. cfg_get(cfg, 'render.max_depth')."""
    cur = config
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def cfg_set(config: dict, dotted_key: str, value: Any) -> None:
    """Set a nested config value using dot notation, creating intermediate dictionaries."""
    cur = config
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_cli_overrides(config: dict, overrides: list[str] | None) -> dict:
    """
    Apply simple key=value overrides after loading YAML.

    Values are parsed with yaml.safe_load, so these work:
        render.render_moving=false
        dataset.target_steps=80
        sam3.prompts='[vehicle, pedestrian]'
    """
    if not overrides:
        return config

    out = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected dotted.key=value, "
                "for example render.render_moving=false"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override with empty key: {item}")
        value = yaml.safe_load(raw_value)
        cfg_set(out, key, value)
    return out


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_data_roots(config: dict) -> None:
    data_root = cfg_get(config, "paths.temp_path", "/project/data")
    os.makedirs(data_root, exist_ok=True)
    os.environ["PY123D_DATA_ROOT"] = data_root
    os.environ["AV2_DATA_ROOT"] = cfg_get(
        config, "paths.av2_path", "/project/data/raw_data/av2"
    )
    os.environ["NUSCENES_DATA_ROOT"] = cfg_get(
        config, "paths.nusc_path", "/project/data/raw_data/nusc"
    )
    os.environ["KITTI360_DATA_ROOT"] = cfg_get(
        config, "paths.kitti360_path", "/project/data/raw_data/KITTI-360"
    )
    os.environ["WOD_PERCEPTION_DATA_ROOT"] = cfg_get(
        config, "paths.waymo_path", "/project/data/raw_data/waymo_percept"
    )


def tensor_to_rgb_numpy(x):
    return (x.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(
        np.uint8
    )


def make_output_dirs(config: dict) -> dict[str, Path | None]:
    root = Path(cfg_get(config, "paths.output_dir"))
    dataset_setup = str(
        cfg_get(config, "dataset.setup", cfg_get(config, "dataset.mode", "stereo"))
    ).strip().lower().replace("-", "_")
    has_real_target_rgb = dataset_setup not in {
        "monocular_shifted",
        "monocular_shift",
        "mono_shifted",
        "mono_shift",
    }

    # Canonical output names. These are intentionally named by view role, not by
    # dataset, so the same code path works for NuScenes and AV2 by config only.
    out: dict[str, Path | None] = {
        "source": root / cfg_get(config, "paths.front_source", "front_source"),
        "target": (
            root / cfg_get(config, "paths.front_target", "front_target")
            if has_real_target_rgb
            else None
        ),
        "warped": root / cfg_get(config, "paths.front_warped", "front_warped"),
        "coverage": root
        / cfg_get(config, "paths.confident_coverage_subdir", "confident_coverage"),
    }

    if bool(cfg_get(config, "save.hole_masks", True)):
        out["static_holes"] = root / cfg_get(
            config, "paths.static_hole_subdir", "static_holes"
        )
        out["dynamic_holes"] = root / cfg_get(
            config, "paths.dynamic_hole_subdir", "dynamic_holes"
        )
    else:
        out["static_holes"] = None
        out["dynamic_holes"] = None

    if bool(cfg_get(config, "save.seginstance_warped", True)):
        out["seginstance_warped"] = root / cfg_get(
            config, "paths.seginstance_warped_subdir", "seginstance_warped"
        )
    else:
        out["seginstance_warped"] = None

    # Optional debug/auxiliary outputs. If the corresponding save flag is false,
    # preprocess.py will not write these files and this function will not create
    # the directory.
    if bool(cfg_get(config, "save.vehicle_masks", True)):
        out["vehicle_masks"] = root / cfg_get(
            config, "paths.vehicle_mask_subdir", "dynamic_masks"
        )
    else:
        out["vehicle_masks"] = None

    if bool(cfg_get(config, "save.sky_masks", True)):
        out["sky_masks"] = root / cfg_get(config, "paths.sky_mask_subdir", "sky_masks")
    else:
        out["sky_masks"] = None

    if bool(cfg_get(config, "save.metadata", True)):
        out["metadata"] = root / cfg_get(config, "paths.metadata_subdir", "metadata")
    else:
        out["metadata"] = None

    if bool(cfg_get(config, "save.top20_reference", True)):
        out["top20_reference"] = root / cfg_get(
            config,
            "paths.top20_reference_subdir",
            "top20_reference",
        )
    else:
        out["top20_reference"] = None

    if bool(cfg_get(config, "save.calibration", True)):
        out["calibration"] = root / cfg_get(
            config,
            "paths.calibration_subdir",
            "calibration",
        )
    else:
        out["calibration"] = None

    for p in out.values():
        if p is not None:
            p.mkdir(parents=True, exist_ok=True)

    return out


def build_scene_filter(config: dict) -> SceneFilter:
    return SceneFilter(
        datasets=cfg_get(config, "dataset.datasets", ["nuscenes"]),
        split_types=cfg_get(config, "dataset.split_types", ["val"]),
        required_scene_modalities=cfg_get(config, "dataset.required_scene_modalities"),
        shuffle=bool(cfg_get(config, "dataset.shuffle_filter", True)),
    )


def get_batch_filter(config: dict):
    xs = cfg_get(config, "dataset.batch_indices", [])
    if xs is None or len(xs) == 0:
        return None
    return set(int(x) for x in xs)
