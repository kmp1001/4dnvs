"""Create the fixed stereo test subset used for periodic training previews."""

import json
from collections import defaultdict
from pathlib import Path

import click

DATA = Path(__file__).resolve().parents[3] / "data/dh_train/nuscenes_stereo_world_scenario_train100_test20"


@click.command()
def main():
    manifest = json.loads((DATA / "dataset_test.json").read_text())
    scenes = manifest["selection"]["scene_ids"][:4]
    groups = defaultdict(list)
    for key, sample in manifest["test"].items():
        groups[sample["scene"]].append((key, sample))
    selected = {}
    for scene in scenes:
        for direction in ("LR", "RL"):
            records = sorted(groups[f"{direction}_{scene}"], key=lambda item: item[1]["frame_index"])
            for start in range(len(records) - 9):
                window = records[start : start + 10]
                indices = [record[1]["frame_index"] for record in window]
                if indices == list(range(indices[0], indices[0] + 10)):
                    selected.update(window)
                    break
            else:
                raise ValueError(f"No ten consecutive retained frames in {direction}_{scene}")
    assert len(selected) == 80
    result = dict(manifest)
    result["test"] = selected
    result["selection"] = dict(manifest["selection"], scene_ids=scenes, purpose="fixed test previews during training")
    output = DATA / "dataset_validation_test80.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Saved {len(selected)} validation-preview frames: four test scenes x two directions x ten consecutive frames: {output}"
    )


if __name__ == "__main__":
    main()
