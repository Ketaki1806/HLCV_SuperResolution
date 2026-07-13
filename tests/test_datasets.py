from pathlib import Path

import pytest

from src.data.coco_dataset import load_coco_dataset
from src.utils.config import load_yaml


def test_coco_dataset_loads_sample(project_root: Path) -> None:
    config = load_yaml(project_root / "configs" / "datasets" / "coco.yaml")
    coco_root = project_root / config["root"] / config["split"]
    if not coco_root.is_dir() and not (Path.home() / "data" / "coco" / config["split"]).is_dir():
        pytest.skip("COCO not available (set COCO_ROOT to the cluster dataset path)")

    dataset = load_coco_dataset(project_root / "configs" / "datasets" / "coco.yaml")

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample.shape == (3, config["patch_size"], config["patch_size"])
