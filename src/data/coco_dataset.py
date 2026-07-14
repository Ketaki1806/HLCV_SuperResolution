import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.data.transforms import random_crop
from src.utils.config import load_yaml

DEFAULT_COCO_CONFIG = Path("configs/datasets/coco.yaml")


def resolve_coco_root(config_root: str | Path) -> Path:
    """Prefer COCO_ROOT from the environment (set on HPC after download)."""
    return Path(os.environ.get("COCO_ROOT", config_root))


def load_coco_dataset(
    config_path: str | Path = DEFAULT_COCO_CONFIG,
    **kwargs,
) -> "COCODataset":
    config = load_yaml(config_path)
    root = resolve_coco_root(config["root"])
    return COCODataset(
        root=root,
        split=config["split"],
        patch_size=config.get("patch_size"),
        **kwargs,
    )


class COCODataset(Dataset):
    """COCO image dataset for super-resolution training and evaluation."""

    SPLITS = ("train2017", "val2017")

    def __init__(
        self,
        root: str | Path,
        split: str = "val2017",
        patch_size: int | None = None,
        return_path: bool = False,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.return_path = return_path

        image_dir = self._resolve_image_dir(self.root, split)
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"COCO split directory not found for {split!r} under {self.root}. "
                "Set COCO_ROOT to the existing cluster dataset path "
                "(e.g. /scratch/teaching/hlcv/hlcv_team019/data/coco)."
            )

        self.image_paths = sorted(image_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _resolve_image_dir(root: Path, split: str) -> Path:
        candidates = [
            root / split,
            root / "images" / split,
            root / "images" if split == "val2017" else root / "images" / split,
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.jpg")):
                return candidate
        return root / split

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")

        if self.patch_size is not None:
            image = random_crop(image, self.patch_size)

        tensor = TF.to_tensor(image)

        if self.return_path:
            return tensor, str(path)
        return tensor
