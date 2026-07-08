# src/data/coco_dataset.py
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF
from pycocotools.coco import COCO

IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# -----------------------------
# BaseDataModule (from assignment)
# -----------------------------
class BaseDataModule:
    def __init__(self, dataset, heldout_split, **loader_kwargs):
        self.dataset       = dataset
        self.loader_kwargs = loader_kwargs
        self.n_samples     = len(self.dataset)
        self.heldout_split = heldout_split
        self.heldout_set   = None

        if self.heldout_split > 0.0:
            self.dataset, self.heldout_set = self._split_data(self.heldout_split)

        self.heldout_kwargs = deepcopy(self.loader_kwargs)
        self.heldout_kwargs.update(dict(shuffle=False))

    def get_loader(self):
        print(f"Initializing DataLoader for {len(self.dataset)} samples with {self.loader_kwargs}")
        return DataLoader(dataset=self.dataset, **self.loader_kwargs)

    def get_heldout_loader(self):
        assert self.heldout_set is not None, \
            "No heldout split created! Set heldout_split > 0.0 during initialization."
        print(f"Initializing heldout DataLoader for {len(self.heldout_set)} samples")
        return DataLoader(dataset=self.heldout_set, **self.heldout_kwargs)

    def _split_data(self, split):
        if split == 0.0:
            return self.dataset, None

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "Validation set larger than dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        len_train     = self.n_samples - len_valid
        train_dataset = Subset(self.dataset, list(range(len_train)))
        eval_dataset  = Subset(self.dataset, list(range(len_train, len_train + len_valid)))
        self.n_samples = len(train_dataset)
        return train_dataset, eval_dataset


# -----------------------------
# COCO Dataset
# -----------------------------
class COCODetectionDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        model_type: Literal["faster_rcnn", "yolo", "detr"] = "faster_rcnn",
        max_images: int | None = None,
    ) -> None:
        self.image_dir  = Path(image_dir)
        self.model_type = model_type
        self.coco       = COCO(str(annotation_file))

        img_ids = sorted(self.coco.getImgIds())
        if max_images:
            img_ids = img_ids[:max_images]
        self.img_ids = img_ids

        print(f"COCODetectionDataset: {len(self.img_ids)} images | model: {model_type}")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id   = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path  = self.image_dir / img_info["file_name"]
        image     = Image.open(img_path).convert("RGB")
        orig_size = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        return {
            "tensor":    self._preprocess(image),
            "image_id":  img_id,
            "filename":  img_info["file_name"],
            "orig_size": orig_size,
            "boxes":     torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4)),
            "labels":    torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,), dtype=torch.int64),
        }

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if self.model_type == "faster_rcnn":
            return TF.to_tensor(image)
        elif self.model_type == "yolo":
            return TF.to_tensor(TF.resize(image, [640, 640]))
        elif self.model_type == "detr":
            return TF.normalize(TF.to_tensor(TF.resize(image, [800, 800])), IMAGENET_MEAN, IMAGENET_STD)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


# -----------------------------
# Collate functions
# -----------------------------
def collate_faster_rcnn(batch: list[dict]) -> tuple[list, list]:
    tensors = [item["tensor"] for item in batch]
    targets = [{k: v for k, v in item.items() if k != "tensor"} for item in batch]
    return tensors, targets


def collate_yolo_detr(batch: list[dict]) -> tuple[torch.Tensor, list]:
    tensors = torch.stack([item["tensor"] for item in batch])
    targets = [{k: v for k, v in item.items() if k != "tensor"} for item in batch]
    return tensors, targets


# -----------------------------
# COCODataModule (combined)
# -----------------------------
class COCODataModule(BaseDataModule):
    """
     This class combines BaseDataModule + COCODetectionDataset.
    This is the only class you import anywhere else.

    Examples:
        # Faster R-CNN — eval only
        dm = COCODataModule(
            image_dir="data/original",
            annotation_file="/annotations/instances_val2017.json",
            model_type="faster_rcnn",
            heldout_split=0.0,
            batch_size=4,
        )
        loader = dm.get_loader()

        # DETR — fine-tuning with train/val split
        dm = COCODataModule(
            image_dir="data/original",
            annotation_file="/annotations/instances_minitrain2017.json",
            model_type="detr",
            heldout_split=0.2,
            batch_size=4,
            shuffle=True,
        )
        train_loader = dm.get_loader()
        val_loader   = dm.get_heldout_loader()
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        model_type: Literal["faster_rcnn", "yolo", "detr"] = "faster_rcnn",
        heldout_split: float = 0.0,
        max_images: int | None = None,
        **loader_kwargs,
    ) -> None:
        dataset = COCODetectionDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            model_type=model_type,
            max_images=max_images,
        )

        if "collate_fn" not in loader_kwargs:
            loader_kwargs["collate_fn"] = (
                collate_faster_rcnn if model_type == "faster_rcnn"
                else collate_yolo_detr
            )

        super().__init__(dataset, heldout_split=heldout_split, **loader_kwargs) 