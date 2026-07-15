"""COCO detection dataloader with SR preprocessing.
For YOLO model, it uses the same collate functions as COCODataModule.

Compatible with data/dataLoader/coco_dataLoader.py:
- Subclasses COCODetectionDataset (same _preprocess, boxes, labels)
- COCOSRDataModule uses the same collate functions as COCODataModule
- Batch format: (tensors, targets) for yolo/detr; (tensor_list, targets) for faster_rcnn
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from data.dataLoader.coco_dataLoader import (
    BaseDataModule,
    COCODetectionDataset,
    collate_faster_rcnn,
    collate_yolo_detr,
)
from src.eval.sr_strategies import STRATEGIES, SRModels, StrategyName

Strategy = StrategyName


class COCOSRDetectionDataset(COCODetectionDataset):
    """Load LR COCO images, apply SR, then use base _preprocess() for detector tensors."""

    def __init__(
        self,
        lr_dir: str | Path,
        annotation_file: str | Path,
        strategy: Strategy,
        sr_models: SRModels,
        model_type: Literal["faster_rcnn", "yolo", "detr"] = "yolo",
        scale: int = 2,
        max_images: int | None = None,
    ) -> None:
        self.lr_dir = Path(lr_dir)
        self.strategy = strategy
        self.sr_models = sr_models
        self.scale = scale

        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if not self.lr_dir.is_dir():
            raise FileNotFoundError(f"LR image dir not found: {self.lr_dir}")

        # Parent stores image_dir; we override __getitem__ to read from lr_dir instead.
        super().__init__(
            image_dir=lr_dir,
            annotation_file=annotation_file,
            model_type=model_type,
            max_images=max_images,
        )
        print(
            f"COCOSRDetectionDataset: {len(self.img_ids)} images | "
            f"strategy={strategy} | model={model_type}"
        )

    def __getitem__(self, idx: int) -> dict:
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        lr_path = self.lr_dir / img_info["file_name"]
        if not lr_path.is_file():
            raise FileNotFoundError(f"LR image missing: {lr_path}")

        lr_image = Image.open(lr_path).convert("RGB")
        hr_size = (int(img_info["width"]), int(img_info["height"]))

        from src.eval.sr_strategies import prepare_image

        prepared = prepare_image(
            strategy=self.strategy,
            lr_image=lr_image,
            hr_size=hr_size,
            models=self.sr_models,
            scale=self.scale,
        )

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        sr_image = prepared.image
        # Same keys as COCODetectionDataset, plus SR-specific fields in targets.
        return {
            "tensor": self._preprocess(sr_image),
            "sr_rgb": TF.to_tensor(sr_image),
            "image_id": img_id,
            "filename": img_info["file_name"],
            "orig_size": hr_size,
            "input_size": sr_image.size,
            "bbox_scale": float(prepared.bbox_scale),
            "strategy": self.strategy,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }


class COCOSRDataModule(BaseDataModule):
    """SR + detection DataModule; same collate/batch contract as COCODataModule."""

    def __init__(
        self,
        lr_dir: str | Path,
        annotation_file: str | Path,
        strategy: Strategy,
        sr_models: SRModels,
        model_type: Literal["faster_rcnn", "yolo", "detr"] = "yolo",
        scale: int = 2,
        heldout_split: float = 0.0,
        max_images: int | None = None,
        **loader_kwargs,
    ) -> None:
        max_im = max_images if max_images and max_images > 0 else None
        dataset = COCOSRDetectionDataset(
            lr_dir=lr_dir,
            annotation_file=annotation_file,
            strategy=strategy,
            sr_models=sr_models,
            model_type=model_type,
            scale=scale,
            max_images=max_im,
        )

        if "collate_fn" not in loader_kwargs:
            loader_kwargs["collate_fn"] = (
                collate_faster_rcnn if model_type == "faster_rcnn" else collate_yolo_detr
            )

        super().__init__(dataset, heldout_split=heldout_split, **loader_kwargs)
