"""Downstream detection eval on downsampled COCO val with SR strategies.

Reuses:
- data/dataLoader/coco_sr_dataLoader.py  (COCOSRDataModule)
- src/eval/yolo_coco.py                  (YOLO + shared COCO conversion)
- scripts/load_faster_rcnn.py            (Faster R-CNN loader)
- src/detr_eval.py                       (DETR loader + predict)
- src/eval/coco_evaluator.py             (mAP@0.5)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataLoader.coco_sr_dataLoader import COCOSRDataModule
from scripts.load_faster_rcnn import load_fasterrcnn, predict_loader_to_coco
from src.detr_eval import load_detr, predict_detr_batch_to_coco
from src.eval.coco_evaluator import GenericCOCOEvaluator
from src.eval.sr_strategies import STRATEGIES, StrategyName, load_sr_models
from src.eval.yolo_coco import (
    DETECTORS,
    DetectorName,
    dataloader_model_type,
    load_yolo_detector,
    predict_yolo_batch_to_coco,
    run_folder_name,
    validate_detector_strategy,
)


def evaluate_strategy(
    detector: DetectorName,
    strategy: StrategyName,
    lr_dir: Path,
    ann_file: Path,
    output_dir: Path,
    device: str,
    max_images: int,
    conf: float,
    scale: int,
    sr_models,
    batch_size: int = 1,
    weights: Path | None = None,
    detr_weights: str = "facebook/detr-resnet-50",
) -> dict:
    validate_detector_strategy(detector, strategy)
    run_dir = output_dir / run_folder_name(detector, strategy)
    run_dir.mkdir(parents=True, exist_ok=True)

    max_im = max_images if max_images > 0 else None
    dm = COCOSRDataModule(
        lr_dir=lr_dir,
        annotation_file=ann_file,
        strategy=strategy,
        sr_models=sr_models,
        model_type=dataloader_model_type(detector),
        scale=scale,
        heldout_split=0.0,
        max_images=max_im,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    loader = dm.get_loader()
    if len(dm.dataset) == 0:
        raise FileNotFoundError(f"No LR images found in {lr_dir} matching {ann_file}")

    anyup_model = sr_models.anyup if strategy == "anyup" else None

    if detector == "yolov8n":
        yolo, image_holder = load_yolo_detector(
            weights=weights or Path("yolov8n.pt"),
            strategy=strategy,
            anyup=anyup_model,
            device=device,
        )
        predictions: list[dict] = []
        for tensors, targets in tqdm(loader, desc=f"{detector} {strategy}"):
            predictions.extend(
                predict_yolo_batch_to_coco(yolo, targets, image_holder, device, conf=conf)
            )

    elif detector == "faster_rcnn":
        model = load_fasterrcnn(device)
        predictions = predict_loader_to_coco(loader, model, device, score_threshold=conf)

    else:
        finetuned = weights if detector == "detr_finetuned" else None
        model = load_detr(
            strategy=strategy,
            device=device,
            anyup=anyup_model,
            weights=detr_weights,
            finetuned_checkpoint=finetuned,
        )
        predictions = []
        for tensors, targets in tqdm(loader, desc=f"{detector} {strategy}"):
            predictions.extend(predict_detr_batch_to_coco(model, tensors, targets, conf=conf))

    pred_path = run_dir / "predictions.json"
    pred_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Wrote {len(predictions)} detections -> {pred_path}")

    metrics = GenericCOCOEvaluator(str(ann_file)).evaluate_predictions(predictions, str(run_dir))

    summary = {
        "detector": detector,
        "strategy": strategy,
        "lr_dir": str(lr_dir),
        "annotation_file": str(ann_file),
        "num_images": len(dm.dataset),
        "num_detections": len(predictions),
        "metrics": metrics,
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate detectors on downsampled COCO val with SR strategies"
    )
    parser.add_argument("--lr-dir", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--detector", choices=list(DETECTORS), default="yolov8n")
    parser.add_argument("--detectors", nargs="+", choices=list(DETECTORS), default=None)
    parser.add_argument("--strategies", nargs="+", choices=list(STRATEGIES), default=list(STRATEGIES))
    parser.add_argument("--weights", type=Path, default=None, help="YOLO or fine-tuned DETR weights")
    parser.add_argument("--detr-weights", default="facebook/detr-resnet-50")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    coco_root = os.environ.get("COCO_ROOT")
    ann_file = args.ann_file or (Path(coco_root) / "annotations/instances_val2017.json" if coco_root else None)
    if ann_file is None or not ann_file.is_file():
        raise SystemExit("ERROR: pass --ann-file or set COCO_ROOT")
    if not args.lr_dir.is_dir():
        raise SystemExit(f"ERROR: LR image dir not found: {args.lr_dir}")

    if args.device and args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but PyTorch has no CUDA support; using cpu")
        device = "cpu"
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    detectors: list[DetectorName] = args.detectors or [args.detector]
    sr_models = load_sr_models(strategies=args.strategies, device=device, scale=args.scale)

    all_summaries: dict[str, dict] = {}
    for detector in detectors:
        for strategy in args.strategies:
            key = run_folder_name(detector, strategy)
            print(f"\n=== {detector} / {strategy} ===")
            try:
                summary = evaluate_strategy(
                    detector=detector,
                    strategy=strategy,
                    lr_dir=args.lr_dir,
                    ann_file=ann_file,
                    output_dir=args.output_dir,
                    device=device,
                    max_images=args.max_images,
                    conf=args.conf,
                    scale=args.scale,
                    sr_models=sr_models,
                    batch_size=args.batch_size,
                    weights=args.weights,
                    detr_weights=args.detr_weights,
                )
                all_summaries[key] = summary
                m = summary["metrics"]
                print(f"{key}: mAP@0.5 all={m['mAP_50_all']:.4f}, small={m['mAP_50_small']:.4f}")
            except Exception as exc:
                print(f"ERROR {key}: {exc}")
                all_summaries[key] = {"error": str(exc)}

    out = args.output_dir / "sr_detection_comparison.json"
    out.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nSaved comparison: {out}")


if __name__ == "__main__":
    main()
