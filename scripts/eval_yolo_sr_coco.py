"""YOLO downstream detection eval on downsampled COCO val2017 with SR strategies.

Runs baseline + ESPCN + FSRCNN + AnyUp (feature-level in YOLO neck), saves COCO
predictions and mAP@0.5 metrics under runs/yolov8n_{strategy}/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.coco_evaluator import GenericCOCOEvaluator
from src.eval.sr_strategies import STRATEGIES, StrategyName, load_sr_models, prepare_image
from src.eval.yolo_coco import iter_coco_val_images, load_yolo_detector, yolo_results_to_coco


def evaluate_strategy(
    strategy: StrategyName,
    lr_dir: Path,
    ann_file: Path,
    output_dir: Path,
    weights: Path,
    device: str,
    max_images: int,
    conf: float,
    scale: int,
    sr_models,
) -> dict:
    run_dir = output_dir / f"yolov8n_{strategy}"
    run_dir.mkdir(parents=True, exist_ok=True)

    images = iter_coco_val_images(lr_dir=lr_dir, ann_file=ann_file, max_images=max_images)
    if not images:
        raise FileNotFoundError(f"No LR images found in {lr_dir} matching {ann_file}")

    anyup_model = sr_models.anyup if strategy == "anyup" else None
    yolo, image_holder = load_yolo_detector(
        weights=weights,
        strategy=strategy,
        anyup=anyup_model,
        device=device,
    )

    predictions: list[dict] = []
    for row in tqdm(images, desc=f"YOLO {strategy}"):
        lr_image = Image.open(row["lr_path"]).convert("RGB")
        prepared = prepare_image(
            strategy=strategy,
            lr_image=lr_image,
            hr_size=row["hr_size"],
            models=sr_models,
            scale=scale,
        )

        arr = np.array(prepared.image.convert("RGB"))
        if image_holder is not None:
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = tensor.to(device)
            image_holder["tensor"] = tensor

        results = yolo.predict(source=arr, conf=conf, imgsz=640, verbose=False)
        preds = yolo_results_to_coco(
            results[0],
            image_id=row["image_id"],
            bbox_scale=prepared.bbox_scale,
            conf_threshold=conf,
        )
        predictions.extend(preds)

    pred_path = run_dir / "predictions.json"
    pred_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Wrote {len(predictions)} detections -> {pred_path}")

    evaluator = GenericCOCOEvaluator(str(ann_file))
    metrics = evaluator.evaluate_predictions(predictions, str(run_dir))

    summary = {
        "detector": "yolov8n",
        "strategy": strategy,
        "lr_dir": str(lr_dir),
        "annotation_file": str(ann_file),
        "num_images": len(images),
        "num_detections": len(predictions),
        "metrics": metrics,
    }
    summary_path = run_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO on downsampled COCO val with SR strategies (baseline/espcn/fsrcnn/anyup)"
    )
    parser.add_argument(
        "--lr-dir",
        type=Path,
        required=True,
        help="Downsampled COCO val2017 images (LR)",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=None,
        help="instances_val2017.json (default: $COCO_ROOT/annotations/instances_val2017.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory for runs/yolov8n_{strategy}/",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES),
        default=list(STRATEGIES),
        help="SR strategies to evaluate",
    )
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images in LR dir")
    parser.add_argument("--conf", type=float, default=0.001, help="Detection confidence threshold")
    parser.add_argument("--scale", type=int, default=2, help="Downsample/SR scale factor")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    coco_root = os.environ.get("COCO_ROOT")
    ann_file = args.ann_file
    if ann_file is None:
        if coco_root:
            ann_file = Path(coco_root) / "annotations" / "instances_val2017.json"
        else:
            raise SystemExit("ERROR: pass --ann-file or set COCO_ROOT")

    if not ann_file.is_file():
        raise SystemExit(f"ERROR: annotations not found: {ann_file}")
    if not args.lr_dir.is_dir():
        raise SystemExit(f"ERROR: LR image dir not found: {args.lr_dir}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"LR dir: {args.lr_dir}")
    print(f"Annotations: {ann_file}")
    print(f"Strategies: {args.strategies}")

    sr_models = load_sr_models(strategies=args.strategies, device=device, scale=args.scale)

    all_summaries: dict[str, dict] = {}
    for strategy in args.strategies:
        print(f"\n=== Strategy: {strategy} ===")
        try:
            summary = evaluate_strategy(
                strategy=strategy,
                lr_dir=args.lr_dir,
                ann_file=ann_file,
                output_dir=args.output_dir,
                weights=args.weights,
                device=device,
                max_images=args.max_images,
                conf=args.conf,
                scale=args.scale,
                sr_models=sr_models,
            )
            all_summaries[strategy] = summary
            mean = summary["metrics"]
            print(
                f"{strategy}: mAP@0.5 all={mean['mAP_50_all']:.4f}, "
                f"small={mean['mAP_50_small']:.4f}, "
                f"medium={mean['mAP_50_medium']:.4f}, "
                f"large={mean['mAP_50_large']:.4f}"
            )
        except Exception as exc:
            print(f"ERROR strategy {strategy}: {exc}")
            all_summaries[strategy] = {"error": str(exc)}

    combined_path = args.output_dir / "yolov8n_sr_comparison.json"
    combined_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nSaved comparison: {combined_path}")


if __name__ == "__main__":
    main()
