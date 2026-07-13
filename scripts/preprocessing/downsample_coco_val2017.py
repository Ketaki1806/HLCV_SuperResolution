from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from quality_metrics import SSIMConfig, psnr, ssim


def _apply_gaussian_blur(img_bgr: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img_bgr, (kernel, kernel), sigma)


def _downsample(img_bgr: np.ndarray, scale: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def _upsample(img_bgr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate downsampled COCO val2017 images (optionally compute PSNR/SSIM vs bicubic-upsampled)."
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path(os.environ.get("COCO_ROOT", "")) if os.environ.get("COCO_ROOT") else None,
        help="COCO root containing val2017/ and annotations/ (default: $COCO_ROOT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to save downsampled images (filenames match original)",
    )
    parser.add_argument("--scale", type=int, default=2, help="Downsample factor (default: 2)")
    parser.add_argument("--blur", action="store_true", help="Apply Gaussian blur before downsampling")
    parser.add_argument("--blur-kernel", type=int, default=5, help="Blur kernel size (odd)")
    parser.add_argument("--blur-sigma", type=float, default=2.0, help="Blur sigma")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Compute PSNR/SSIM between original and (LR bicubic-upsampled back to HR).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Write metrics JSON here (default: <output-dir>/downsample_quality.json)",
    )
    args = parser.parse_args()

    if args.coco_root is None:
        raise SystemExit("ERROR: --coco-root not provided and $COCO_ROOT is not set.")

    img_dir = args.coco_root / "val2017"
    ann_file = args.coco_root / "annotations" / "instances_val2017.json"

    if not img_dir.is_dir():
        raise SystemExit(f"ERROR: val2017 dir not found: {img_dir}")
    if not ann_file.is_file():
        raise SystemExit(f"ERROR: COCO annotations not found: {ann_file}")

    if args.scale < 2:
        raise SystemExit("ERROR: --scale must be >= 2")
    if args.blur_kernel % 2 == 0:
        raise SystemExit("ERROR: --blur-kernel must be odd")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = args.metrics_out or (args.output_dir / "downsample_quality.json")

    coco = COCO(str(ann_file))
    img_ids = sorted(coco.getImgIds())
    if args.max_images and args.max_images > 0:
        img_ids = img_ids[: args.max_images]

    quality_rows: list[dict] = []

    for img_id in tqdm(img_ids, desc="Downsampling COCO val2017"):
        info = coco.loadImgs(img_id)[0]
        fname = info["file_name"]
        src_path = img_dir / fname
        dst_path = args.output_dir / fname

        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        if args.blur:
            img_proc = _apply_gaussian_blur(img, kernel=args.blur_kernel, sigma=args.blur_sigma)
        else:
            img_proc = img

        lr = _downsample(img_proc, scale=args.scale)
        cv2.imwrite(str(dst_path), lr)

        if args.metrics:
            up = _upsample(lr, size_wh=(img.shape[1], img.shape[0]))
            row = {
                "image_id": int(img_id),
                "file_name": fname,
                "orig_size": [int(img.shape[1]), int(img.shape[0])],
                "lr_size": [int(lr.shape[1]), int(lr.shape[0])],
                "psnr_db": psnr(img, up),
                "ssim": ssim(img, up, cfg=SSIMConfig()),
            }
            quality_rows.append(row)

    summary = {
        "config": {
            "coco_root": str(args.coco_root),
            "split": "val2017",
            "output_dir": str(args.output_dir),
            "scale": int(args.scale),
            "blur": bool(args.blur),
            "blur_kernel": int(args.blur_kernel),
            "blur_sigma": float(args.blur_sigma),
            "metrics": bool(args.metrics),
        },
        "count_images": len(img_ids),
        "count_metrics": len(quality_rows),
        "metrics_mean": {
            "psnr_db": float(np.mean([r["psnr_db"] for r in quality_rows])) if quality_rows else None,
            "ssim": float(np.mean([r["ssim"] for r in quality_rows])) if quality_rows else None,
        },
        "per_image": quality_rows,
    }

    if args.metrics:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote metrics: {metrics_out}")

    print(f"\nDone. Wrote LR images to: {args.output_dir}")


if __name__ == "__main__":
    main()

