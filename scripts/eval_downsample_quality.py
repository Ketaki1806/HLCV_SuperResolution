"""PSNR/SSIM for downsampled images: compare HR vs bicubic-upsampled LR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing"))

from quality_metrics import SSIMConfig, psnr, ssim  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(folder: Path, max_images: int) -> list[Path]:
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:max_images] if max_images > 0 else paths


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _bicubic_upscale(lr_bgr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    up_bgr = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(up_bgr, cv2.COLOR_BGR2RGB)


def evaluate_downsample_bicubic(
    hr_dir: Path,
    lr_dir: Path,
    max_images: int = 0,
) -> dict:
    hr_paths = _collect_images(hr_dir, max_images)
    if not hr_paths:
        raise FileNotFoundError(f"No images in {hr_dir}")

    per_image: list[dict] = []
    psnr_scores: list[float] = []
    ssim_scores: list[float] = []

    for hr_path in hr_paths:
        lr_path = lr_dir / hr_path.name
        if not lr_path.is_file():
            print(f"skip (missing LR): {hr_path.name}")
            continue

        hr = _read_rgb(hr_path)
        lr_bgr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
        if lr_bgr is None:
            print(f"skip (bad LR): {hr_path.name}")
            continue

        up = _bicubic_upscale(lr_bgr, size_wh=(hr.shape[1], hr.shape[0]))
        p = psnr(hr, up)
        s = ssim(hr, up, cfg=SSIMConfig())

        row = {
            "file_name": hr_path.name,
            "hr_size": [int(hr.shape[1]), int(hr.shape[0])],
            "lr_size": [int(lr_bgr.shape[1]), int(lr_bgr.shape[0])],
            "psnr_db": p,
            "ssim": s,
        }
        per_image.append(row)
        psnr_scores.append(p)
        ssim_scores.append(s)
        print(f"{hr_path.name}: PSNR={p:.2f} dB, SSIM={s:.4f}")

    return {
        "config": {
            "hr_dir": str(hr_dir),
            "lr_dir": str(lr_dir),
            "max_images": max_images,
            "method": "bicubic_upsample_vs_hr",
        },
        "count": len(per_image),
        "mean": {
            "psnr_db": float(np.mean(psnr_scores)) if psnr_scores else None,
            "ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
        },
        "per_image": per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PSNR/SSIM: HR vs bicubic-upsampled low-res (downsampling quality check)"
    )
    parser.add_argument("--hr-dir", type=Path, required=True, help="Original HR images")
    parser.add_argument("--lr-dir", type=Path, required=True, help="Downsampled LR images")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images (0 = all)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/downsample_quality.json"),
        help="JSON output path",
    )
    args = parser.parse_args()

    summary = evaluate_downsample_bicubic(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        max_images=args.max_images,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mean = summary["mean"]
    print(f"\nMean ({summary['count']} images): PSNR={mean['psnr_db']:.2f} dB, SSIM={mean['ssim']:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
