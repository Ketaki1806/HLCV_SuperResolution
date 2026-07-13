"""Compute PSNR/SSIM of upsampled images against HR references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing"))

from quality_metrics import SSIMConfig, psnr, ssim  # noqa: E402

from src.data.degradation import downscale_bicubic  # noqa: E402
from src.models.espcn import load_espcn, upscale_image as espcn_upscale  # noqa: E402
from src.models.fsrcnn import load_fsrcnn, upscale_image as fsrcnn_upscale  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(folder: Path, max_images: int) -> list[Path]:
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:max_images] if max_images > 0 else paths


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _bicubic_upscale(lr: Image.Image, hr_size: tuple[int, int]) -> Image.Image:
    return lr.resize(hr_size, Image.BICUBIC)


def _score_pair(hr: Image.Image, sr: Image.Image) -> dict[str, float]:
    hr_np = _to_rgb_array(hr)
    sr_np = _to_rgb_array(sr)
    if hr_np.shape != sr_np.shape:
        sr = sr.resize(hr.size, Image.BICUBIC)
        sr_np = _to_rgb_array(sr)
    return {
        "psnr_db": psnr(hr_np, sr_np),
        "ssim": ssim(hr_np, sr_np, cfg=SSIMConfig()),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float | None]:
    if not rows:
        return {"psnr_db": None, "ssim": None}
    return {
        "psnr_db": float(np.mean([r["psnr_db"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
    }


def evaluate_upsamplers(
    hr_dir: Path,
    lr_dir: Path | None,
    scale: int,
    methods: list[str],
    max_images: int,
    device: str | None,
) -> dict:
    import torch

    hr_paths = _collect_images(hr_dir, max_images)
    if not hr_paths:
        raise FileNotFoundError(f"No images found in {hr_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fsrcnn_model = None
    espcn_model = None
    if "fsrcnn" in methods:
        fsrcnn_model = load_fsrcnn(scale=scale, device=device)
    if "espcn" in methods:
        espcn_model = load_espcn(scale=scale, device=device)

    per_method: dict[str, list[dict]] = {m: [] for m in methods}
    per_image: list[dict] = []

    for hr_path in hr_paths:
        hr = Image.open(hr_path).convert("RGB")
        if lr_dir is not None:
            lr_path = lr_dir / hr_path.name
            if not lr_path.is_file():
                print(f"skip (missing LR): {hr_path.name}")
                continue
            lr = Image.open(lr_path).convert("RGB")
        else:
            lr = downscale_bicubic(hr, scale=scale)

        row: dict = {"file_name": hr_path.name, "methods": {}}

        if "bicubic" in methods:
            sr = _bicubic_upscale(lr, hr.size)
            scores = _score_pair(hr, sr)
            row["methods"]["bicubic"] = scores
            per_method["bicubic"].append(scores)

        if "fsrcnn" in methods and fsrcnn_model is not None:
            sr = fsrcnn_upscale(fsrcnn_model, lr)
            scores = _score_pair(hr, sr)
            row["methods"]["fsrcnn"] = scores
            per_method["fsrcnn"].append(scores)

        if "espcn" in methods and espcn_model is not None:
            sr = espcn_upscale(espcn_model, lr)
            scores = _score_pair(hr, sr)
            row["methods"]["espcn"] = scores
            per_method["espcn"].append(scores)

        per_image.append(row)
        print(f"{hr_path.name}: " + ", ".join(f"{k} PSNR={v['psnr_db']:.2f} SSIM={v['ssim']:.4f}" for k, v in row["methods"].items()))

    summary = {
        "config": {
            "hr_dir": str(hr_dir),
            "lr_dir": str(lr_dir) if lr_dir else None,
            "scale": scale,
            "methods": methods,
            "max_images": max_images,
            "device": device,
        },
        "mean": {method: _mean_metrics(per_method[method]) for method in methods},
        "per_image": per_image,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PSNR/SSIM: compare upsamplers vs HR reference")
    parser.add_argument("--hr-dir", type=Path, required=True, help="High-resolution reference images")
    parser.add_argument("--lr-dir", type=Path, default=None, help="Low-res inputs (optional; else downscale HR)")
    parser.add_argument("--scale", type=int, default=2, help="Downscale factor if LR is generated")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bicubic", "fsrcnn", "espcn"],
        default=["bicubic", "fsrcnn", "espcn"],
    )
    parser.add_argument("--max-images", type=int, default=0, help="Limit images (0 = all)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sr_quality.json"),
        help="Where to save JSON summary",
    )
    args = parser.parse_args()

    summary = evaluate_upsamplers(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        scale=args.scale,
        methods=args.methods,
        max_images=args.max_images,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nMean scores vs HR:")
    for method, scores in summary["mean"].items():
        print(f"  {method}: PSNR={scores['psnr_db']:.2f} dB, SSIM={scores['ssim']:.4f}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
