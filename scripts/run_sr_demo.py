"""Run FSRCNN 2x on sample_lr/ images and save HR outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.visualize import save_comparison, save_image
from src.models.anyup import load_anyup
from src.models.fsrcnn import load_fsrcnn, upscale_image
from src.utils.config import load_yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(input_dir: Path, max_images: int) -> list[Path]:
    paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths[:max_images]


def run_sr_demo(
    input_dir: Path,
    output_dir: Path,
    max_images: int = 10,
    device: str | None = None,
) -> int:
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}. "
            f"Run: python scripts/create_sample_lr.py --input sample --output sample_lr"
        )

    images = collect_images(input_dir, max_images)
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    config_path = PROJECT_ROOT / "configs" / "models" / "fsrcnn.yaml"
    config = load_yaml(config_path) if config_path.is_file() else {}
    scale = int(config.get("scale", 2))

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading FSRCNN x{scale} on {device}")
    model = load_fsrcnn(
        scale=scale,
        d=int(config.get("d", 56)),
        s=int(config.get("s", 12)),
        m=int(config.get("m", 4)),
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for image_path in images:
        stem = image_path.stem
        lr_image = Image.open(image_path).convert("RGB")
        sr_image = upscale_image(model, lr_image)

        save_image(lr_image, output_dir / f"{stem}_lr.png")
        save_image(sr_image, output_dir / f"{stem}_fsrcnn_x{scale}.png")
        save_comparison(lr_image, sr_image, output_dir / f"{stem}_compare.png")
        print(f"processed {image_path.name} -> {lr_image.size} -> {sr_image.size}")
        processed += 1

    print("Verifying AnyUp loader...")
    try:
        load_anyup(device=device)
        print("AnyUp: loaded successfully")
    except Exception as exc:
        print(f"AnyUp: load failed ({exc})")

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FSRCNN on sample_lr images")
    parser.add_argument("--input", type=Path, default=Path("sample_lr"))
    parser.add_argument("--output", type=Path, default=Path("results/figures/sr_demo"))
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    count = run_sr_demo(
        input_dir=args.input,
        output_dir=args.output,
        max_images=args.max_images,
        device=args.device,
    )
    print(f"Done. Processed {count} image(s). Outputs in {args.output}")


if __name__ == "__main__":
    main()
