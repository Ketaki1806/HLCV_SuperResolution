# TEMP: replaced when downsampling branch merges.

"""Create low-resolution sample images in sample_lr/ from sample/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.degradation import downscale_bicubic

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(input_dir: Path, max_images: int) -> list[Path]:
    paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths[:max_images]


def create_sample_lr(
    input_dir: Path,
    output_dir: Path,
    scale: int = 4,
    max_images: int = 10,
    force: bool = False,
) -> int:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = collect_images(input_dir, max_images)
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for image_path in images:
        output_path = output_dir / image_path.name
        if output_path.exists() and not force:
            print(f"skip (exists): {output_path.name}")
            continue

        image = Image.open(image_path).convert("RGB")
        lr_image = downscale_bicubic(image, scale=scale)
        lr_image.save(output_path)
        print(f"wrote {output_path.name} ({image.size} -> {lr_image.size})")
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LR images in sample_lr/")
    parser.add_argument("--input", type=Path, default=Path("sample"))
    parser.add_argument("--output", type=Path, default=Path("sample_lr"))
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Integer downscale factor (default: 4 = quarter resolution)",
    )
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Overwrite existing LR files")
    args = parser.parse_args()

    count = create_sample_lr(
        input_dir=args.input,
        output_dir=args.output,
        scale=args.scale,
        max_images=args.max_images,
        force=args.force,
    )
    print(f"Done. Wrote {count} image(s) to {args.output}")


if __name__ == "__main__":
    main()
