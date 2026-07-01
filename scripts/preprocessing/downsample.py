import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os

def apply_gaussian_blur(img, kernel_size=3, sigma=1.0):
    """
    Blur before downsampling to simulate real-world lens blur.
    kernel_size must be odd (3, 5, 7...).
    Higher sigma = more blur.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def downsample_image(img, scale=2, interpolation=cv2.INTER_CUBIC):
    """
    Shrink image by 'scale' factor.
    INTER_CUBIC is standard for SR research (matches what SR models expect).
    """
    h, w = img.shape[:2]
    new_w, new_h = w // scale, h // scale
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

def process_dataset(
    input_dir,
    output_dir,
    scale=2,
    apply_blur=True,
    blur_kernel=5,  # modified to 5 for better blur effect
    blur_sigma=2, # modified to 2 for stronger blur
    min_size=64,        # skip images too small to downsample meaningfully
    extensions=(".jpg", ".jpeg", ".png")
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    ]

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    skipped = []
    processed = []
    metadata = []  # track original and output sizes for verification

    for img_path in tqdm(image_paths, desc="Downsampling"):
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"  Warning: could not read {img_path.name}, skipping.")
            skipped.append(img_path.name)
            continue

        h, w = img.shape[:2]

        # Skip images that are too small
        if h < min_size * scale or w < min_size * scale:
            skipped.append(img_path.name)
            continue

        orig_h, orig_w = h, w

        # Step 1: optional blur
        if apply_blur:
            img = apply_gaussian_blur(img, kernel_size=blur_kernel, sigma=blur_sigma)

        # Step 2: downsample
        img_lr = downsample_image(img, scale=scale)
        lr_h, lr_w = img_lr.shape[:2]

        # Save with same filename so it maps 1:1 to original
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img_lr)

        processed.append(img_path.name)
        metadata.append({
            "filename": img_path.name,
            "original_size": [orig_w, orig_h],
            "lr_size": [lr_w, lr_h],
            "scale": scale,
            "blur_applied": apply_blur,
        })

    # Saving metadata — useful for the team to verify consistency
    meta_path = output_dir / "downsample_metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump({
            "config": {
                "scale": scale,
                "apply_blur": apply_blur,
                "blur_kernel": blur_kernel,
                "blur_sigma": blur_sigma,
                "interpolation": "INTER_CUBIC",
            },
            "total_processed": len(processed),
            "total_skipped": len(skipped),
            "images": metadata
        }, f, indent=2)

    print(f"\nDone.")
    print(f"  Processed : {len(processed)} images")
    print(f"  Skipped   : {len(skipped)} images")
    print(f"  Output    : {output_dir}")
    print(f"  Metadata  : {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shared downsampling script for SR project")
    parser.add_argument("--input",        default="data/original",  help="Folder of original images")
    parser.add_argument("--output",       default="data/low_res",   help="Where to save low-res images")
    parser.add_argument("--scale",        type=int,   default=2,    help="Downsample factor (2 = half size)")
    parser.add_argument("--blur",         action="store_true",       help="Apply Gaussian blur before downsampling")
    parser.add_argument("--blur-kernel",  type=int,   default=3,    help="Blur kernel size (must be odd)")
    parser.add_argument("--blur-sigma",   type=float, default=1.0,  help="Blur strength")
    parser.add_argument("--min-size",     type=int,   default=64,   help="Skip images smaller than this after scaling")
    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        scale=args.scale,
        apply_blur=args.blur,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        min_size=args.min_size,
    )