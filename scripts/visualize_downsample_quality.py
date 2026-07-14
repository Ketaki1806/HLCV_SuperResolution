"""Plot PSNR/SSIM charts from downsample_quality.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(data: dict) -> tuple[list[dict], dict | None, dict | None]:
    """Support eval_downsample_quality and downsample_coco_val2017 JSON formats."""
    per_image = data.get("per_image", [])
    mean = data.get("mean") or data.get("metrics_mean")
    config = data.get("config")
    return per_image, mean, config


def _extract_scores(per_image: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    psnr = np.array([r["psnr_db"] for r in per_image], dtype=float)
    ssim = np.array([r["ssim"] for r in per_image], dtype=float)
    names = [r.get("file_name", f"image_{i}") for i, r in enumerate(per_image)]
    return psnr, ssim, names


def plot_downsample_quality(
    json_path: Path,
    output_dir: Path,
    dpi: int = 150,
) -> list[Path]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    per_image, mean, config = _load_rows(data)
    if not per_image:
        raise ValueError(f"No per_image entries in {json_path}")

    psnr, ssim, names = _extract_scores(per_image)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # --- Overview: 2x2 panel ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    title_bits = []
    if config:
        scale = config.get("scale")
        blur = config.get("blur")
        if scale is not None:
            title_bits.append(f"scale={scale}")
        if blur:
            title_bits.append("blur")
    subtitle = ", ".join(title_bits) if title_bits else "bicubic downsample"
    fig.suptitle(f"Downsample Quality ({len(per_image)} images, {subtitle})", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(psnr, bins=min(20, max(5, len(psnr) // 5)), color="#3498DB", edgecolor="white", alpha=0.9)
    ax.axvline(np.mean(psnr), color="#E74C3C", linestyle="--", linewidth=2, label=f"mean={np.mean(psnr):.2f} dB")
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("Count")
    ax.set_title("PSNR distribution")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[0, 1]
    ax.hist(ssim, bins=min(20, max(5, len(ssim) // 5)), color="#2ECC71", edgecolor="white", alpha=0.9)
    ax.axvline(np.mean(ssim), color="#E74C3C", linestyle="--", linewidth=2, label=f"mean={np.mean(ssim):.4f}")
    ax.set_xlabel("SSIM")
    ax.set_ylabel("Count")
    ax.set_title("SSIM distribution")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1, 0]
    ax.scatter(psnr, ssim, alpha=0.65, c="#9B59B6", edgecolors="white", linewidths=0.5, s=40)
    ax.axvline(np.mean(psnr), color="#E74C3C", linestyle=":", alpha=0.7)
    ax.axhline(np.mean(ssim), color="#E74C3C", linestyle=":", alpha=0.7)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("SSIM")
    ax.set_title("PSNR vs SSIM")
    ax.grid(linestyle="--", alpha=0.4)

    ax = axes[1, 1]
    metrics = ["PSNR (dB)", "SSIM"]
    values = [float(np.mean(psnr)), float(np.mean(ssim))]
    colors = ["#3498DB", "#2ECC71"]
    bars = ax.bar(metrics, values, color=colors, width=0.5)
    ax.set_title("Mean metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, values):
        fmt = f"{val:.2f}" if bar.get_x() < 0.5 else f"{val:.4f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt, ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    overview_path = output_dir / "downsample_quality_overview.png"
    fig.savefig(overview_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(overview_path)

    # --- Per-image PSNR (sorted) ---
    order = np.argsort(psnr)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(psnr))
    ax.bar(x, psnr[order], color="#5499C7", width=0.8)
    ax.axhline(np.mean(psnr), color="#E74C3C", linestyle="--", linewidth=1.5, label=f"mean={np.mean(psnr):.2f} dB")
    ax.set_xlabel("Image (sorted by PSNR, low → high)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-image PSNR")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    psnr_path = output_dir / "downsample_quality_psnr_per_image.png"
    fig.savefig(psnr_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(psnr_path)

    # --- Per-image SSIM (sorted) ---
    order = np.argsort(ssim)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ssim))
    ax.bar(x, ssim[order], color="#58D68D", width=0.8)
    ax.axhline(np.mean(ssim), color="#E74C3C", linestyle="--", linewidth=1.5, label=f"mean={np.mean(ssim):.4f}")
    ax.set_xlabel("Image (sorted by SSIM, low → high)")
    ax.set_ylabel("SSIM")
    ax.set_title("Per-image SSIM")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    ssim_path = output_dir / "downsample_quality_ssim_per_image.png"
    fig.savefig(ssim_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(ssim_path)

    # --- Worst / best table chart (top 5 each by PSNR) ---
    worst_idx = np.argsort(psnr)[:5]
    best_idx = np.argsort(psnr)[-5:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, idxs, title in [
        (axes[0], worst_idx, "Lowest PSNR (worst)"),
        (axes[1], best_idx, "Highest PSNR (best)"),
    ]:
        labels = [names[i][:20] for i in idxs]
        vals = [psnr[i] for i in idxs]
        ax.barh(labels, vals, color="#E67E22" if "Lowest" in title else "#27AE60")
        ax.set_xlabel("PSNR (dB)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    extremes_path = output_dir / "downsample_quality_extremes.png"
    fig.savefig(extremes_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(extremes_path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize downsample_quality.json as charts")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/downsample_quality.json"),
        help="Path to downsample_quality.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/downsample_quality"),
        help="Directory for PNG charts",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"ERROR: JSON not found: {args.input}")

    saved = plot_downsample_quality(args.input, args.output_dir, dpi=args.dpi)
    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
