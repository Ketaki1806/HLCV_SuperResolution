"""Analyze JSON experiment outputs and generate charts.

Reads detection runs, downsample quality, and SR quality JSON files.
Also recomputes corrected mAP from saved predictions (fixes center-xywh bug).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STRATEGIES = ["baseline", "espcn2x", "fsrcnn2x", "anyup"]
STRATEGY_LABELS = {
    "baseline": "Baseline (LR)",
    "espcn2x": "ESPCN 2×",
    "fsrcnn2x": "FSRCNN 2×",
    "anyup": "AnyUp",
}
COLORS = {
    "baseline": "#7F8C8D",
    "espcn2x": "#A9CCE3",
    "fsrcnn2x": "#5499C7",
    "anyup": "#E74C3C",
}


def load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fix_center_xywh_predictions(preds: list[dict]) -> list[dict]:
    """Convert stored center-format xywh (old bug) to COCO top-left xywh."""
    fixed: list[dict] = []
    for p in preds:
        x, y, w, h = p["bbox"]
        fixed.append(
            {
                **p,
                "bbox": [float(x - w / 2), float(y - h / 2), float(w), float(h)],
            }
        )
    return fixed


def recompute_map(ann_file: Path, preds: list[dict]) -> dict[str, float]:
    coco = COCO(str(ann_file))
    import tempfile
    import os

    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        os.write(fd, json.dumps(preds).encode())
        os.close(fd)
        dt = coco.loadRes(tmp)
        ev = COCOeval(coco, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        def _map50_area(area_idx: int) -> float:
            p = ev.eval["precision"][0, :, :, area_idx, :]
            valid = p[p > -1]
            return float(valid.mean()) if valid.size else 0.0

        return {
            "mAP_50_all": float(ev.stats[1]),
            "mAP_50_small": _map50_area(1),
            "mAP_50_medium": _map50_area(2),
            "mAP_50_large": _map50_area(3),
        }
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def plot_detection_comparison(
    reported: dict,
    corrected: dict,
    output_dir: Path,
) -> None:
    """Side-by-side reported vs corrected mAP@0.5 for all strategies."""
    metrics = ["mAP_50_all", "mAP_50_small", "mAP_50_medium", "mAP_50_large"]
    metric_labels = ["All", "Small", "Medium", "Large"]
    x = np.arange(len(metrics))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, title, data in [
        (axes[0], "Reported (buggy pipeline)", reported),
        (axes[1], "Corrected (fixed bbox + mAP)", corrected),
    ]:
        for i, strat in enumerate(STRATEGIES):
            if strat not in data:
                continue
            vals = [data[strat].get(m, 0) * 100 for m in metrics]
            offset = (i - 1.5) * width
            ax.bar(x + offset, vals, width, label=STRATEGY_LABELS[strat], color=COLORS[strat])
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("mAP @ 0.5 (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.suptitle("YOLOv8n Detection mAP — Reported vs Corrected", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "detection_map_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_detection_counts(counts: dict[str, int], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [STRATEGY_LABELS[s] for s in STRATEGIES if s in counts]
    vals = [counts[s] for s in STRATEGIES if s in counts]
    bars = ax.bar(labels, vals, color=[COLORS[s] for s in STRATEGIES if s in counts])
    ax.set_title("Total Detections (conf ≥ 0.001)", fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "detection_counts.png", dpi=200)
    plt.close(fig)


def plot_downsample_quality(path: Path, output_dir: Path) -> None:
    data = load_json(path)
    if not data or "per_image" not in data:
        return
    psnr = [r["psnr_db"] for r in data["per_image"]]
    ssim = [r["ssim"] for r in data["per_image"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(psnr, bins=40, color="#3498DB", edgecolor="white")
    axes[0].axvline(data["metrics_mean"]["psnr_db"], color="red", linestyle="--", label=f"mean={data['metrics_mean']['psnr_db']:.2f} dB")
    axes[0].set_title("PSNR (HR vs bicubic-up LR)")
    axes[0].set_xlabel("PSNR (dB)")
    axes[0].legend()

    axes[1].hist(ssim, bins=40, color="#2ECC71", edgecolor="white")
    axes[1].axvline(data["metrics_mean"]["ssim"], color="red", linestyle="--", label=f"mean={data['metrics_mean']['ssim']:.4f}")
    axes[1].set_title("SSIM (HR vs bicubic-up LR)")
    axes[1].set_xlabel("SSIM")
    axes[1].legend()

    axes[2].scatter(psnr, ssim, alpha=0.25, s=8, color="#8E44AD")
    axes[2].set_xlabel("PSNR (dB)")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("PSNR vs SSIM per image")
    fig.suptitle(f"COCO val2017 Downsample Quality (n={len(psnr)})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "downsample_quality.png", dpi=200)
    plt.close(fig)


def plot_sr_quality(path: Path, output_dir: Path) -> None:
    data = load_json(path)
    if not data or "mean" not in data:
        return
    methods = list(data["mean"].keys())
    psnr = [data["mean"][m]["psnr_db"] for m in methods]
    ssim = [data["mean"][m]["ssim"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(methods))
    axes[0].bar(x, psnr, color=["#95A5A6", "#5499C7", "#A9CCE3"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_title("Mean PSNR by upsampling method")
    axes[0].set_ylabel("PSNR (dB)")

    axes[1].bar(x, ssim, color=["#95A5A6", "#5499C7", "#A9CCE3"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_title("Mean SSIM by upsampling method")
    axes[1].set_ylabel("SSIM")
    fig.suptitle("SR Quality on Sample Images", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "sr_quality_sample.png", dpi=200)
    plt.close(fig)


def plot_sr_vs_detection(corrected: dict, sr_mean: dict | None, output_dir: Path) -> None:
    """Scatter SR quality proxy vs corrected detection mAP (where available)."""
    if not sr_mean:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    points = {
        "baseline": ("bicubic", corrected.get("baseline", {}).get("mAP_50_all", 0)),
        "espcn2x": ("espcn", corrected.get("espcn2x", {}).get("mAP_50_all", 0)),
        "fsrcnn2x": ("fsrcnn", corrected.get("fsrcnn2x", {}).get("mAP_50_all", 0)),
    }
    for strat, (sr_key, map50) in points.items():
        if sr_key not in sr_mean:
            continue
        ax.scatter(sr_mean[sr_key]["psnr_db"], map50 * 100, s=120, color=COLORS[strat], label=STRATEGY_LABELS[strat])
        ax.annotate(STRATEGY_LABELS[strat], (sr_mean[sr_key]["psnr_db"], map50 * 100), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("SR PSNR (dB) — sample images only")
    ax.set_ylabel("Corrected mAP@0.5 (%)")
    ax.set_title("SR Quality vs Detection (sample SR metrics)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "sr_vs_detection.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze JSON results and generate charts")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--ann-file", type=Path, default=Path("data/coco/annotations/instances_val2017.json"))
    parser.add_argument(
        "--downsample-json",
        type=Path,
        default=Path("data/preprocessed/val2017_lr_x2/downsample_quality.json"),
    )
    parser.add_argument("--sr-quality-json", type=Path, default=Path("results/sr_quality_sample.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = load_json(args.runs_dir / "sr_detection_comparison.json") or {}
    reported: dict[str, dict] = {}
    corrected: dict[str, dict] = {}
    counts: dict[str, int] = {}

    for strat in STRATEGIES:
        key = f"yolov8n_{strat}"
        run_dir = args.runs_dir / key
        summary = load_json(run_dir / "eval_summary.json")
        preds = load_json(run_dir / "predictions.json")

        if summary:
            reported[strat] = summary.get("metrics", {})
            counts[strat] = int(summary.get("num_detections", 0))

        if preds and args.ann_file.is_file():
            print(f"Recomputing corrected mAP for {key} ({len(preds):,} dets)...")
            fixed = fix_center_xywh_predictions(preds)
            corrected[strat] = recompute_map(args.ann_file, fixed)

    plot_detection_comparison(reported, corrected, args.output_dir)
    plot_detection_counts(counts, args.output_dir)
    plot_downsample_quality(args.downsample_json, args.output_dir)
    plot_sr_quality(args.sr_quality_json, args.output_dir)

    sr_data = load_json(args.sr_quality_json)
    sr_mean = sr_data.get("mean") if isinstance(sr_data, dict) else None
    plot_sr_vs_detection(corrected, sr_mean, args.output_dir)

    # Save corrected summary JSON
    out = {
        "reported_metrics": reported,
        "corrected_metrics": corrected,
        "detection_counts": counts,
        "notes": [
            "Reported metrics used center-format YOLO xywh as top-left COCO boxes.",
            "Corrected metrics convert center xywh -> top-left and use proper COCO mAP extraction.",
            "Re-run eval_sr_coco.py after code fixes for official corrected predictions.json files.",
        ],
    }
    summary_path = args.output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nCharts saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print("\nCorrected mAP@0.5 (all):")
    for strat in STRATEGIES:
        if strat in corrected:
            r = reported.get(strat, {}).get("mAP_50_all", 0)
            c = corrected[strat]["mAP_50_all"]
            print(f"  {STRATEGY_LABELS[strat]:16s}  reported={r*100:.4f}%  corrected={c*100:.2f}%")


if __name__ == "__main__":
    main()
