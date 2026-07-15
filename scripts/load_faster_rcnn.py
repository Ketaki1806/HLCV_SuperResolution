from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import coco_dataLoader


# -----------------------------
# Load Faster R-CNN
# -----------------------------
def load_fasterrcnn(device: str = "cpu"):
    """
    Load pretrained Faster R-CNN (COCO weights).
    Used as fixed detector for SR evaluation.
    """

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model.to(device)


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess(image: Image.Image) -> torch.Tensor:
    """PIL → Tensor"""
    return TF.to_tensor(image)


# -----------------------------
# Inference
# -----------------------------
def detect(
    model,
    image: Image.Image,
    device: str = "cpu",
    score_threshold: float = 0.5,
):
    """
    Run Faster R-CNN detection on an image.

    Returns filtered detections:
    - boxes
    - labels
    - scores
    """

    img_tensor = preprocess(image).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    scores = outputs["scores"]
    keep   = scores >= score_threshold

    return {
        "boxes": outputs["boxes"][keep],
        "labels": outputs["labels"][keep],
        "scores": outputs["scores"][keep],
    }


# -----------------------------
# Simple wrapper (optional)
# -----------------------------
class FasterRCNNDetector:
    """
    Optional wrapper so all detectors (YOLO / DETR / Faster R-CNN)
    can share same interface style.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = load_fasterrcnn(device)

    def __call__(self, image: Image.Image):
        return detect(self.model, image, self.device)


# -----------------------------
# Batch inference over a folder
# -----------------------------
def run_on_folder(image_dir: str, device: str, score_threshold: float = 0.5) -> list[dict]:
    """
    Run Faster R-CNN on all images in a folder using the dataloader.
    Returns a list of results, one per image.
    """
    model  = load_fasterrcnn(device)
    dm = coco_dataLoader.COCODataModule(
        image_dir="data/coco",
        annotation_file="data/annotations/instances_val2017.json",
        model_type="faster_rcnn",
        heldout_split=0.0,
        batch_size=4,
        max_images=20,
        shuffle=False,
    )
    loader = dm.get_loader()

    for tensors, targets in loader:
        outputs = model(tensors)

    all_results = []

    for tensors, meta in tqdm(loader, desc="Detecting"):
        tensors = [t.to(device) for t in tensors]

        with torch.no_grad():
            outputs = model(tensors)

        for out, m in zip(outputs, meta):
            scores = out["scores"]
            keep   = scores >= score_threshold
            all_results.append({
                "filename": m["filename"],
                "boxes":    out["boxes"][keep].cpu().tolist(),
                "labels":   out["labels"][keep].cpu().tolist(),
                "scores":   out["scores"][keep].cpu().tolist(),
            })

    return all_results


# -----------------------------
# Main — run from command line
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Faster R-CNN on image folder")
    parser.add_argument("--input",     default="data/coco",  help="Folder of images to run detection on")
    parser.add_argument("--output",    default="results/faster_rcnn_raw.json", help="Where to save results")
    parser.add_argument("--threshold", type=float, default=0.5,  help="Score threshold")
    parser.add_argument("--device",    default=None,             help="cuda or cpu")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running on  : {args.input}")

    results = run_on_folder(args.input, device, args.threshold)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} images processed.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()