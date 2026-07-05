# SR + Object Detection Project — Documentation

## Project Overview
Comparing super-resolution techniques (FSRCNN, AnyUp, ESPCN) as preprocessing 
pipelines for object detection (Faster R-CNN, YOLOv8n, Detr) on COCO dataset.

---

## Setup

### Environment
- OS: Windows 11
- Python: 3.x
- Virtual env: `.venv`

### Dependencies
```bash
pip install opencv-python numpy tqdm matplotlib
```

---

## Changelog

### [30-July-2026] — Preprocessing Pipeline

**What was done**
- Set up shared downsampling script for all three detection models
- Tested locally on 100 images from COCO val2017

**Config used**
```bash
python scripts\preprocessing\downsample.py \
  --input data\original \
  --output data\preprocessed\low_res\scale_x2 \
  --scale 2 \
  --blur \
  --blur-kernel 5 \
  --blur-sigma 2
```

**Commad to run and verify downsampling**
    python scripts\preprocessing\downsample.py --input data\original --output data\preprocessed\low_res\scale_x2 --scale 2 --blur --blur-sigma 1.0  

    python scripts\preprocessing\verify.py



**Result**
- 100 images (for testing) processed, 0 skipped
- Output: data\preprocessed\low_res\scale_x2\
- Verified visually via verification.png — dimensions halved as expected

**Decisions made**
- Chose blur + bicubic downsample only (no noise, no JPEG compression)
- Reason: standard degradation pipeline matching SR model training conditions
- Scale ×2 to match pretrained SR model weights (FSRCNN_x2, ESPCN_x2, YOLOv8n)

---

### [05-July-2026] — Models evaluation and Figures plot

**What was done**
- Models evaluation and Figures plot

**Config used**

**Result**

**Decisions made**
- For evaluation of the model, the output of the model should be converted to the following format:
  ```
  [
    {
        "image_id": int,
        "category_id": int,
        "bbox": [x, y, width, height],
        "score": float
    }
  ]
  ```
- Figures:
  1. For each model, compare all mAP for different methods. (4)
  2. For small-sized objects, compare different models for each method. (1)

- Store the weights in the runs/weights
- Store the models' evaluation results in the corresponding runs/model_strat/predictions.json. Here, model_strat is such a combination:
  ```
  detectors = ["yolov8n", "faster_rcnn", "detr_zero_shot", "detr_finetuned"]
  strategies = ["baseline", "espcn2x", "fsrcnn2x", "anyup"]
  ```
  I have created the example yolov8n_baseline
---

### [04-07-2026] — Added ESPCN Super-Resolution Model

**Changes**
- Added ESPCN model for image super-resolution.
- Created `src/models/espcn.py` with a simple interface (`load_espcn()` and `upscale_image()`).
- Connected ESPCN to the existing `run_sr_demo.py` pipeline.
- Added automatic download of pretrained ESPCN models (.pb files).
- Supported ×2, ×3, and ×4 upscaling.

**Result**
- ESPCN now works end-to-end in the project pipeline.
- It takes low-resolution images and generates high-resolution outputs successfully.
- All outputs are saved and can be used for further evaluation.

**Decisions made**
- Used OpenCV version of ESPCN because pretrained models are easily available.
- Kept the same input/output format so it works with the rest of the project without changes.
- Added automatic download so users do not need to manually download model files.

---
