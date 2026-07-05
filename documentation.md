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

**Result**
- 100 images processed, 0 skipped
- Output: data\preprocessed\low_res\scale_x2\
- Verified visually via verification.png — dimensions halved as expected

**Decisions made**
- Chose blur + bicubic downsample only (no noise, no JPEG compression)
- Reason: standard degradation pipeline matching SR model training conditions
- Scale ×2 to match pretrained SR model weights (FSRCNN_x2, ESPCN_x2)

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
---

### [Date] — <next change title>

**What was done**

**Config used**

**Result**

**Decisions made**

---