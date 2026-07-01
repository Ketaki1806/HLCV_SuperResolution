# scripts/preprocessing/verify.py
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np

orig_dir = Path("data/original")
lr_dir   = Path("data/preprocessed/low_res/scale_x2")

print("Looking in:", lr_dir)
files = list(lr_dir.glob("*.jpg"))
print("Found files:", len(files))

if len(files) == 0:
    print("No jpg files found in low_res folder.")
    exit()

# Pick 1 image only for detailed comparison
sample = random.choice(files)
print("Sample:", sample.name)

orig = cv2.cvtColor(cv2.imread(str(orig_dir / sample.name)), cv2.COLOR_BGR2RGB)
lr   = cv2.cvtColor(cv2.imread(str(lr_dir  / sample.name)), cv2.COLOR_BGR2RGB)

# Crop a small patch from center to make blur visible
h, w  = orig.shape[:2]
cx, cy = w // 2, h // 2
patch_size = 100  # 100x100 crop from center of original
orig_patch = orig[cy-patch_size:cy+patch_size, cx-patch_size:cx+patch_size]

# Same crop from lr — but lr is half the size so coordinates are halved
lh, lw = lr.shape[:2]
lcx, lcy = lw // 2, lh // 2
p = patch_size // 2
lr_patch = lr[lcy-p:lcy+p, lcx-p:lcx+p]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Full images
axes[0][0].imshow(orig);       axes[0][0].set_title(f"Original  {orig.shape[1]}x{orig.shape[0]}"); axes[0][0].axis("off")
axes[0][1].imshow(lr);         axes[0][1].set_title(f"Low-res   {lr.shape[1]}x{lr.shape[0]}");    axes[0][1].axis("off")

# Zoomed patches
axes[1][0].imshow(orig_patch); axes[1][0].set_title("Original — center crop");  axes[1][0].axis("off")
axes[1][1].imshow(lr_patch);   axes[1][1].set_title("Low-res — center crop");   axes[1][1].axis("off")

plt.suptitle(sample.name, fontsize=10, color="gray")
plt.tight_layout()
plt.savefig("data/verification.png", dpi=200)
print("Saved data/verification.png")