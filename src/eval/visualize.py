from __future__ import annotations

from pathlib import Path

from PIL import Image


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_comparison(lr: Image.Image, sr: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gap = 8
    width = lr.width + gap + sr.width
    height = max(lr.height, sr.height)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(lr, (0, 0))
    canvas.paste(sr, (lr.width + gap, 0))
    canvas.save(path)
