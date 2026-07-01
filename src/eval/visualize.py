from __future__ import annotations

from pathlib import Path

from PIL import Image


def save_image(image: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_comparison(left: Image.Image, right: Image.Image, path: str | Path) -> None:
    """Save a side-by-side comparison (left | right)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    left = left.convert("RGB")
    right = right.convert("RGB")
    height = max(left.height, right.height)

    if left.height != height:
        new_width = int(left.width * height / left.height)
        left = left.resize((new_width, height), Image.BICUBIC)
    if right.height != height:
        new_width = int(right.width * height / right.height)
        right = right.resize((new_width, height), Image.BICUBIC)

    canvas = Image.new("RGB", (left.width + right.width, height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(path)
