# TEMP: replaced when downsampling branch merges.

from PIL import Image


def downscale_bicubic(image: Image.Image, scale: int = 2) -> Image.Image:
    """Bicubic downscale by the given integer factor."""
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    width, height = image.size
    new_width = max(1, width // scale)
    new_height = max(1, height // scale)
    return image.resize((new_width, new_height), Image.BICUBIC)
