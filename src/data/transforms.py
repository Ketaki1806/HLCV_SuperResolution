import random

from PIL import Image


def random_crop(image: Image.Image, patch_size: int) -> Image.Image:
    width, height = image.size
    if width < patch_size or height < patch_size:
        raise ValueError(
            f"Image {width}x{height} is smaller than patch_size={patch_size}"
        )

    left = random.randint(0, width - patch_size)
    top = random.randint(0, height - patch_size)
    return image.crop((left, top, left + patch_size, top + patch_size))
