from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Download URLs
# -----------------------------
BASE_URL = (
    "https://huggingface.co/spaces/"
    "PabloGabrielSch/AI_Resolution_Upscaler_And_Resizer/"
    "resolve/7888e9bba89c44423e0d9919602abb75d5b3ce2b/models"
)

ESPCN_MODEL_URLS = {
    2: f"{BASE_URL}/ESPCN_x2.pb",
    3: f"{BASE_URL}/ESPCN_x3.pb",
    4: f"{BASE_URL}/ESPCN_x4.pb",
}


def _checkpoint_path(scale: int, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"ESPCN_x{scale}.pb"


def download_espcn_model(scale: int, checkpoint_dir: Path) -> Path:
    if scale not in ESPCN_MODEL_URLS:
        raise ValueError(f"Unsupported scale x{scale}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = _checkpoint_path(scale, checkpoint_dir)

    if model_path.exists():
        return model_path

    print(f"Downloading ESPCN x{scale} model...")

    urlretrieve(
        ESPCN_MODEL_URLS[scale],
        str(model_path),
    )

    print("Download complete.")

    return model_path


def load_espcn(
    scale: int = 2,
    checkpoint_dir: str | Path = "artifacts/checkpoints",
    device: str | None = None,
):
    """
    Load OpenCV ESPCN model.

    device:
        cpu
        cuda
    """

    model_path = download_espcn_model(scale, Path(checkpoint_dir))

    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    sr.readModel(str(model_path))
    sr.setModel("espcn", scale)

    if device is not None and device.lower() == "cuda":
        try:
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using OpenCV CUDA backend.")
        except Exception:
            print("CUDA backend unavailable. Using CPU.")

    return sr


def upscale_image(model, image: Image.Image) -> Image.Image:
    """
    Same API as FSRCNN implementation.
    Input : PIL Image
    Output: PIL Image
    """

    rgb = np.array(image)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    sr = model.upsample(bgr)

    sr = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(sr)