from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image

# Primary + fallback sources (HuggingFace can return 500 intermittently).
ESPCN_MODEL_URLS: dict[int, list[str]] = {
    2: [
        "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x2.pb",
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
        "https://huggingface.co/spaces/PabloGabrielSch/AI_Resolution_Upscaler_And_Resizer/resolve/main/models/ESPCN_x2.pb",
    ],
    3: [
        "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x3.pb",
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb",
        "https://huggingface.co/spaces/PabloGabrielSch/AI_Resolution_Upscaler_And_Resizer/resolve/main/models/ESPCN_x3.pb",
    ],
    4: [
        "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x4.pb",
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb",
        "https://huggingface.co/spaces/PabloGabrielSch/AI_Resolution_Upscaler_And_Resizer/resolve/main/models/ESPCN_x4.pb",
    ],
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
    errors: list[str] = []
    for url in ESPCN_MODEL_URLS[scale]:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response:
                model_path.write_bytes(response.read())
            print(f"Download complete: {url}")
            return model_path
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append(f"{url} -> {exc}")

    raise RuntimeError(
        f"Failed to download ESPCN x{scale} model. Tried:\n  " + "\n  ".join(errors)
    )


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