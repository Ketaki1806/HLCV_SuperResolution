from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def psnr(ref: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.

    - ref/pred are HxWxC or HxW images
    - expects same shape
    """
    if ref.shape != pred.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape} pred={pred.shape}")

    ref_f = _to_float01(ref)
    pred_f = _to_float01(pred)
    mse = float(np.mean((ref_f - pred_f) ** 2))
    if mse <= eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@dataclass(frozen=True)
class SSIMConfig:
    # 11x11 Gaussian window is a common default
    win_size: int = 11
    sigma: float = 1.5
    k1: float = 0.01
    k2: float = 0.03


def ssim(ref: np.ndarray, pred: np.ndarray, cfg: SSIMConfig = SSIMConfig()) -> float:
    """
    Structural Similarity Index (single-scale), averaged over channels.

    Implementation uses Gaussian filtering similarly to the classic SSIM paper.
    Expects images in uint8 [0,255] or float [0,1], same shape.
    """
    if ref.shape != pred.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape} pred={pred.shape}")

    ref_f = _to_float01(ref)
    pred_f = _to_float01(pred)

    if ref_f.ndim == 2:
        ref_f = ref_f[..., None]
        pred_f = pred_f[..., None]

    if cfg.win_size % 2 == 0 or cfg.win_size < 3:
        raise ValueError("win_size must be odd and >= 3")

    # constants for L=1
    c1 = (cfg.k1 ** 2)
    c2 = (cfg.k2 ** 2)

    scores: list[float] = []
    for c in range(ref_f.shape[2]):
        x = ref_f[..., c]
        y = pred_f[..., c]

        mu_x = cv2.GaussianBlur(x, (cfg.win_size, cfg.win_size), cfg.sigma)
        mu_y = cv2.GaussianBlur(y, (cfg.win_size, cfg.win_size), cfg.sigma)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = cv2.GaussianBlur(x * x, (cfg.win_size, cfg.win_size), cfg.sigma) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (cfg.win_size, cfg.win_size), cfg.sigma) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (cfg.win_size, cfg.win_size), cfg.sigma) - mu_xy

        num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        ssim_map = num / (den + 1e-12)
        scores.append(float(np.mean(ssim_map)))

    return float(np.mean(scores))

