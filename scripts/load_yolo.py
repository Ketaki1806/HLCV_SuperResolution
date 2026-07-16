"""Verify YOLOv8 encoder/decoder split and AnyUp loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.anyup import load_anyup
from src.models.yolo import load_yolov8


def main() -> None:
    parser = argparse.ArgumentParser(description="Load YOLOv8 encoder/decoder and AnyUp")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading YOLOv8n (full split)...")
    yolo = load_yolov8(device=device)
    encoder = yolo.encoder
    decoder = yolo.decoder
    print(f"  encoder layers: 0–{encoder.end_layer}")
    print(f"  decoder layers: {decoder.start_layer}–{decoder.end_layer - 1}")
    print(f"  upsample hook layers (AnyUp): {yolo.upsample_layers}")

    print("\nLoading encoder and decoder separately (same split instance)...")
    enc = yolo.encoder
    dec = yolo.decoder
    enc_params = sum(p.numel() for p in enc.layers[: enc.end_layer + 1].parameters())
    dec_params = sum(p.numel() for p in dec.layers[dec.start_layer :].parameters())
    print(f"  encoder params: {enc_params:,}")
    print(f"  decoder params: {dec_params:,}")

    x = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        enc_out = enc(x)
        split_out = dec(enc_out)
        full_out = dec(enc(x))

    print(f"\nForward check:")
    print(f"  encoder feature shape: {tuple(enc_out.features.shape)}")
    print(f"  decoder output shape: {tuple(split_out.shape)}")
    print(f"  encoder+decoder consistent: {torch.allclose(split_out, full_out, atol=1e-5)}")

    print("\nLoading AnyUp...")
    try:
        anyup = load_anyup(device=device)
        print(f"  AnyUp loaded: {type(anyup).__name__}")
    except Exception as exc:
        print(f"  AnyUp load failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
