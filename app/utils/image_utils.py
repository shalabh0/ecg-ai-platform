"""
Image Utilities — Mobile ECG AI Platform
Helpers for decoding, encoding, and manipulating images across the pipeline.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Decode ───────────────────────────────────────────────────────────────────

def decode_image(data: bytes) -> Optional[np.ndarray]:
    """
    Decode raw image bytes (JPEG / PNG / TIFF / BMP) into a numpy array.

    Returns
    -------
    np.ndarray  H×W×3 uint8 RGB, or None on failure.
    """
    try:
        buf = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None — unsupported format or corrupt data.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        logger.debug("Decoded image: shape=%s  dtype=%s", rgb.shape, rgb.dtype)
        return rgb
    except Exception as exc:
        logger.error("decode_image failed: %s", exc)
        return None


def decode_image_from_b64(b64_str: str) -> Optional[np.ndarray]:
    """Decode a base64-encoded image string."""
    try:
        raw = base64.b64decode(b64_str)
        return decode_image(raw)
    except Exception as exc:
        logger.error("decode_image_from_b64 failed: %s", exc)
        return None


# ─── Encode ───────────────────────────────────────────────────────────────────

def encode_image_bytes(image: np.ndarray, fmt: str = ".jpg", quality: int = 90) -> bytes:
    """
    Encode a numpy array to compressed image bytes.

    Parameters
    ----------
    image : np.ndarray  H×W×3 RGB uint8
    fmt   : str         ".jpg" or ".png"
    quality : int       JPEG quality (1–100)
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(fmt, bgr, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for format '{fmt}'.")
    return buf.tobytes()


def encode_image_base64(image: np.ndarray, fmt: str = ".jpg", quality: int = 85) -> str:
    """Encode numpy image to a base64 string (data URL ready)."""
    raw = encode_image_bytes(image, fmt=fmt, quality=quality)
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = "image/jpeg" if fmt.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


# ─── Resize helpers ───────────────────────────────────────────────────────────

def resize_keep_aspect(
    image: np.ndarray,
    max_side: int = 1024,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize so the longest side ≤ max_side, preserving aspect ratio."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def pad_to_square(image: np.ndarray, fill: int = 128) -> np.ndarray:
    """Zero-pad image to a square with the given fill value."""
    h, w = image.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, image.shape[2] if image.ndim == 3 else 1), fill, dtype=image.dtype)
    if image.ndim == 2:
        canvas = np.full((side, side), fill, dtype=image.dtype)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    canvas[pad_y : pad_y + h, pad_x : pad_x + w] = image
    return canvas


# ─── Channel utilities ────────────────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert H×W×C or H×W to H×W grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure output is H×W×3 uint8 RGB."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def image_stats(image: np.ndarray) -> dict:
    """Return basic statistics about the image for logging/debugging."""
    arr = image.astype(np.float32)
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std()), 3),
    }


def draw_pipeline_overlay(image: np.ndarray, label: str, confidence: float) -> np.ndarray:
    """
    Overlay prediction label and confidence bar on a copy of the image.
    Useful for debugging/demo outputs.
    """
    out = image.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{label}  {confidence * 100:.1f}%"
    scale = max(0.5, w / 800)
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    # Background bar
    cv2.rectangle(out, (0, h - th - 20), (min(tw + 20, w), h), (0, 0, 0), -1)
    # Text
    cv2.putText(out, text, (10, h - 10), font, scale, (0, 255, 120), thickness, cv2.LINE_AA)
    # Confidence bar
    bar_w = int((w - 4) * confidence)
    cv2.rectangle(out, (2, h - 4), (bar_w, h - 1), (0, 200, 100), -1)
    return out