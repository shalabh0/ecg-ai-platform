"""
Preprocessing — Mobile ECG AI Platform
Converts a raw ECG image into a normalised tensor ready for CNN inference.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

# Target spatial dimensions for the CNN (H × W)
TARGET_SIZE: Tuple[int, int] = (224, 224)

# ImageNet-style normalisation constants (per-channel mean / std)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CLAHE parameters for adaptive contrast enhancement
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


# ─── Preprocessor ────────────────────────────────────────────────────────────

class ECGPreprocessor:
    """
    Stage 4 — Preprocessing.

    Pipeline
    --------
    1. Convert to RGB (drop alpha if present).
    2. Adaptive histogram equalisation (CLAHE) on luminance channel.
    3. Bilateral filter to reduce noise while preserving edges.
    4. Resize to TARGET_SIZE with aspect-ratio-preserving letterbox padding.
    5. Normalise to float32 in [0, 1].
    6. Apply ImageNet mean/std normalisation.
    7. Return CHW tensor  shape=(1, 3, H, W)  suitable for PyTorch / ONNX.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = TARGET_SIZE,
        mean: np.ndarray = IMAGENET_MEAN,
        std: np.ndarray = IMAGENET_STD,
        clahe_clip: float = CLAHE_CLIP_LIMIT,
        bilateral_d: int = 9,
    ):
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=CLAHE_TILE_GRID
        )
        self._bilateral_d = bilateral_d

    # ── Public entry-point ───────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image : np.ndarray
            H×W or H×W×C uint8 image (RGB / RGBA / gray).

        Returns
        -------
        np.ndarray
            Float32 array of shape (1, 3, H, W) — batch dimension included.
        """
        img = self._to_rgb(image)
        img = self._apply_clahe(img)
        img = self._denoise(img)
        img = self._letterbox(img, self.target_size)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        # HWC → CHW → NCHW
        tensor = np.transpose(img, (2, 0, 1))[np.newaxis]
        logger.debug("Preprocessor output shape: %s  dtype: %s", tensor.shape, tensor.dtype)
        return tensor

    # ── Steps ────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        """Ensure output is H×W×3 uint8 RGB."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        # If already 3-channel assume RGB (caller's responsibility)
        return image.astype(np.uint8)

    def _apply_clahe(self, rgb: np.ndarray) -> np.ndarray:
        """CLAHE on L-channel of Lab colour space for adaptive contrast."""
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self._clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    def _denoise(self, rgb: np.ndarray) -> np.ndarray:
        """Bilateral filter: preserves ECG waveform edges."""
        return cv2.bilateralFilter(
            rgb, d=self._bilateral_d, sigmaColor=75, sigmaSpace=75
        )

    @staticmethod
    def _letterbox(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize while preserving aspect ratio, pad remainder with 128 (gray).
        size = (target_h, target_w)
        """
        target_h, target_w = size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        pad_y = (target_h - new_h) // 2
        pad_x = (target_w - new_w) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        return canvas

    # ── Inverse (for debugging) ──────────────────────────────────────────────

    def denormalize(self, tensor: np.ndarray) -> np.ndarray:
        """Convert normalised NCHW tensor back to HWC uint8 for display."""
        img = tensor[0]  # CHW
        img = np.transpose(img, (1, 2, 0))  # HWC
        img = (img * self.std + self.mean) * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)