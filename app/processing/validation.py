"""
Validation — Mobile ECG AI Platform
Checks that an uploaded image is structurally sound before any heavy processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ─── Thresholds ──────────────────────────────────────────────────────────────

MIN_WIDTH = 224
MIN_HEIGHT = 224
MAX_WIDTH = 8_000
MAX_HEIGHT = 8_000
MIN_CHANNELS = 1
MAX_CHANNELS = 4


# ─── Result ──────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed: bool
    message: str


# ─── Validator ───────────────────────────────────────────────────────────────

class ImageValidator:
    """
    Stage 1 — Validation.

    Checks performed
    ----------------
    1. Input is a numpy array (image already decoded upstream).
    2. Array is 2-D (grayscale) or 3-D (H×W×C).
    3. Spatial dimensions are within [MIN, MAX] bounds.
    4. Channel count is 1–4.
    5. Array contains finite values (no NaN / Inf).
    6. Array is not blank (pure white or pure black).
    """

    def validate(self, image: np.ndarray) -> ValidationResult:
        checks = [
            self._check_type,
            self._check_ndim,
            self._check_size,
            self._check_channels,
            self._check_finite,
            self._check_not_blank,
        ]

        for check in checks:
            result = check(image)
            if not result.passed:
                logger.warning("Validation failed at %s: %s", check.__name__, result.message)
                return result

        logger.info(
            "Validation passed  shape=%s  dtype=%s", image.shape, image.dtype
        )
        return ValidationResult(passed=True, message="All checks passed.")

    # ── Individual checks ────────────────────────────────────────────────────

    @staticmethod
    def _check_type(image) -> ValidationResult:
        if not isinstance(image, np.ndarray):
            return ValidationResult(False, f"Expected numpy array, got {type(image).__name__}.")
        return ValidationResult(True, "")

    @staticmethod
    def _check_ndim(image: np.ndarray) -> ValidationResult:
        if image.ndim not in (2, 3):
            return ValidationResult(
                False,
                f"Image must be 2-D or 3-D, got {image.ndim}-D array.",
            )
        return ValidationResult(True, "")

    @staticmethod
    def _check_size(image: np.ndarray) -> ValidationResult:
        h, w = image.shape[:2]
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return ValidationResult(
                False,
                f"Image too small ({w}×{h}). Minimum is {MIN_WIDTH}×{MIN_HEIGHT}.",
            )
        if w > MAX_WIDTH or h > MAX_HEIGHT:
            return ValidationResult(
                False,
                f"Image too large ({w}×{h}). Maximum is {MAX_WIDTH}×{MAX_HEIGHT}.",
            )
        return ValidationResult(True, "")

    @staticmethod
    def _check_channels(image: np.ndarray) -> ValidationResult:
        if image.ndim == 2:
            return ValidationResult(True, "")
        c = image.shape[2]
        if not (MIN_CHANNELS <= c <= MAX_CHANNELS):
            return ValidationResult(
                False,
                f"Unexpected channel count {c}. Expected 1–{MAX_CHANNELS}.",
            )
        return ValidationResult(True, "")

    @staticmethod
    def _check_finite(image: np.ndarray) -> ValidationResult:
        arr = image.astype(np.float32)
        if not np.all(np.isfinite(arr)):
            return ValidationResult(False, "Image contains NaN or Inf values.")
        return ValidationResult(True, "")

    @staticmethod
    def _check_not_blank(image: np.ndarray) -> ValidationResult:
        arr = image.astype(np.float32)
        # Normalise to [0, 1] regardless of dtype
        if arr.max() > 1.0:
            arr = arr / 255.0
        std = float(arr.std())
        if std < 0.01:
            return ValidationResult(
                False,
                f"Image appears blank (pixel std={std:.4f} < 0.01).",
            )
        return ValidationResult(True, "")