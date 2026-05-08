"""
Quality Assessment — Mobile ECG AI Platform
Evaluates sharpness, noise, contrast, and ECG-specific signal quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Thresholds ──────────────────────────────────────────────────────────────

BLUR_THRESHOLD = 80.0       # Laplacian variance; below → too blurry
SNR_THRESHOLD_DB = 10.0     # Signal-to-noise ratio in dB; below → noisy
CONTRAST_THRESHOLD = 30.0   # RMS contrast; below → low contrast
MIN_SCORE = 0.45            # Composite score below this → reject


# ─── Result ──────────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    score: float             # 0.0 – 1.0 composite
    acceptable: bool
    reason: str
    snr_db: float = 0.0
    blur_score: float = 0.0
    contrast: float = 0.0
    metrics: dict = field(default_factory=dict)


# ─── Assessor ────────────────────────────────────────────────────────────────

class QualityAssessor:
    """
    Stage 2 — Quality Assessment.

    Metrics computed
    ----------------
    - Blur (Laplacian variance): higher = sharper.
    - SNR (dB): estimated from local mean/std on a grid.
    - RMS Contrast: global pixel standard deviation.
    - Baseline wander score: low-frequency power ratio.
    - Composite score: weighted mean of normalised sub-scores.
    """

    def assess(self, image: np.ndarray) -> QualityResult:
        gray = self._to_gray(image)

        blur = self._blur_score(gray)
        snr = self._snr_db(gray)
        contrast = self._rms_contrast(gray)
        baseline_ok = self._baseline_wander_ok(gray)

        # Normalise each metric to [0, 1]
        blur_norm = min(blur / 300.0, 1.0)
        snr_norm = min(max((snr - 0.0) / 40.0, 0.0), 1.0)
        contrast_norm = min(contrast / 80.0, 1.0)
        baseline_norm = 1.0 if baseline_ok else 0.5

        score = (
            0.35 * blur_norm
            + 0.30 * snr_norm
            + 0.20 * contrast_norm
            + 0.15 * baseline_norm
        )
        score = round(float(score), 4)
        acceptable = score >= MIN_SCORE

        if not acceptable:
            if blur < BLUR_THRESHOLD:
                reason = f"Image is too blurry (Laplacian variance={blur:.1f}, threshold={BLUR_THRESHOLD})."
            elif snr < SNR_THRESHOLD_DB:
                reason = f"Signal-to-noise ratio too low ({snr:.1f} dB, threshold={SNR_THRESHOLD_DB} dB)."
            elif contrast < CONTRAST_THRESHOLD:
                reason = f"Image contrast too low (RMS={contrast:.1f}, threshold={CONTRAST_THRESHOLD})."
            else:
                reason = f"Composite quality score {score:.2f} below threshold {MIN_SCORE}."
        else:
            reason = "Quality acceptable."

        logger.info(
            "Quality  score=%.3f  blur=%.1f  snr=%.1f dB  contrast=%.1f  acceptable=%s",
            score, blur, snr, contrast, acceptable,
        )

        return QualityResult(
            score=score,
            acceptable=acceptable,
            reason=reason,
            snr_db=round(snr, 2),
            blur_score=round(blur, 2),
            contrast=round(contrast, 2),
            metrics={
                "blur_norm": round(blur_norm, 3),
                "snr_norm": round(snr_norm, 3),
                "contrast_norm": round(contrast_norm, 3),
                "baseline_norm": round(baseline_norm, 3),
            },
        )

    # ── Sub-metrics ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.float32)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    @staticmethod
    def _blur_score(gray: np.ndarray) -> float:
        """Laplacian variance — proxy for focus/sharpness."""
        lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
        return float(lap.var())

    @staticmethod
    def _snr_db(gray: np.ndarray) -> float:
        """
        Estimate SNR by sampling 16×16 patches on a grid.
        SNR = 20 * log10(mean_signal / mean_noise)
        """
        h, w = gray.shape
        patch_size = 16
        means, stds = [], []
        for r in range(0, h - patch_size, patch_size * 2):
            for c in range(0, w - patch_size, patch_size * 2):
                patch = gray[r : r + patch_size, c : c + patch_size]
                means.append(float(patch.mean()))
                stds.append(float(patch.std()) + 1e-6)

        if not means:
            return 0.0
        signal = np.mean(means)
        noise = np.mean(stds)
        if noise < 1e-6 or signal < 1e-6:
            return 40.0  # essentially perfect
        return float(20 * np.log10(signal / noise))

    @staticmethod
    def _rms_contrast(gray: np.ndarray) -> float:
        """Root-mean-square contrast (global pixel std)."""
        return float(gray.std())

    @staticmethod
    def _baseline_wander_ok(gray: np.ndarray) -> bool:
        """
        Rough baseline-wander check: compare low-frequency energy ratio.
        Applies a strong Gaussian blur and measures ratio to original variance.
        A high ratio → most energy is low-frequency → likely wander.
        """
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=gray.shape[1] // 8)
        orig_var = float(gray.var()) + 1e-6
        blur_var = float(blurred.var())
        ratio = blur_var / orig_var
        return ratio < 0.70  # more than 30 % high-freq energy = ok