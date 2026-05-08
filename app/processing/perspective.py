"""
Perspective Correction — Mobile ECG AI Platform
Detects and rectifies camera-induced perspective distortion in ECG printout photos.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

SKEW_ANGLE_THRESHOLD_DEG = 2.0   # correct only if skew exceeds this
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 80


# ─── Corrector ───────────────────────────────────────────────────────────────

class PerspectiveCorrector:
    """
    Stage 3 — Perspective Correction.

    Strategy
    --------
    1. Convert to grayscale and detect edges (Canny).
    2. Run Probabilistic Hough Line Transform to find dominant lines.
    3. Estimate document skew angle from the median line angle.
    4. If skew exceeds threshold, rotate the image to deskew.
    5. Optionally detect document quad (4-corner) for full perspective warp.

    Returns the corrected image and a flag indicating whether correction was applied.
    """

    def correct(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        gray = self._to_gray(image)

        # ── Attempt full quad-warp first ──────────────────────────────────────
        quad = self._detect_document_quad(gray)
        if quad is not None:
            warped = self._four_point_transform(image, quad)
            logger.info("Perspective correction: 4-point warp applied.")
            return warped, True

        # ── Fall back to skew-only rotation ──────────────────────────────────
        angle = self._estimate_skew(gray)
        if abs(angle) > SKEW_ANGLE_THRESHOLD_DEG:
            corrected = self._rotate(image, angle)
            logger.info("Perspective correction: deskew %.2f°.", angle)
            return corrected, True

        logger.info("Perspective correction: no correction needed.")
        return image.copy(), False

    # ── Gray conversion ──────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # ── Skew estimation ──────────────────────────────────────────────────────

    @staticmethod
    def _estimate_skew(gray: np.ndarray) -> float:
        """Median angle of Hough lines relative to horizontal."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=HOUGH_RHO,
            theta=HOUGH_THETA,
            threshold=HOUGH_THRESHOLD,
            minLineLength=gray.shape[1] // 4,
            maxLineGap=20,
        )
        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Keep only near-horizontal lines (±45°)
                if -45 < angle < 45:
                    angles.append(angle)

        if not angles:
            return 0.0
        return float(np.median(angles))

    # ── Rotation ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        # Expand canvas to avoid cropping corners
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy
        return cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # ── Document quad detection ──────────────────────────────────────────────

    @staticmethod
    def _detect_document_quad(gray: np.ndarray) -> np.ndarray | None:
        """
        Attempts to find a 4-corner document contour in the image.
        Returns a (4, 2) float32 array of corner points, or None.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 120)
        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        image_area = gray.shape[0] * gray.shape[1]

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            area = cv2.contourArea(cnt)
            # Accept if roughly quadrilateral and covers at least 20 % of frame
            if len(approx) == 4 and area > 0.20 * image_area:
                return approx.reshape(4, 2).astype(np.float32)

        return None

    # ── 4-point perspective warp ─────────────────────────────────────────────

    @staticmethod
    def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Classic bird's-eye-view warp given 4 corner points."""
        # Order: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        w_a = np.linalg.norm(br - bl)
        w_b = np.linalg.norm(tr - tl)
        max_w = max(int(w_a), int(w_b))

        h_a = np.linalg.norm(tr - br)
        h_b = np.linalg.norm(tl - bl)
        max_h = max(int(h_a), int(h_b))

        dst = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
            dtype=np.float32,
        )
        src = np.array([tl, tr, br, bl], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (max_w, max_h))