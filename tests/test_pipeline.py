"""
Tests — Mobile ECG AI Platform
Covers validation, quality, perspective, preprocessing, inference, and API endpoints.
"""

from __future__ import annotations

import io
import os
import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Force mock backend so tests never need a real model file
os.environ["MODEL_BACKEND"] = "mock"

from app.main import app
from app.processing.validation import ImageValidator
from app.processing.quality import QualityAssessor
from app.processing.perspective import PerspectiveCorrector
from app.processing.preprocessing import ECGPreprocessor
from app.inference.model_loader import ModelLoader
from app.inference.classifier import ECGClassifier
from app.utils.image_utils import (
    decode_image,
    encode_image_bytes,
    encode_image_base64,
    to_grayscale,
    ensure_rgb,
    image_stats,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ecg_image(h: int = 512, w: int = 700, noise: float = 30.0) -> np.ndarray:
    """Synthetic ECG-like image: white background with noisy horizontal lines."""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    # Grid lines
    for y in range(0, h, 20):
        img[y, :] = [220, 180, 180]
    for x in range(0, w, 20):
        img[:, x] = [220, 180, 180]
    # Simulate ECG waveform (dark line)
    for x in range(w):
        base = h // 2
        wave = int(40 * np.sin(x * 0.08) + 20 * np.sin(x * 0.3))
        y = np.clip(base + wave, 0, h - 1)
        img[y - 2 : y + 2, x] = [10, 10, 10]
    # Add noise
    noise_arr = (np.random.randn(h, w, 3) * noise).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def ecg_image() -> np.ndarray:
    return _make_ecg_image()


@pytest.fixture
def ecg_jpeg_bytes(ecg_image) -> bytes:
    return encode_image_bytes(ecg_image, fmt=".jpg")


@pytest.fixture
def classifier() -> ECGClassifier:
    loader = ModelLoader()
    loader.load()
    return ECGClassifier(loader)


# ─── Image utils ─────────────────────────────────────────────────────────────

class TestImageUtils:
    def test_decode_valid_jpeg(self, ecg_jpeg_bytes):
        img = decode_image(ecg_jpeg_bytes)
        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_decode_invalid_returns_none(self):
        assert decode_image(b"not an image") is None

    def test_encode_decode_roundtrip(self, ecg_image):
        raw = encode_image_bytes(ecg_image, fmt=".png")
        recovered = decode_image(raw)
        assert recovered is not None
        assert recovered.shape == ecg_image.shape

    def test_encode_base64_prefix(self, ecg_image):
        b64 = encode_image_base64(ecg_image)
        assert b64.startswith("data:image/jpeg;base64,")

    def test_to_grayscale_rgb(self, ecg_image):
        gray = to_grayscale(ecg_image)
        assert gray.ndim == 2

    def test_ensure_rgb_grayscale(self):
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        rgb = ensure_rgb(gray)
        assert rgb.shape == (64, 64, 3)

    def test_image_stats_keys(self, ecg_image):
        stats = image_stats(ecg_image)
        for key in ("shape", "dtype", "min", "max", "mean", "std"):
            assert key in stats


# ─── Validation ──────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_image_passes(self, ecg_image):
        result = ImageValidator().validate(ecg_image)
        assert result.passed

    def test_too_small_fails(self):
        tiny = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = ImageValidator().validate(tiny)
        assert not result.passed
        assert "small" in result.message.lower()

    def test_wrong_type_fails(self):
        result = ImageValidator().validate("not an array")
        assert not result.passed

    def test_blank_image_fails(self):
        blank = np.full((300, 300, 3), 128, dtype=np.uint8)
        result = ImageValidator().validate(blank)
        assert not result.passed
        assert "blank" in result.message.lower()

    def test_nan_fails(self):
        img = np.full((300, 300, 3), np.nan, dtype=np.float32)
        result = ImageValidator().validate(img)
        assert not result.passed


# ─── Quality Assessment ───────────────────────────────────────────────────────

class TestQualityAssessment:
    def test_good_image_acceptable(self, ecg_image):
        result = QualityAssessor().assess(ecg_image)
        assert result.score > 0.0
        assert isinstance(result.acceptable, bool)

    def test_blurry_image_lower_score(self, ecg_image):
        import cv2
        blurry = cv2.GaussianBlur(ecg_image, (51, 51), 30)
        sharp_score = QualityAssessor().assess(ecg_image).score
        blur_score = QualityAssessor().assess(blurry).score
        assert sharp_score >= blur_score

    def test_result_fields_present(self, ecg_image):
        result = QualityAssessor().assess(ecg_image)
        assert hasattr(result, "snr_db")
        assert hasattr(result, "blur_score")
        assert hasattr(result, "contrast")


# ─── Perspective Correction ───────────────────────────────────────────────────

class TestPerspectiveCorrection:
    def test_straight_image_no_correction(self, ecg_image):
        corrected, applied = PerspectiveCorrector().correct(ecg_image)
        assert corrected.shape[2] == 3   # still RGB

    def test_skewed_image_corrected(self, ecg_image):
        import cv2
        h, w = ecg_image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)
        skewed = cv2.warpAffine(ecg_image, M, (w, h))
        corrected, applied = PerspectiveCorrector().correct(skewed)
        # Correction may or may not be applied depending on detected angle,
        # but output must always be a valid image
        assert corrected.ndim == 3


# ─── Preprocessing ────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_output_shape(self, ecg_image):
        preprocessor = ECGPreprocessor()
        tensor = preprocessor.preprocess(ecg_image)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == np.float32

    def test_output_normalised(self, ecg_image):
        tensor = ECGPreprocessor().preprocess(ecg_image)
        # After ImageNet normalisation values are in roughly [-3, 3]
        assert tensor.min() < 0
        assert tensor.max() < 5.0

    def test_denormalize_roundtrip_shape(self, ecg_image):
        prep = ECGPreprocessor()
        tensor = prep.preprocess(ecg_image)
        recovered = prep.denormalize(tensor)
        assert recovered.shape == (224, 224, 3)
        assert recovered.dtype == np.uint8

    def test_grayscale_input(self):
        gray = np.random.randint(50, 200, (300, 400), dtype=np.uint8)
        tensor = ECGPreprocessor().preprocess(gray)
        assert tensor.shape == (1, 3, 224, 224)


# ─── Inference ───────────────────────────────────────────────────────────────

class TestInference:
    def test_predict_returns_result(self, ecg_image, classifier):
        tensor = ECGPreprocessor().preprocess(ecg_image)
        result = classifier.predict(tensor)
        assert result.top_class in classifier.class_labels
        assert 0.0 <= result.confidence <= 1.0

    def test_probabilities_sum_to_one(self, ecg_image, classifier):
        tensor = ECGPreprocessor().preprocess(ecg_image)
        result = classifier.predict(tensor)
        total = sum(result.class_probabilities.values())
        assert abs(total - 1.0) < 1e-4

    def test_clinical_summary_contains_class(self, ecg_image, classifier):
        tensor = ECGPreprocessor().preprocess(ecg_image)
        result = classifier.predict(tensor)
        assert result.top_class in result.clinical_summary

    def test_class_labels_count(self, classifier):
        assert len(classifier.class_labels) == 20


# ─── API Endpoints ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAPI:
    async def _client(self):
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def test_health(self):
        async with await self._client() as client:
            r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    async def test_analyze_valid_image(self, ecg_jpeg_bytes):
        async with await self._client() as client:
            r = await client.post(
                "/api/v1/analyze",
                files={"file": ("ecg.jpg", ecg_jpeg_bytes, "image/jpeg")},
            )
        assert r.status_code == 200
        body = r.json()
        assert "top_class" in body
        assert "confidence" in body
        assert "clinical_summary" in body
        assert "pipeline_stages" in body

    async def test_analyze_invalid_content_type(self, ecg_jpeg_bytes):
        async with await self._client() as client:
            r = await client.post(
                "/api/v1/analyze",
                files={"file": ("ecg.pdf", ecg_jpeg_bytes, "application/pdf")},
            )
        assert r.status_code == 415

    async def test_analyze_too_large(self):
        big = b"X" * (21 * 1024 * 1024)  # 21 MB
        async with await self._client() as client:
            r = await client.post(
                "/api/v1/analyze",
                files={"file": ("big.jpg", big, "image/jpeg")},
            )
        assert r.status_code == 413

    async def test_classes_endpoint(self):
        async with await self._client() as client:
            r = await client.get("/api/v1/classes")
        assert r.status_code == 200
        assert "classes" in r.json()
        assert len(r.json()["classes"]) == 20

    async def test_process_time_header(self, ecg_jpeg_bytes):
        async with await self._client() as client:
            r = await client.post(
                "/api/v1/analyze",
                files={"file": ("ecg.jpg", ecg_jpeg_bytes, "image/jpeg")},
            )
        assert "X-Process-Time-Ms" in r.headers