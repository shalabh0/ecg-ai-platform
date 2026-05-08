"""
Model Loader — Mobile ECG AI Platform
Loads CNN weights once at startup; supports ONNX Runtime, PyTorch, TFLite, and a mock backend.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

# Backend is auto-detected unless overridden via env var MODEL_BACKEND
# Values: "onnx" | "pytorch" | "tflite" | "mock"
BACKEND_OVERRIDE: Optional[str] = os.getenv("MODEL_BACKEND")

# Filenames to look for in MODELS_DIR
ONNX_FILENAME = os.getenv("ONNX_MODEL_FILENAME", "ecg_model.onnx")
PYTORCH_FILENAME = os.getenv("PYTORCH_MODEL_FILENAME", "ecg_model.pt")
TFLITE_FILENAME = os.getenv("TFLITE_MODEL_FILENAME", "ecg_model.tflite")

NUM_CLASSES = int(os.getenv("NUM_CLASSES", "20"))
INPUT_SHAPE = (1, 3, 224, 224)   # NCHW


# ─── Backend enum ─────────────────────────────────────────────────────────────

class Backend(str, Enum):
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TFLITE = "tflite"
    MOCK = "mock"


# ─── Model Loader ─────────────────────────────────────────────────────────────

class ModelLoader:
    """
    Loads and exposes the ECG CNN model for inference.

    Auto-detection priority (if MODEL_BACKEND env var not set):
      1. ONNX Runtime  (fast, cross-platform)
      2. PyTorch       (full-precision)
      3. TFLite        (edge deployment)
      4. Mock          (testing / demo)

    Call .load() once at startup, then .run(tensor) for each inference.
    """

    def __init__(self):
        self._session = None
        self._backend: Optional[Backend] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        backend = self._resolve_backend()
        self._backend = backend
        logger.info("Loading model with backend: %s", backend)

        loaders = {
            Backend.ONNX: self._load_onnx,
            Backend.PYTORCH: self._load_pytorch,
            Backend.TFLITE: self._load_tflite,
            Backend.MOCK: self._load_mock,
        }
        loaders[backend]()
        logger.info("Model loaded successfully via %s backend.", backend)

    def run(self, tensor: np.ndarray) -> np.ndarray:
        """
        Run a single forward pass.

        Parameters
        ----------
        tensor : np.ndarray  shape=(1, 3, 224, 224)  float32

        Returns
        -------
        np.ndarray  shape=(1, NUM_CLASSES)  raw logits
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        runners = {
            Backend.ONNX: self._run_onnx,
            Backend.PYTORCH: self._run_pytorch,
            Backend.TFLITE: self._run_tflite,
            Backend.MOCK: self._run_mock,
        }
        return runners[self._backend](tensor)

    @property
    def backend(self) -> Optional[Backend]:
        return self._backend

    # ── Backend resolution ────────────────────────────────────────────────────

    def _resolve_backend(self) -> Backend:
        if BACKEND_OVERRIDE:
            return Backend(BACKEND_OVERRIDE.lower())

        if (MODELS_DIR / ONNX_FILENAME).exists():
            try:
                import onnxruntime  # noqa: F401
                return Backend.ONNX
            except ImportError:
                logger.warning("ONNX model found but onnxruntime not installed.")

        if (MODELS_DIR / PYTORCH_FILENAME).exists():
            try:
                import torch  # noqa: F401
                return Backend.PYTORCH
            except ImportError:
                logger.warning("PyTorch model found but torch not installed.")

        if (MODELS_DIR / TFLITE_FILENAME).exists():
            try:
                import tflite_runtime.interpreter  # noqa: F401
                return Backend.TFLITE
            except ImportError:
                logger.warning("TFLite model found but tflite_runtime not installed.")

        logger.warning("No trained model found in %s — using MOCK backend.", MODELS_DIR)
        return Backend.MOCK

    # ── ONNX ──────────────────────────────────────────────────────────────────

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        model_path = str(MODELS_DIR / ONNX_FILENAME)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "4"))

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ONNX session created. Input: %s", self._input_name)

    def _run_onnx(self, tensor: np.ndarray) -> np.ndarray:
        outputs = self._session.run(None, {self._input_name: tensor})
        return outputs[0]   # (1, num_classes)

    # ── PyTorch ───────────────────────────────────────────────────────────────

    def _load_pytorch(self) -> None:
        import torch

        model_path = MODELS_DIR / PYTORCH_FILENAME
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch_device = torch.device(device)
        self._session = torch.load(str(model_path), map_location=self._torch_device)
        self._session.eval()
        logger.info("PyTorch model loaded on device: %s", device)

    def _run_pytorch(self, tensor: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = torch.from_numpy(tensor).to(self._torch_device)
            logits = self._session(x)
            return logits.cpu().numpy()

    # ── TFLite ────────────────────────────────────────────────────────────────

    def _load_tflite(self) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        model_path = str(MODELS_DIR / TFLITE_FILENAME)
        self._session = Interpreter(model_path=model_path)
        self._session.allocate_tensors()
        self._tflite_input = self._session.get_input_details()[0]
        self._tflite_output = self._session.get_output_details()[0]
        logger.info("TFLite interpreter ready.")

    def _run_tflite(self, tensor: np.ndarray) -> np.ndarray:
        # TFLite expects NHWC; our tensor is NCHW → transpose
        nhwc = np.transpose(tensor, (0, 2, 3, 1))
        self._session.set_tensor(self._tflite_input["index"], nhwc)
        self._session.invoke()
        output = self._session.get_tensor(self._tflite_output["index"])
        return output   # (1, num_classes)

    # ── Mock ──────────────────────────────────────────────────────────────────

    def _load_mock(self) -> None:
        """Deterministic mock that returns reproducible random logits."""
        self._session = "mock"
        logger.warning(
            "MOCK backend active — outputs are random and NOT clinically meaningful."
        )

    def _run_mock(self, tensor: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(seed=int(abs(tensor.sum()) * 1e4) % (2**31))
        logits = rng.standard_normal((1, NUM_CLASSES)).astype(np.float32)
        # Bias toward "Normal Sinus Rhythm" (index 0) for demo realism
        logits[0, 0] += 1.5
        return logits