"""
API Routes — Mobile ECG AI Platform
Exposes endpoints for ECG image upload and analysis.
"""

import io
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.inference.classifier import ECGClassifier
from app.processing.perspective import PerspectiveCorrector
from app.processing.preprocessing import ECGPreprocessor
from app.processing.quality import QualityAssessor
from app.processing.validation import ImageValidator
from app.schemas.response import (
    ECGAnalysisResponse,
    PipelineStage,
    PipelineStatus,
    QualityReport,
)
from app.utils.image_utils import decode_image, encode_image_base64

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Dependency: classifier ──────────────────────────────────────────────────

def get_classifier(request: Request) -> ECGClassifier:
    model_loader = request.app.state.model_loader
    return ECGClassifier(model_loader)


# ─── Helper: run full pipeline ───────────────────────────────────────────────

async def _run_pipeline(
    image_bytes: bytes,
    classifier: ECGClassifier,
    return_debug_images: bool = False,
) -> ECGAnalysisResponse:
    stages: list[PipelineStage] = []

    def _stage(name: str, ok: bool, detail: str = "") -> None:
        stages.append(PipelineStage(name=name, status=PipelineStatus.OK if ok else PipelineStatus.FAILED, detail=detail))

    # ── 1. Decode ────────────────────────────────────────────────────────────
    image = decode_image(image_bytes)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode image bytes.")

    # ── 2. Validation ────────────────────────────────────────────────────────
    validator = ImageValidator()
    val_result = validator.validate(image)
    _stage("Validation", val_result.passed, val_result.message)
    if not val_result.passed:
        raise HTTPException(status_code=422, detail=f"Validation failed: {val_result.message}")

    # ── 3. Quality Assessment ────────────────────────────────────────────────
    assessor = QualityAssessor()
    quality = assessor.assess(image)
    _stage("Quality Assessment", quality.acceptable, f"Score={quality.score:.2f}")
    if not quality.acceptable:
        raise HTTPException(
            status_code=422,
            detail=f"Image quality too low (score={quality.score:.2f}). {quality.reason}",
        )

    # ── 4. Perspective Correction ────────────────────────────────────────────
    corrector = PerspectiveCorrector()
    corrected, correction_applied = corrector.correct(image)
    _stage("Perspective Correction", True, "Applied" if correction_applied else "Not needed")

    # ── 5. Preprocessing ─────────────────────────────────────────────────────
    preprocessor = ECGPreprocessor()
    tensor = preprocessor.preprocess(corrected)
    _stage("Preprocessing", True, f"Output shape: {list(tensor.shape)}")

    # ── 6. CNN Inference ─────────────────────────────────────────────────────
    inference_result = classifier.predict(tensor)
    _stage("CNN Inference", True, f"Top class: {inference_result.top_class}")

    # ── 7. Clinical Assistance ───────────────────────────────────────────────
    clinical = inference_result.clinical_summary
    _stage("Clinical Assistance Output", True)

    debug_image_b64 = encode_image_base64(corrected) if return_debug_images else None

    return ECGAnalysisResponse(
        pipeline_stages=stages,
        quality_report=QualityReport(
            score=quality.score,
            acceptable=quality.acceptable,
            reason=quality.reason,
            snr_db=quality.snr_db,
            blur_score=quality.blur_score,
        ),
        top_class=inference_result.top_class,
        confidence=inference_result.confidence,
        class_probabilities=inference_result.class_probabilities,
        clinical_summary=clinical,
        corrected_image_b64=debug_image_b64,
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=ECGAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze an ECG image",
    description=(
        "Upload a mobile-captured ECG image (JPEG / PNG / TIFF). "
        "Returns classification, clinical assistance output, and pipeline diagnostics."
    ),
    tags=["ECG Analysis"],
)
async def analyze_ecg(
    file: Annotated[UploadFile, File(description="ECG image file (JPEG/PNG/TIFF, max 20 MB)")],
    return_debug_image: bool = False,
    classifier: ECGClassifier = Depends(get_classifier),
):
    # ── Size guard (20 MB) ────────────────────────────────────────────────────
    MAX_BYTES = 20 * 1024 * 1024
    image_bytes = await file.read()
    if len(image_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")

    # ── MIME guard ────────────────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/png", "image/tiff"}
    content_type = file.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Allowed: {allowed_types}",
        )

    logger.info("Received ECG image: %s  size=%d bytes", file.filename, len(image_bytes))

    return await _run_pipeline(image_bytes, classifier, return_debug_images=return_debug_image)


@router.post(
    "/analyze/batch",
    response_model=list[ECGAnalysisResponse],
    status_code=status.HTTP_200_OK,
    summary="Analyze multiple ECG images",
    tags=["ECG Analysis"],
)
async def analyze_ecg_batch(
    files: Annotated[list[UploadFile], File(description="Multiple ECG image files (max 10)")],
    classifier: ECGClassifier = Depends(get_classifier),
):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Batch limited to 10 images.")

    results = []
    for f in files:
        image_bytes = await f.read()
        try:
            result = await _run_pipeline(image_bytes, classifier)
        except HTTPException as exc:
            # Return partial failure inline rather than aborting whole batch
            result = ECGAnalysisResponse(
                pipeline_stages=[PipelineStage(name="Batch Item", status=PipelineStatus.FAILED, detail=exc.detail)],
                quality_report=None,
                top_class="ERROR",
                confidence=0.0,
                class_probabilities={},
                clinical_summary=exc.detail,
            )
        results.append(result)

    return results


@router.get(
    "/classes",
    summary="List supported ECG rhythm classes",
    tags=["ECG Analysis"],
)
async def list_classes(classifier: ECGClassifier = Depends(get_classifier)):
    return {"classes": classifier.class_labels}