"""
Response Schemas — Mobile ECG AI Platform
Pydantic models that define the API contract returned to callers.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class PipelineStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


# ─── Sub-schemas ──────────────────────────────────────────────────────────────

class PipelineStage(BaseModel):
    name: str = Field(..., description="Name of the pipeline stage.")
    status: PipelineStatus = Field(..., description="Execution status.")
    detail: str = Field("", description="Human-readable detail or error message.")

    model_config = {"use_enum_values": True}


class QualityReport(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Composite quality score (0–1).")
    acceptable: bool = Field(..., description="Whether the image meets minimum quality requirements.")
    reason: str = Field("", description="Human-readable quality assessment summary.")
    snr_db: float = Field(0.0, description="Estimated signal-to-noise ratio in dB.")
    blur_score: float = Field(0.0, description="Laplacian variance (sharpness proxy; higher = sharper).")

    model_config = {"json_schema_extra": {"example": {
        "score": 0.82,
        "acceptable": True,
        "reason": "Quality acceptable.",
        "snr_db": 24.5,
        "blur_score": 210.3,
    }}}


# ─── Main response ────────────────────────────────────────────────────────────

class ECGAnalysisResponse(BaseModel):
    """Top-level API response for a single ECG image analysis."""

    pipeline_stages: List[PipelineStage] = Field(
        ...,
        description="Status of each pipeline stage, in execution order.",
    )
    quality_report: Optional[QualityReport] = Field(
        None,
        description="Quality assessment results. Null if validation failed before quality check.",
    )
    top_class: str = Field(
        ...,
        description="Predicted ECG rhythm / finding with highest probability.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence for the top predicted class (0–1).",
    )
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability for every supported ECG class, sorted descending.",
    )
    clinical_summary: str = Field(
        ...,
        description=(
            "AI-generated clinical assistance text for the predicted finding. "
            "This is NOT a diagnosis — physician review is mandatory."
        ),
    )
    corrected_image_b64: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded perspective-corrected image (data URI). "
            "Only present when return_debug_image=true."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "pipeline_stages": [
                    {"name": "Validation", "status": "ok", "detail": "All checks passed."},
                    {"name": "Quality Assessment", "status": "ok", "detail": "Score=0.82"},
                    {"name": "Perspective Correction", "status": "ok", "detail": "Applied"},
                    {"name": "Preprocessing", "status": "ok", "detail": "Output shape: [1, 3, 224, 224]"},
                    {"name": "CNN Inference", "status": "ok", "detail": "Top class: Normal Sinus Rhythm"},
                    {"name": "Clinical Assistance Output", "status": "ok", "detail": ""},
                ],
                "quality_report": {
                    "score": 0.82,
                    "acceptable": True,
                    "reason": "Quality acceptable.",
                    "snr_db": 24.5,
                    "blur_score": 210.3,
                },
                "top_class": "Normal Sinus Rhythm",
                "confidence": 0.9412,
                "class_probabilities": {
                    "Normal Sinus Rhythm": 0.9412,
                    "Sinus Bradycardia": 0.0231,
                },
                "clinical_summary": (
                    "[AI-Assisted Clinical Note — NOT a diagnosis]\n\n"
                    "Detected pattern: Normal Sinus Rhythm (confidence 94.1 %)\n\n"
                    "Heart rhythm appears within normal limits…"
                ),
                "corrected_image_b64": None,
            }
        }
    }