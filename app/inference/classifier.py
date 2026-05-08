"""
ECG Classifier — Mobile ECG AI Platform
Runs inference on a preprocessed tensor and generates clinical assistance output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from app.inference.model_loader import ModelLoader

logger = logging.getLogger(__name__)

# ─── ECG Rhythm Classes ───────────────────────────────────────────────────────

ECG_CLASSES: List[str] = [
    "Normal Sinus Rhythm",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "Atrial Fibrillation",
    "Atrial Flutter",
    "Supraventricular Tachycardia (SVT)",
    "Premature Atrial Contractions (PAC)",
    "Premature Ventricular Contractions (PVC)",
    "Left Bundle Branch Block (LBBB)",
    "Right Bundle Branch Block (RBBB)",
    "First-Degree AV Block",
    "Second-Degree AV Block (Mobitz I)",
    "Second-Degree AV Block (Mobitz II)",
    "Third-Degree (Complete) AV Block",
    "ST Elevation (STEMI pattern)",
    "ST Depression",
    "T-Wave Inversion",
    "Long QT Syndrome",
    "Ventricular Tachycardia",
    "Ventricular Fibrillation",
]

# ─── Clinical assistance templates ───────────────────────────────────────────

_CLINICAL_NOTES: Dict[str, str] = {
    "Normal Sinus Rhythm": (
        "Heart rhythm appears within normal limits. Rate 60–100 bpm with regular P-QRS-T morphology. "
        "No immediate intervention indicated; routine follow-up as clinically appropriate."
    ),
    "Sinus Bradycardia": (
        "Heart rate below 60 bpm in a sinus pattern. Consider physiological causes (athletes, sleep) "
        "vs. pathological (hypothyroidism, medication effect, sick sinus syndrome). "
        "Haemodynamic stability assessment recommended."
    ),
    "Sinus Tachycardia": (
        "Heart rate above 100 bpm with sinus morphology. Investigate underlying cause: fever, pain, "
        "hypovolaemia, anaemia, hyperthyroidism, anxiety, or pharmacological effect. "
        "Treat aetiology rather than rate."
    ),
    "Atrial Fibrillation": (
        "Irregularly irregular rhythm with absent P waves — consistent with atrial fibrillation. "
        "Assess stroke risk (CHA₂DS₂-VASc) and bleeding risk (HAS-BLED). "
        "Anticoagulation and rate/rhythm control strategy warranted. Urgent evaluation if haemodynamically unstable."
    ),
    "Atrial Flutter": (
        "Regular sawtooth flutter waves (~300 bpm atrial rate) with typical 2:1 or 4:1 AV conduction. "
        "Consider cardioversion, rate control, or ablation. Thromboembolic risk similar to AF."
    ),
    "Supraventricular Tachycardia (SVT)": (
        "Narrow-complex regular tachycardia. First-line: vagal manoeuvres; second-line: IV adenosine. "
        "If haemodynamically unstable, synchronised DC cardioversion. Electrophysiology referral for recurrent SVT."
    ),
    "Premature Atrial Contractions (PAC)": (
        "Isolated early ectopic beats arising from atria. Usually benign; reassurance often sufficient. "
        "Reduce caffeine, alcohol, and stress. Evaluate for underlying structural disease if frequent."
    ),
    "Premature Ventricular Contractions (PVC)": (
        "Early wide-complex beats of ventricular origin. Assess burden (> 10 % may cause cardiomyopathy). "
        "Exclude electrolyte imbalance, ischaemia, and cardiomyopathy. Consider β-blocker or referral if symptomatic."
    ),
    "Left Bundle Branch Block (LBBB)": (
        "LBBB pattern: QRS ≥ 120 ms, broad notched R in V5/V6, deep S in V1. "
        "New LBBB in the context of chest pain may represent STEMI-equivalent — urgent coronary angiography may be required. "
        "Echocardiogram to assess LV function."
    ),
    "Right Bundle Branch Block (RBBB)": (
        "RBBB pattern: rSR′ in V1, broad S in V5/V6, QRS ≥ 120 ms. "
        "Isolated RBBB is often benign. Consider pulmonary embolism if new onset with clinical suspicion. "
        "Evaluate for structural heart disease."
    ),
    "First-Degree AV Block": (
        "PR interval > 200 ms. Generally benign; no treatment required. "
        "Review medications (digoxin, β-blockers, calcium channel blockers). Monitor for progression."
    ),
    "Second-Degree AV Block (Mobitz I)": (
        "Wenckebach: progressive PR lengthening followed by dropped beat. Usually benign and vagally mediated. "
        "Review medications; if symptomatic or inferior MI related, seek cardiology input."
    ),
    "Second-Degree AV Block (Mobitz II)": (
        "Fixed PR with sudden non-conducted P waves. High risk of progression to complete heart block. "
        "Urgent pacing evaluation required — temporary pacing may be necessary."
    ),
    "Third-Degree (Complete) AV Block": (
        "Complete AV dissociation. Haemodynamic emergency. "
        "Immediate cardiology consultation; transcutaneous or transvenous pacing required. "
        "Permanent pacemaker implantation likely indicated."
    ),
    "ST Elevation (STEMI pattern)": (
        "ST elevation consistent with STEMI pattern. ACTIVATE CATH LAB. "
        "Aspirin + P2Y12 inhibitor loading, anticoagulation, and urgent primary PCI (door-to-balloon < 90 min). "
        "This is a time-critical emergency — do not delay reperfusion therapy."
    ),
    "ST Depression": (
        "ST depression may indicate subendocardial ischaemia, NSTEMI, or posterior STEMI (V1–V3 depression). "
        "Serial ECGs and troponins. Risk stratify with GRACE or TIMI score. "
        "Cardiology review urgently."
    ),
    "T-Wave Inversion": (
        "T-wave inversion may reflect ischaemia, myocarditis, cardiomyopathy, PE, or normal variant. "
        "Correlate with symptoms, troponin, and clinical context. "
        "Diffuse T-wave inversion in V1–V4 raises suspicion for Wellens syndrome."
    ),
    "Long QT Syndrome": (
        "Prolonged QTc (> 450 ms men, > 460 ms women) increases risk of torsades de pointes. "
        "Review QT-prolonging medications (azithromycin, haloperidol, methadone, etc.). "
        "Correct electrolytes (K⁺, Mg²⁺). Cardiology and genetics referral. Avoid triggering medications."
    ),
    "Ventricular Tachycardia": (
        "Wide-complex regular tachycardia — ventricular tachycardia until proven otherwise. "
        "If pulseless: CPR + defibrillation. If pulse present but unstable: synchronised cardioversion. "
        "If stable: IV amiodarone or procainamide. Urgent cardiology."
    ),
    "Ventricular Fibrillation": (
        "CARDIAC ARREST. Chaotic, disorganised ventricular activity — no effective output. "
        "INITIATE CPR IMMEDIATELY. DEFIBRILLATE as soon as possible. "
        "Follow ALS/ACLS protocol. Adrenaline 1 mg IV every 3–5 min."
    ),
}

_DEFAULT_NOTE = (
    "Pattern identified by automated AI analysis. "
    "Clinical correlation and physician review are mandatory before any management decision."
)

# ─── Inference result ─────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    top_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    clinical_summary: str = field(init=False)

    def __post_init__(self):
        note = _CLINICAL_NOTES.get(self.top_class, _DEFAULT_NOTE)
        self.clinical_summary = (
            f"[AI-Assisted Clinical Note — NOT a diagnosis]\n\n"
            f"Detected pattern: {self.top_class} (confidence {self.confidence * 100:.1f} %)\n\n"
            f"{note}\n\n"
            f"⚠️  This output is intended to assist, not replace, clinical judgement. "
            f"Always confirm findings with a qualified clinician."
        )


# ─── Classifier ───────────────────────────────────────────────────────────────

class ECGClassifier:
    """
    Stage 5 — CNN Inference + Stage 6 — Clinical Assistance Output.

    The classifier wraps whatever backend the ModelLoader provides
    (ONNX Runtime, PyTorch, TFLite, or a mock) and returns an InferenceResult.
    """

    def __init__(self, model_loader: ModelLoader):
        self._loader = model_loader

    @property
    def class_labels(self) -> List[str]:
        return ECG_CLASSES

    def predict(self, tensor: np.ndarray) -> InferenceResult:
        """
        Parameters
        ----------
        tensor : np.ndarray
            Float32 array of shape (1, 3, H, W), normalised.

        Returns
        -------
        InferenceResult
        """
        logits = self._loader.run(tensor)          # shape: (1, num_classes)
        probs = self._softmax(logits[0])           # shape: (num_classes,)

        # Pad or trim if model output dim differs from class list
        n = len(ECG_CLASSES)
        if len(probs) < n:
            probs = np.pad(probs, (0, n - len(probs)))
        probs = probs[:n]
        probs = probs / probs.sum()                # renormalise

        top_idx = int(np.argmax(probs))
        top_class = ECG_CLASSES[top_idx]
        confidence = float(probs[top_idx])

        class_probs = {
            cls: round(float(p), 4)
            for cls, p in zip(ECG_CLASSES, probs)
        }
        # Sort descending for readability
        class_probs = dict(sorted(class_probs.items(), key=lambda x: -x[1]))

        logger.info(
            "Inference  top=%s  confidence=%.3f", top_class, confidence
        )

        return InferenceResult(
            top_class=top_class,
            confidence=round(confidence, 4),
            class_probabilities=class_probs,
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()