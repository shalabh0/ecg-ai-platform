# Mobile ECG AI

![Status](https://img.shields.io/badge/status-WIP-orange)

Production-oriented Mobile ECG AI Clinical Assistance Platform.

The system processes mobile-captured ECG sheet images, validates image quality, applies preprocessing, and performs CNN-based ECG interpretation.

---

## Pipeline

```text
Mobile ECG Photo
        ↓
Validation
        ↓
Quality Assessment
        ↓
Perspective Correction
        ↓
Preprocessing
        ↓
CNN ECG Classification
        ↓
Clinical Assistance Output