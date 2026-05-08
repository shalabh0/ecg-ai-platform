export interface PipelineStage {
  name: string;
  status: "ok" | "failed";
  detail: string;
}

export interface QualityReport {
  score: number;
  acceptable: boolean;
  reason: string;
  snr_db: number;
  blur_score: number;
}

export interface ECGAnalysisResponse {
  pipeline_stages: PipelineStage[];
  quality_report: QualityReport;
  top_class: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  clinical_summary: string;
  corrected_image_b64: string | null;
}