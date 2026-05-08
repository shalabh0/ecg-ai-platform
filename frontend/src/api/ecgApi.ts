import { ECGAnalysisResponse } from "../types/ecg";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function analyzeECG(file: File): Promise<ECGAnalysisResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/v1/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${res.status}: ${err}`);
  }
  return res.json();
}

export async function healthCheck(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}