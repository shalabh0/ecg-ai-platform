import { create } from "zustand";
import { ECGAnalysisResponse } from "../types/ecg";

interface BatchItem {
  id: string;
  file: File;
  status: "pending" | "processing" | "done" | "error";
  result?: ECGAnalysisResponse;
  error?: string;
}

interface AnalysisStore {
  result: ECGAnalysisResponse | null;
  loading: boolean;
  error: string | null;
  file: File | null;
  batchItems: BatchItem[];
  setResult: (r: ECGAnalysisResponse) => void;
  setLoading: (v: boolean) => void;
  setError: (e: string | null) => void;
  setFile: (f: File | null) => void;
  reset: () => void;
  setBatchItems: (items: BatchItem[]) => void;
  updateBatchItem: (id: string, patch: Partial<BatchItem>) => void;
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  result: null,
  loading: false,
  error: null,
  file: null,
  batchItems: [],
  setResult: (r) => set({ result: r }),
  setLoading: (v) => set({ loading: v }),
  setError: (e) => set({ error: e }),
  setFile: (f) => set({ file: f }),
  reset: () => set({ result: null, loading: false, error: null, file: null }),
  setBatchItems: (items) => set({ batchItems: items }),
  updateBatchItem: (id, patch) =>
    set((s) => ({
      batchItems: s.batchItems.map((i) => (i.id === id ? { ...i, ...patch } : i)),
    })),
}));