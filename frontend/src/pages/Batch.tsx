import { useNavigate } from "react-router-dom";
import { useCallback } from "react";
import { toast } from "sonner";
import { analyzeECG } from "../api/ecgApi";
import { useAnalysisStore } from "../store/analysisStore";
import BatchTable from "../components/BatchTable";
import { v4 as uuidv4 } from "uuid";

export default function Batch() {
  const nav = useNavigate();
  const { batchItems, setBatchItems, updateBatchItem } = useAnalysisStore();

  const handleFiles = useCallback(
    async (files: FileList) => {
      const newItems = Array.from(files).slice(0, 10).map((f) => ({
        id: uuidv4(),
        file: f,
        status: "pending" as const,
      }));
      setBatchItems([...batchItems, ...newItems]);

      for (const item of newItems) {
        updateBatchItem(item.id, { status: "processing" });
        try {
          const result = await analyzeECG(item.file);
          updateBatchItem(item.id, { status: "done", result });
        } catch (e: unknown) {
          const msg = e instanceof Error ? e.message : "Failed";
          updateBatchItem(item.id, { status: "error", error: msg });
          toast.error(`${item.file.name}: ${msg}`);
        }
      }
    },
    [batchItems, setBatchItems, updateBatchItem]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  return (
    <div className="min-h-screen" style={{ background: "#0A0F1E" }}>
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            repeating-linear-gradient(0deg,transparent,transparent 99px,rgba(0,212,255,0.05) 100px),
            repeating-linear-gradient(90deg,transparent,transparent 99px,rgba(0,212,255,0.05) 100px),
            repeating-linear-gradient(0deg,transparent,transparent 19px,rgba(0,212,255,0.018) 20px),
            repeating-linear-gradient(90deg,transparent,transparent 19px,rgba(0,212,255,0.018) 20px)
          `,
        }}
      />

      <nav className="relative z-10 flex items-center justify-between px-8 py-5 border-b border-white/5">
        <button onClick={() => nav("/")} className="flex items-center gap-2 text-white/50 hover:text-white transition-colors">
          <span className="text-cyan-400">♥</span>
          <span className="text-sm tracking-wider">ECG·AI</span>
        </button>
        <span className="text-white/30 text-xs tracking-widest uppercase">BATCH ANALYSIS</span>
        <button onClick={() => nav("/analyze")} className="text-xs text-white/30 hover:text-cyan-400 tracking-widest uppercase transition-colors">
          SINGLE →
        </button>
      </nav>

      <div className="relative z-10 max-w-4xl mx-auto px-4 py-10 space-y-6">
        <h2 className="text-white/60 text-xs tracking-widest uppercase">UPLOAD UP TO 10 FILES</h2>

        {/* Drop zone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => document.getElementById("batch-input")?.click()}
          className="w-full rounded-lg border border-dashed border-cyan-400/25 flex flex-col items-center justify-center py-12 gap-3 cursor-pointer hover:border-cyan-400/50 transition-colors"
          style={{ background: "rgba(0,212,255,0.02)" }}
        >
          <input
            id="batch-input"
            type="file"
            multiple
            accept="image/png,image/jpeg,image/tiff"
            className="hidden"
            onChange={(e) => { if (e.target.files) handleFiles(e.target.files); }}
          />
          <span className="text-cyan-400/40 text-3xl">⊕</span>
          <p className="text-white/40 text-xs tracking-widest uppercase">DRAG & DROP UP TO 10 ECG IMAGES</p>
        </div>

        <BatchTable />
      </div>
    </div>
  );
}