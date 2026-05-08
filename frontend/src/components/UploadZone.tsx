import { useCallback, useState } from "react";
import { useAnalysisStore } from "../store/analysisStore";

interface Props {
  onFile: (f: File) => void;
}

export default function UploadZone({ onFile }: Props) {
  const [drag, setDrag] = useState(false);
  const file = useAnalysisStore((s) => s.file);
  const setFile = useAnalysisStore((s) => s.setFile);
  const reset = useAnalysisStore((s) => s.reset);

  const handleFile = useCallback(
    (f: File) => {
      setFile(f);
      onFile(f);
    },
    [onFile, setFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const preview = file ? URL.createObjectURL(file) : null;

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      onClick={() => { if (!file) document.getElementById("ecg-file-input")?.click(); }}
      className="relative w-full rounded-lg cursor-pointer transition-all duration-300 overflow-hidden"
      style={{
        border: `1.5px dashed ${drag ? "#00D4FF" : "rgba(0,212,255,0.3)"}`,
        boxShadow: drag ? "0 0 24px rgba(0,212,255,0.25)" : "none",
        background: `
          repeating-linear-gradient(0deg, transparent, transparent 19px, rgba(0,212,255,0.04) 20px),
          repeating-linear-gradient(90deg, transparent, transparent 19px, rgba(0,212,255,0.04) 20px),
          #0A0F1E
        `,
        minHeight: 220,
      }}
    >
      <input
        id="ecg-file-input"
        type="file"
        accept="image/png,image/jpeg,image/tiff"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />

      {file && preview ? (
        <div className="relative w-full h-full" style={{ minHeight: 220 }}>
          <img
            src={preview}
            alt="ECG preview"
            className="w-full h-full object-contain"
            style={{
              maxHeight: 280,
              backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(255,255,255,0.03) 2px)",
            }}
          />
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background: "repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(0,0,0,0.18) 2px)",
            }}
          />
          <button
            onClick={(e) => { e.stopPropagation(); reset(); }}
            className="absolute top-2 right-2 bg-black/60 border border-white/10 text-white/60 hover:text-white text-xs px-2 py-1 rounded transition-colors"
          >
            ✕ CLEAR
          </button>
          <div className="absolute bottom-2 left-3 text-cyan-400/70 text-xs tracking-widest">
            {file.name} · {(file.size / 1024).toFixed(1)} KB
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-14 gap-4">
          <div className="text-cyan-400/50 text-4xl select-none">⊕</div>
          <div className="text-center">
            <p className="text-white/70 text-sm tracking-widest uppercase">
              {drag ? "DROP TO ANALYZE" : "DRAG & DROP ECG IMAGE"}
            </p>
            <p className="text-white/30 text-xs mt-1 tracking-wider">
              PNG · JPEG · TIFF · MAX 20MB
            </p>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); document.getElementById("ecg-file-input")?.click(); }}
            className="mt-2 px-6 py-2 border border-cyan-400/40 text-cyan-400 text-xs tracking-widest uppercase hover:bg-cyan-400/10 transition-colors rounded"
          >
            BROWSE FILES
          </button>
        </div>
      )}
    </div>
  );
}