import { motion } from "framer-motion";
import { PipelineStage } from "../types/ecg";

const STAGE_ICONS: Record<string, string> = {
  Validation: "⬡",
  "Quality Assessment": "◈",
  "Perspective Correction": "◫",
  Preprocessing: "◳",
  "CNN Inference": "◉",
  "Clinical Assistance Output": "✦",
};

interface Props {
  stages: PipelineStage[];
  loading?: boolean;
  totalStages?: number;
}

export default function PipelineProgress({ stages, loading, totalStages = 6 }: Props) {
  const progress = (stages.length / totalStages) * 100;

  return (
    <div className="w-full space-y-1">
      {/* Stage list */}
      {stages.map((stage, i) => (
        <motion.div
          key={stage.name}
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.05, duration: 0.25 }}
          className="relative flex items-start gap-3 py-2"
        >
          {/* Connector line */}
          {i < stages.length - 1 && (
            <div
              className="absolute top-7 bottom-0 w-px bg-white/10"
              style={{ left: 19 }}
            />
          )}

          {/* Status icon */}
          <div
            className="flex-shrink-0 w-9 h-9 rounded border flex items-center justify-center text-sm"
            style={{
              borderColor: stage.status === "ok" ? "rgba(0,212,255,0.4)" : "rgba(255,80,80,0.4)",
              background: stage.status === "ok" ? "rgba(0,212,255,0.07)" : "rgba(255,80,80,0.07)",
              color: stage.status === "ok" ? "#00D4FF" : "#FF5050",
            }}
          >
            {stage.status === "ok" ? STAGE_ICONS[stage.name] ?? "✓" : "✕"}
          </div>

          <div className="flex-1 min-w-0 pt-1">
            <div className="flex items-center gap-2">
              <span className="text-white/80 text-xs tracking-widest uppercase">{stage.name}</span>
              <span
                className="text-xs px-1.5 py-0.5 rounded tracking-widest"
                style={{
                  background: stage.status === "ok" ? "rgba(0,212,255,0.1)" : "rgba(255,80,80,0.1)",
                  color: stage.status === "ok" ? "#00D4FF" : "#FF5050",
                }}
              >
                {stage.status.toUpperCase()}
              </span>
            </div>
            {stage.detail && (
              <p className="text-white/40 text-xs mt-0.5 truncate">{stage.detail}</p>
            )}
          </div>
        </motion.div>
      ))}

      {/* Processing pulse */}
      {loading && stages.length < totalStages && (
        <motion.div
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ repeat: Infinity, duration: 1.2 }}
          className="flex items-center gap-3 py-2"
        >
          <div className="w-9 h-9 rounded border border-cyan-400/30 flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          </div>
          <span className="text-cyan-400/70 text-xs tracking-widest uppercase">PROCESSING…</span>
        </motion.div>
      )}

      {/* Progress bar */}
      <div className="mt-4 h-1 w-full bg-white/5 rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.4 }}
          style={{
            background: "linear-gradient(90deg, #0066FF, #00D4FF)",
            boxShadow: "0 0 10px rgba(0,212,255,0.6)",
          }}
        />
      </div>
      <p className="text-white/30 text-xs tracking-widest text-right">
        {stages.length}/{totalStages} STAGES
      </p>
    </div>
  );
}