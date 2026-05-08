import { useState } from "react";
import { motion } from "framer-motion";

interface Props {
  probabilities: Record<string, number>;
}

export default function ProbabilityChart({ probabilities }: Props) {
  const [expanded, setExpanded] = useState(false);
  const sorted = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .filter(([, v]) => v > 0);
  const visible = expanded ? sorted : sorted.slice(0, 5);

  return (
    <div className="space-y-2">
      <h3 className="text-white/50 text-xs tracking-widest uppercase mb-3">CLASS PROBABILITIES</h3>
      {visible.map(([cls, prob], i) => (
        <motion.div
          key={cls}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.04 }}
          className="space-y-1"
        >
          <div className="flex justify-between text-xs">
            <span className="text-white/70 truncate max-w-[70%]">{cls}</span>
            <span className="text-cyan-400/80 tabular-nums">{(prob * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1 bg-white/5 rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${prob * 100}%` }}
              transition={{ duration: 0.8, delay: i * 0.04, ease: "easeOut" }}
              style={{
                background: i === 0
                  ? "linear-gradient(90deg,#0066FF,#00D4FF)"
                  : "rgba(0,212,255,0.35)",
                boxShadow: i === 0 ? "0 0 6px rgba(0,212,255,0.5)" : "none",
              }}
            />
          </div>
        </motion.div>
      ))}

      {sorted.length > 5 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-white/30 hover:text-cyan-400 tracking-widest uppercase mt-2 transition-colors"
        >
          {expanded ? "▲ SHOW LESS" : `▼ SHOW ${sorted.length - 5} MORE`}
        </button>
      )}
    </div>
  );
}