import { motion } from "framer-motion";
import { useEffect, useState } from "react";

interface Props {
  topClass: string;
  confidence: number;
}

function confidenceColor(c: number) {
  if (c >= 0.7) return { stroke: "#00FF88", border: "rgba(0,255,136,0.3)", text: "#00FF88", label: "HIGH" };
  if (c >= 0.5) return { stroke: "#FFB800", border: "rgba(255,184,0,0.3)", text: "#FFB800", label: "MODERATE" };
  return { stroke: "#FF5050", border: "rgba(255,80,80,0.3)", text: "#FF5050", label: "LOW" };
}

export default function ResultCard({ topClass, confidence }: Props) {
  const [animated, setAnimated] = useState(0);
  const col = confidenceColor(confidence);
  const r = 54;
  const circ = Math.PI * r; // half-circle circumference
  const dash = circ * animated;

  useEffect(() => {
    const timeout = setTimeout(() => setAnimated(confidence), 100);
    return () => clearTimeout(timeout);
  }, [confidence]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg p-6 border"
      style={{ borderColor: col.border, background: "rgba(10,15,30,0.95)" }}
    >
      {/* Half-circle gauge */}
      <div className="flex flex-col items-center mb-4">
        <svg width="140" height="80" viewBox="0 0 140 80">
          {/* Track */}
          <path
            d="M 14 76 A 56 56 0 0 1 126 76"
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Fill arc */}
          <motion.path
            d="M 14 76 A 56 56 0 0 1 126 76"
            fill="none"
            stroke={col.stroke}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${circ} ${circ}`}
            initial={{ strokeDashoffset: circ }}
            animate={{ strokeDashoffset: circ - dash }}
            transition={{ duration: 1.2, ease: "easeOut" }}
            style={{ filter: `drop-shadow(0 0 6px ${col.stroke})` }}
          />
          {/* Center text */}
          <text x="70" y="68" textAnchor="middle" fill="white" fontSize="18" fontFamily="JetBrains Mono" fontWeight="600">
            {(confidence * 100).toFixed(1)}%
          </text>
        </svg>
        <span
          className="text-xs tracking-widest px-2 py-0.5 rounded mt-1"
          style={{ background: `rgba(${col.stroke === "#00FF88" ? "0,255,136" : col.stroke === "#FFB800" ? "255,184,0" : "255,80,80"},0.12)`, color: col.text }}
        >
          {col.label} CONFIDENCE
        </span>
      </div>

      {/* Class name */}
      <h2 className="text-xl font-semibold text-center text-white tracking-wide mb-4">{topClass}</h2>

      {/* Confidence bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-white/40 tracking-widest uppercase">
          <span>CONFIDENCE</span>
          <span style={{ color: col.text }}>{(confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
          <motion.div
            className="h-full rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ duration: 1.2, ease: "easeOut" }}
            style={{ background: col.stroke, boxShadow: `0 0 8px ${col.stroke}` }}
          />
        </div>
      </div>
    </motion.div>
  );
}