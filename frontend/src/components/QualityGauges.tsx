import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { QualityReport } from "../types/ecg";

interface GaugeProps {
  label: string;
  value: number;       // 0-1 normalized
  display: string;
  color: string;
  delay?: number;
}

function HalfGauge({ label, value, display, color, delay = 0 }: GaugeProps) {
  const [anim, setAnim] = useState(0);
  const r = 40;
  const circ = Math.PI * r;

  useEffect(() => {
    const t = setTimeout(() => setAnim(value), 200 + delay * 1000);
    return () => clearTimeout(t);
  }, [value, delay]);

  // Needle dot position on arc
  const angle = Math.PI * anim; // 0 = left, PI = right
  const nx = 55 - r * Math.cos(angle);
  const ny = 55 - r * Math.sin(angle);

  return (
    <div className="flex flex-col items-center">
      <svg width="110" height="62" viewBox="0 0 110 62">
        <path d="M 15 55 A 40 40 0 0 1 95 55" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" strokeLinecap="round" />
        <motion.path
          d="M 15 55 A 40 40 0 0 1 95 55"
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${circ} ${circ}`}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: circ - circ * anim }}
          transition={{ duration: 1.0, delay, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 4px ${color})` }}
        />
        {/* Needle dot */}
        <motion.circle
          cx={nx}
          cy={ny}
          r="3.5"
          fill={color}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay + 0.9 }}
        />
        <text x="55" y="52" textAnchor="middle" fill="white" fontSize="11" fontFamily="JetBrains Mono" fontWeight="500">
          {display}
        </text>
      </svg>
      <span className="text-white/40 text-xs tracking-widest uppercase mt-1">{label}</span>
    </div>
  );
}

interface Props {
  report: QualityReport;
}

function gaugeColor(norm: number) {
  if (norm >= 0.7) return "#00FF88";
  if (norm >= 0.4) return "#FFB800";
  return "#FF5050";
}

export default function QualityGauges({ report }: Props) {
  const scoreNorm = report.score;
  const snrNorm = Math.min(report.snr_db / 30, 1);
  const blurNorm = Math.min(report.blur_score / 100000, 1);

  return (
    <div>
      <h3 className="text-white/50 text-xs tracking-widest uppercase mb-4">SIGNAL QUALITY</h3>
      <div className="grid grid-cols-3 gap-2">
        <HalfGauge label="QUALITY" value={scoreNorm} display={report.score.toFixed(2)} color={gaugeColor(scoreNorm)} delay={0} />
        <HalfGauge label="SNR" value={snrNorm} display={`${report.snr_db.toFixed(1)}dB`} color={gaugeColor(snrNorm)} delay={0.15} />
        <HalfGauge label="SHARPNESS" value={blurNorm} display={`${(report.blur_score / 1000).toFixed(0)}k`} color={gaugeColor(blurNorm)} delay={0.3} />
      </div>
      <p className="text-center text-xs mt-2 tracking-wider" style={{ color: report.acceptable ? "#00FF88" : "#FF5050" }}>
        {report.reason}
      </p>
    </div>
  );
}