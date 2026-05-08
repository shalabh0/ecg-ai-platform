import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import ECGWaveform from "../components/ECGWaveform";

const FEATURES = [
  { icon: "◈", title: "6-STAGE PIPELINE", desc: "Validation → Quality → Perspective → Preprocessing → CNN → Clinical output" },
  { icon: "◉", title: "20-CLASS DETECTION", desc: "NSR, AF, Flutter, LBBB, RBBB, ST changes, AV blocks and more" },
  { icon: "✦", title: "CLINICAL NOTES", desc: "Structured AI-assisted summaries with confidence grading and disclaimers" },
];

export default function Landing() {
  const nav = useNavigate();

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#0A0F1E" }}>
      {/* ECG paper grid */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            repeating-linear-gradient(0deg, transparent, transparent 99px, rgba(0,212,255,0.06) 100px),
            repeating-linear-gradient(90deg, transparent, transparent 99px, rgba(0,212,255,0.06) 100px),
            repeating-linear-gradient(0deg, transparent, transparent 19px, rgba(0,212,255,0.02) 20px),
            repeating-linear-gradient(90deg, transparent, transparent 19px, rgba(0,212,255,0.02) 20px)
          `,
        }}
      />

      {/* Nav */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-5 border-b border-white/5">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-lg">♥</span>
          <span className="text-white font-semibold tracking-wider text-sm">ECG·AI</span>
        </div>
        <div className="flex gap-6 text-xs text-white/40 tracking-widest uppercase">
          <button onClick={() => nav("/analyze")} className="hover:text-cyan-400 transition-colors">ANALYZE</button>
          <button onClick={() => nav("/batch")} className="hover:text-cyan-400 transition-colors">BATCH</button>
        </div>
      </nav>

      {/* Hero */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 py-24 text-center">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-2xl w-full"
        >
          <div className="mb-4 text-cyan-400/60 text-xs tracking-[0.3em] uppercase">
            MOBILE · ECG · AI · PLATFORM
          </div>
          <h1 className="text-4xl md:text-5xl font-semibold text-white mb-4 leading-tight tracking-tight">
            AI-Assisted<br />
            <span style={{ color: "#00D4FF" }}>ECG Analysis</span>
          </h1>
          <p className="text-white/40 text-sm tracking-wider mb-10">
            Upload. Analyze. Assist.
          </p>

          {/* Animated waveform */}
          <div className="mb-10 overflow-hidden rounded" style={{ height: 80 }}>
            <ECGWaveform width={800} height={80} />
          </div>

          <div className="flex gap-4 justify-center">
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => nav("/analyze")}
              className="px-8 py-3 text-sm tracking-widest uppercase font-medium text-black rounded"
              style={{ background: "linear-gradient(135deg, #0066FF, #00D4FF)", boxShadow: "0 0 24px rgba(0,212,255,0.3)" }}
            >
              ANALYZE ECG →
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => nav("/batch")}
              className="px-8 py-3 text-sm tracking-widest uppercase border border-white/15 text-white/70 rounded hover:border-cyan-400/40 hover:text-white transition-colors"
            >
              BATCH MODE
            </motion.button>
          </div>
        </motion.div>

        {/* Feature cards */}
        <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl w-full">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.1 }}
              className="rounded-lg p-5 border border-white/5 text-left"
              style={{ background: "rgba(255,255,255,0.02)" }}
            >
              <div className="text-cyan-400 text-xl mb-3">{f.icon}</div>
              <h3 className="text-white/80 text-xs tracking-widest uppercase mb-2">{f.title}</h3>
              <p className="text-white/35 text-xs leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </main>

      <footer className="relative z-10 text-center py-4 text-white/15 text-xs tracking-widest border-t border-white/5">
        NOT FOR CLINICAL USE · AI-ASSISTED ONLY · ALWAYS CONFIRM WITH A QUALIFIED CLINICIAN
      </footer>
    </div>
  );
}