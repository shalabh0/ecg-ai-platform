import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  summary: string;
}

export default function ClinicalSummary({ summary }: Props) {
  const [open, setOpen] = useState(true);

  return (
    <div className="rounded-lg border border-amber-500/20 overflow-hidden" style={{ background: "rgba(10,15,30,0.98)" }}>
      {/* Disclaimer banner */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-amber-500/15" style={{ background: "rgba(255,184,0,0.07)" }}>
        <span className="text-amber-400 text-sm">⚠</span>
        <span className="text-amber-400/90 text-xs tracking-widest uppercase font-medium flex-1">
          AI-ASSISTED · NOT A CLINICAL DIAGNOSIS
        </span>
        <button
          onClick={() => setOpen(!open)}
          className="text-white/30 hover:text-white/60 text-xs tracking-widest transition-colors"
        >
          {open ? "▲ HIDE" : "▼ SHOW"}
        </button>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <div
              className="px-4 py-4 max-h-72 overflow-y-auto text-white/70 text-sm leading-relaxed whitespace-pre-wrap"
              style={{
                scrollbarWidth: "thin",
                scrollbarColor: "rgba(0,212,255,0.2) transparent",
              }}
            >
              {summary}
            </div>

            {/* Scanline footer */}
            <div
              className="h-6 border-t border-white/5"
              style={{
                background: "repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(255,255,255,0.015) 2px)",
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}