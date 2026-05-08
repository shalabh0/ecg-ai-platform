import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useState } from "react";
import { toast } from "sonner";
import UploadZone from "../components/UploadZone";
import PipelineProgress from "../components/PipelineProgress";
import ResultCard from "../components/ResultCard";
import ProbabilityChart from "../components/ProbabilityChart";
import QualityGauges from "../components/QualityGauges";
import ClinicalSummary from "../components/ClinicalSummary";
import { analyzeECG } from "../api/ecgApi";
import { useAnalysisStore } from "../store/analysisStore";
import { PipelineStage } from "../types/ecg";

export default function Analyzer() {
  const nav = useNavigate();
  const { file, result, loading, setResult, setLoading, setError } = useAnalysisStore();
  const [stages, setStages] = useState<PipelineStage[]>([]);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setStages([]);
    setError(null);
    try {
      const res = await analyzeECG(file);
      // Animate stages in one by one
      for (let i = 0; i < res.pipeline_stages.length; i++) {
        await new Promise((r) => setTimeout(r, 120));
        setStages((prev) => [...prev, res.pipeline_stages[i]]);
      }
      setResult(res);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Analysis failed";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen" style={{ background: "#0A0F1E" }}>
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            repeating-linear-gradient(0deg, transparent, transparent 99px, rgba(0,212,255,0.05) 100px),
            repeating-linear-gradient(90deg, transparent, transparent 99px, rgba(0,212,255,0.05) 100px),
            repeating-linear-gradient(0deg, transparent, transparent 19px, rgba(0,212,255,0.018) 20px),
            repeating-linear-gradient(90deg, transparent, transparent 19px, rgba(0,212,255,0.018) 20px)
          `,
        }}
      />

      <nav className="relative z-10 flex items-center justify-between px-8 py-5 border-b border-white/5">
        <button onClick={() => nav("/")} className="flex items-center gap-2 text-white/50 hover:text-white transition-colors">
          <span className="text-cyan-400">♥</span>
          <span className="text-sm tracking-wider">ECG·AI</span>
        </button>
        <span className="text-white/30 text-xs tracking-widest uppercase">SINGLE ANALYSIS</span>
        <button onClick={() => nav("/batch")} className="text-xs text-white/30 hover:text-cyan-400 tracking-widest uppercase transition-colors">
          BATCH →
        </button>
      </nav>

      <div className="relative z-10 max-w-5xl mx-auto px-4 py-10 grid md:grid-cols-2 gap-8">
        {/* Left panel */}
        <div className="space-y-5">
          <h2 className="text-white/60 text-xs tracking-widest uppercase">INPUT</h2>
          <UploadZone onFile={() => { setStages([]); setResult(null as never); }} />

          <motion.button
            whileHover={{ scale: file && !loading ? 1.02 : 1 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="w-full py-3 text-sm tracking-widest uppercase font-medium rounded transition-all"
            style={{
              background: file && !loading ? "linear-gradient(135deg,#0066FF,#00D4FF)" : "rgba(255,255,255,0.05)",
              color: file && !loading ? "#000" : "rgba(255,255,255,0.2)",
              cursor: file && !loading ? "pointer" : "not-allowed",
              boxShadow: file && !loading ? "0 0 20px rgba(0,212,255,0.25)" : "none",
            }}
          >
            {loading ? "ANALYZING…" : "ANALYZE ECG"}
          </motion.button>

          {/* Pipeline */}
          {(stages.length > 0 || loading) && (
            <div className="border border-white/5 rounded-lg p-4" style={{ background: "rgba(255,255,255,0.015)" }}>
              <h3 className="text-white/40 text-xs tracking-widest uppercase mb-4">PIPELINE</h3>
              <PipelineProgress stages={stages} loading={loading} />
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="space-y-5">
          {result ? (
            <>
              <h2 className="text-white/60 text-xs tracking-widest uppercase">RESULTS</h2>
              <ResultCard topClass={result.top_class} confidence={result.confidence} />

              <div className="border border-white/5 rounded-lg p-4" style={{ background: "rgba(255,255,255,0.015)" }}>
                <ProbabilityChart probabilities={result.class_probabilities} />
              </div>

              <div className="border border-white/5 rounded-lg p-4" style={{ background: "rgba(255,255,255,0.015)" }}>
                <QualityGauges report={result.quality_report} />
              </div>

              <ClinicalSummary summary={result.clinical_summary} />
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-center">
              <div className="text-white/10 text-5xl mb-4">◈</div>
              <p className="text-white/20 text-xs tracking-widest uppercase">RESULTS APPEAR HERE</p>
              <p className="text-white/10 text-xs mt-1">Upload an ECG image and click ANALYZE</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}