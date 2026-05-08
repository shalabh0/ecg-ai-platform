import { motion } from "framer-motion";
import { useAnalysisStore } from "../store/analysisStore";

export default function BatchTable() {
  const items = useAnalysisStore((s) => s.batchItems);

  const exportCSV = () => {
    const rows = [
      ["Filename", "Top Class", "Confidence", "Quality Score", "Status"],
      ...items.map((i) => [
        i.file.name,
        i.result?.top_class ?? "-",
        i.result ? (i.result.confidence * 100).toFixed(1) + "%" : "-",
        i.result ? i.result.quality_report.score.toFixed(3) : "-",
        i.status,
      ]),
    ];
    const csv = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ecg-batch-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!items.length) return null;

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <h3 className="text-white/50 text-xs tracking-widest uppercase">BATCH RESULTS — {items.length} FILES</h3>
        <button
          onClick={exportCSV}
          className="text-xs px-3 py-1.5 border border-cyan-400/30 text-cyan-400 rounded hover:bg-cyan-400/10 tracking-widest uppercase transition-colors"
        >
          ↓ EXPORT CSV
        </button>
      </div>

      <div className="border border-white/5 rounded-lg overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-white/5" style={{ background: "rgba(255,255,255,0.03)" }}>
              {["FILE", "TOP CLASS", "CONFIDENCE", "QUALITY", "STATUS"].map((h) => (
                <th key={h} className="px-3 py-2 text-left text-white/30 tracking-widest font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((item, i) => (
              <motion.tr
                key={item.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.04 }}
                className="border-b border-white/5 hover:bg-white/2 transition-colors"
              >
                <td className="px-3 py-2.5 text-white/60 max-w-[140px] truncate">{item.file.name}</td>
                <td className="px-3 py-2.5 text-white/80">{item.result?.top_class ?? "—"}</td>
                <td className="px-3 py-2.5 text-cyan-400">
                  {item.result ? `${(item.result.confidence * 100).toFixed(1)}%` : "—"}
                </td>
                <td className="px-3 py-2.5 text-white/50">
                  {item.result ? item.result.quality_report.score.toFixed(3) : "—"}
                </td>
                <td className="px-3 py-2.5">
                  {item.status === "processing" && (
                    <span className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping inline-block" />
                      <span className="text-cyan-400/70 tracking-widest">RUNNING</span>
                    </span>
                  )}
                  {item.status === "done" && <span className="text-green-400 tracking-widest">✓ DONE</span>}
                  {item.status === "error" && <span className="text-red-400 tracking-widest">✕ ERROR</span>}
                  {item.status === "pending" && <span className="text-white/30 tracking-widest">PENDING</span>}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}