import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import Landing from "./pages/Landing";
import Analyzer from "./pages/Analyzer";
import Batch from "./pages/Batch";

const qc = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Toaster
          theme="dark"
          toastOptions={{
            style: { background: "#0A0F1E", border: "1px solid rgba(255,255,255,0.08)", color: "white", fontFamily: "JetBrains Mono" },
          }}
        />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/analyze" element={<Analyzer />} />
          <Route path="/batch" element={<Batch />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}