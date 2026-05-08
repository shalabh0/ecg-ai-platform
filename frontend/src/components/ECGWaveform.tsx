import { useEffect, useRef } from "react";

interface Props {
  width?: number;
  height?: number;
  color?: string;
  speed?: number;
}

export default function ECGWaveform({
  width = 1200,
  height = 80,
  color = "#00D4FF",
  speed = 1.2,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offsetRef = useRef(0);
  const rafRef = useRef<number>(0);

  // Genuine P-QRS-T morphology points (normalized 0-1 x, -1 to 1 y)
  const PQRST: [number, number][] = [
    [0, 0], [0.05, 0], [0.08, 0.15], [0.12, 0.15], [0.15, 0],
    [0.18, 0], [0.21, -0.08], [0.24, 0.9], [0.26, -0.3], [0.29, 0],
    [0.32, 0], [0.36, 0.25], [0.44, 0.25], [0.48, 0], [0.52, 0],
    [1.0, 0],
  ];

  function buildPath(offset: number, w: number, h: number): Path2D {
    const path = new Path2D();
    const mid = h / 2;
    const amp = h * 0.38;
    const beatW = w * 0.35;
    const totalBeats = Math.ceil(w / beatW) + 2;

    for (let b = -1; b < totalBeats; b++) {
      const bx = b * beatW - (offset % beatW);
      PQRST.forEach(([nx, ny], i) => {
        const px = bx + nx * beatW;
        const py = mid - ny * amp;
        if (b === -1 && i === 0) path.moveTo(px, py);
        else path.lineTo(px, py);
      });
    }
    return path;
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    function draw() {
      ctx.clearRect(0, 0, width, height);

      // Glow layer
      ctx.save();
      ctx.filter = "blur(4px)";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = 0.45;
      ctx.stroke(buildPath(offsetRef.current, width, height));
      ctx.restore();

      // Crisp layer with fade mask
      const grad = ctx.createLinearGradient(0, 0, width, 0);
      grad.addColorStop(0, "transparent");
      grad.addColorStop(0.08, color);
      grad.addColorStop(0.92, color);
      grad.addColorStop(1, "transparent");

      ctx.save();
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 1;
      ctx.stroke(buildPath(offsetRef.current, width, height));
      ctx.restore();

      offsetRef.current += speed;
      rafRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [width, height, color, speed]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="w-full"
      style={{ imageRendering: "crisp-edges" }}
    />
  );
}