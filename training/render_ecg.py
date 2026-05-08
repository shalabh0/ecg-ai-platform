# training/render_ecg.py
# Standard 12-lead ECG renderer — fixed 4×3 grid + rhythm strip
# Layout: I aVR V1 V4 / II aVL V2 V5 / III aVF V3 V6 / II (rhythm)

import os
import sys
import ast
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import wfdb

# ─── Constants ────────────────────────────────────────────────────────────────

# Fixed 4-column × 3-row layout — columns left→right, rows top→bottom
LEAD_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II"

# PTB-XL signal array order (fixed — do not reorder)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4",  "V5",  "V6"]
LEAD_INDEX = {name: i for i, name in enumerate(LEAD_NAMES)}

# ── Paper & colour ────────────────────────────────────────────────────────────
PAPER_COLOR  = "#FFFFFF"
GRID_MAJOR   = "#FFAAAA"
GRID_MINOR   = "#FFDDDD"
SIGNAL_COLOR = "#000000"
SIGNAL_LW    = 0.65

# ── Figure geometry ───────────────────────────────────────────────────────────
FIG_W   = 11.0      # inches
FIG_H   =  8.5      # inches
DPI     = 100       # → 1100 × 850 px output

# ── Timing ────────────────────────────────────────────────────────────────────
STRIP_SEC   = 2.5   # seconds per lead strip  (4 × 2.5 = 10 s total)
RHYTHM_SEC  = 10.0  # full 10-second rhythm strip

# ── Grid tick intervals (standard clinical ECG paper) ─────────────────────────
MAJOR_T = 0.20      # 5 mm = 0.2 s
MINOR_T = 0.04      # 1 mm = 0.04 s
MAJOR_V = 0.50      # 5 mm = 0.5 mV
MINOR_V = 0.10      # 1 mm = 0.1 mV

# ── Y-axis fixed window: ±1.8 mV covers >99% of PTB-XL leads cleanly ─────────
Y_MIN = -1.8
Y_MAX =  1.8


# ─── Header ───────────────────────────────────────────────────────────────────

def _draw_header(ax, record_id: str, label: str, fs: int) -> None:
    """Single text-only row: record ID | paper settings | label badge."""
    ax.set_facecolor(PAPER_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    label_color = "#006600" if label.lower() == "normal" else "#CC0000"

    ax.text(0.01, 0.50, f"Record: {record_id}",
            transform=ax.transAxes, fontsize=7.5,
            va="center", ha="left", family="monospace", color="#222222")

    ax.text(0.50, 0.50,
            f"25 mm/s    10 mm/mV    {fs} Hz    Leads: 12",
            transform=ax.transAxes, fontsize=7.5,
            va="center", ha="center", family="monospace", color="#444444")

    ax.text(0.99, 0.50, f"◉  {label.upper()}",
            transform=ax.transAxes, fontsize=7.5,
            va="center", ha="right", family="monospace",
            color=label_color, fontweight="bold")


# ─── Single lead strip ────────────────────────────────────────────────────────

def _draw_strip(ax, signal: np.ndarray, fs: int,
                lead_name: str, duration_s: float,
                is_rhythm: bool = False) -> None:
    """
    Render one ECG strip into ax.

    Parameters
    ----------
    signal     : 1-D numpy array, length = int(duration_s * fs)
    fs         : sampling rate in Hz
    lead_name  : string shown in top-left corner
    duration_s : x-axis span in seconds
    is_rhythm  : if True, draw a faint separator line on top spine
    """
    n = int(duration_s * fs)
    sig = signal[:n] if len(signal) >= n else np.pad(signal, (0, n - len(signal)))
    t   = np.arange(n) / fs

    # ── Background ──────────────────────────────────────────────────────────
    ax.set_facecolor(PAPER_COLOR)

    # ── Grid ────────────────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(ticker.MultipleLocator(MAJOR_T))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(MINOR_T))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(MAJOR_V))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(MINOR_V))

    ax.grid(which="major", color=GRID_MAJOR, linewidth=0.55, zorder=1)
    ax.grid(which="minor", color=GRID_MINOR, linewidth=0.25, zorder=1)
    ax.set_axisbelow(True)

    # ── Calibration pulse: 1 mV square, 40 ms wide, before t=0 ─────────────
    # Sits in the left margin; doesn't touch the signal window
    cal_t  = [-0.05, -0.05, -0.01, -0.01]
    cal_mv = [-0.5,   0.5,   0.5,  -0.5]
    ax.plot(cal_t, cal_mv,
            color=SIGNAL_COLOR, linewidth=SIGNAL_LW,
            solid_capstyle="butt", zorder=3)

    # ── ECG signal ──────────────────────────────────────────────────────────
    ax.plot(t, sig,
            color=SIGNAL_COLOR, linewidth=SIGNAL_LW,
            antialiased=True, zorder=3)

    # ── Axis limits ─────────────────────────────────────────────────────────
    ax.set_xlim(-0.06, duration_s)   # -0.06 gives space for cal pulse
    ax.set_ylim(Y_MIN, Y_MAX)

    # ── Lead label ──────────────────────────────────────────────────────────
    ax.text(0.01, 0.94, lead_name,
            transform=ax.transAxes,
            fontsize=6 if not is_rhythm else 7,
            va="top", ha="left",
            family="monospace", fontweight="bold", color="#000000",
            zorder=4,
            bbox=dict(boxstyle="square,pad=0.15",
                      fc=PAPER_COLOR, ec="none", alpha=0.85))

    # ── Cosmetics ───────────────────────────────────────────────────────────
    for sp in ax.spines.values():
        sp.set_visible(False)

    if is_rhythm:                       # faint top border separates rhythm row
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_color("#CCCCCC")
        ax.spines["top"].set_linewidth(0.6)

    ax.tick_params(which="both", length=0,
                   labelbottom=False, labelleft=False)


# ─── Master render function ───────────────────────────────────────────────────

def render_ecg(record_path: str, output_path: str,
               label: str = "", patient_id: str = "") -> bool:
    """
    Load a WFDB record and render a standard 12-lead ECG PNG.

    Grid layout (rows top → bottom):
        Row 0  │ Header bar  (spans all 4 columns)
        Row 1  │ I    │ aVR │ V1 │ V4
        Row 2  │ II   │ aVL │ V2 │ V5
        Row 3  │ III  │ aVF │ V3 │ V6
        Row 4  │ II rhythm strip (spans all 4 columns)

    Returns True on success, False on failure.
    """
    try:
        # ── 1. Load signal ─────────────────────────────────────────────────
        record = wfdb.rdrecord(record_path)
        sig    = record.p_signal          # (samples, 12)
        fs     = record.fs

        if sig is None or sig.ndim != 2 or sig.shape[1] < 12:
            print(f"[WARN] {record_path}: bad signal shape {getattr(sig,'shape',None)}")
            return False

        total_needed = int(RHYTHM_SEC * fs)
        sig = sig[:total_needed, :]       # cap at 10 s; pad handled in _draw_strip

        pid = patient_id or Path(record_path).stem

        # ── 2. Build figure ────────────────────────────────────────────────
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=PAPER_COLOR)

        # Fixed row height ratios:
        #   header=0.05 | 3 lead rows × 0.22 | rhythm=0.29
        gs = gridspec.GridSpec(
            nrows=5, ncols=4,
            figure=fig,
            height_ratios=[0.05, 0.22, 0.22, 0.22, 0.29],
            hspace=0.0,
            wspace=0.0,
            left=0.01,
            right=0.99,
            top=0.99,
            bottom=0.01,
        )

        # ── 3. Header (row 0, all columns) ─────────────────────────────────
        ax_hdr = fig.add_subplot(gs[0, :])
        _draw_header(ax_hdr, pid, label, fs)

        # ── 4. 12-lead rows (rows 1–3) ─────────────────────────────────────
        for row_i, lead_row in enumerate(LEAD_LAYOUT):          # 0,1,2 → gs row 1,2,3
            for col_i, lead_name in enumerate(lead_row):        # 0,1,2,3
                ax = fig.add_subplot(gs[row_i + 1, col_i])
                lead_sig = sig[:, LEAD_INDEX[lead_name]]
                _draw_strip(ax, lead_sig, fs, lead_name, STRIP_SEC)

        # ── 5. Rhythm strip (row 4, all columns) ───────────────────────────
        ax_rhy = fig.add_subplot(gs[4, :])
        rhythm_sig = sig[:, LEAD_INDEX[RHYTHM_LEAD]]
        _draw_strip(ax_rhy, rhythm_sig, fs, f"{RHYTHM_LEAD} — Rhythm Strip",
                    RHYTHM_SEC, is_rhythm=True)

        # ── 6. Save ────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=DPI,
                    bbox_inches="tight", pad_inches=0.05,
                    facecolor=PAPER_COLOR)
        plt.close(fig)
        return True

    except Exception as exc:
        print(f"[ERROR] {record_path}: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        plt.close("all")
        return False


# ─── Batch rendering ──────────────────────────────────────────────────────────

def render_dataset(ptbxl_dir: str, output_base: str,
                   max_per_class: int = 1000, use_lr: bool = True) -> dict:
    """
    Render balanced normal / abnormal PNGs from ptbxl_database.csv.
    Skips already-rendered files (resumable).
    """
    csv_path = os.path.join(ptbxl_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found in {ptbxl_dir}")

    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)   # safe Python repr parse

    # Signal quality filter (PTB-XL quality fields: 0=good, 1=noisy, 2=very noisy)
    if "baseline_drift" in df.columns and "static_noise" in df.columns:
     before = len(df)
     df["baseline_drift"] = pd.to_numeric(df["baseline_drift"], errors="coerce")
     df["static_noise"]   = pd.to_numeric(df["static_noise"],   errors="coerce")
     df = df.dropna(subset=["baseline_drift", "static_noise"])
     df = df[(df["baseline_drift"] <= 1) & (df["static_noise"] <= 1)]
     print(f"[quality] Dropped {before - len(df)} low-quality records. {len(df)} remain.")

    def _label(scp: dict) -> str:
        return "normal" if scp.get("NORM", 0) >= 50.0 else "abnormal"

    df["label"] = df["scp_codes"].apply(_label)

    normal_df   = df[df["label"] == "normal"].sample(
                      min(max_per_class, df["label"].eq("normal").sum()),
                      random_state=42)
    abnormal_df = df[df["label"] == "abnormal"].sample(
                      min(max_per_class, df["label"].eq("abnormal").sum()),
                      random_state=42)
    subset = pd.concat([normal_df, abnormal_df]).sample(frac=1, random_state=42)

    print(f"[batch] Rendering {len(subset)} records "
          f"({len(normal_df)} normal | {len(abnormal_df)} abnormal)…")

    fn_col = "filename_lr" if use_lr else "filename_hr"
    counts = {"normal": 0, "abnormal": 0, "failed": 0}

    for ecg_id, row in subset.iterrows():
        label    = row["label"]
        out_path = os.path.join(output_base, label, f"{ecg_id:05d}.png")

        if os.path.exists(out_path):                            # resumable
            counts[label] += 1
            continue

        # filename_lr stores e.g. "records100/00000/00001_lr"
        rec_rel  = row[fn_col]
        rec_path = os.path.join(ptbxl_dir, rec_rel)

        ok = render_ecg(rec_path, out_path,
                        label=label, patient_id=str(ecg_id))

        if ok:
            counts[label] += 1
            done = counts["normal"] + counts["abnormal"]
            if done % 100 == 0:
                print(f"  [{done}/{len(subset)}] "
                      f"normal={counts['normal']} "
                      f"abnormal={counts['abnormal']} "
                      f"failed={counts['failed']}")
        else:
            counts["failed"] += 1

    print(f"\n[done] {counts}")
    return counts


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser(
        description="Render PTB-XL WFDB records as 12-lead ECG PNGs")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("single", help="Render one record")
    s.add_argument("--record", required=True,
                   help="WFDB record path (no extension)")
    s.add_argument("--out",    required=True, help="Output PNG path")
    s.add_argument("--label",  default="",    help="normal | abnormal")

    b = sub.add_parser("batch", help="Render full dataset")
    b.add_argument("--ptbxl-dir",     default="datasets/ptbxl_raw")
    b.add_argument("--output-dir",    default="datasets/rendered")
    b.add_argument("--max-per-class", type=int, default=1000)
    b.add_argument("--hires",         action="store_true",
                   help="Use 500 Hz records")

    return p.parse_args()


if __name__ == "__main__":
    args = _args()

    if args.cmd == "single":
        ok = render_ecg(args.record, args.out, label=args.label)
        print(f"{'Saved' if ok else 'FAILED'}: {args.out}")
        sys.exit(0 if ok else 1)

    elif args.cmd == "batch":
        counts = render_dataset(
            ptbxl_dir=args.ptbxl_dir,
            output_base=args.output_dir,
            max_per_class=args.max_per_class,
            use_lr=not args.hires,
        )
        sys.exit(0 if counts["failed"] == 0 else 1)

    else:
        print("Usage:\n"
              "  python render_ecg.py single --record <path> --out <file.png> --label normal\n"
              "  python render_ecg.py batch  --ptbxl-dir <dir> --output-dir <dir>")
        sys.exit(1)