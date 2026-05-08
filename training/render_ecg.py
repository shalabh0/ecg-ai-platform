"""
training/render_ecg.py
======================
PTB-XL waveform → realistic 12-lead ECG PNG

MVP v1 — no OpenCV, no augmentation, no fancy tricks.
Get this rendering right before anything else.

Output directories
------------------
datasets/rendered/normal/
datasets/rendered/abnormal/

Usage
-----
# Render a single record (test it works)
python training/render_ecg.py --record datasets/ptbxl_raw/records100/00000/00001_hr --label normal --out datasets/rendered/normal/00001.png

# Render entire dataset (after this works)
python training/render_ecg.py --all
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, must be before pyplot import

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import wfdb

# ─── Layout constants ─────────────────────────────────────────────────────────

# Standard 12-lead layout: 4 columns × 3 rows + 1 rhythm strip
LEAD_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II"

# PTB-XL lead order (index in the signal array)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_INDEX = {name: i for i, name in enumerate(LEAD_NAMES)}

# Paper settings (clinical ECG standard)
MM_PER_SEC   = 25    # 25 mm/s paper speed
MM_PER_MV    = 10    # 10 mm/mV gain
PAPER_COLOR  = "#FFFFFF"
GRID_MAJOR   = "#FFAAAA"   # red ECG grid (major)
GRID_MINOR   = "#FFCCCC"   # red ECG grid (minor)
SIGNAL_COLOR = "#000000"
SIGNAL_WIDTH = 0.6

# Figure size in inches at 100 DPI → 1100×850 px
FIG_WIDTH_IN  = 11.0
FIG_HEIGHT_IN = 8.5
DPI = 100

# How many seconds to show per lead strip
SECONDS_PER_STRIP = 2.5    # 4 columns × 2.5 s = 10 s total (standard ECG)
RHYTHM_SECONDS    = 10.0   # full 10 s for rhythm strip

# Sampling rate of PTB-XL low-res records (100 Hz) vs high-res (500 Hz)
# We use 100 Hz records to keep rendering fast
FS = 100   # samples/second for _lr records; use 500 for _hr

# ─── Rendering ────────────────────────────────────────────────────────────────

def render_ecg(
    record_path: str,
    output_path: str,
    label: str = "",
    patient_id: str = "",
    fs: int = FS,
) -> bool:
    """
    Render a WFDB ECG record as a clinical-style 12-lead PNG.

    Parameters
    ----------
    record_path : str
        Path to WFDB record WITHOUT extension (e.g. "datasets/ptbxl_raw/records100/00001_lr")
    output_path : str
        Where to save the PNG
    label : str
        "normal" or "abnormal" — printed on image header
    patient_id : str
        Printed in header. Defaults to basename of record_path.
    fs : int
        Sampling rate (100 for _lr records, 500 for _hr records)

    Returns
    -------
    bool
        True on success, False on failure
    """
    try:
        # ── Load signal ────────────────────────────────────────────────────
        samples_needed = int(RHYTHM_SECONDS * fs)
        record = wfdb.rdrecord(record_path, sampto=samples_needed)
        sig = record.p_signal          # shape: (samples, 12)
        actual_fs = record.fs

        if sig is None or sig.shape[1] < 12:
            print(f"[WARN] Skipping {record_path}: insufficient leads ({sig.shape if sig is not None else 'None'})")
            return False

        # Use actual sampling rate from record
        fs = actual_fs
        samples_needed = min(int(RHYTHM_SECONDS * fs), sig.shape[0])
        sig = sig[:samples_needed, :]

        # ── Figure setup ───────────────────────────────────────────────────
        fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=DPI)
        fig.patch.set_facecolor(PAPER_COLOR)

        # Grid: 3 rows of leads + 1 rhythm strip + 1 title row
        # Proportions: title=0.06, each lead row=0.22, rhythm=0.24
        row_heights = [0.06, 0.22, 0.22, 0.22, 0.24]
        gs = fig.add_gridspec(
            5, 4,
            height_ratios=row_heights,
            hspace=0.0,
            wspace=0.0,
            left=0.04, right=0.98,
            top=0.97, bottom=0.02,
        )

        # ── Header row ─────────────────────────────────────────────────────
        header_ax = fig.add_subplot(gs[0, :])
        _render_header(header_ax, patient_id or Path(record_path).stem, label, fs)

        # ── 12-lead rows ────────────────────────────────────────────────────
        strip_samples = int(SECONDS_PER_STRIP * fs)
        for row_idx, lead_row in enumerate(LEAD_LAYOUT):
            for col_idx, lead_name in enumerate(lead_row):
                ax = fig.add_subplot(gs[row_idx + 1, col_idx])
                lead_idx = LEAD_INDEX[lead_name]
                lead_sig = sig[:strip_samples, lead_idx]
                _render_lead_strip(ax, lead_sig, lead_name, fs, SECONDS_PER_STRIP)

        # ── Rhythm strip (full 10 s, Lead II) ──────────────────────────────
        rhythm_ax = fig.add_subplot(gs[4, :])
        lead_idx  = LEAD_INDEX[RHYTHM_LEAD]
        rhythm_sig = sig[:, lead_idx]
        _render_lead_strip(
            rhythm_ax, rhythm_sig, f"{RHYTHM_LEAD} (rhythm)",
            fs, RHYTHM_SECONDS, is_rhythm=True
        )

        # ── Save ───────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0,
                    facecolor=PAPER_COLOR)
        plt.close(fig)
        return True

    except Exception as exc:
        print(f"[ERROR] render_ecg failed for {record_path}: {exc}", file=sys.stderr)
        plt.close("all")
        return False


def _render_header(ax, patient_id: str, label: str, fs: int) -> None:
    """Render the ECG paper header bar (no spines, just text)."""
    ax.set_facecolor(PAPER_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    header_left  = f"ID: {patient_id}"
    header_mid   = f"25 mm/s   10 mm/mV   {fs} Hz"
    header_right = f"Label: {label.upper()}" if label else ""

    ax.text(0.01, 0.5, header_left,  transform=ax.transAxes, fontsize=7,
            va="center", ha="left",  family="monospace", color="#333333")
    ax.text(0.50, 0.5, header_mid,   transform=ax.transAxes, fontsize=7,
            va="center", ha="center",family="monospace", color="#333333")
    ax.text(0.99, 0.5, header_right, transform=ax.transAxes, fontsize=7,
            va="center", ha="right", family="monospace",
            color="#AA0000" if label == "abnormal" else "#006600")


def _render_lead_strip(
    ax,
    signal: np.ndarray,
    lead_name: str,
    fs: int,
    duration_s: float,
    is_rhythm: bool = False,
) -> None:
    """
    Render a single ECG lead strip onto an axis.

    Grid: 1 large square = 5 mm × 5 mm = 0.2 s × 0.5 mV
          1 small square = 1 mm × 1 mm = 0.04 s × 0.1 mV
    """
    ax.set_facecolor(PAPER_COLOR)

    # Time axis
    t = np.arange(len(signal)) / fs

    # ── Grid ────────────────────────────────────────────────────────────────
    # Minor grid: 1 mm = 0.04 s horizontal, 0.1 mV vertical
    # Major grid: 5 mm = 0.2 s horizontal, 0.5 mV vertical
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    ax.grid(which="minor", color=GRID_MINOR, linewidth=0.3, alpha=0.8)
    ax.grid(which="major", color=GRID_MAJOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)

    # ── Signal ──────────────────────────────────────────────────────────────
    ax.plot(t, signal, color=SIGNAL_COLOR, linewidth=SIGNAL_WIDTH,
            antialiased=True, rasterized=False)

    # ── Axis limits ─────────────────────────────────────────────────────────
    ax.set_xlim(0, duration_s)
    ax.set_ylim(-2.0, 2.0)    # ±2 mV covers most ECG amplitudes

    # ── Lead label ──────────────────────────────────────────────────────────
    ax.text(0.01, 0.90, lead_name, transform=ax.transAxes,
            fontsize=6 if not is_rhythm else 7,
            va="top", ha="left", family="monospace",
            color="#000000", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.1", fc=PAPER_COLOR, ec="none", alpha=0.8))

    # ── Calibration pulse (1 mV square wave at start) ───────────────────────
    # Standard ECG shows a 1 mV cal pulse at the beginning
    cal_width = 0.04   # 40 ms wide
    ax.plot([0, 0, cal_width, cal_width], [-0.5, 0.5, 0.5, -0.5],
            color=SIGNAL_COLOR, linewidth=SIGNAL_WIDTH, antialiased=True)

    # ── Spines: only bottom for non-rhythm, all hidden for clean look ────────
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove tick labels — clean clinical look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which="both", length=0)


# ─── Batch rendering ──────────────────────────────────────────────────────────

def render_dataset(
    ptbxl_dir: str,
    output_base: str,
    max_per_class: int = 1000,
    use_lr: bool = True,         # True = 100 Hz, False = 500 Hz
) -> dict:
    """
    Render all PTB-XL records → PNG images.

    Parameters
    ----------
    ptbxl_dir    : root of extracted PTB-XL dataset (contains ptbxl_database.csv)
    output_base  : base output dir (normal/ and abnormal/ created inside)
    max_per_class: cap per class to keep dataset balanced
    use_lr       : use low-res (100 Hz) records — faster rendering

    Returns dict with counts: {"normal": N, "abnormal": M, "failed": K}
    """
    db_path = os.path.join(ptbxl_dir, "ptbxl_database.csv")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found in {ptbxl_dir}")

    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(
        lambda s: json.loads(s.replace("'", '"'))
    )

    # Determine label: NORM in scp_codes with likelihood ≥ 50 → normal
    def get_label(scp_codes: dict) -> str:
        if "NORM" in scp_codes and scp_codes["NORM"] >= 50.0:
            return "normal"
        return "abnormal"

    df["label"] = df["scp_codes"].apply(get_label)

    # Balance: sample equal numbers from each class
    normal_df   = df[df["label"] == "normal"].sample(
        min(max_per_class, (df["label"] == "normal").sum()), random_state=42)
    abnormal_df = df[df["label"] == "abnormal"].sample(
        min(max_per_class, (df["label"] == "abnormal").sum()), random_state=42)
    subset = pd.concat([normal_df, abnormal_df]).sample(frac=1, random_state=42)

    print(f"Rendering {len(subset)} records "
          f"({len(normal_df)} normal, {len(abnormal_df)} abnormal)…")

    os.makedirs(os.path.join(output_base, "normal"),   exist_ok=True)
    os.makedirs(os.path.join(output_base, "abnormal"), exist_ok=True)

    counts = {"normal": 0, "abnormal": 0, "failed": 0}
    field = "filename_lr" if use_lr else "filename_hr"

    for ecg_id, row in subset.iterrows():
        rec_rel  = row[field]                              # e.g. records100/00000/00001_lr
        rec_path = os.path.join(ptbxl_dir, rec_rel)
        label    = row["label"]
        out_name = f"{ecg_id:05d}.png"
        out_path = os.path.join(output_base, label, out_name)

        if os.path.exists(out_path):
            counts[label] += 1
            continue

        ok = render_ecg(
            record_path=rec_path,
            output_path=out_path,
            label=label,
            patient_id=str(ecg_id),
            fs=100 if use_lr else 500,
        )

        if ok:
            counts[label] += 1
            if (counts["normal"] + counts["abnormal"]) % 100 == 0:
                print(f"  Progress: normal={counts['normal']}  "
                      f"abnormal={counts['abnormal']}  "
                      f"failed={counts['failed']}")
        else:
            counts["failed"] += 1

    print(f"\nDone: {counts}")
    return counts


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Render PTB-XL records as ECG PNG images")
    sub = p.add_subparsers(dest="cmd")

    # Single record
    single = sub.add_parser("single", help="Render one record")
    single.add_argument("--record",  required=True, help="WFDB record path (no extension)")
    single.add_argument("--out",     required=True, help="Output PNG path")
    single.add_argument("--label",   default="",    help="normal or abnormal")
    single.add_argument("--fs",      type=int, default=100)

    # Batch
    batch = sub.add_parser("batch", help="Render full PTB-XL dataset")
    batch.add_argument("--ptbxl-dir",   default="datasets/ptbxl_raw",
                       help="Root of extracted PTB-XL dataset")
    batch.add_argument("--output-dir",  default="datasets/rendered",
                       help="Output base directory")
    batch.add_argument("--max-per-class", type=int, default=1000)
    batch.add_argument("--hires",  action="store_true",
                       help="Use 500 Hz records (slower, higher quality)")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.cmd == "single":
        ok = render_ecg(
            record_path=args.record,
            output_path=args.out,
            label=args.label,
            fs=args.fs,
        )
        if ok:
            print(f"Saved: {args.out}")
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.cmd == "batch":
        counts = render_dataset(
            ptbxl_dir=args.ptbxl_dir,
            output_base=args.output_dir,
            max_per_class=args.max_per_class,
            use_lr=not args.hires,
        )
        total = counts["normal"] + counts["abnormal"]
        failed = counts["failed"]
        print(f"\nRendering complete: {total} images saved, {failed} failed.")
        sys.exit(0 if failed == 0 else 1)

    else:
        # No subcommand — show quick test
        print("No command given. Running quick render test on a dummy signal…")

        # Generate a synthetic ECG-like signal for testing without PTB-XL
        import tempfile
        fs_test = 100
        duration = 10
        t = np.linspace(0, duration, duration * fs_test)
        # Synthetic 12-lead: QRS spike + baseline noise
        base = 0.05 * np.sin(2 * np.pi * 0.3 * t)   # slow baseline
        qrs_times = np.arange(0.8, duration, 0.85)
        sig_test = base.copy()
        for qt in qrs_times:
            mask = (t >= qt) & (t < qt + 0.12)
            sig_test[mask] += np.linspace(0, 1.5, mask.sum()) * np.sin(
                np.linspace(0, np.pi, mask.sum()))

        # Create a fake 12-lead array by offsetting copies
        sig_12 = np.column_stack([
            sig_test * (0.8 + 0.1 * i) for i in range(12)
        ])

        # Write to temp WFDB record
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_path = os.path.join(tmpdir, "test_ecg")
            wfdb.wrsamp(
                "test_ecg",
                fs=fs_test,
                units=["mV"] * 12,
                sig_name=LEAD_NAMES,
                p_signal=sig_12,
                fmt=["16"] * 12,
                write_dir=tmpdir,
            )
            out_path = "datasets/rendered/test_render.png"
            ok = render_ecg(rec_path, out_path, label="normal", patient_id="TEST-001")
            if ok:
                print(f"\nTest render saved: {out_path}")
                print("Open it and verify it looks like a real ECG printout.")
            else:
                print("Test render failed — check errors above.")