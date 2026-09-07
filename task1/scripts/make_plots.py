#!/usr/bin/env python3
"""
make_plots.py -- Generate the required plots from aggregated_results_*.csv.

Produces (into --out-dir, default docs/figures/):
  mpki_vs_size_tile_K{K}.png       L1-D MPKI vs matrix size, one line per tile size      (Task 1B)
  speedup_vs_size_tile_K{K}.png    Speedup vs matrix size, one line per tile size        (Task 1B)
  speedup_vs_size_simd_K{K}.png    Speedup vs matrix size, one line per SIMD config      (Task 1C)
  final_comparison_K{K}.png        Grouped bar chart: best of each technique vs size     (Task 1D, Figure 1.1)
  speedup_vs_size_reorder.png      Speedup vs matrix size, K=3 and K=5 lines             (Task 1A)
  speedup_vs_size_unroll.png       Speedup vs matrix size, K=3 and K=5 lines (best unroll)(Task 1A)

The last two are K-independent (both K values are plotted as separate lines on the
same figure) and are written on every run regardless of --k.

Usage (from task1/):
    python3 scripts/make_plots.py docs/aggregated_results_20260906_193511.csv --k 3
    python3 scripts/make_plots.py docs/aggregated_results_20260906_193511.csv --k 5
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TILE_LABELS = ["tileSize8", "tileSize16", "tileSize32", "tileSize64",
               "tileSize128", "tileSize256", "tileSize512", "tileSize1024"]
TILE_DISPLAY = {l: f"Tile={l.replace('tileSize', '')}" for l in TILE_LABELS}

SIMD_LABELS = ["128bits1unroll", "128bits2unroll", "128bits4unroll",
               "256bits1unroll", "256bits2unroll", "512bits1unroll"]
SIMD_DISPLAY = {
    "128bits1unroll": "128-bit, 1x unroll",
    "128bits2unroll": "128-bit, 2x unroll",
    "128bits4unroll": "128-bit, 4x unroll",
    "256bits1unroll": "256-bit, 1x unroll",
    "256bits2unroll": "256-bit, 2x unroll",
    "512bits1unroll": "512-bit, 1x unroll",
}

# Best-of-each-technique picks for the Task 1D comparison chart.
# (profile_stage, label, table_stage, display name)
FINAL_COMPARISON_SERIES = [
    ("unroll", "unrolling8combo_16acc", "unroll", "Loop Unrolling (best)"),
    ("reorder", "reorder8combo", "reorder", "Loop Reordering"),
    ("tile", "tileSize256", "tile", "Tiling (best)"),
    ("simd", "256bits2unroll", "simd", "SIMD (best)"),
    ("optimized", "tileSize32SIMDbits256", "optimized", "Tiling+SIMD (Tile=32, 256-bit)"),
    ("optimized", "tileSize32SIMDbits512", "optimized", "Tiling+SIMD (Tile=32, 512-bit)"),
]

SIZES = [512, 1024, 2048, 4096]


def load(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def index_by(rows, stage, table_stage, k):
    """label -> {H: row}"""
    out = defaultdict(dict)
    for r in rows:
        if r["profile_stage"] == stage and r["table_stage"] == table_stage and int(r["K"]) == k:
            out[r["label"]][int(r["H"])] = r
    return out


def plot_mpki_vs_size_tile(rows, k, out_dir):
    by_label = index_by(rows, "tile", "tile", k)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in TILE_LABELS:
        sizes_present = [h for h in SIZES if h in by_label.get(label, {})]
        ys = [float(by_label[label][h]["isolated_mpki"]) for h in sizes_present]
        ax.plot(sizes_present, ys, marker="o", label=TILE_DISPLAY[label])
    ax.set_xlabel("Matrix Size (H = W)")
    ax.set_ylabel("L1-D MPKI (isolated)")
    ax.set_title(f"L1-D MPKI vs. Matrix Size by Tile Size (K={k})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"mpki_vs_size_tile_K{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_speedup_vs_size_tile(rows, k, out_dir):
    by_label = index_by(rows, "tile", "tile", k)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in TILE_LABELS:
        sizes_present = [h for h in SIZES if h in by_label.get(label, {})]
        ys = [float(by_label[label][h]["speedup"]) for h in sizes_present]
        ax.plot(sizes_present, ys, marker="o", label=TILE_DISPLAY[label])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Matrix Size (H = W)")
    ax.set_ylabel("Speedup vs. Naive")
    ax.set_title(f"Speedup vs. Matrix Size by Tile Size (K={k})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"speedup_vs_size_tile_K{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_speedup_vs_size_simd(rows, k, out_dir):
    by_label = index_by(rows, "simd", "simd", k)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in SIMD_LABELS:
        sizes_present = [h for h in SIZES if h in by_label.get(label, {})]
        ys = [float(by_label[label][h]["speedup"]) for h in sizes_present]
        ax.plot(sizes_present, ys, marker="o", label=SIMD_DISPLAY[label])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Matrix Size (H = W)")
    ax.set_ylabel("Speedup vs. Naive")
    ax.set_title(f"Speedup vs. Matrix Size by SIMD Width/Unroll (K={k})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"speedup_vs_size_simd_K{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_speedup_vs_size_reorder(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in (3, 5):
        by_label = index_by(rows, "reorder", "reorder", k)
        row_map = by_label.get("reorder8combo", {})
        sizes_present = [h for h in SIZES if h in row_map]
        ys = [float(row_map[h]["speedup"]) for h in sizes_present]
        ax.plot(sizes_present, ys, marker="o", label=f"K={k}")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="1x (Naive baseline)")
    ax.set_xlabel("Matrix Size (H = W)")
    ax.set_ylabel("Speedup vs. Naive")
    ax.set_title("Loop Reordering: Speedup vs. Matrix Size")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "speedup_vs_size_reorder.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_speedup_vs_size_unroll(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in (3, 5):
        by_label = index_by(rows, "unroll", "unroll", k)
        row_map = by_label.get("unrolling8combo_16acc", {})
        sizes_present = [h for h in SIZES if h in row_map]
        ys = [float(row_map[h]["speedup"]) for h in sizes_present]
        ax.plot(sizes_present, ys, marker="o", label=f"K={k}")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="1x (Naive baseline)")
    ax.set_xlabel("Matrix Size (H = W)")
    ax.set_ylabel("Speedup vs. Naive")
    ax.set_title("Loop Unrolling (16 Accumulators): Speedup vs. Matrix Size")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "speedup_vs_size_unroll.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_final_comparison(rows, k, out_dir):
    series_data = []
    for stage, label, table_stage, display in FINAL_COMPARISON_SERIES:
        by_label = index_by(rows, stage, table_stage, k)
        row_map = by_label.get(label, {})
        ys = [float(row_map[h]["speedup"]) if h in row_map else 0.0 for h in SIZES]
        series_data.append((display, ys))

    n_series = len(series_data)
    n_sizes = len(SIZES)
    x = range(n_sizes)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (display, ys) in enumerate(series_data):
        offsets = [xi + (i - n_series / 2) * width + width / 2 for xi in x]
        ax.bar(offsets, ys, width=width, label=display)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="1x (Naive baseline)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Matrix Size (H = W)")
    ax.set_ylabel("Speedup vs. Naive")
    ax.set_ylim(0.8, 20)
    ax.set_title(f"Best-of-Technique Speedup Comparison (K={k})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"final_comparison_K{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="path to aggregated_results_*.csv")
    ap.add_argument("--k", type=int, default=3, help="K value to plot (default 3)")
    ap.add_argument("--out-dir", default="docs/figures", help="output directory (default docs/figures)")
    args = ap.parse_args()

    rows = load(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_mpki_vs_size_tile(rows, args.k, out_dir)
    p2 = plot_speedup_vs_size_tile(rows, args.k, out_dir)
    p3 = plot_speedup_vs_size_simd(rows, args.k, out_dir)
    p4 = plot_final_comparison(rows, args.k, out_dir)
    p5 = plot_speedup_vs_size_reorder(rows, out_dir)
    p6 = plot_speedup_vs_size_unroll(rows, out_dir)

    for p in (p1, p2, p3, p4, p5, p6):
        print(f"[written] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())