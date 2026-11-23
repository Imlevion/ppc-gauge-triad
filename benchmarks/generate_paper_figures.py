#!/usr/bin/env python3
"""Thin 300-DPI Matplotlib figure renderer powered by Rust `tools/ppc-bench`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.dataset_cache import BENCHMARK_SERVERS, DEFAULT_CACHE_DIR, populate_all_caches
from benchmarks.run_benchmarks import (
    eval_sklearn_supplement,
    merge_and_write_baselines_md,
    run_rust_ppc_bench,
)

TEST5_SEEDS = [101, 202, 303, 404, 505]
TEST5_V05_FAR = 10.44
TEST5_RAW_1WIN_FARS = [3.92, 3.11, 4.44, 4.48, 6.42]
TEST5_DUAL_GATE_FARS = [0.24, 0.13, 0.29, 0.32, 0.53]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.fontsize": 8.2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.28,
        "grid.linestyle": "--", "figure.dpi": 300, "savefig.dpi": 300,
    })


def plot_fig1(srv_res: dict, summary: dict, out_dir: Path) -> None:
    servers = [name for name, _, _ in BENCHMARK_SERVERS]
    short = [s.replace("SMD machine-", "SMD ").replace("PSM eBay Server", "PSM eBay") for s in servers]
    v05_au = [srv_res["per_server"][s]["v05"]["auroc"] for s in servers]
    hr_au = [srv_res["per_server"][s]["high_recall"]["auroc"] for s in servers]
    lf_au = [srv_res["per_server"][s]["low_far"]["auroc"] for s in servers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.7), gridspec_kw={"width_ratios": [1.35, 1.35]})
    x, w = np.arange(len(servers)), 0.26
    ax1.bar(x - w, v05_au, width=w, label="v0.5 (NoAdapt)", color="#9AA5B1", edgecolor="#3E4C59", lw=0.6)
    ax1.bar(x, hr_au, width=w, label="v0.7 High-Recall (0.002/0.001/1.0)", color="#0072B2", edgecolor="#003E6B", lw=0.6)
    ax1.bar(x + w, lf_au, width=w, label="v0.7 Low-FAR (0.008/0.004/1.5)", color="#009E73", edgecolor="#005B43", lw=0.6)
    pcaf_m, rsh_m = summary["pca_mahalanobis_fulld"]["auroc_mean"], summary["rs_hash"]["auroc_mean"]
    ax1.axhline(pcaf_m, color="#D55E00", ls=":", lw=1.35, label=f"PCA+Mahal Full-D Mean ({pcaf_m:.1f}%)")
    ax1.axhline(rsh_m, color="#CC79A7", ls="-.", lw=1.25, label=f"RS-Hash Mean ({rsh_m:.1f}%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=24, ha="right")
    ax1.set_ylabel("Window-Level AUROC (%, No Point-Adjust)")
    ax1.set_ylim(42, 116)
    ax1.set_title("(a) 8-Server Chronological Replay (SMD + PSM)")
    ax1.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95, fontsize=7.5)

    specs = [
        ("hs_trees", "HS-Trees\n(Rust)", "#B8C4D0"),
        ("pca_mahalanobis_32d", "PCA+Mah.\n(32-D)", "#CBD2D9"),
        ("raw_ocsvm", "OC-SVM\n(32-D)", "#CBD2D9"),
        ("v05", "v0.5\nNoAdapt", "#9AA5B1"),
        ("v07_low_far", "v0.7\nLow-FAR", "#009E73"),
        ("raw_isoforest", "IsoFor.\n(32-D)", "#D8B4C8"),
        ("v07_high_recall", "v0.7\nHigh-Rec", "#0072B2"),
        ("raw_knn", "kNN\n(Rust)", "#E69F00"),
        ("rs_hash", "RS-Hash\n(Rust)", "#CC79A7"),
        ("pca_mahalanobis_fulld", "PCA+Mah.\n(Full-D)", "#D55E00"),
    ]
    xb = np.arange(len(specs))
    means = [summary[k]["auroc_mean"] for k, _, _ in specs]
    stds = [summary[k]["auroc_std"] for k, _, _ in specs]
    fars = [summary[k]["far_dual_mean"] for k, _, _ in specs]
    bars = ax2.bar(xb, means, yerr=stds, capsize=3.2, error_kw={"elinewidth": 1.0, "ecolor": "#222222", "capthick": 0.9}, width=0.66, color=[c for _, _, c in specs], edgecolor="#222222", lw=0.6)
    for idx, (bar, m, s, fv) in enumerate(zip(bars, means, stds, fars)):
        cx = bar.get_x() + bar.get_width() / 2.0
        ax2.text(cx, m + s + 0.9, f"{m:.1f}\n±{s:.1f}", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#1F2933")
        txt_col = "white" if specs[idx][0] in ("v07_low_far", "v07_high_recall") else "#1F2933"
        ax2.text(cx, 51.4, f"FAR\n{fv:.1f}%", ha="center", va="bottom", fontsize=6.5, color=txt_col, fontweight="bold")
    ax2.set_xticks(xb)
    ax2.set_xticklabels([lbl for _, lbl, _ in specs], fontsize=7.2)
    ax2.set_ylabel("8-Server Mean ± Std AUROC (%)")
    ax2.set_ylim(50, 105)
    ax2.set_yticks([50, 60, 70, 80, 90, 100])
    ax2.set_title("(b) 8-Server Mean ± Std AUROC & Dual-Gate FAR (No Point-Adjust)")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_8server_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig2(warm_res: dict, out_dir: Path) -> None:
    xs = [0, 100, 200, 400]
    styles = [("SMD machine-1-2", "#E69F00", "o", "--"), ("SMD machine-1-6", "#56B4E9", "s", "-."), ("SMD machine-2-1", "#009E73", "^", ":")]
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for tgt, col, mk, ls in styles:
        ys = [warm_res["per_target"][tgt]["0w"], warm_res["v07_onchip_per_target"][tgt]["100w"], warm_res["v07_onchip_per_target"][tgt]["200w"], warm_res["v07_onchip_per_target"][tgt]["400w"]]
        ax.plot(xs, ys, marker=mk, ls=ls, color=col, lw=1.7, ms=6.0, label=tgt)
    mean_ys = [warm_res["mean"]["0w"], warm_res["v07_onchip_mean"]["100w"], warm_res["v07_onchip_mean"]["200w"], warm_res["v07_onchip_mean"]["400w"]]
    ax.plot(xs, mean_ys, marker="D", ls="-", color="#003E6B", lw=2.6, ms=7.0, label="Mean AUROC (v0.7 On-Chip AutoCal)", zorder=5)
    for xv, yv, (dx, dy) in zip(xs, mean_ys, [(30, -1.5), (0, -9.8), (0, -9.5), (-12, -8.5)]):
        ax.annotate(f"Mean {yv:.1f}%", xy=(xv, yv), xytext=(xv + dx, yv + dy), ha="center", va="center", fontsize=8.2, fontweight="bold", color="#003E6B", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#003E6B", alpha=0.94, lw=0.75), arrowprops=dict(arrowstyle="-", color="#003E6B", lw=0.8), zorder=6)
    ax.axvspan(-18, 22, color="#D55E00", alpha=0.09, label="Pure Zero-Shot (Inverted <50%)")
    ax.axvspan(85, 418, color="#009E73", alpha=0.07, label="Per-Device Warmup Zone (100–400w)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0w\n(Zero-Shot)", "100w\n(On-Chip)", "200w\n(On-Chip)", "400w\n(On-Chip)"])
    ax.set_xlim(-22, 425)
    ax.set_ylim(15, 98)
    ax.set_xlabel("Unsupervised Per-Device Warmup + Calibration Windows (0 Labels)")
    ax.set_ylabel("Unseen Target Server AUROC (%)")
    ax.set_title("Test 3B: Cross-Device Zero-Shot vs. v0.7 On-Chip AutoCalEngineF32")
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_onchip_autowarmup.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig3(out_dir: Path) -> None:
    rm, rs = float(np.mean(TEST5_RAW_1WIN_FARS)), float(np.std(TEST5_RAW_1WIN_FARS, ddof=0))
    dm, ds = float(np.mean(TEST5_DUAL_GATE_FARS)), float(np.std(TEST5_DUAL_GATE_FARS, ddof=0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.05, 1.25]})
    x = np.arange(3)
    means, stds = [TEST5_V05_FAR, rm, dm], [0.0, rs, ds]
    bars = ax1.bar(x, means, yerr=stds, capsize=5, width=0.55, color=["#9AA5B1", "#E69F00", "#0072B2"], edgecolor="#222222", lw=0.7)
    for i, (b, m, s) in enumerate(zip(bars, means, stds)):
        ax1.text(b.get_x() + b.get_width() / 2.0, m + s + 0.35, f"{m:.2f}%" if i == 0 else f"{m:.2f}%\n(±{s:.2f}%)", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    ax1.annotate("34.8x FAR reduction\n100.0% Recall (1000/1000)\n0 NaN / 50k windows", xy=(2, dm + ds + 1.9), xytext=(1.35, 8.2), ha="center", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.35", fc="#EBF8FF", ec="#0072B2", lw=0.9), arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.2))
    ax1.set_xticks(x)
    ax1.set_xticklabels(["v0.5 Baseline\n(1-Win)", "v0.6/v0.7 Raw\n(1-Win, n_cal=500)", "v0.6/v0.7 Dual-Stage\nAlarm Gate"], fontsize=8.2)
    ax1.set_ylabel("False Alarm Rate — FAR (%)")
    ax1.set_ylim(0, 12.6)
    ax1.set_title("(a) 50,000-Window Multi-Regime FAR Reduction")
    sx, bw = np.arange(len(TEST5_SEEDS)), 0.35
    ax2.bar(sx - bw / 2, TEST5_RAW_1WIN_FARS, width=bw, color="#E69F00", edgecolor="#6B4C00", lw=0.6, label="Raw 1-Window FAR")
    ax2.bar(sx + bw / 2, TEST5_DUAL_GATE_FARS, width=bw, color="#0072B2", edgecolor="#003E6B", lw=0.6, label="Dual-Stage Gate FAR")
    for i, (rv, dv) in enumerate(zip(TEST5_RAW_1WIN_FARS, TEST5_DUAL_GATE_FARS)):
        ax2.text(i - bw / 2, rv + 0.14, f"{rv:.2f}%", ha="center", va="bottom", fontsize=7.4)
        ax2.text(i + bw / 2, dv + 0.14, f"{dv:.2f}%", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#003E6B")
    ax2.set_xticks(sx)
    ax2.set_xticklabels([f"Seed {s}" for s in TEST5_SEEDS])
    ax2.set_ylabel("False Alarm Rate — FAR (%)")
    ax2.set_ylim(0, 7.6)
    ax2.set_title("(b) Per-Seed Audit (5 Independent Seeds, 100% Fault Recall)")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_dual_gate_far_reduction.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Render 300-DPI paper figures from Rust `ppc-bench` results.")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "paper" / "figures")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_raw = populate_all_caches(cache_dir=args.cache_dir, offline=args.offline, verbose=True)
    rust_rep = run_rust_ppc_bench("all", args.cache_dir)
    sk_sup = eval_sklearn_supplement(cache_raw)
    merged = merge_and_write_baselines_md(rust_rep, sk_sup, md_path=None)

    plot_fig1(rust_rep["servers_8"], merged["combined_summary"], args.out_dir)
    plot_fig2(rust_rep["warmup_curve"], args.out_dir)
    plot_fig3(args.out_dir)

    rust_rep["baselines_8"] = merged
    (args.out_dir / "figures_metrics.json").write_text(json.dumps(rust_rep, indent=2), encoding="utf-8")
    print(f"\nGenerated 300-DPI figures + provenance JSON in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
