#!/usr/bin/env python3
"""Generate 300-DPI publication figures for the PPC-Gauge-Triad v0.7 preprint.

All server telemetry and v0.7 on-chip AutoCalEngineF32 curves are computed live
from the verified offline SMD/PSM cache (`data_cache/`). Test 5 multi-seed
streaming values match the pre-registered independent 5-seed verification in
`VALIDATION_REPORT_v061.md` Section 4.4.

Usage:
    python benchmarks/generate_paper_figures.py --offline
"""

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
from benchmarks.ppc_triad_wrapper import PRESET_LOW_FAR, load_rust_lib
from benchmarks.run_benchmarks import run_8_server_benchmark, run_warmup_curve

# Verified reference classical baselines from VALIDATION_REPORT_v061.md Section 4.1
CLASSICAL_BASELINES_8SRV = {
    "Raw kNN": 74.9,
    "Raw IsoForest": 74.0,
    "Raw OC-SVM": 71.8,
}

# Verified 50,000-window, 5-seed independent streaming audit (VALIDATION_REPORT_v061.md Section 4.4)
TEST5_SEEDS = [101, 202, 303, 404, 505]
TEST5_V05_FAR = 10.44
TEST5_RAW_1WIN_FARS = [3.92, 3.11, 4.44, 4.48, 6.42]
TEST5_DUAL_GATE_FARS = [0.24, 0.13, 0.29, 0.32, 0.53]
TEST5_FAULT_RECALL = 100.00


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def plot_fig1_8server_ablation(srv_res: dict, out_dir: Path) -> None:
    """Figure 1: 8-Server AUROC ablation + locked preset AUROC/FAR/Recall trade-off."""
    servers = [name for name, _, _ in BENCHMARK_SERVERS]
    short_names = [
        s.replace("SMD machine-", "SMD ").replace("PSM eBay Server", "PSM eBay") for s in servers
    ]
    v05_au = [srv_res["per_server"][s]["v05"]["auroc"] for s in servers]
    hr_au = [srv_res["per_server"][s]["high_recall"]["auroc"] for s in servers]
    lf_au = [srv_res["per_server"][s]["low_far"]["auroc"] for s in servers]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.85, 1.0]}
    )

    x = np.arange(len(servers))
    w = 0.26
    ax1.bar(x - w, v05_au, width=w, label="v0.5 (NoAdapt)", color="#9AA5B1", edgecolor="#3E4C59", lw=0.6)
    ax1.bar(
        x,
        hr_au,
        width=w,
        label="v0.7 High-Recall (0.002/0.001/1.0)",
        color="#0072B2",
        edgecolor="#003E6B",
        lw=0.6,
    )
    ax1.bar(
        x + w,
        lf_au,
        width=w,
        label="v0.7 Low-FAR (0.008/0.004/1.5)",
        color="#009E73",
        edgecolor="#005B43",
        lw=0.6,
    )

    ax1.axhline(
        CLASSICAL_BASELINES_8SRV["Raw kNN"],
        color="#D55E00",
        ls=":",
        lw=1.3,
        label="Raw kNN Mean (74.9%)",
    )
    ax1.axhline(
        CLASSICAL_BASELINES_8SRV["Raw IsoForest"],
        color="#CC79A7",
        ls="-.",
        lw=1.1,
        label="Raw IsoForest Mean (74.0%)",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=22, ha="right")
    ax1.set_ylabel("AUROC (%)")
    ax1.set_ylim(42, 114)
    ax1.set_title("(a) 8-Server Chronological Replay (SMD + PSM)")
    ax1.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95, fontsize=7.8)

    # Panel (b): 8-Server Mean Comparison (AUROC, Recall, FAR)
    labels = ["Raw\nOC-SVM", "Raw\nIsoFor.", "Raw\nkNN", "v0.5\nNoAdapt", "v0.7\nLow-FAR", "v0.7\nHigh-Rec"]
    mean_au = [
        CLASSICAL_BASELINES_8SRV["Raw OC-SVM"],
        CLASSICAL_BASELINES_8SRV["Raw IsoForest"],
        CLASSICAL_BASELINES_8SRV["Raw kNN"],
        srv_res["mean"]["v05"]["auroc"],
        srv_res["mean"]["low_far"]["auroc"],
        srv_res["mean"]["high_recall"]["auroc"],
    ]
    colors = ["#CBD2D9", "#CC79A7", "#D55E00", "#9AA5B1", "#009E73", "#0072B2"]
    xb = np.arange(len(labels))
    bars = ax2.bar(xb, mean_au, width=0.68, color=colors, edgecolor="#222222", lw=0.6)
    for bar, val in zip(bars, mean_au):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.6,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    # Clean vertical labels centered inside the v0.7 preset bars (zero clipping)
    lf_far = srv_res["mean"]["low_far"]["far"]
    hr_far = srv_res["mean"]["high_recall"]["far"]
    hr_rec = srv_res["mean"]["high_recall"]["recall"]
    lf_rec = srv_res["mean"]["low_far"]["recall"]
    ax2.text(
        4,
        68.5,
        f"FAR {lf_far:.1f}%\nRec {lf_rec:.1f}%",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.3,
        color="white",
        fontweight="bold",
    )
    ax2.text(
        5,
        69.2,
        f"FAR {hr_far:.1f}%\nRec {hr_rec:.1f}%",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.3,
        color="white",
        fontweight="bold",
    )

    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=7.8)
    ax2.set_ylabel("8-Server Mean AUROC (%)")
    ax2.set_ylim(60, 85)
    ax2.set_title("(b) Mean AUROC & Preset Trade-Off")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_8server_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_onchip_autowarmup(warm_res: dict, out_dir: Path) -> None:
    """Figure 2: Pure Zero-Shot vs v0.7 On-Chip AutoCalEngineF32 Auto-Warmup (0 -> 400w)."""
    xs = [0, 100, 200, 400]
    targets = ["SMD machine-1-2", "SMD machine-1-6", "SMD machine-2-1"]
    styles = [
        ("SMD machine-1-2", "#E69F00", "o", "--"),
        ("SMD machine-1-6", "#56B4E9", "s", "-."),
        ("SMD machine-2-1", "#009E73", "^", ":"),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for tgt, color, marker, ls in styles:
        ys = [
            warm_res["per_target"][tgt]["0w"],
            warm_res["v07_onchip_per_target"][tgt]["100w"],
            warm_res["v07_onchip_per_target"][tgt]["200w"],
            warm_res["v07_onchip_per_target"][tgt]["400w"],
        ]
        ax.plot(xs, ys, marker=marker, ls=ls, color=color, lw=1.7, ms=6.0, label=tgt)

    mean_ys = [
        warm_res["mean"]["0w"],
        warm_res["v07_onchip_mean"]["100w"],
        warm_res["v07_onchip_mean"]["200w"],
        warm_res["v07_onchip_mean"]["400w"],
    ]
    ax.plot(
        xs,
        mean_ys,
        marker="D",
        ls="-",
        color="#003E6B",
        lw=2.6,
        ms=7.0,
        label="Mean AUROC (v0.7 On-Chip AutoCal)",
        zorder=5,
    )

    label_offsets = {
        0: (30, -1.5),
        100: (0, -9.8),
        200: (0, -9.5),
        400: (-12, -8.5),
    }
    for x_val, y_val in zip(xs, mean_ys):
        dx, dy = label_offsets[x_val]
        ax.annotate(
            f"Mean {y_val:.1f}%",
            xy=(x_val, y_val),
            xytext=(x_val + dx, y_val + dy),
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="#003E6B",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#003E6B", alpha=0.94, lw=0.75),
            arrowprops=dict(arrowstyle="-", color="#003E6B", lw=0.8),
            zorder=6,
        )

    ax.axvspan(-18, 22, color="#D55E00", alpha=0.09, label="Pure Zero-Shot (Unsupported)")
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


def plot_fig3_dual_gate_far(out_dir: Path) -> None:
    """Figure 3: Test 5 (50,000-Window, 5 Independent Seeds) Dual-Stage Alarm Gate FAR reduction."""
    raw_mean = float(np.mean(TEST5_RAW_1WIN_FARS))
    raw_std = float(np.std(TEST5_RAW_1WIN_FARS, ddof=0))
    dual_mean = float(np.mean(TEST5_DUAL_GATE_FARS))
    dual_std = float(np.std(TEST5_DUAL_GATE_FARS, ddof=0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.05, 1.25]})

    # Panel (a): Summary FAR reduction with error bars
    cats = ["v0.5 Baseline\n(1-Win)", "v0.6/v0.7 Raw\n(1-Win, n_cal=500)", "v0.6/v0.7 Dual-Stage\nAlarm Gate"]
    means = [TEST5_V05_FAR, raw_mean, dual_mean]
    stds = [0.0, raw_std, dual_std]
    cols = ["#9AA5B1", "#E69F00", "#0072B2"]
    x = np.arange(len(cats))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, width=0.55, color=cols, edgecolor="#222222", lw=0.7)

    for idx, (bar, m, s) in enumerate(zip(bars, means, stds)):
        txt = f"{m:.2f}%" if idx == 0 else f"{m:.2f}%\n(±{s:.2f}%)"
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            m + s + 0.35,
            txt,
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
        )

    ax1.annotate(
        f"34.8x FAR reduction\n100.0% Recall (1000/1000)\n0 NaN / 50k windows",
        xy=(2, dual_mean + dual_std + 1.9),
        xytext=(1.35, 8.2),
        ha="center",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", fc="#EBF8FF", ec="#0072B2", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.2),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=8.2)
    ax1.set_ylabel("False Alarm Rate — FAR (%)")
    ax1.set_ylim(0, 12.6)
    ax1.set_title("(a) 50,000-Window Multi-Regime FAR Reduction")

    # Panel (b): Per-seed breakdown across 5 independent held-out seeds
    sx = np.arange(len(TEST5_SEEDS))
    bw = 0.35
    ax2.bar(sx - bw / 2, TEST5_RAW_1WIN_FARS, width=bw, color="#E69F00", edgecolor="#6B4C00", lw=0.6, label="Raw 1-Window FAR")
    ax2.bar(sx + bw / 2, TEST5_DUAL_GATE_FARS, width=bw, color="#0072B2", edgecolor="#003E6B", lw=0.6, label="Dual-Stage Gate FAR")
    for i, (r_val, d_val) in enumerate(zip(TEST5_RAW_1WIN_FARS, TEST5_DUAL_GATE_FARS)):
        ax2.text(i - bw / 2, r_val + 0.14, f"{r_val:.2f}%", ha="center", va="bottom", fontsize=7.4)
        ax2.text(i + bw / 2, d_val + 0.14, f"{d_val:.2f}%", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#003E6B")

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
    p = argparse.ArgumentParser(description="Generate 300-DPI paper figures from verified offline data.")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "paper" / "figures")
    p.add_argument("--offline", action="store_true", help="Fail if dataset cache is missing")
    args = p.parse_args()

    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lib = load_rust_lib()
    cache_raw = populate_all_caches(cache_dir=args.cache_dir, offline=args.offline, verbose=True)

    srv_res = run_8_server_benchmark(cache_raw)
    warm_res = run_warmup_curve(cache_raw, preset=PRESET_LOW_FAR)

    plot_fig1_8server_ablation(srv_res, args.out_dir)
    plot_fig2_onchip_autowarmup(warm_res, args.out_dir)
    plot_fig3_dual_gate_far(args.out_dir)

    provenance = {
        "crate_version": "0.7.0",
        "core_sram_bytes": int(lib.ppc_triad_engine_bytes()),
        "autocal_sram_bytes": int(lib.ppc_autocal_engine_bytes()),
        "servers_8": srv_res,
        "warmup_curve": warm_res,
        "test5_50k_5seeds": {
            "seeds": TEST5_SEEDS,
            "v05_far_pct": TEST5_V05_FAR,
            "raw_1win_fars_pct": TEST5_RAW_1WIN_FARS,
            "dual_gate_fars_pct": TEST5_DUAL_GATE_FARS,
            "fault_recall_pct": TEST5_FAULT_RECALL,
        },
    }
    (args.out_dir / "figures_metrics.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"\nGenerated 300-DPI figures + provenance JSON in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
