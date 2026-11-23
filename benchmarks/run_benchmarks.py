#!/usr/bin/env python3
"""Local Benchmark & Verification Runner for PPC-Gauge-Triad v0.6.1.

Replaces Google Colab with a reproducible local CLI powered by the native Rust
`cdylib` and a local `.npz` dataset cache (`data_cache/`) for SMD and PSM.

Examples:
    python benchmarks/run_benchmarks.py --l1-parity
    python benchmarks/run_benchmarks.py --cache-only
    python benchmarks/run_benchmarks.py --servers --warmup-curve
    python benchmarks/run_benchmarks.py --all --offline
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Allow running directly via `python benchmarks/run_benchmarks.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.baselines import (
    BASELINE_DISPLAY_NAMES,
    DEFAULT_SEEDS,
    eval_all_baselines_on_server,
    summarize_across_servers,
)
from benchmarks.dataset_cache import (
    BENCHMARK_SERVERS,
    DEFAULT_CACHE_DIR,
    populate_all_caches,
)
from benchmarks.ppc_triad_wrapper import (
    PRESET_HIGH_RECALL,
    PRESET_LOW_FAR,
    PRESET_V05_BASELINE,
    NumpyTriadRefV061,
    OperationalPreset,
    RustAutoCalV07,
    RustTriadV061,
    compute_auroc,
    conformal_threshold,
    dual_gate_scalar,
    load_rust_lib,
    to_windows_v6,
    triad_norm_score,
)


def run_l1_8_parity() -> Dict[str, Any]:
    """L1.8 Cross-Implementation Parity: Pure-Python NumPy f32 vs Rust cdylib f32."""
    print("\n=== L1.8 CROSS-IMPLEMENTATION PARITY (Python NumPy vs Rust cdylib f32) ===")
    rng = np.random.default_rng(1108)
    w, d = 8, 4
    n_train, n_test = 150, 100
    x_tr = rng.standard_normal((n_train, w, d)).astype(np.float32)
    x_te = (rng.standard_normal((n_test, w, d)) * 1.2 + 0.4).astype(np.float32)

    rs = RustTriadV061(w, d, k_max=3, lam0=5e-3, alpha_min=0.01, e_alpha=0.008, dc_alpha=0.004)
    py = NumpyTriadRefV061(w, d, k_max=3, lam0=5e-3, alpha_min=0.01, e_alpha=0.008, dc_alpha=0.004)

    rs_tr = rs.run(x_tr, learn=True)
    py_tr = py.run(x_tr, learn=True)

    rs_te = rs.run(x_te, learn=False, shape_gate=50.0, adapt_env=True, huber_clip=1.5)
    py_te = py.run(x_te, learn=False, shape_gate=50.0, adapt_env=True, huber_clip=1.5)

    # Relative/absolute diffs on score channels
    max_diff_sh = float(np.max(np.abs(rs_te[0] - py_te[0])))
    max_diff_en = float(np.max(np.abs(rs_te[1] - py_te[1])))
    max_diff_lv = float(np.max(np.abs(rs_te[2] - py_te[2])))
    # Canonical signature parity (direct check on initial 1-window canonicalize equivalence)
    # Using score diff normalized by scale + direct train step-0 diff
    max_diff_tr_en = float(np.max(np.abs(rs_tr[1] - py_tr[1])))
    max_diff_tr_lv = float(np.max(np.abs(rs_tr[2] - py_tr[2])))
    alarm_mismatch = int(np.sum(rs_te[3] != py_te[3]))

    # Canonicalize parity check across 100 windows via Rust score vs Python score on fixed identity precision
    # With k_max=1, lam0=1.0, first window spawns proto (mean=u0, P=I), second window returns ||u1-u0||^2
    canon_diffs = []
    for i in range(50):
        pair = x_tr[2 * i : 2 * i + 2]
        r1 = RustTriadV061(w, d, k_max=1, lam0=1.0, alpha_min=0.01)
        p1 = NumpyTriadRefV061(w, d, k_max=1, lam0=1.0, alpha_min=0.01)
        r_sh, _, _, _ = r1.run(pair, learn=True)
        p_sh, _, _, _ = p1.run(pair, learn=True)
        canon_diffs.append(abs(float(r_sh[1] - p_sh[1])))
    max_canon_diff = float(np.max(canon_diffs))

    passed = (max_canon_diff < 1e-5) and (max_diff_en < 1e-4) and (max_diff_lv < 1e-4) and (alarm_mismatch == 0)
    print(f"  Canonical shape L2^2 max_diff : {max_canon_diff:.3e}  (criterion < 1e-5)")
    print(f"  Stream s_shape max_diff       : {max_diff_sh:.3e}")
    print(f"  Stream s_energy max_diff      : {max_diff_en:.3e}")
    print(f"  Stream s_level max_diff       : {max_diff_lv:.3e}")
    print(f"  Alarm mismatches              : {alarm_mismatch} / {n_test}")
    print(f"  Status                        : {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise AssertionError(
            f"L1.8 parity failed: max_canon_diff={max_canon_diff:.3e}, "
            f"en={max_diff_en:.3e}, lv={max_diff_lv:.3e}, alarm_mismatch={alarm_mismatch}"
        )

    return {
        "max_canon_diff": max_canon_diff,
        "max_stream_sh_diff": max_diff_sh,
        "max_stream_en_diff": max_diff_en,
        "max_stream_lv_diff": max_diff_lv,
        "max_train_en_diff": max_diff_tr_en,
        "max_train_lv_diff": max_diff_tr_lv,
        "alarm_mismatch": alarm_mismatch,
        "passed": passed,
    }


def eval_preset_on_server(
    tr_r: np.ndarray,
    te_r: np.ndarray,
    yl_r: np.ndarray,
    preset: OperationalPreset,
) -> Dict[str, float]:
    """Evaluate a locked a priori preset chronologically on a server telemetry stream.

    Evaluation protocol: non-overlapping w=8 windows, window-level binary labels
    (any anomalous sample inside window -> 1), strictly WITHOUT point-adjust.
    """
    wtr, wte, yte, _ = to_windows_v6(tr_r, te_r, yl_r, use_pca_resid=preset.use_pca_resid)
    kf = int(len(wtr) * 0.75)
    eng = RustTriadV061(
        w=8,
        d=4,
        k_max=3,
        lam0=5e-3,
        alpha_min=0.01,
        e_alpha=preset.e_alpha,
        dc_alpha=preset.dc_alpha,
    )
    eng.run(wtr[:kf], learn=True)
    shc, enc, lvc, _ = eng.run(wtr[kf:], learn=False)
    thr_sh = conformal_threshold(shc, 0.05)
    thr_sc = conformal_threshold(triad_norm_score(shc, enc, lvc, shc, enc, lvc), 0.05)
    sht, ent, lvt, _ = eng.run(
        wte,
        learn=False,
        shape_gate=thr_sh,
        adapt_env=preset.adapt_env,
        dc_relock=preset.dc_relock,
        huber_clip=preset.huber_clip,
        sev_mult=preset.sev_mult,
    )
    sc = triad_norm_score(shc, enc, lvc, sht, ent, lvt)
    al_raw = sc > thr_sc
    al_dual = dual_gate_scalar(sc, thr_sc, preset.sev_mult)
    al = al_dual if preset.use_dual else al_raw
    auroc = compute_auroc(yte, sc) * 100.0
    far = float(np.mean(al[yte == 0])) * 100.0
    rec = float(np.mean(al[yte == 1])) * 100.0
    far_raw = float(np.mean(al_raw[yte == 0])) * 100.0
    rec_raw = float(np.mean(al_raw[yte == 1])) * 100.0
    far_dual = float(np.mean(al_dual[yte == 0])) * 100.0
    rec_dual = float(np.mean(al_dual[yte == 1])) * 100.0
    return {
        "auroc": auroc,
        "far": far,
        "recall": rec,
        "far_raw": far_raw,
        "recall_raw": rec_raw,
        "far_dual": far_dual,
        "recall_dual": rec_dual,
    }


def run_8_server_benchmark(
    cache_raw: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Run 8-server chronological replay across locked presets (v0.5, High-Recall, Low-FAR)."""
    print("\n=== 8-SERVER REAL-WORLD CHRONOLOGICAL REPLAY (CACHED SMD + PSM, NO POINT-ADJUST) ===")
    header = (
        f"{'Server':<16} | {'v0.5 (NoAdapt)':^20} | "
        f"{'High-Recall (0.002)':^20} | {'Low-FAR (0.008)':^20}"
    )
    subhdr = (
        f"{'':<16} | {'AUROC':>6} {'FAR':>6} {'Rec':>6} | "
        f"{'AUROC':>6} {'FAR':>6} {'Rec':>6} | {'AUROC':>6} {'FAR':>6} {'Rec':>6}"
    )
    print(header)
    print(subhdr)
    print("-" * len(subhdr))

    acc: Dict[str, List[Dict[str, float]]] = {"v05": [], "high_recall": [], "low_far": []}
    per_server: Dict[str, Dict[str, Dict[str, float]]] = {}

    for display_name, _, _ in BENCHMARK_SERVERS:
        tr_r, te_r, yl_r = cache_raw[display_name]
        r_v5 = eval_preset_on_server(tr_r, te_r, yl_r, PRESET_V05_BASELINE)
        r_hr = eval_preset_on_server(tr_r, te_r, yl_r, PRESET_HIGH_RECALL)
        r_lf = eval_preset_on_server(tr_r, te_r, yl_r, PRESET_LOW_FAR)
        acc["v05"].append(r_v5)
        acc["high_recall"].append(r_hr)
        acc["low_far"].append(r_lf)
        per_server[display_name] = {
            "v05": r_v5,
            "high_recall": r_hr,
            "low_far": r_lf,
        }
        print(
            f"{display_name:<16} | "
            f"{r_v5['auroc']:5.1f}% {r_v5['far']:5.1f}% {r_v5['recall']:5.1f}% | "
            f"{r_hr['auroc']:5.1f}% {r_hr['far']:5.1f}% {r_hr['recall']:5.1f}% | "
            f"{r_lf['auroc']:5.1f}% {r_lf['far']:5.1f}% {r_lf['recall']:5.1f}%"
        )

    print("-" * len(subhdr))
    means: Dict[str, Dict[str, float]] = {}
    stds: Dict[str, Dict[str, float]] = {}
    mean_parts = []
    std_parts = []
    for key in ("v05", "high_recall", "low_far"):
        au = np.array([r["auroc"] for r in acc[key]], dtype=np.float64)
        far = np.array([r["far"] for r in acc[key]], dtype=np.float64)
        rec = np.array([r["recall"] for r in acc[key]], dtype=np.float64)
        far_raw = np.array([r["far_raw"] for r in acc[key]], dtype=np.float64)
        rec_raw = np.array([r["recall_raw"] for r in acc[key]], dtype=np.float64)
        far_dual = np.array([r["far_dual"] for r in acc[key]], dtype=np.float64)
        rec_dual = np.array([r["recall_dual"] for r in acc[key]], dtype=np.float64)
        means[key] = {
            "auroc": float(au.mean()),
            "far": float(far.mean()),
            "recall": float(rec.mean()),
            "far_raw": float(far_raw.mean()),
            "recall_raw": float(rec_raw.mean()),
            "far_dual": float(far_dual.mean()),
            "recall_dual": float(rec_dual.mean()),
        }
        stds[key] = {
            "auroc": float(au.std(ddof=0)),
            "far": float(far.std(ddof=0)),
            "recall": float(rec.std(ddof=0)),
            "far_raw": float(far_raw.std(ddof=0)),
            "recall_raw": float(rec_raw.std(ddof=0)),
            "far_dual": float(far_dual.std(ddof=0)),
            "recall_dual": float(rec_dual.std(ddof=0)),
        }
        mean_parts.append(f"{au.mean():5.1f}% {far.mean():5.1f}% {rec.mean():5.1f}%")
        std_parts.append(f"{au.std(ddof=0):5.1f}% {far.std(ddof=0):5.1f}% {rec.std(ddof=0):5.1f}%")
    print(f"{'8-SERVER MEAN':<16} | " + " | ".join(mean_parts))
    print(f"{'8-SERVER STD':<16} | " + " | ".join(std_parts))

    return {"per_server": per_server, "mean": means, "std": stds}


def render_baselines_markdown(
    baseline_res: Dict[str, Any],
    srv_res: Dict[str, Any],
) -> str:
    """Format honest comparative baseline benchmark table (mean +- std AUROC & FAR)."""
    method_order = [
        ("v07_high_recall", "PPC-Gauge-Triad v0.7 High-Recall (`0.002/0.001/1.0`)", "Streaming (Rust `f32`, 32-D)"),
        ("v07_low_far", "PPC-Gauge-Triad v0.7 Low-FAR (`0.008/0.004/1.5`)", "Streaming (Rust `f32`, 32-D)"),
        ("v05", "PPC-Gauge-Triad v0.5 (`NoAdapt`)", "Static (Rust `f32`, 32-D 4PC)"),
        ("hs_trees", "Half-Space Trees (Tan et al. 2011, 5 seeds)", "Streaming (32-D, `psi=250`)"),
        ("rs_hash", "RS-Hash / RS-Stream (Sathe & Aggarwal 2016, 5 seeds)", "Streaming (32-D, `W=1000`)"),
        ("pca_mahalanobis_32d", "PCA + Mahalanobis (`0.95` var, matched 32-D)", "Batch (32-D)"),
        ("pca_mahalanobis_fulld", "PCA + Mahalanobis (`0.95` var, full raw `w*D`)", "Batch (Full `w*D`)"),
        ("raw_isoforest", "Isolation Forest (`n=100`, 5 seeds)", "Batch (32-D)"),
        ("raw_knn", "kNN (`k=5`, Euclidean)", "Batch (32-D)"),
        ("raw_ocsvm", "OC-SVM (`RBF`, `nu=0.05`)", "Batch (32-D)"),
    ]

    summary = baseline_res["combined_summary"]
    per_srv = baseline_res["combined_per_server"]
    servers = [name for name, _, _ in BENCHMARK_SERVERS]

    # Rank methods by 8-server mean AUROC descending for honest winner/loser reporting
    ranked = sorted(method_order, key=lambda item: summary[item[0]]["auroc_mean"], reverse=True)
    best_key, best_name, _ = ranked[0]
    best_au = summary[best_key]["auroc_mean"]
    hr_au = summary["v07_high_recall"]["auroc_mean"]
    lf_au = summary["v07_low_far"]["auroc_mean"]

    lines: List[str] = [
        "# Fair Baseline Comparison — 8-Server SMD + PSM Chronological Replay",
        "",
        "## Evaluation Protocol (Identical Across All Methods)",
        "",
        "- **Datasets:** 7 SMD servers (`machine-1-1`, `1-6`, `2-1`, `1-2`, `1-3`, `2-2`, `3-1`) + PSM eBay Server (`8` servers total, loaded from local offline `.npz` cache).",
        "- **Windowing:** Non-overlapping windows of length `w = 8` (`to_windows_v6`, `3PC+Resid` `d = 4` -> flattened `32-D` window vectors; `max_tr_win = 2000` normal training windows). An additional `PCA + Mahalanobis (Full-D Raw)` row is evaluated on standardized full-channel `w * D` windows (`304-D` for SMD, `200-D` for PSM) as a fairness check against dimensionality compression.",
        "- **Chronological Split:** First `75%` of `wtr` for fitting/warmup, held-out last `25%` of `wtr` for split-conformal calibration (`alpha = 0.05`). Test windows `wte` are evaluated strictly chronologically.",
        "- **Labeling & Metrics:** **Window-level binary ground truth WITHOUT point-adjust** (`y_win = 1` iff any timestep inside the 8-sample window is labeled anomalous). **No point-adjustment (PA) is applied anywhere.**",
        "- **Thresholding Fairness:** Every method's threshold is calibrated identically via finite-sample split-conformal quantile (`alpha = 0.05`) on the held-out `25%` calibration split. Both **Raw 1-Window FAR / Recall** and **Dual-Stage Gated FAR / Recall** (`> 2.5x thr` immediate OR 2-consecutive windows) are reported so the dual-stage alarm gate never biases cross-method FAR comparisons.",
        "- **Variance & Seeds:** Randomized methods (`Half-Space Trees`, `RS-Hash`, `Isolation Forest`) are averaged over 5 fixed seeds `[0, 1, 2, 3, 4]` per server. Reported `mean ± std` is across the 8 servers (`ddof = 0`).",
        "- **Locked Engine Presets:** `e_alpha`, `dc_alpha`, `huber_clip`, and `dc_relock` were **not** modified for this benchmark.",
        "",
        "## 8-Server Summary Table (Mean ± Std Across 8 Servers, No Point-Adjust)",
        "",
        "| Method | Category | AUROC (%) | Raw 1-Win FAR (%) | Dual-Gate FAR (%) | Dual-Gate Recall (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for key, label, cat in method_order:
        s = summary[key]
        lines.append(
            f"| {label} | {cat} | "
            f"**{s['auroc_mean']:.1f} ± {s['auroc_std']:.1f}** | "
            f"{s['far_raw_mean']:.1f} ± {s['far_raw_std']:.1f} | "
            f"{s['far_dual_mean']:.1f} ± {s['far_dual_std']:.1f} | "
            f"{s['recall_dual_mean']:.1f} ± {s['recall_dual_std']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Honest Findings & Where PPC-Gauge-Triad Wins or Loses",
            "",
        ]
    )

    beating_hr = [
        (label, summary[k]["auroc_mean"], summary[k]["auroc_std"])
        for k, label, _ in ranked
        if k not in ("v07_high_recall", "v07_low_far", "v05") and summary[k]["auroc_mean"] > hr_au
    ]
    beating_lf = [
        (label, summary[k]["auroc_mean"], summary[k]["auroc_std"])
        for k, label, _ in ranked
        if k not in ("v07_high_recall", "v07_low_far", "v05") and summary[k]["auroc_mean"] > lf_au
    ]

    lines.append(
        f"- **Highest 8-Server Mean AUROC:** {best_name} at **{best_au:.1f} ± {summary[best_key]['auroc_std']:.1f}%**."
    )
    if beating_hr:
        beat_str = "; ".join(f"{name} ({m:.1f} ± {sd:.1f}%)" for name, m, sd in beating_hr)
        lines.append(
            f"- **Baselines outperforming PPC-Gauge-Triad v0.7 High-Recall ({hr_au:.1f}%):** {beat_str}. "
            f"Reported as-is without post-hoc preset changes."
        )
    else:
        lines.append(
            f"- **vs. PPC-Gauge-Triad v0.7 High-Recall ({hr_au:.1f} ± {summary['v07_high_recall']['auroc_std']:.1f}%):** "
            f"No tested baseline exceeded `v0.7 High-Recall` on 8-server mean AUROC, though `v0.7 High-Recall` carries higher FAR (`{summary['v07_high_recall']['far_dual_mean']:.1f}%` dual-gate)."
        )
    if beating_lf:
        beat_lf_str = "; ".join(f"{name} ({m:.1f} ± {sd:.1f}%)" for name, m, sd in beating_lf)
        lines.append(
            f"- **Baselines outperforming PPC-Gauge-Triad v0.7 Low-FAR ({lf_au:.1f}%):** {beat_lf_str}."
        )

    lines.extend(
        [
            "- **Memory / Footprint Context:** Batch baselines (`kNN`, `OC-SVM`, `Isolation Forest`, `PCA+Mahalanobis`) require storing training buffers, trees, support vectors, or full `w*D` covariance/SVD decompositions in host RAM (`std` Python/heap). Streaming baselines (`Half-Space Trees`, `RS-Hash`) maintain multi-tree mass tables or `100 x 4 x 1024` Count-Min Sketches + FIFO ring buffers (~hundreds of KB to MBs). PPC-Gauge-Triad trades off raw offline modeling capacity for a fixed `< 16 KB` `#![no_std]` static struct footprint (`0 B` heap, `0` matrix inversions).",
            "- **Deep Learning Baselines (USAD / TranAD) — `PENDING / Future Work`:** PyTorch is not installed in this lightweight offline verification environment (`numpy` + `scikit-learn` only). Deep temporal baselines (`USAD`, `TranAD`) are therefore explicitly marked as **future work** and are not estimated or copied from external papers whose preprocessing/point-adjust protocols differ.",
            "",
            "## Per-Server AUROC (%) Breakdown (No Point-Adjust)",
            "",
            "| Server | v0.7 High-Rec | v0.7 Low-FAR | HS-Trees | RS-Hash | PCA+Mahal (32-D) | PCA+Mahal (Full-D) | IsoForest | kNN | OC-SVM |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for sname in servers:
        row = per_srv[sname]
        lines.append(
            f"| {sname} | "
            f"{row['v07_high_recall']['auroc']:.1f} | "
            f"{row['v07_low_far']['auroc']:.1f} | "
            f"{row['hs_trees']['auroc']:.1f} | "
            f"{row['rs_hash']['auroc']:.1f} | "
            f"{row['pca_mahalanobis_32d']['auroc']:.1f} | "
            f"{row['pca_mahalanobis_fulld']['auroc']:.1f} | "
            f"{row['raw_isoforest']['auroc']:.1f} | "
            f"{row['raw_knn']['auroc']:.1f} | "
            f"{row['raw_ocsvm']['auroc']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Server Dual-Gate FAR (%) Breakdown (`alpha = 0.05` Split-Conformal)",
            "",
            "| Server | v0.7 High-Rec | v0.7 Low-FAR | HS-Trees | RS-Hash | PCA+Mahal (32-D) | PCA+Mahal (Full-D) | IsoForest | kNN | OC-SVM |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for sname in servers:
        row = per_srv[sname]
        lines.append(
            f"| {sname} | "
            f"{row['v07_high_recall']['far_dual']:.1f} | "
            f"{row['v07_low_far']['far_dual']:.1f} | "
            f"{row['hs_trees']['far_dual']:.1f} | "
            f"{row['rs_hash']['far_dual']:.1f} | "
            f"{row['pca_mahalanobis_32d']['far_dual']:.1f} | "
            f"{row['pca_mahalanobis_fulld']['far_dual']:.1f} | "
            f"{row['raw_isoforest']['far_dual']:.1f} | "
            f"{row['raw_knn']['far_dual']:.1f} | "
            f"{row['raw_ocsvm']['far_dual']:.1f} |"
        )

    lines.append("")
    lines.append(
        "*Reproduce directly from local offline cache via `python benchmarks/run_benchmarks.py --baselines --offline`.*"
    )
    lines.append("")
    return "\n".join(lines)


def run_baseline_comparison(
    cache_raw: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    srv_res: Dict[str, Any] | None = None,
    write_md_path: Path | None = REPO_ROOT / "benchmarks" / "results_baselines.md",
) -> Dict[str, Any]:
    """Run fair streaming + batch baselines on all 8 servers and write results_baselines.md."""
    if srv_res is None:
        srv_res = run_8_server_benchmark(cache_raw)

    print("\n=== FAIR BASELINE COMPARISON (HS-Trees, RS-Hash, PCA+Mahalanobis, Batch) ===")
    print(f"Seeds for randomized methods: {list(DEFAULT_SEEDS)} | Protocol: window-level w=8, NO point-adjust")

    baseline_keys = list(BASELINE_DISPLAY_NAMES.keys())
    combined_per_server: Dict[str, Dict[str, Dict[str, float]]] = {}

    for display_name, _, _ in BENCHMARK_SERVERS:
        tr_r, te_r, yl_r = cache_raw[display_name]
        b_res = eval_all_baselines_on_server(tr_r, te_r, yl_r, seeds=DEFAULT_SEEDS)
        combined_per_server[display_name] = {
            "v05": srv_res["per_server"][display_name]["v05"],
            "v07_high_recall": srv_res["per_server"][display_name]["high_recall"],
            "v07_low_far": srv_res["per_server"][display_name]["low_far"],
            **b_res,
        }
        print(
            f"  {display_name:<16} | "
            f"HST={b_res['hs_trees']['auroc']:5.1f}% | "
            f"RSH={b_res['rs_hash']['auroc']:5.1f}% | "
            f"PCA+M32={b_res['pca_mahalanobis_32d']['auroc']:5.1f}% | "
            f"PCA+MFull={b_res['pca_mahalanobis_fulld']['auroc']:5.1f}% | "
            f"IsoF={b_res['raw_isoforest']['auroc']:5.1f}% | "
            f"kNN={b_res['raw_knn']['auroc']:5.1f}% | "
            f"OCSVM={b_res['raw_ocsvm']['auroc']:5.1f}%"
        )

    all_keys = ["v07_high_recall", "v07_low_far", "v05"] + baseline_keys
    combined_summary = summarize_across_servers(combined_per_server, all_keys)

    print("\n  8-SERVER MEAN +- STD SUMMARY (NO POINT-ADJUST):")
    print(f"  {'Method':<26} | {'AUROC (%)':>14} | {'Raw FAR (%)':>14} | {'Dual FAR (%)':>14} | {'Dual Rec (%)':>14}")
    print("  " + "-" * 92)
    for k in all_keys:
        s = combined_summary[k]
        print(
            f"  {k:<26} | "
            f"{s['auroc_mean']:5.1f} +- {s['auroc_std']:4.1f}% | "
            f"{s['far_raw_mean']:5.1f} +- {s['far_raw_std']:4.1f}% | "
            f"{s['far_dual_mean']:5.1f} +- {s['far_dual_std']:4.1f}% | "
            f"{s['recall_dual_mean']:5.1f} +- {s['recall_dual_std']:4.1f}%"
        )
    print("  Deep baselines (USAD / TranAD): PENDING / Future Work (PyTorch not installed offline).")

    result = {
        "seeds": list(DEFAULT_SEEDS),
        "protocol": "non-overlapping w=8 window-level evaluation without point-adjust",
        "combined_per_server": combined_per_server,
        "combined_summary": combined_summary,
        "deep_baselines": "PENDING / Future Work (PyTorch not installed offline)",
    }

    if write_md_path is not None:
        md_text = render_baselines_markdown(result, srv_res)
        write_md_path.write_text(md_text, encoding="utf-8")
        print(f"  Wrote baseline comparison report -> {write_md_path}")

    return result


def run_warmup_curve(
    cache_raw: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    preset: OperationalPreset = PRESET_LOW_FAR,
) -> Dict[str, Any]:
    """Test 3: Pure Zero-Shot vs Zero-Touch On-Device Auto-Warmup (100/200/400w)."""
    print(
        f"\n=== TEST 3: CROSS-DEVICE ZERO-SHOT vs AUTO-WARMUP CURVE (preset={preset.name}) ==="
    )
    tr11, te11, yl11 = cache_raw["SMD machine-1-1"]
    wtr_src, _, _, (mu_s, sd_s, pca_s, rmu_s, rsd_s) = to_windows_v6(
        tr11, te11, yl11, use_pca_resid=True
    )
    kf_s = int(len(wtr_src) * 0.75)
    e_src = RustTriadV061(
        8, 4, k_max=3, lam0=5e-3, alpha_min=0.01, e_alpha=preset.e_alpha, dc_alpha=preset.dc_alpha
    )
    e_src.run(wtr_src[:kf_s], learn=True)
    sh_cs, en_cs, lv_cs, _ = e_src.run(wtr_src[kf_s:], learn=False)

    warm_counts = [0, 100, 200, 400]
    targets = ["SMD machine-1-2", "SMD machine-1-6", "SMD machine-2-1"]
    print(
        f"{'Unseen Target':<18}"
        + "".join(f"{('ZeroShot' if n == 0 else f'AutoWarm {n}w'):>14}" for n in warm_counts)
    )
    print("-" * (18 + 14 * len(warm_counts)))

    cols: Dict[int, List[float]] = {n: [] for n in warm_counts}
    per_target: Dict[str, Dict[str, float]] = {}
    zero_shot_diagnostics: Dict[str, Dict[str, float]] = {}

    for tgt in targets:
        tr_t, te_t, yl_t = cache_raw[tgt]
        row = f"{tgt:<18}"
        per_target[tgt] = {}
        for nw in warm_counts:
            if nw == 0:
                z_std = (te_t - mu_s) / sd_s
                p_t = pca_s.transform(z_std)
                r_t = np.linalg.norm(z_std - pca_s.inverse_transform(p_t), axis=1, keepdims=True)
                z_t = np.hstack([p_t, (r_t - rmu_s) / rsd_s])
                nte = len(z_t) // 8
                wt = z_t[: nte * 8].reshape(nte, 8, 4)
                yt = (yl_t[: nte * 8].reshape(nte, 8).max(axis=1) > 0).astype(np.int32)
                ee = RustTriadV061(
                    8,
                    4,
                    k_max=3,
                    lam0=5e-3,
                    alpha_min=0.01,
                    e_alpha=preset.e_alpha,
                    dc_alpha=preset.dc_alpha,
                )
                ee.run(wtr_src[:kf_s], learn=True)
                sht, ent, lvt, _ = ee.run(
                    wt, learn=False, adapt_env=True, huber_clip=preset.huber_clip
                )
                sc_zs = triad_norm_score(sh_cs, en_cs, lv_cs, sht, ent, lvt)
                au = compute_auroc(yt, sc_zs) * 100.0
                zero_shot_diagnostics[tgt] = {
                    "auroc": float(au),
                    "inverted_auroc": float(100.0 - au),
                    "normal_median_score": float(np.median(sc_zs[yt == 0])),
                    "fault_median_score": float(np.median(sc_zs[yt == 1])),
                    "normal_mean_score": float(np.mean(sc_zs[yt == 0])),
                    "fault_mean_score": float(np.mean(sc_zs[yt == 1])),
                }
            else:
                wtr_t, wte_t, yte_t, _ = to_windows_v6(
                    tr_t[: nw * 8], te_t, yl_t, max_tr_win=nw, use_pca_resid=True
                )
                kf = int(len(wtr_t) * 0.75)
                ee = RustTriadV061(
                    8,
                    4,
                    k_max=3,
                    lam0=5e-3,
                    alpha_min=0.01,
                    e_alpha=preset.e_alpha,
                    dc_alpha=preset.dc_alpha,
                )
                ee.run(wtr_t[:kf], learn=True)
                shc, enc, lvc, _ = ee.run(wtr_t[kf:], learn=False)
                thr_sh = conformal_threshold(shc, 0.05)
                sht, ent, lvt, _ = ee.run(
                    wte_t,
                    learn=False,
                    shape_gate=thr_sh,
                    adapt_env=True,
                    huber_clip=preset.huber_clip,
                )
                au = compute_auroc(yte_t, triad_norm_score(shc, enc, lvc, sht, ent, lvt)) * 100.0
            cols[nw].append(au)
            per_target[tgt][f"{nw}w"] = float(au)
            row += f"{au:13.1f}%"
        print(row)

    print("-" * (18 + 14 * len(warm_counts)))
    mean_row = {f"{n}w": float(np.mean(cols[n])) for n in warm_counts}
    print(f"{'MEAN AUROC':<18}" + "".join(f"{mean_row[f'{n}w']:13.1f}%" for n in warm_counts))
    print("\n  Zero-Shot (0w) Score-Direction Inversion Diagnostic (normal vs fault scores):")
    for tgt, diag in zero_shot_diagnostics.items():
        print(
            f"    {tgt:<16}: AUROC={diag['auroc']:4.1f}% (100-AUROC={diag['inverted_auroc']:4.1f}%) | "
            f"normal med={diag['normal_median_score']:.2f} vs fault med={diag['fault_median_score']:.2f}"
        )

    # v0.7 On-Chip AutoCalEngineF32 verification (Warmup -> Calibrate -> Monitor inside #![no_std] Rust)
    print(
        f"\n=== TEST 3B (v0.7 ON-CHIP AutoCalEngineF32): ZERO-TOUCH AUTO-WARMUP (100/200/400w) ==="
    )
    onchip_counts = [100, 200, 400]
    print(
        f"{'Unseen Target':<18}"
        + "".join(f"{f'OnChip {n}w':>14}" for n in onchip_counts)
    )
    print("-" * (18 + 14 * len(onchip_counts)))
    oc_cols: Dict[int, List[float]] = {n: [] for n in onchip_counts}
    oc_per_target: Dict[str, Dict[str, float]] = {}
    for tgt in targets:
        tr_t, te_t, yl_t = cache_raw[tgt]
        row = f"{tgt:<18}"
        oc_per_target[tgt] = {}
        for nw in onchip_counts:
            wtr_t, wte_t, yte_t, _ = to_windows_v6(
                tr_t[: nw * 8], te_t, yl_t, max_tr_win=nw, use_pca_resid=True
            )
            n_warm = int(len(wtr_t) * 0.75)
            n_cal = len(wtr_t) - n_warm
            stream = np.concatenate([wtr_t, wte_t], axis=0)
            ac = RustAutoCalV07(
                w=8,
                d=4,
                k_max=3,
                lam0=5e-3,
                alpha_min=0.01,
                e_alpha=preset.e_alpha,
                dc_alpha=preset.dc_alpha,
                n_warmup=n_warm,
                n_cal=n_cal,
                alpha=0.05,
                adapt_env=preset.adapt_env,
                sev_mult=preset.sev_mult,
                dc_relock=preset.dc_relock,
                huber_clip=preset.huber_clip,
                use_dual=preset.use_dual,
            )
            states, _, _, _, s_comb, _ = ac.run(stream)
            assert int(states[0]) == 0 and int(states[n_warm]) == 1 and int(states[len(wtr_t)]) == 2
            au = compute_auroc(yte_t, s_comb[len(wtr_t) :]) * 100.0
            oc_cols[nw].append(au)
            oc_per_target[tgt][f"{nw}w"] = float(au)
            row += f"{au:13.1f}%"
        print(row)
    print("-" * (18 + 14 * len(onchip_counts)))
    oc_mean = {f"{n}w": float(np.mean(oc_cols[n])) for n in onchip_counts}
    print(f"{'MEAN AUROC':<18}" + "".join(f"{oc_mean[f'{n}w']:13.1f}%" for n in onchip_counts))

    return {
        "preset": preset.name,
        "per_target": per_target,
        "mean": mean_row,
        "zero_shot_diagnostics": zero_shot_diagnostics,
        "v07_onchip_per_target": oc_per_target,
        "v07_onchip_mean": oc_mean,
    }


def run_host_latency_benchmark(n_batch: int = 200, n_repeats: int = 500) -> Dict[str, Any]:
    """Measure native Rust host CPU throughput/latency (NOT Level 3 MCU silicon)."""
    print(f"\n=== HOST CPU LATENCY BENCHMARK ({platform.machine()} / {platform.system()}) ===")
    rng = np.random.default_rng(42)
    x_warm = rng.standard_normal((200, 8, 4)).astype(np.float32)
    x_batch = np.ascontiguousarray(rng.standard_normal((n_batch, 8, 4)), dtype=np.float32)

    eng = RustTriadV061(8, 4, k_max=3, lam0=5e-3, alpha_min=0.01, e_alpha=0.008, dc_alpha=0.004)
    eng.run(x_warm, learn=True)

    # Warm JIT/caches
    eng.run(x_batch, learn=False, adapt_env=True, huber_clip=1.5)

    def measure(learn: bool) -> Dict[str, float]:
        per_win_us = []
        for _ in range(n_repeats):
            t0 = time.perf_counter_ns()
            eng.run(x_batch, learn=learn, adapt_env=True, huber_clip=1.5)
            dt_ns = time.perf_counter_ns() - t0
            per_win_us.append((dt_ns / 1000.0) / n_batch)
        arr = np.asarray(per_win_us, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        return {"p50_us": p50, "p95_us": p95, "p99_us": p99, "win_per_sec": float(1e6 / p50)}

    score_only = measure(learn=False)
    live_learn = measure(learn=True)
    print(
        f"  Score-Only (8x4)      : p50={score_only['p50_us']:.3f} us | "
        f"p95={score_only['p95_us']:.3f} us | p99={score_only['p99_us']:.3f} us "
        f"({score_only['win_per_sec']/1e3:.1f}k win/s)"
    )
    print(
        f"  Live Learn+Score (8x4): p50={live_learn['p50_us']:.3f} us | "
        f"p95={live_learn['p95_us']:.3f} us | p99={live_learn['p99_us']:.3f} us "
        f"({live_learn['win_per_sec']/1e3:.1f}k win/s)"
    )
    print("  Note: Host CPU benchmark only; Level 3 MCU silicon measurements remain pending.")
    return {"score_only": score_only, "live_learn": live_learn}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local benchmark & verification runner for PPC-Gauge-Triad v0.6.1 (no Colab)."
    )
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Dataset cache folder")
    p.add_argument("--offline", action="store_true", help="Never download; fail if cache missing")
    p.add_argument("--force-refresh", action="store_true", help="Re-download SMD/PSM datasets")
    p.add_argument("--rebuild", action="store_true", help="Force `cargo build --release`")
    p.add_argument("--cache-only", action="store_true", help="Populate/verify SMD & PSM cache only")
    p.add_argument("--l1-parity", action="store_true", help="Run L1.8 Python vs Rust cdylib parity")
    p.add_argument("--servers", action="store_true", help="Run 8-server SMD/PSM ablation")
    p.add_argument("--baselines", action="store_true", help="Run fair streaming/batch baselines and write results_baselines.md")
    p.add_argument("--warmup-curve", action="store_true", help="Run Test 3 auto-warmup curve")
    p.add_argument("--latency", action="store_true", help="Run host CPU latency benchmark")
    p.add_argument("--all", action="store_true", help="Run all benchmarks and checks")
    p.add_argument("--save-json", type=Path, default=None, help="Save results summary to JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_any = any(
        [args.cache_only, args.l1_parity, args.servers, args.baselines, args.warmup_curve, args.latency, args.all]
    )
    if not run_any:
        # Sensible default: run L1.8 parity + 8-server + warmup curve
        args.l1_parity = True
        args.servers = True
        args.warmup_curve = True

    lib = load_rust_lib(rebuild=args.rebuild)
    sram_bytes = int(lib.ppc_triad_engine_bytes())
    autocal_sram_bytes = int(lib.ppc_autocal_engine_bytes())
    print(
        f"PPC-Gauge-Triad v0.7.1 native library loaded | "
        f"Core SRAM = {sram_bytes} B ({sram_bytes / 1024.0:.2f} KB) | "
        f"AutoCalEngine SRAM = {autocal_sram_bytes} B ({autocal_sram_bytes / 1024.0:.2f} KB)"
    )

    report: Dict[str, Any] = {
        "crate_version": "0.7.1",
        "sram_bytes": sram_bytes,
        "autocal_sram_bytes": autocal_sram_bytes,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    if args.l1_parity or args.all:
        report["l1_8_parity"] = run_l1_8_parity()

    need_data = args.cache_only or args.servers or args.baselines or args.warmup_curve or args.all
    cache_raw = None
    if need_data:
        t0 = time.perf_counter()
        cache_raw = populate_all_caches(
            cache_dir=args.cache_dir,
            offline=args.offline,
            force_refresh=args.force_refresh,
            verbose=True,
        )
        print(f"[cache] Loaded all {len(cache_raw)} servers from {args.cache_dir} in {time.perf_counter() - t0:.2f}s")

    srv_res = None
    if args.servers or args.baselines or args.all:
        assert cache_raw is not None
        srv_res = run_8_server_benchmark(cache_raw)
        report["servers_8"] = srv_res

    if args.baselines or args.all:
        assert cache_raw is not None
        report["baselines"] = run_baseline_comparison(cache_raw, srv_res=srv_res)

    if args.warmup_curve or args.all:
        assert cache_raw is not None
        report["warmup_curve"] = run_warmup_curve(cache_raw, preset=PRESET_LOW_FAR)

    if args.latency or args.all:
        report["host_latency"] = run_host_latency_benchmark()

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report -> {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
