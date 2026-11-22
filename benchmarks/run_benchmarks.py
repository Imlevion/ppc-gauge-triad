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
) -> Tuple[float, float, float]:
    """Evaluate a locked a priori preset chronologically on a server telemetry stream."""
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
    al = dual_gate_scalar(sc, thr_sc, preset.sev_mult) if preset.use_dual else (sc > thr_sc)
    auroc = compute_auroc(yte, sc) * 100.0
    far = float(np.mean(al[yte == 0])) * 100.0
    rec = float(np.mean(al[yte == 1])) * 100.0
    return auroc, far, rec


def run_8_server_benchmark(
    cache_raw: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Run 8-server chronological replay across locked presets (v0.5, High-Recall, Low-FAR)."""
    print("\n=== 8-SERVER REAL-WORLD CHRONOLOGICAL REPLAY (CACHED SMD + PSM) ===")
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

    acc: Dict[str, List[Tuple[float, float, float]]] = {"v05": [], "high_recall": [], "low_far": []}
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
            "v05": {"auroc": r_v5[0], "far": r_v5[1], "recall": r_v5[2]},
            "high_recall": {"auroc": r_hr[0], "far": r_hr[1], "recall": r_hr[2]},
            "low_far": {"auroc": r_lf[0], "far": r_lf[1], "recall": r_lf[2]},
        }
        print(
            f"{display_name:<16} | "
            f"{r_v5[0]:5.1f}% {r_v5[1]:5.1f}% {r_v5[2]:5.1f}% | "
            f"{r_hr[0]:5.1f}% {r_hr[1]:5.1f}% {r_hr[2]:5.1f}% | "
            f"{r_lf[0]:5.1f}% {r_lf[1]:5.1f}% {r_lf[2]:5.1f}%"
        )

    print("-" * len(subhdr))
    means: Dict[str, Dict[str, float]] = {}
    row_parts = []
    for key in ("v05", "high_recall", "low_far"):
        arr = np.asarray(acc[key], dtype=np.float64)
        m_au, m_far, m_rec = arr.mean(axis=0)
        means[key] = {"auroc": float(m_au), "far": float(m_far), "recall": float(m_rec)}
        row_parts.append(f"{m_au:5.1f}% {m_far:5.1f}% {m_rec:5.1f}%")
    print(f"{'8-SERVER MEAN':<16} | " + " | ".join(row_parts))

    return {"per_server": per_server, "mean": means}


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
                au = compute_auroc(yt, triad_norm_score(sh_cs, en_cs, lv_cs, sht, ent, lvt)) * 100.0
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
    p.add_argument("--warmup-curve", action="store_true", help="Run Test 3 auto-warmup curve")
    p.add_argument("--latency", action="store_true", help="Run host CPU latency benchmark")
    p.add_argument("--all", action="store_true", help="Run all benchmarks and checks")
    p.add_argument("--save-json", type=Path, default=None, help="Save results summary to JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_any = any([args.cache_only, args.l1_parity, args.servers, args.warmup_curve, args.latency, args.all])
    if not run_any:
        # Sensible default: run L1.8 parity + 8-server + warmup curve
        args.l1_parity = True
        args.servers = True
        args.warmup_curve = True

    lib = load_rust_lib(rebuild=args.rebuild)
    sram_bytes = int(lib.ppc_triad_engine_bytes())
    autocal_sram_bytes = int(lib.ppc_autocal_engine_bytes())
    print(
        f"PPC-Gauge-Triad v0.7.0 native library loaded | "
        f"Core SRAM = {sram_bytes} B ({sram_bytes / 1024.0:.2f} KB) | "
        f"AutoCalEngine SRAM = {autocal_sram_bytes} B ({autocal_sram_bytes / 1024.0:.2f} KB)"
    )

    report: Dict[str, Any] = {
        "crate_version": "0.7.0",
        "sram_bytes": sram_bytes,
        "autocal_sram_bytes": autocal_sram_bytes,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    if args.l1_parity or args.all:
        report["l1_8_parity"] = run_l1_8_parity()

    need_data = args.cache_only or args.servers or args.warmup_curve or args.all
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

    if args.servers or args.all:
        assert cache_raw is not None
        report["servers_8"] = run_8_server_benchmark(cache_raw)

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
