//! `ppc-bench` — Rust-first CLI for PPC-Gauge-Triad v0.7.1.
//!
//! Executes core 8-server SMD+PSM chronological replay, Test 3 / Test 3B
//! on-chip `AutoCalEngineF32` warmup curves, native Rust streaming & window
//! baselines (`Half-Space Trees`, `RS-Hash`, `PCA+Mahalanobis`, `kNN`),
//! native host CPU latency profiling, and Level 2.5 ARM Cortex-M4F `.elf`
//! footprint inspection directly in Rust without Python runtime overhead.

mod baselines;
mod elf;
mod npz;
mod pca;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ppc_gauge_triad_rs::{
    ppc_autocal_engine_bytes, ppc_triad_engine_bytes, AutoCalEngineF32, GaugeTriadEngineF32,
};

use crate::baselines::{
    compute_auroc, conformal_threshold, dual_gate_scalar, eval_rust_baselines_on_server,
    median_f64, triad_norm_score, AggregatedMetrics, RustBaselineServerResult, SplitMix64,
    DEFAULT_SEEDS,
};
use crate::elf::inspect_elf32;
use crate::npz::{load_server_npz, ServerDataset};
use crate::pca::{project_zero_shot, to_windows_v6};

const W: usize = 8;
const D: usize = 4;

const SERVERS: [(&str, &str); 8] = [
    ("SMD machine-1-1", "smd_machine-1-1.npz"),
    ("SMD machine-1-6", "smd_machine-1-6.npz"),
    ("SMD machine-2-1", "smd_machine-2-1.npz"),
    ("PSM eBay Server", "psm_ebay.npz"),
    ("SMD machine-1-2", "smd_machine-1-2.npz"),
    ("SMD machine-1-3", "smd_machine-1-3.npz"),
    ("SMD machine-2-2", "smd_machine-2-2.npz"),
    ("SMD machine-3-1", "smd_machine-3-1.npz"),
];

#[derive(Debug, Clone, Copy)]
struct Preset {
    name: &'static str,
    e_alpha: f32,
    dc_alpha: f32,
    huber_clip: f32,
    dc_relock: u8,
    use_pca_resid: bool,
    adapt_env: bool,
    use_dual: bool,
    sev_mult: f32,
}

const PRESET_V05: Preset = Preset {
    name: "v05",
    e_alpha: 0.10,
    dc_alpha: 0.05,
    huber_clip: 0.0,
    dc_relock: 0,
    use_pca_resid: false,
    adapt_env: false,
    use_dual: false,
    sev_mult: 2.5,
};

const PRESET_HIGH_RECALL: Preset = Preset {
    name: "high_recall",
    e_alpha: 0.002,
    dc_alpha: 0.001,
    huber_clip: 1.0,
    dc_relock: 0,
    use_pca_resid: true,
    adapt_env: true,
    use_dual: true,
    sev_mult: 2.5,
};

const PRESET_LOW_FAR: Preset = Preset {
    name: "low_far",
    e_alpha: 0.008,
    dc_alpha: 0.004,
    huber_clip: 1.5,
    dc_relock: 0,
    use_pca_resid: true,
    adapt_env: true,
    use_dual: true,
    sev_mult: 2.5,
};

fn make_dct_f32(w: usize) -> Vec<f32> {
    let wm1 = w - 1;
    let mut b = vec![0.0_f64; w * wm1];
    for k in 0..wm1 {
        let freq = (k + 1) as f64;
        let mut norm_sq = 0.0_f64;
        for t in 0..w {
            let v = (std::f64::consts::PI * (t as f64 + 0.5) * freq / (w as f64)).cos();
            b[t * wm1 + k] = v;
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        for t in 0..w {
            b[t * wm1 + k] /= norm;
        }
    }
    b.into_iter().map(|v| v as f32).collect()
}

#[derive(Debug, Clone, Copy)]
struct PresetServerMetrics {
    auroc: f64,
    far: f64,
    recall: f64,
    far_raw: f64,
    recall_raw: f64,
    far_dual: f64,
    recall_dual: f64,
}

fn run_engine_stream(
    eng: &mut GaugeTriadEngineF32,
    windows_f64: &[f64],
    n_win: usize,
    learn: bool,
    shape_gate: f32,
    adapt_env: bool,
    dc_relock: u8,
    huber_clip: f32,
    sev_mult: f32,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let stride = W * D;
    let mut sh = Vec::with_capacity(n_win);
    let mut en = Vec::with_capacity(n_win);
    let mut lv = Vec::with_capacity(n_win);
    let mut buf = [0.0_f32; W * D];
    for i in 0..n_win {
        let src = &windows_f64[i * stride..(i + 1) * stride];
        for j in 0..stride {
            buf[j] = src[j] as f32;
        }
        let (s, e, l, _) = eng.step_v61(
            &buf,
            u8::from(learn),
            shape_gate,
            u8::from(adapt_env),
            10.0,
            10.0,
            sev_mult,
            dc_relock,
            huber_clip,
        );
        sh.push(s as f64);
        en.push(e as f64);
        lv.push(l as f64);
    }
    (sh, en, lv)
}

fn eval_preset_on_server(ds: &ServerDataset, preset: Preset, dct: &[f32]) -> PresetServerMetrics {
    let win = to_windows_v6(
        &ds.train,
        &ds.test,
        &ds.test_label,
        W,
        D,
        2000,
        preset.use_pca_resid,
        None,
    );
    let kf = ((win.n_tr as f64) * 0.75) as usize;
    let n_cal = win.n_tr - kf;
    let stride = W * D;

    let mut eng = GaugeTriadEngineF32::zeroed();
    eng.init(
        W,
        D,
        3,
        5e-3,
        0.01,
        preset.e_alpha,
        preset.dc_alpha,
        dct,
    );

    run_engine_stream(&mut eng, &win.w_tr[..kf * stride], kf, true, 1e20, false, 0, 1.5, 2.5);
    let (shc, enc, lvc) = run_engine_stream(
        &mut eng,
        &win.w_tr[kf * stride..],
        n_cal,
        false,
        1e20,
        false,
        0,
        1.5,
        2.5,
    );
    let thr_sh = conformal_threshold(&shc, 0.05) as f32;
    let cal_comb = triad_norm_score(&shc, &enc, &lvc, &shc, &enc, &lvc);
    let thr_sc = conformal_threshold(&cal_comb, 0.05);

    let (sht, ent, lvt) = run_engine_stream(
        &mut eng,
        &win.w_te,
        win.n_te,
        false,
        thr_sh,
        preset.adapt_env,
        preset.dc_relock,
        preset.huber_clip,
        preset.sev_mult,
    );
    let sc = triad_norm_score(&shc, &enc, &lvc, &sht, &ent, &lvt);
    let al_raw: Vec<bool> = sc.iter().map(|&v| v > thr_sc).collect();
    let al_dual = dual_gate_scalar(&sc, thr_sc, preset.sev_mult as f64);

    let mut n_neg = 0usize;
    let mut n_pos = 0usize;
    let mut fp_raw = 0usize;
    let mut tp_raw = 0usize;
    let mut fp_dual = 0usize;
    let mut tp_dual = 0usize;
    for i in 0..win.y_te.len() {
        if win.y_te[i] == 0 {
            n_neg += 1;
            if al_raw[i] { fp_raw += 1; }
            if al_dual[i] { fp_dual += 1; }
        } else {
            n_pos += 1;
            if al_raw[i] { tp_raw += 1; }
            if al_dual[i] { tp_dual += 1; }
        }
    }
    let far_raw = (fp_raw as f64 / n_neg.max(1) as f64) * 100.0;
    let recall_raw = (tp_raw as f64 / n_pos.max(1) as f64) * 100.0;
    let far_dual = (fp_dual as f64 / n_neg.max(1) as f64) * 100.0;
    let recall_dual = (tp_dual as f64 / n_pos.max(1) as f64) * 100.0;
    let (far, recall) = if preset.use_dual {
        (far_dual, recall_dual)
    } else {
        (far_raw, recall_raw)
    };
    PresetServerMetrics {
        auroc: compute_auroc(&win.y_te, &sc) * 100.0,
        far,
        recall,
        far_raw,
        recall_raw,
        far_dual,
        recall_dual,
    }
}

fn mean_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len().max(1) as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn load_all_servers(cache_dir: &Path) -> Result<Vec<(&'static str, ServerDataset)>, String> {
    let t0 = Instant::now();
    let mut loaded = Vec::with_capacity(SERVERS.len());
    for &(name, fname) in &SERVERS {
        let p = cache_dir.join(fname);
        if !p.exists() {
            return Err(format!(
                "Missing offline cache file: {}. Run `python benchmarks/run_benchmarks.py --cache-only` once to populate it.",
                p.display()
            ));
        }
        loaded.push((name, load_server_npz(&p)?));
    }
    println!(
        "[rust-cache] Loaded all {} servers from {} in {:.2}s",
        loaded.len(),
        cache_dir.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(loaded)
}

struct ServersReport {
    per_server: Vec<(&'static str, PresetServerMetrics, PresetServerMetrics, PresetServerMetrics)>,
}

fn run_servers(datasets: &[(&'static str, ServerDataset)], dct: &[f32]) -> ServersReport {
    println!("\n=== 8-SERVER CHRONOLOGICAL REPLAY (NATIVE RUST, NO POINT-ADJUST) ===");
    println!(
        "{:<16} | {:^20} | {:^20} | {:^20}",
        "Server", "v0.5 (NoAdapt)", "High-Recall (0.002)", "Low-FAR (0.008)"
    );
    println!(
        "{:<16} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}",
        "", "AUROC", "FAR", "Rec", "AUROC", "FAR", "Rec", "AUROC", "FAR", "Rec"
    );
    println!("{}", "-".repeat(85));

    let mut per_server = Vec::with_capacity(datasets.len());
    for (name, ds) in datasets {
        let m_v5 = eval_preset_on_server(ds, PRESET_V05, dct);
        let m_hr = eval_preset_on_server(ds, PRESET_HIGH_RECALL, dct);
        let m_lf = eval_preset_on_server(ds, PRESET_LOW_FAR, dct);
        println!(
            "{:<16} | {:5.1}% {:5.1}% {:5.1}% | {:5.1}% {:5.1}% {:5.1}% | {:5.1}% {:5.1}% {:5.1}%",
            name, m_v5.auroc, m_v5.far, m_v5.recall, m_hr.auroc, m_hr.far, m_hr.recall, m_lf.auroc, m_lf.far, m_lf.recall
        );
        per_server.push((*name, m_v5, m_hr, m_lf));
    }
    println!("{}", "-".repeat(85));

    let stats = |sel: fn(&( &'static str, PresetServerMetrics, PresetServerMetrics, PresetServerMetrics)) -> PresetServerMetrics| {
        let au: Vec<f64> = per_server.iter().map(|r| sel(r).auroc).collect();
        let far: Vec<f64> = per_server.iter().map(|r| sel(r).far).collect();
        let rec: Vec<f64> = per_server.iter().map(|r| sel(r).recall).collect();
        (mean_std(&au), mean_std(&far), mean_std(&rec))
    };
    let (v5_au, v5_f, v5_r) = stats(|r| r.1);
    let (hr_au, hr_f, hr_r) = stats(|r| r.2);
    let (lf_au, lf_f, lf_r) = stats(|r| r.3);
    println!(
        "{:<16} | {:5.1}% {:5.1}% {:5.1}% | {:5.1}% {:5.1}% {:5.1}% | {:5.1}% {:5.1}% {:5.1}%",
        "8-SERVER MEAN", v5_au.0, v5_f.0, v5_r.0, hr_au.0, hr_f.0, hr_r.0, lf_au.0, lf_f.0, lf_r.0
    );
    println!(
        "{:<16} | {:5.1}% {:5.1}% {:5.1}% | {:5.1}% {:5.1}% {:5.1}% | {:5.1}% {:5.1}% {:5.1}%",
        "8-SERVER STD", v5_au.1, v5_f.1, v5_r.1, hr_au.1, hr_f.1, hr_r.1, lf_au.1, lf_f.1, lf_r.1
    );

    ServersReport { per_server }
}

struct BaselinesReport {
    per_server: Vec<(&'static str, RustBaselineServerResult)>,
}

fn run_baselines(datasets: &[(&'static str, ServerDataset)]) -> BaselinesReport {
    let t0 = Instant::now();
    println!("\n=== NATIVE RUST BASELINES (HS-Trees, RS-Hash, PCA+Mahalanobis, kNN) ===");
    println!("Seeds: {:?} | Protocol: non-overlapping w=8, window-level, NO point-adjust", DEFAULT_SEEDS);
    let mut per_server = Vec::with_capacity(datasets.len());
    for (name, ds) in datasets {
        let res = eval_rust_baselines_on_server(ds);
        println!(
            "  {:<16} | HST={:5.1}% | RSH={:5.1}% | PCA+M32={:5.1}% | PCA+MFull={:5.1}% | kNN={:5.1}%",
            name,
            res.hs_trees.auroc,
            res.rs_hash.auroc,
            res.pca_mahalanobis_32d.auroc,
            res.pca_mahalanobis_fulld.auroc,
            res.raw_knn.auroc
        );
        per_server.push((*name, res));
    }

    println!("\n  8-SERVER MEAN +- STD (NATIVE RUST, completed in {:.2}s):", t0.elapsed().as_secs_f64());
    let rows: [(&str, fn(&RustBaselineServerResult) -> AggregatedMetrics); 5] = [
        ("hs_trees", |r| r.hs_trees),
        ("rs_hash", |r| r.rs_hash),
        ("pca_mahalanobis_32d", |r| r.pca_mahalanobis_32d),
        ("pca_mahalanobis_fulld", |r| r.pca_mahalanobis_fulld),
        ("raw_knn", |r| r.raw_knn),
    ];
    for (label, getter) in rows {
        let au: Vec<f64> = per_server.iter().map(|(_, r)| getter(r).auroc).collect();
        let fr: Vec<f64> = per_server.iter().map(|(_, r)| getter(r).far_raw).collect();
        let fd: Vec<f64> = per_server.iter().map(|(_, r)| getter(r).far_dual).collect();
        let rd: Vec<f64> = per_server.iter().map(|(_, r)| getter(r).recall_dual).collect();
        let (ma, sa) = mean_std(&au);
        let (mfr, sfr) = mean_std(&fr);
        let (mfd, sfd) = mean_std(&fd);
        let (mrd, srd) = mean_std(&rd);
        println!(
            "  {:<24} | AUROC {:5.1} +- {:4.1}% | RawFAR {:5.1} +- {:4.1}% | DualFAR {:5.1} +- {:4.1}% | DualRec {:5.1} +- {:4.1}%",
            label, ma, sa, mfr, sfr, mfd, sfd, mrd, srd
        );
    }

    BaselinesReport { per_server }
}

struct WarmupTargetReport {
    name: &'static str,
    zs_0w: f64,
    manual_100w: f64,
    manual_200w: f64,
    manual_400w: f64,
    onchip_100w: f64,
    onchip_200w: f64,
    onchip_400w: f64,
    zs_normal_med: f64,
    zs_fault_med: f64,
}

fn run_warmup(datasets: &[(&'static str, ServerDataset)], dct: &[f32]) -> Vec<WarmupTargetReport> {
    let preset = PRESET_LOW_FAR;
    println!(
        "\n=== TEST 3 & 3B: ZERO-SHOT vs v0.7 ON-CHIP AutoCalEngineF32 (NATIVE RUST, preset={}) ===",
        preset.name
    );
    let src_ds = &datasets
        .iter()
        .find(|(n, _)| *n == "SMD machine-1-1")
        .expect("SMD machine-1-1")
        .1;
    let src_win = to_windows_v6(&src_ds.train, &src_ds.test, &src_ds.test_label, W, D, 2000, true, None);
    let kf_s = ((src_win.n_tr as f64) * 0.75) as usize;
    let stride = W * D;

    let mut e_src = GaugeTriadEngineF32::zeroed();
    e_src.init(W, D, 3, 5e-3, 0.01, preset.e_alpha, preset.dc_alpha, dct);
    run_engine_stream(&mut e_src, &src_win.w_tr[..kf_s * stride], kf_s, true, 1e20, false, 0, 1.5, 2.5);
    let (sh_cs, en_cs, lv_cs) = run_engine_stream(
        &mut e_src,
        &src_win.w_tr[kf_s * stride..],
        src_win.n_tr - kf_s,
        false,
        1e20,
        false,
        0,
        1.5,
        2.5,
    );

    let targets = ["SMD machine-1-2", "SMD machine-1-6", "SMD machine-2-1"];
    println!("{:<18}{:>12}{:>12}{:>12}{:>12}  | {:>11}{:>11}{:>11}", "Unseen Target", "ZeroShot", "Auto 100w", "Auto 200w", "Auto 400w", "OnChip100", "OnChip200", "OnChip400");
    println!("{}", "-".repeat(102));

    let mut out = Vec::new();
    for tgt in targets {
        let ds = &datasets.iter().find(|(n, _)| *n == tgt).expect("target").1;

        // 0w Zero-Shot
        let (wt_zs, nte_zs, yt_zs) = project_zero_shot(&ds.test, &ds.test_label, &src_win.meta, W, D);
        let mut ee = GaugeTriadEngineF32::zeroed();
        ee.init(W, D, 3, 5e-3, 0.01, preset.e_alpha, preset.dc_alpha, dct);
        run_engine_stream(&mut ee, &src_win.w_tr[..kf_s * stride], kf_s, true, 1e20, false, 0, 1.5, 2.5);
        let (sht, ent, lvt) = run_engine_stream(&mut ee, &wt_zs, nte_zs, false, 1e20, true, 0, preset.huber_clip, 2.5);
        let sc_zs = triad_norm_score(&sh_cs, &en_cs, &lv_cs, &sht, &ent, &lvt);
        let zs_0w = compute_auroc(&yt_zs, &sc_zs) * 100.0;
        let norm_scores: Vec<f64> = sc_zs.iter().zip(&yt_zs).filter(|(_, &y)| y == 0).map(|(&s, _)| s).collect();
        let fault_scores: Vec<f64> = sc_zs.iter().zip(&yt_zs).filter(|(_, &y)| y == 1).map(|(&s, _)| s).collect();
        let zs_normal_med = median_f64(&norm_scores);
        let zs_fault_med = median_f64(&fault_scores);

        let mut man_vals = [0.0_f64; 3];
        let mut oc_vals = [0.0_f64; 3];
        for (idx, &nw) in [100usize, 200, 400].iter().enumerate() {
            let win_t = to_windows_v6(&ds.train, &ds.test, &ds.test_label, W, D, nw, true, Some(nw * W));
            let kf = ((win_t.n_tr as f64) * 0.75) as usize;
            let n_cal = win_t.n_tr - kf;

            // Manual split-conformal
            let mut em = GaugeTriadEngineF32::zeroed();
            em.init(W, D, 3, 5e-3, 0.01, preset.e_alpha, preset.dc_alpha, dct);
            run_engine_stream(&mut em, &win_t.w_tr[..kf * stride], kf, true, 1e20, false, 0, 1.5, 2.5);
            let (shc, enc, lvc) = run_engine_stream(&mut em, &win_t.w_tr[kf * stride..], n_cal, false, 1e20, false, 0, 1.5, 2.5);
            let thr_sh = conformal_threshold(&shc, 0.05) as f32;
            let (sht, ent, lvt) = run_engine_stream(&mut em, &win_t.w_te, win_t.n_te, false, thr_sh, true, 0, preset.huber_clip, 2.5);
            let sc_m = triad_norm_score(&shc, &enc, &lvc, &sht, &ent, &lvt);
            man_vals[idx] = compute_auroc(&win_t.y_te, &sc_m) * 100.0;

            // v0.7 On-Chip AutoCalEngineF32
            let mut ac = AutoCalEngineF32::zeroed();
            ac.init(
                W, D, 3, 5e-3, 0.01, preset.e_alpha, preset.dc_alpha, dct,
                kf as u16, n_cal as u16, 0.05,
                u8::from(preset.adapt_env), preset.sev_mult, preset.dc_relock, preset.huber_clip,
                u8::from(preset.use_dual),
            );
            let mut buf = [0.0_f32; W * D];
            for i in 0..win_t.n_tr {
                for j in 0..stride {
                    buf[j] = win_t.w_tr[i * stride + j] as f32;
                }
                let _ = ac.step(&buf);
            }
            let mut oc_scores = Vec::with_capacity(win_t.n_te);
            for i in 0..win_t.n_te {
                for j in 0..stride {
                    buf[j] = win_t.w_te[i * stride + j] as f32;
                }
                let r = ac.step(&buf);
                assert_eq!(r.state, 2);
                oc_scores.push(r.s_combined as f64);
            }
            oc_vals[idx] = compute_auroc(&win_t.y_te, &oc_scores) * 100.0;
        }

        println!(
            "{:<18}{:11.1}%{:11.1}%{:11.1}%{:11.1}%  | {:10.1}%{:10.1}%{:10.1}%",
            tgt, zs_0w, man_vals[0], man_vals[1], man_vals[2], oc_vals[0], oc_vals[1], oc_vals[2]
        );
        out.push(WarmupTargetReport {
            name: tgt,
            zs_0w,
            manual_100w: man_vals[0],
            manual_200w: man_vals[1],
            manual_400w: man_vals[2],
            onchip_100w: oc_vals[0],
            onchip_200w: oc_vals[1],
            onchip_400w: oc_vals[2],
            zs_normal_med,
            zs_fault_med,
        });
    }
    println!("{}", "-".repeat(102));
    let m0 = out.iter().map(|r| r.zs_0w).sum::<f64>() / out.len() as f64;
    let m1 = out.iter().map(|r| r.manual_100w).sum::<f64>() / out.len() as f64;
    let m2 = out.iter().map(|r| r.manual_200w).sum::<f64>() / out.len() as f64;
    let m4 = out.iter().map(|r| r.manual_400w).sum::<f64>() / out.len() as f64;
    let o1 = out.iter().map(|r| r.onchip_100w).sum::<f64>() / out.len() as f64;
    let o2 = out.iter().map(|r| r.onchip_200w).sum::<f64>() / out.len() as f64;
    let o4 = out.iter().map(|r| r.onchip_400w).sum::<f64>() / out.len() as f64;
    println!(
        "{:<18}{:11.1}%{:11.1}%{:11.1}%{:11.1}%  | {:10.1}%{:10.1}%{:10.1}%",
        "MEAN AUROC", m0, m1, m2, m4, o1, o2, o4
    );
    println!("\n  Zero-Shot (0w) Score-Direction Inversion Diagnostic:");
    for r in &out {
        println!(
            "    {:<16}: AUROC={:4.1}% (100-AUROC={:4.1}%) | normal med={:.2} vs fault med={:.2}",
            r.name, r.zs_0w, 100.0 - r.zs_0w, r.zs_normal_med, r.zs_fault_med
        );
    }
    out
}

struct LatencyReport {
    score_p50_us: f64,
    score_p95_us: f64,
    score_p99_us: f64,
    learn_p50_us: f64,
    learn_p95_us: f64,
    learn_p99_us: f64,
}

fn run_latency(dct: &[f32]) -> LatencyReport {
    println!("\n=== NATIVE RUST HOST CPU LATENCY BENCHMARK (ZERO FFI OVERHEAD) ===");
    let n_batch = 200usize;
    let n_repeats = 500usize;
    let mut rng = SplitMix64::new(42);
    let mut gen = |n: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(n * W * D);
        for _ in 0..(n * W * D) {
            out.push((rng.next_f64() * 2.0 - 1.0) as f32);
        }
        out
    };
    let warm = gen(200);
    let batch = gen(n_batch);

    let mut eng = GaugeTriadEngineF32::zeroed();
    eng.init(W, D, 3, 5e-3, 0.01, 0.008, 0.004, dct);
    for i in 0..200 {
        eng.step_v61(&warm[i * 32..(i + 1) * 32], 1, 1e20, 0, 10.0, 10.0, 2.5, 0, 1.5);
    }

    let measure = |eng: &mut GaugeTriadEngineF32, learn: u8| -> (f64, f64, f64) {
        let mut us = Vec::with_capacity(n_repeats);
        let mut sink = 0.0_f32;
        for _ in 0..n_repeats {
            let t0 = Instant::now();
            for i in 0..n_batch {
                let (s, _, _, _) =
                    eng.step_v61(&batch[i * 32..(i + 1) * 32], learn, 1e20, 1, 10.0, 10.0, 2.5, 0, 1.5);
                sink += s;
            }
            let dt = t0.elapsed().as_nanos() as f64;
            us.push((dt / 1000.0) / (n_batch as f64));
        }
        std::hint::black_box(sink);
        us.sort_by(|a, b| a.total_cmp(b));
        let p = |q: f64| us[((q * ((us.len() - 1) as f64)).round() as usize).min(us.len() - 1)];
        (p(0.50), p(0.95), p(0.99))
    };

    let (sp50, sp95, sp99) = measure(&mut eng, 0);
    let (lp50, lp95, lp99) = measure(&mut eng, 1);
    println!(
        "  Score-Only (8x4)      : p50={:.3} us | p95={:.3} us | p99={:.3} us ({:.1}k win/s)",
        sp50, sp95, sp99, 1e3 / sp50
    );
    println!(
        "  Live Learn+Score (8x4): p50={:.3} us | p95={:.3} us | p99={:.3} us ({:.1}k win/s)",
        lp50, lp95, lp99, 1e3 / lp50
    );
    println!("  Note: Host CPU benchmark only; Level 3 physical MCU silicon remains PENDING.");
    LatencyReport {
        score_p50_us: sp50,
        score_p95_us: sp95,
        score_p99_us: sp99,
        learn_p50_us: lp50,
        learn_p95_us: lp95,
        learn_p99_us: lp99,
    }
}

fn fmt_preset_json(m: PresetServerMetrics) -> String {
    format!(
        "{{\"auroc\": {:.6}, \"far\": {:.6}, \"recall\": {:.6}, \"far_raw\": {:.6}, \"recall_raw\": {:.6}, \"far_dual\": {:.6}, \"recall_dual\": {:.6}}}",
        m.auroc, m.far, m.recall, m.far_raw, m.recall_raw, m.far_dual, m.recall_dual
    )
}

fn fmt_agg_json(m: AggregatedMetrics) -> String {
    format!(
        "{{\"auroc\": {:.6}, \"auroc_seed_std\": {:.6}, \"far_raw\": {:.6}, \"recall_raw\": {:.6}, \"far_dual\": {:.6}, \"recall_dual\": {:.6}}}",
        m.auroc, m.auroc_seed_std, m.far_raw, m.recall_raw, m.far_dual, m.recall_dual
    )
}

fn build_report_json(
    srv: Option<&ServersReport>,
    base: Option<&BaselinesReport>,
    warm: Option<&[WarmupTargetReport]>,
    lat: Option<&LatencyReport>,
) -> String {
    let mut parts = Vec::new();
    parts.push("  \"crate_version\": \"0.7.1\"".to_string());
    parts.push(format!("  \"core_sram_bytes\": {}", ppc_triad_engine_bytes()));
    parts.push(format!("  \"autocal_sram_bytes\": {}", ppc_autocal_engine_bytes()));

    if let Some(s) = srv {
        let mut srv_entries = Vec::new();
        for (name, v5, hr, lf) in &s.per_server {
            srv_entries.push(format!(
                "      \"{}\": {{\n        \"v05\": {},\n        \"high_recall\": {},\n        \"low_far\": {}\n      }}",
                name,
                fmt_preset_json(*v5),
                fmt_preset_json(*hr),
                fmt_preset_json(*lf)
            ));
        }
        parts.push(format!(
            "  \"servers_8\": {{\n    \"per_server\": {{\n{}\n    }}\n  }}",
            srv_entries.join(",\n")
        ));
    }

    if let Some(b) = base {
        let mut b_entries = Vec::new();
        for (name, r) in &b.per_server {
            b_entries.push(format!(
                "      \"{}\": {{\n        \"hs_trees\": {},\n        \"rs_hash\": {},\n        \"pca_mahalanobis_32d\": {},\n        \"pca_mahalanobis_fulld\": {},\n        \"raw_knn\": {}\n      }}",
                name,
                fmt_agg_json(r.hs_trees),
                fmt_agg_json(r.rs_hash),
                fmt_agg_json(r.pca_mahalanobis_32d),
                fmt_agg_json(r.pca_mahalanobis_fulld),
                fmt_agg_json(r.raw_knn)
            ));
        }
        parts.push(format!(
            "  \"rust_baselines_8\": {{\n    \"seeds\": [0, 1, 2, 3, 4],\n    \"protocol\": \"non-overlapping w=8 window-level evaluation without point-adjust\",\n    \"per_server\": {{\n{}\n    }}\n  }}",
            b_entries.join(",\n")
        ));
    }

    if let Some(w) = warm {
        let mut pt = Vec::new();
        let mut oc = Vec::new();
        let mut diag = Vec::new();
        for r in w {
            pt.push(format!(
                "      \"{}\": {{\"0w\": {:.6}, \"100w\": {:.6}, \"200w\": {:.6}, \"400w\": {:.6}}}",
                r.name, r.zs_0w, r.manual_100w, r.manual_200w, r.manual_400w
            ));
            oc.push(format!(
                "      \"{}\": {{\"100w\": {:.6}, \"200w\": {:.6}, \"400w\": {:.6}}}",
                r.name, r.onchip_100w, r.onchip_200w, r.onchip_400w
            ));
            diag.push(format!(
                "      \"{}\": {{\"auroc\": {:.6}, \"inverted_auroc\": {:.6}, \"normal_median_score\": {:.6}, \"fault_median_score\": {:.6}}}",
                r.name, r.zs_0w, 100.0 - r.zs_0w, r.zs_normal_med, r.zs_fault_med
            ));
        }
        let m0 = w.iter().map(|r| r.zs_0w).sum::<f64>() / w.len() as f64;
        let m1 = w.iter().map(|r| r.manual_100w).sum::<f64>() / w.len() as f64;
        let m2 = w.iter().map(|r| r.manual_200w).sum::<f64>() / w.len() as f64;
        let m4 = w.iter().map(|r| r.manual_400w).sum::<f64>() / w.len() as f64;
        let o1 = w.iter().map(|r| r.onchip_100w).sum::<f64>() / w.len() as f64;
        let o2 = w.iter().map(|r| r.onchip_200w).sum::<f64>() / w.len() as f64;
        let o4 = w.iter().map(|r| r.onchip_400w).sum::<f64>() / w.len() as f64;
        parts.push(format!(
            "  \"warmup_curve\": {{\n    \"preset\": \"low_far\",\n    \"per_target\": {{\n{}\n    }},\n    \"mean\": {{\"0w\": {:.6}, \"100w\": {:.6}, \"200w\": {:.6}, \"400w\": {:.6}}},\n    \"zero_shot_diagnostics\": {{\n{}\n    }},\n    \"v07_onchip_per_target\": {{\n{}\n    }},\n    \"v07_onchip_mean\": {{\"100w\": {:.6}, \"200w\": {:.6}, \"400w\": {:.6}}}\n  }}",
            pt.join(",\n"), m0, m1, m2, m4, diag.join(",\n"), oc.join(",\n"), o1, o2, o4
        ));
    }

    if let Some(l) = lat {
        parts.push(format!(
            "  \"host_latency\": {{\n    \"score_only\": {{\"p50_us\": {:.4}, \"p95_us\": {:.4}, \"p99_us\": {:.4}}},\n    \"live_learn\": {{\"p50_us\": {:.4}, \"p95_us\": {:.4}, \"p99_us\": {:.4}}}\n  }}",
            l.score_p50_us, l.score_p95_us, l.score_p99_us, l.learn_p50_us, l.learn_p95_us, l.learn_p99_us
        ));
    }

    format!("{{\n{}\n}}\n", parts.join(",\n"))
}

fn print_usage() {
    println!("ppc-bench (PPC-Gauge-Triad v0.7.1 Rust-first CLI)");
    println!("Usage:");
    println!("  ppc-bench inspect-elf [PATH] [--json] [--check-budget]");
    println!("  ppc-bench servers   [--cache-dir DIR] [--save-json PATH]");
    println!("  ppc-bench baselines [--cache-dir DIR] [--save-json PATH]");
    println!("  ppc-bench warmup    [--cache-dir DIR] [--save-json PATH]");
    println!("  ppc-bench latency   [--save-json PATH]");
    println!("  ppc-bench all       [--cache-dir DIR] [--save-json PATH]");
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let cmd = args[0].as_str();
    if cmd == "inspect-elf" {
        let mut path = PathBuf::from(
            "firmware/cortex-m4f-eval/target/thumbv7em-none-eabihf/release/ppc-m4f-eval",
        );
        let mut as_json = false;
        let mut check_budget = false;
        for a in &args[1..] {
            if a == "--json" {
                as_json = true;
            } else if a == "--check-budget" {
                check_budget = true;
            } else if !a.starts_with('-') {
                path = PathBuf::from(a);
            }
        }
        match inspect_elf32(&path) {
            Ok(rep) => {
                if as_json {
                    println!("{}", rep.to_json());
                } else {
                    rep.print_table();
                }
                if check_budget {
                    let data_sz = rep.breakdown.get(".data").copied().unwrap_or(0);
                    if !rep.is_arm {
                        eprintln!("ELF_ASSERT_FAIL: expected EM_ARM (40), got {}", rep.e_machine);
                        return ExitCode::FAILURE;
                    }
                    if data_sz != 0 {
                        eprintln!("ELF_ASSERT_FAIL: expected .data == 0 B, got {} B", data_sz);
                        return ExitCode::FAILURE;
                    }
                    if rep.total_static_ram_bytes >= 16_384 {
                        eprintln!(
                            "ELF_ASSERT_FAIL: Static RAM {} B >= 16384 B",
                            rep.total_static_ram_bytes
                        );
                        return ExitCode::FAILURE;
                    }
                    println!(
                        "ELF_ASSERT_PASS: Flash={} B ({:.2} KB), .data=0 B, StaticRAM={} B ({:.2} KB)",
                        rep.total_flash_bytes,
                        rep.total_flash_bytes as f64 / 1024.0,
                        rep.total_static_ram_bytes,
                        rep.total_static_ram_bytes as f64 / 1024.0
                    );
                }
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let mut cache_dir = PathBuf::from("data_cache");
    let mut save_json: Option<PathBuf> = None;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--cache-dir" if i + 1 < args.len() => {
                cache_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--save-json" if i + 1 < args.len() => {
                save_json = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let core_b = ppc_triad_engine_bytes();
    let ac_b = ppc_autocal_engine_bytes();
    println!(
        "PPC-Gauge-Triad v0.7.1 (Rust native CLI) | Core SRAM = {} B ({:.2} KB) | AutoCalEngine SRAM = {} B ({:.2} KB)",
        core_b,
        core_b as f64 / 1024.0,
        ac_b,
        ac_b as f64 / 1024.0
    );
    let dct = make_dct_f32(W);

    let need_data = matches!(cmd, "servers" | "baselines" | "warmup" | "all");
    let datasets = if need_data {
        match load_all_servers(&cache_dir) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    let srv = if matches!(cmd, "servers" | "baselines" | "all") {
        Some(run_servers(datasets.as_ref().unwrap(), &dct))
    } else {
        None
    };
    let base = if matches!(cmd, "baselines" | "all") {
        Some(run_baselines(datasets.as_ref().unwrap()))
    } else {
        None
    };
    let warm = if matches!(cmd, "warmup" | "all") {
        Some(run_warmup(datasets.as_ref().unwrap(), &dct))
    } else {
        None
    };
    let lat = if matches!(cmd, "latency" | "all") {
        Some(run_latency(&dct))
    } else {
        None
    };

    if !matches!(cmd, "servers" | "baselines" | "warmup" | "latency" | "all") {
        eprintln!("Unknown subcommand: {cmd}");
        print_usage();
        return ExitCode::FAILURE;
    }

    if let Some(path) = save_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let json = build_report_json(srv.as_ref(), base.as_ref(), warm.as_deref(), lat.as_ref());
        if let Err(e) = fs::write(&path, json) {
            eprintln!("Failed to write {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
        println!("\nSaved Rust benchmark JSON -> {}", path.display());
    }

    ExitCode::SUCCESS
}
