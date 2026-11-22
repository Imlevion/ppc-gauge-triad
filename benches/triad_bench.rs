//! Criterion host CPU micro-benchmarks for `ppc_gauge_triad_rs` v0.7.0.
//!
//! Note: These measure host CPU software latency only; Level 3 MCU silicon
//! cycle counts and power profiling remain pending physical hardware verification.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ppc_gauge_triad_rs::{AutoCalEngineF32, GaugeTriadEngineF32, MAX_D, MAX_M};

const W: usize = 8;
const D: usize = 4;

fn make_dct_f32(w: usize) -> Vec<f32> {
    let wm1 = w - 1;
    let mut b = vec![0.0f64; w * wm1];
    for k in 0..wm1 {
        let freq = (k + 1) as f64;
        let mut norm_sq = 0.0f64;
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

fn make_windows(n: usize) -> Vec<[f32; W * D]> {
    let mut wins = Vec::with_capacity(n);
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    for i in 0..n {
        let mut w = [0.0f32; W * D];
        for t in 0..W {
            let tt = t as f32 * 0.55 + (i as f32) * 0.1;
            for j in 0..D {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((s >> 40) as f32 / ((1u32 << 24) as f32) - 0.5) * 0.2;
                w[t * D + j] = (tt + j as f32 * 0.7).sin() + noise;
            }
        }
        wins.push(w);
    }
    wins
}

fn bench_ppc_triad_v07(c: &mut Criterion) {
    let b_dct = make_dct_f32(W);
    let wins = make_windows(512);

    let mut eng = GaugeTriadEngineF32::zeroed();
    eng.init(W, D, 3, 5e-3, 0.01, 0.008, 0.004, &b_dct);
    for w in wins.iter().take(200) {
        eng.step_v61(w, 1, 1e20, 1, 10.0, 10.0, 2.5, 0, 1.5);
    }

    c.bench_function("canonicalize_8x4", |b| {
        let mut u = [0.0f32; MAX_M];
        let mut dc = [0.0f32; MAX_D];
        let mut idx = 0usize;
        b.iter(|| {
            let win = &wins[idx & 511];
            idx = idx.wrapping_add(1);
            let e = eng.canonicalize(black_box(win), &mut u, &mut dc);
            black_box((e, u[0], dc[0]))
        });
    });

    c.bench_function("step_v61_score_only_8x4", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let win = &wins[idx & 511];
            idx = idx.wrapping_add(1);
            black_box(eng.step_v61(black_box(win), 0, 50.0, 1, 10.0, 10.0, 2.5, 0, 1.5))
        });
    });

    c.bench_function("step_v61_live_learn_8x4", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let win = &wins[idx & 511];
            idx = idx.wrapping_add(1);
            black_box(eng.step_v61(black_box(win), 1, 500.0, 1, 10.0, 10.0, 2.5, 0, 1.5))
        });
    });

    let mut auto_mon = AutoCalEngineF32::zeroed();
    auto_mon.init(W, D, 3, 5e-3, 0.01, 0.008, 0.004, &b_dct, 300, 100, 0.05, 1, 2.5, 0, 1.5, 1);
    for w in wins.iter().take(400) {
        auto_mon.step(w);
    }

    c.bench_function("autocal_v07_monitor_step_8x4", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let win = &wins[idx & 511];
            idx = idx.wrapping_add(1);
            black_box(auto_mon.step(black_box(win)))
        });
    });

    c.bench_function("autocal_v07_full_400w_warmup_calibrate", |b| {
        let mut tmp = AutoCalEngineF32::zeroed();
        b.iter(|| {
            tmp.init(W, D, 3, 5e-3, 0.01, 0.008, 0.004, &b_dct, 300, 100, 0.05, 1, 2.5, 0, 1.5, 1);
            for w in wins.iter().take(400) {
                black_box(tmp.step(black_box(w)));
            }
            black_box(tmp.cal.thr_combined)
        });
    });
}

criterion_group!(benches, bench_ppc_triad_v07);
criterion_main!(benches);
