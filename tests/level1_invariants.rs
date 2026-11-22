//! Level 1 Algorithmic Invariant & Embedded Constraint Verification Suite
//! for `ppc_gauge_triad_rs` v0.6.1.
//!
//! Covers all 9 pre-registered Level 1 criteria from `VALIDATION_REPORT_v061.md`:
//!   - L1.1: SO(4) Rotation Invariance (f32)         — max_err < 1e-4
//!   - L1.2: O(4) Chirality Sensitivity (det = -1)   — min_PPC > 0.05, max_Gram < 1e-5
//!   - L1.3: Scale Invariance [1e-3 .. 1e4]          — max_err < 1e-4
//!   - L1.4: DC Offset Invariance (±10 / ±1000)      — < 1e-5 / < 5e-4
//!   - L1.5: Combined Sim+(4) Transform              — max_err < 1e-4
//!   - L1.6: Rank-Deficient & Near-Singular (5 cases)— 0 NaN, 0 Inf, finite >= 0
//!   - L1.7: Precision f32 vs f64                    — max_diff < 1e-5, r > 0.9999
//!   - L1.8: C-FFI & Reference Parity                — exact 0.0 FFI parity, < 1e-5 ref parity
//!   - L1.9: Live Online Stability (up to 1,000,000) — 0 NaN/Inf, asym == 0.0, bounded
//!
//! Plus embedded & v0.6.1 behavioral invariants from `AGENTS.md`:
//!   - Struct SRAM footprint < 16 KB
//!   - Dual-Stage Alarm Gate (immediate severe trip vs 2-window confirmation)
//!   - Huber-Clamped EMA (Server mode) vs Stable-Plateau DC Re-Lock (Thermal Chip mode)

use core::mem::size_of;
use ppc_gauge_triad_rs::{
    ppc_triad_engine_bytes, ppc_triad_init, ppc_triad_run_stream_v61, GaugeTriadEngineF32,
};

const W: usize = 8;
const D: usize = 4;
const MAX_W: usize = 8;
const MAX_D: usize = 4;
const MAX_M: usize = 26;
const MAX_K: usize = 4;

// ---------------------------------------------------------------------------
// Deterministic PRNG (SplitMix64 + Box-Muller) — zero external dependencies
// ---------------------------------------------------------------------------

struct Rng64 {
    state: u64,
    spare: Option<f64>,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self { state: seed, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        // Uniform in (0, 1)
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        if u <= 1e-15 { 1e-15 } else { u }
    }

    fn next_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    fn next_normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

// ---------------------------------------------------------------------------
// Linear Algebra & Engine Helpers
// ---------------------------------------------------------------------------

fn make_dct_f64(w: usize) -> Vec<f64> {
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
    b
}

fn make_dct_f32(w: usize) -> Vec<f32> {
    make_dct_f64(w).into_iter().map(|v| v as f32).collect()
}

fn zeroed_engine() -> GaugeTriadEngineF32 {
    GaugeTriadEngineF32 {
        w: 0,
        d: 0,
        m: 0,
        k_max: 0,
        k_active: 0,
        lam0: 0.0,
        alpha_min: 0.0,
        e_alpha: 0.0,
        dc_alpha: 0.0,
        e_ema: 0.0,
        e_var: 0.0,
        dc_ema: [0.0; MAX_D],
        dc_var: [0.0; MAX_D],
        prev_dc: [0.0; MAX_D],
        e_init: 0,
        streak: 0,
        dc_streak: 0,
        sh_alarm_streak: 0,
        en_alarm_streak: 0,
        lv_alarm_streak: 0,
        b_dct: [0.0; MAX_W * MAX_W],
        means: [[0.0; MAX_M]; MAX_K],
        precs: [[0.0; MAX_M * MAX_M]; MAX_K],
        counts: [0; MAX_K],
    }
}

fn init_default_engine(
    w: usize,
    d: usize,
    k_max: usize,
    lam0: f32,
    alpha_min: f32,
    e_alpha: f32,
    dc_alpha: f32,
) -> GaugeTriadEngineF32 {
    let mut eng = zeroed_engine();
    let b_dct = make_dct_f32(w);
    eng.init(w, d, k_max, lam0, alpha_min, e_alpha, dc_alpha, &b_dct);
    eng
}

/// Exact determinant of a d x d row-major f64 matrix via partial-pivot Gaussian elimination.
fn det_f64(mat: &[f64], d: usize) -> f64 {
    let mut a = mat.to_vec();
    let mut sign = 1.0f64;
    for col in 0..d {
        let mut pivot_row = col;
        let mut max_val = a[col * d + col].abs();
        for r in (col + 1)..d {
            let v = a[r * d + col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = r;
            }
        }
        if max_val < 1e-15 {
            return 0.0;
        }
        if pivot_row != col {
            for k in 0..d {
                a.swap(col * d + k, pivot_row * d + k);
            }
            sign = -sign;
        }
        let piv = a[col * d + col];
        for r in (col + 1)..d {
            let factor = a[r * d + col] / piv;
            for k in col..d {
                a[r * d + k] -= factor * a[col * d + k];
            }
        }
    }
    let mut prod = sign;
    for i in 0..d {
        prod *= a[i * d + i];
    }
    prod
}

/// Generate a Haar-like random orthogonal matrix Q in SO(d) (det = +1) or O(d)\SO(d) (det = -1).
fn random_orthogonal(rng: &mut Rng64, d: usize, target_det: f64) -> Vec<f64> {
    // Modified Gram-Schmidt on Gaussian columns
    let mut cols = vec![vec![0.0f64; d]; d];
    for c in 0..d {
        loop {
            let mut v: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
            for prev in cols.iter().take(c) {
                let dot: f64 = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for i in 0..d {
                    v[i] -= dot * prev[i];
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-6 {
                for x in &mut v {
                    *x /= norm;
                }
                cols[c] = v;
                break;
            }
        }
    }
    let mut q = vec![0.0f64; d * d];
    for r in 0..d {
        for c in 0..d {
            q[r * d + c] = cols[c][r];
        }
    }
    let det = det_f64(&q, d);
    if det * target_det < 0.0 {
        for r in 0..d {
            q[r * d + (d - 1)] = -q[r * d + (d - 1)];
        }
    }
    let final_det = det_f64(&q, d);
    assert!(
        (final_det - target_det).abs() < 1e-11,
        "Expected det {target_det}, got {final_det}"
    );
    q
}

/// Right-multiply window X (w x d) by Q (d x d): X' = X * Q.
fn matmul_win_q(x: &[f64], q: &[f64], w: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; w * d];
    for t in 0..w {
        for j in 0..d {
            let mut acc = 0.0f64;
            for k in 0..d {
                acc += x[t * d + k] * q[k * d + j];
            }
            out[t * d + j] = acc;
        }
    }
    out
}

/// Generate a well-conditioned full-rank random window X (w x d) so QR diagonals
/// stay away from exact sign-flip discontinuity boundaries.
fn sample_well_conditioned_window(rng: &mut Rng64, eng: &GaugeTriadEngineF32) -> Vec<f64> {
    let b_dct = make_dct_f64(eng.w);
    loop {
        let x: Vec<f64> = (0..(eng.w * eng.d)).map(|_| rng.next_normal()).collect();
        let (u, _, diag) = canonicalize_f64(&x, eng.w, eng.d, &b_dct);
        // Ensure all d-1 leading diagonals are bounded away from 0 (avoiding measure-zero sign-flip boundary)
        // and the last row has non-trivial chiral energy.
        let min_lead_diag = diag[..(eng.d - 1)]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if min_lead_diag > 0.08 && diag[eng.d - 1].abs() > 0.05 && u.iter().all(|v| v.is_finite()) {
            return x;
        }
    }
}

/// Exact f64 reference implementation of `GaugeTriadEngineF32::canonicalize`.
fn canonicalize_f64(x_win: &[f64], w: usize, d: usize, b_dct: &[f64]) -> (Vec<f64>, f64, Vec<f64>) {
    let wm1 = w - 1;
    let mut m = 0usize;
    for row_k in 0..wm1 {
        m += if row_k < d { row_k + 1 } else { d };
    }
    let mut xc = vec![0.0f64; w * d];
    let mut norm_sq = 0.0f64;
    for j in 0..d {
        let mut mean = 0.0f64;
        for t in 0..w {
            mean += x_win[t * d + j];
        }
        mean /= w as f64;
        for t in 0..w {
            let v = x_win[t * d + j] - mean;
            xc[t * d + j] = v;
            norm_sq += v * v;
        }
    }
    let norm_raw = norm_sq.sqrt();
    let log_e = (norm_raw + 1e-6).ln();
    if norm_raw < 1e-12 {
        return (vec![0.0f64; m], log_e, vec![0.0f64; d]);
    }
    let inv_norm = 1.0 / norm_raw;

    let mut r_mat = vec![0.0f64; d * wm1];
    for j in 0..d {
        for k in 0..wm1 {
            let mut acc = 0.0f64;
            for t in 0..w {
                acc += b_dct[t * wm1 + k] * (xc[t * d + j] * inv_norm);
            }
            r_mat[j * wm1 + k] = acc;
        }
    }

    for col in 0..d {
        let mut row = d - 1;
        while row > col {
            let a = r_mat[(row - 1) * wm1 + col];
            let b = r_mat[row * wm1 + col];
            let rad = (a * a + b * b).sqrt();
            if rad > 1e-7 {
                let c = a / rad;
                let s = -b / rad;
                for k in col..wm1 {
                    let r1 = r_mat[(row - 1) * wm1 + k];
                    let r2 = r_mat[row * wm1 + k];
                    r_mat[(row - 1) * wm1 + k] = c * r1 - s * r2;
                    r_mat[row * wm1 + k] = s * r1 + c * r2;
                }
                r_mat[row * wm1 + col] = 0.0;
            }
            row -= 1;
        }
    }

    let mut det_sign: i32 = 1;
    let mut col_sign = vec![1.0f64; d];
    for i in 0..(d - 1) {
        if r_mat[i * wm1 + i] < 0.0 {
            col_sign[i] = -1.0;
            det_sign = -det_sign;
        }
    }
    if det_sign < 0 {
        col_sign[d - 1] = -1.0;
    }

    let mut diag = vec![0.0f64; d];
    for i in 0..d {
        diag[i] = r_mat[i * wm1 + i] * col_sign[i];
    }

    let mut u_out = vec![0.0f64; m];
    let mut idx = 0usize;
    for row_k in 0..wm1 {
        let max_j = if row_k < d { row_k + 1 } else { d };
        for col_j in 0..max_j {
            u_out[idx] = r_mat[col_j * wm1 + row_k] * col_sign[col_j];
            idx += 1;
        }
    }
    (u_out, log_e, diag)
}

/// Centered, unit-Frobenius temporal Gram matrix G(X) = X_c_norm * X_c_norm^T (w x w) in f32.
fn temporal_gram_f32(x: &[f32], w: usize, d: usize) -> Vec<f32> {
    let mut xc = vec![0.0f32; w * d];
    let mut norm_sq = 0.0f32;
    for j in 0..d {
        let mut mean = 0.0f32;
        for t in 0..w {
            mean += x[t * d + j];
        }
        mean /= w as f32;
        for t in 0..w {
            let v = x[t * d + j] - mean;
            xc[t * d + j] = v;
            norm_sq += v * v;
        }
    }
    let inv = 1.0 / norm_sq.sqrt().max(1e-12);
    for v in &mut xc {
        *v *= inv;
    }
    let mut g = vec![0.0f32; w * w];
    for t1 in 0..w {
        for t2 in 0..w {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += xc[t1 * d + j] * xc[t2 * d + j];
            }
            g[t1 * w + t2] = acc;
        }
    }
    g
}

fn to_f32_vec(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn l2_dist_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn pearson_r(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va * vb).sqrt()
}

// ---------------------------------------------------------------------------
// Level 1 Tests (L1.1 through L1.9)
// ---------------------------------------------------------------------------

#[test]
fn l1_1_so4_rotation_invariance() {
    let eng = init_default_engine(W, D, 3, 5e-3, 0.01, 0.008, 0.004);
    let mut rng = Rng64::new(1101);
    let mut max_err = 0.0f32;
    let mut max_e_err = 0.0f32;

    for _ in 0..100 {
        let x64 = sample_well_conditioned_window(&mut rng, &eng);
        let q = random_orthogonal(&mut rng, D, 1.0);
        let x_rot64 = matmul_win_q(&x64, &q, W, D);

        let x = to_f32_vec(&x64);
        let x_rot = to_f32_vec(&x_rot64);

        let mut u1 = [0.0f32; MAX_M];
        let mut dc1 = [0.0f32; MAX_D];
        let mut u2 = [0.0f32; MAX_M];
        let mut dc2 = [0.0f32; MAX_D];

        let e1 = eng.canonicalize(&x, &mut u1, &mut dc1);
        let e2 = eng.canonicalize(&x_rot, &mut u2, &mut dc2);

        max_err = max_err.max(max_abs_diff_f32(&u1[..eng.m], &u2[..eng.m]));
        max_e_err = max_e_err.max((e1 - e2).abs());
    }

    println!("L1.1 SO(4) Rotation: max_err={max_err:.3e}, max_e_err={max_e_err:.3e}");
    assert!(
        max_err < 1e-4,
        "L1.1 SO(4) rotation invariance failed: max_err = {max_err:.3e} >= 1e-4"
    );
    assert!(
        max_e_err < 1e-4,
        "L1.1 log-energy rotation invariance failed: max_e_err = {max_e_err:.3e}"
    );
}

#[test]
fn l1_2_o4_chirality_sensitivity() {
    let eng = init_default_engine(W, D, 3, 5e-3, 0.01, 0.008, 0.004);
    let mut rng = Rng64::new(1102);
    let mut min_ppc = f32::INFINITY;
    let mut max_gram = 0.0f32;

    for _ in 0..100 {
        let x64 = sample_well_conditioned_window(&mut rng, &eng);
        let q_neg = random_orthogonal(&mut rng, D, -1.0);
        let x_ref64 = matmul_win_q(&x64, &q_neg, W, D);

        let x = to_f32_vec(&x64);
        let x_ref = to_f32_vec(&x_ref64);

        let mut u1 = [0.0f32; MAX_M];
        let mut dc1 = [0.0f32; MAX_D];
        let mut u2 = [0.0f32; MAX_M];
        let mut dc2 = [0.0f32; MAX_D];

        eng.canonicalize(&x, &mut u1, &mut dc1);
        eng.canonicalize(&x_ref, &mut u2, &mut dc2);

        let ppc_dist = l2_dist_f32(&u1[..eng.m], &u2[..eng.m]);
        min_ppc = min_ppc.min(ppc_dist);

        let g1 = temporal_gram_f32(&x, W, D);
        let g2 = temporal_gram_f32(&x_ref, W, D);
        max_gram = max_gram.max(max_abs_diff_f32(&g1, &g2));
    }

    println!("L1.2 O(4) Chirality: min_PPC={min_ppc:.4}, max_Gram={max_gram:.3e}");
    assert!(
        min_ppc > 0.05,
        "L1.2 O(4) chirality sensitivity failed: min_PPC = {min_ppc:.4} <= 0.05"
    );
    assert!(
        max_gram < 1e-5,
        "L1.2 Gram reflection invariance sanity failed: max_Gram = {max_gram:.3e} >= 1e-5"
    );
}

#[test]
fn l1_3_scale_invariance() {
    let eng = init_default_engine(W, D, 3, 5e-3, 0.01, 0.008, 0.004);
    let mut rng = Rng64::new(1103);
    let scales = [1e-3f64, 1e-2, 0.1, 0.5, 2.0, 10.0, 100.0, 1e3, 1e4];
    let mut max_err = 0.0f32;

    for _ in 0..50 {
        let x64 = sample_well_conditioned_window(&mut rng, &eng);
        let x = to_f32_vec(&x64);
        let mut u_base = [0.0f32; MAX_M];
        let mut dc_base = [0.0f32; MAX_D];
        eng.canonicalize(&x, &mut u_base, &mut dc_base);

        for &s in &scales {
            let xs: Vec<f32> = x64.iter().map(|&v| (v * s) as f32).collect();
            let mut u_s = [0.0f32; MAX_M];
            let mut dc_s = [0.0f32; MAX_D];
            eng.canonicalize(&xs, &mut u_s, &mut dc_s);
            max_err = max_err.max(max_abs_diff_f32(&u_base[..eng.m], &u_s[..eng.m]));
        }
    }

    println!("L1.3 Scale [1e-3..1e4]: max_err={max_err:.3e}");
    assert!(
        max_err < 1e-4,
        "L1.3 scale invariance [1e-3..1e4] failed: max_err = {max_err:.3e} >= 1e-4"
    );
}

#[test]
fn l1_4_dc_offset_invariance() {
    let eng = init_default_engine(W, D, 3, 5e-3, 0.01, 0.008, 0.004);
    let mut rng = Rng64::new(1104);
    let mut max_err_10 = 0.0f32;
    let mut max_err_1000 = 0.0f32;

    for _ in 0..50 {
        let x64 = sample_well_conditioned_window(&mut rng, &eng);
        let x = to_f32_vec(&x64);
        let mut u_base = [0.0f32; MAX_M];
        let mut dc_base = [0.0f32; MAX_D];
        eng.canonicalize(&x, &mut u_base, &mut dc_base);

        // ±10 DC offset
        let off10: Vec<f64> = (0..D).map(|_| rng.next_uniform(-10.0, 10.0)).collect();
        let mut x_off10 = vec![0.0f32; W * D];
        for t in 0..W {
            for j in 0..D {
                x_off10[t * D + j] = (x64[t * D + j] + off10[j]) as f32;
            }
        }
        let mut u10 = [0.0f32; MAX_M];
        let mut dc10 = [0.0f32; MAX_D];
        eng.canonicalize(&x_off10, &mut u10, &mut dc10);
        max_err_10 = max_err_10.max(max_abs_diff_f32(&u_base[..eng.m], &u10[..eng.m]));

        // Verify DC channel accurately captures offset
        for j in 0..D {
            let dc_shift_err = ((dc10[j] - dc_base[j]) - off10[j] as f32).abs();
            assert!(dc_shift_err < 1e-4, "DC mean shift mismatch: {dc_shift_err}");
        }

        // ±1000 DC offset
        let off1000: Vec<f64> = (0..D).map(|_| rng.next_uniform(-1000.0, 1000.0)).collect();
        let mut x_off1000 = vec![0.0f32; W * D];
        for t in 0..W {
            for j in 0..D {
                x_off1000[t * D + j] = (x64[t * D + j] + off1000[j]) as f32;
            }
        }
        let mut u1000 = [0.0f32; MAX_M];
        let mut dc1000 = [0.0f32; MAX_D];
        eng.canonicalize(&x_off1000, &mut u1000, &mut dc1000);
        max_err_1000 = max_err_1000.max(max_abs_diff_f32(&u_base[..eng.m], &u1000[..eng.m]));
    }

    println!("L1.4 DC Offset: max_err(±10)={max_err_10:.3e}, max_err(±1000)={max_err_1000:.3e}");
    assert!(
        max_err_10 < 1e-5,
        "L1.4 DC offset (±10) invariance failed: max_err = {max_err_10:.3e} >= 1e-5"
    );
    assert!(
        max_err_1000 < 5e-4,
        "L1.4 DC offset (±1000) invariance failed: max_err = {max_err_1000:.3e} >= 5e-4"
    );
}

#[test]
fn l1_5_combined_sim_plus_4_transform() {
    let eng = init_default_engine(W, D, 3, 5e-3, 0.01, 0.008, 0.004);
    let mut rng = Rng64::new(1105);
    let mut max_err = 0.0f32;

    for _ in 0..100 {
        let x64 = sample_well_conditioned_window(&mut rng, &eng);
        let q = random_orthogonal(&mut rng, D, 1.0);
        let scale = 10.0f64.powf(rng.next_uniform(-2.0, 2.0));
        let dc_off: Vec<f64> = (0..D).map(|_| rng.next_uniform(-10.0, 10.0)).collect();

        let x_rot = matmul_win_q(&x64, &q, W, D);
        let mut x_sim = vec![0.0f32; W * D];
        for t in 0..W {
            for j in 0..D {
                x_sim[t * D + j] = (scale * x_rot[t * D + j] + dc_off[j]) as f32;
            }
        }

        let x = to_f32_vec(&x64);
        let mut u1 = [0.0f32; MAX_M];
        let mut dc1 = [0.0f32; MAX_D];
        let mut u2 = [0.0f32; MAX_M];
        let mut dc2 = [0.0f32; MAX_D];

        eng.canonicalize(&x, &mut u1, &mut dc1);
        eng.canonicalize(&x_sim, &mut u2, &mut dc2);

        max_err = max_err.max(max_abs_diff_f32(&u1[..eng.m], &u2[..eng.m]));
    }

    println!("L1.5 Combined Sim+(4): max_err={max_err:.3e}");
    assert!(
        max_err < 1e-4,
        "L1.5 combined Sim+(4) invariance failed: max_err = {max_err:.3e} >= 1e-4"
    );
}

#[test]
fn l1_6_rank_deficient_and_near_singular_five_cases() {
    let mut eng = init_default_engine(W, D, 3, 5e-3, 0.01, 0.008, 0.004);
    let mut rng = Rng64::new(1106);

    // Warm up on 100 normal windows first so prototype bank and EMA are populated
    for _ in 0..100 {
        let x: Vec<f32> = (0..(W * D)).map(|_| rng.next_normal() as f32).collect();
        let (sh, en, lv, _) = eng.step_v61(&x, 1, 1e20, 1, 10.0, 10.0, 2.5, 0, 1.5);
        assert!(sh.is_finite() && sh >= 0.0);
        assert!(en.is_finite() && en >= 0.0);
        assert!(lv.is_finite() && lv >= 0.0);
    }

    // Case 1: All zeros
    let c1_zeros = vec![0.0f32; W * D];

    // Case 2: Pure constant DC (zero fluctuation)
    let mut c2_const = vec![0.0f32; W * D];
    for t in 0..W {
        c2_const[t * D] = 3.5;
        c2_const[t * D + 1] = -12.0;
        c2_const[t * D + 2] = 100.0;
        c2_const[t * D + 3] = 0.25;
    }

    // Case 3: Exact rank-1 window (all channels scalar multiples of one temporal vector)
    let mut c3_rank1 = vec![0.0f32; W * D];
    let weights = [1.0f32, -2.0, 0.5, 3.0];
    for t in 0..W {
        let s = (t as f32 * 0.7).sin();
        for j in 0..D {
            c3_rank1[t * D + j] = s * weights[j];
        }
    }

    // Case 4: Exact rank-2 window (ch2 = ch0 + ch1, ch3 = 2*ch0 - ch1)
    let mut c4_rank2 = vec![0.0f32; W * D];
    for t in 0..W {
        let a = (t as f32 * 0.5).sin();
        let b = (t as f32 * 1.1).cos();
        c4_rank2[t * D] = a;
        c4_rank2[t * D + 1] = b;
        c4_rank2[t * D + 2] = a + b;
        c4_rank2[t * D + 3] = 2.0 * a - b;
    }

    // Case 5: Near-singular / tiny 1e-13 perturbation + subnormal-adjacent scale
    let mut c5_near_sing = vec![0.0f32; W * D];
    for t in 0..W {
        for j in 0..D {
            c5_near_sing[t * D + j] = 5.0 + 1e-13 * (rng.next_normal() as f32);
        }
    }

    let cases: [(&str, Vec<f32>); 5] = [
        ("all_zeros", c1_zeros),
        ("constant_dc", c2_const),
        ("rank_1", c3_rank1),
        ("rank_2", c4_rank2),
        ("near_singular", c5_near_sing),
    ];

    for (name, win) in &cases {
        let mut u = [0.0f32; MAX_M];
        let mut dc = [0.0f32; MAX_D];
        let log_e = eng.canonicalize(win, &mut u, &mut dc);
        assert!(log_e.is_finite(), "L1.6 {name}: log_e not finite");
        assert!(
            u[..eng.m].iter().all(|v| v.is_finite()),
            "L1.6 {name}: u contains NaN/Inf"
        );
        assert!(
            dc[..eng.d].iter().all(|v| v.is_finite()),
            "L1.6 {name}: dc contains NaN/Inf"
        );

        // Test both inference (learn=0) and online learning (learn=1)
        for &learn in &[0u8, 1u8] {
            let (sh, en, lv, _al) = eng.step_v61(win, learn, 1e20, 1, 10.0, 10.0, 2.5, 0, 1.5);
            assert!(
                sh.is_finite() && sh >= 0.0,
                "L1.6 {name} (learn={learn}): invalid s_shape = {sh}"
            );
            assert!(
                en.is_finite() && en >= 0.0,
                "L1.6 {name} (learn={learn}): invalid s_energy = {en}"
            );
            assert!(
                lv.is_finite() && lv >= 0.0,
                "L1.6 {name} (learn={learn}): invalid s_level = {lv}"
            );
        }
    }
}

#[test]
fn l1_7_precision_f32_vs_f64() {
    let eng = init_default_engine(W, D, 3, 5e-3, 0.01, 0.008, 0.004);
    let b_dct64 = make_dct_f64(W);
    let mut rng = Rng64::new(1107);

    let mut all_f32 = Vec::with_capacity(200 * eng.m);
    let mut all_f64 = Vec::with_capacity(200 * eng.m);
    let mut max_diff = 0.0f64;

    for _ in 0..200 {
        let x64 = sample_well_conditioned_window(&mut rng, &eng);
        let x32 = to_f32_vec(&x64);
        // Feed identical f32-representable input to both f32 and f64 paths to measure pure arithmetic drift
        let x32_as_64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();

        let mut u32 = [0.0f32; MAX_M];
        let mut dc32 = [0.0f32; MAX_D];
        eng.canonicalize(&x32, &mut u32, &mut dc32);

        let (u64, _, _) = canonicalize_f64(&x32_as_64, W, D, &b_dct64);

        for i in 0..eng.m {
            let diff = (u32[i] as f64 - u64[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            all_f32.push(u32[i] as f64);
            all_f64.push(u64[i]);
        }
    }

    let r = pearson_r(&all_f32, &all_f64);
    println!("L1.7 f32 vs f64: max_diff={max_diff:.3e}, r={r:.8}");
    assert!(
        max_diff < 1e-5,
        "L1.7 f32 vs f64 max_diff failed: {max_diff:.3e} >= 1e-5"
    );
    assert!(
        r > 0.9999,
        "L1.7 f32 vs f64 correlation failed: r = {r:.8} <= 0.9999"
    );
}

#[test]
fn l1_8_ffi_and_direct_api_exact_parity() {
    // Verifies bit-exact parity between the C FFI entrypoints used by Python ctypes
    // and the direct Rust GaugeTriadEngineF32 API.
    // (Cross-language Python NumPy vs Rust cdylib parity is additionally verified in benchmarks/run_benchmarks.py.)
    let b_dct = make_dct_f32(W);
    let mut rng = Rng64::new(1108);
    let n_train = 120usize;
    let n_test = 80usize;

    let train_x: Vec<f32> = (0..(n_train * W * D))
        .map(|_| rng.next_normal() as f32)
        .collect();
    let test_x: Vec<f32> = (0..(n_test * W * D))
        .map(|_| (rng.next_normal() * 1.3 + 0.5) as f32)
        .collect();

    // Direct Rust engine
    let mut eng_rs = init_default_engine(W, D, 3, 5e-3, 0.01, 0.008, 0.004);
    for i in 0..n_train {
        let win = &train_x[i * W * D..(i + 1) * W * D];
        eng_rs.step_v61(win, 1, 1e20, 0, 10.0, 10.0, 2.5, 0, 1.5);
    }
    let mut rs_sh = vec![0.0f32; n_test];
    let mut rs_en = vec![0.0f32; n_test];
    let mut rs_lv = vec![0.0f32; n_test];
    let mut rs_al = vec![0u8; n_test];
    for i in 0..n_test {
        let win = &test_x[i * W * D..(i + 1) * W * D];
        let (sh, en, lv, al) = eng_rs.step_v61(win, 0, 25.0, 1, 10.0, 10.0, 2.5, 0, 1.5);
        rs_sh[i] = sh;
        rs_en[i] = en;
        rs_lv[i] = lv;
        rs_al[i] = al;
    }

    // C FFI path
    let mut eng_ffi = zeroed_engine();
    let mut ffi_sh_tr = vec![0.0f32; n_train];
    let mut ffi_en_tr = vec![0.0f32; n_train];
    let mut ffi_lv_tr = vec![0.0f32; n_train];
    let mut ffi_al_tr = vec![0u8; n_train];

    let mut ffi_sh = vec![0.0f32; n_test];
    let mut ffi_en = vec![0.0f32; n_test];
    let mut ffi_lv = vec![0.0f32; n_test];
    let mut ffi_al = vec![0u8; n_test];

    unsafe {
        ppc_triad_init(
            &mut eng_ffi as *mut GaugeTriadEngineF32,
            W,
            D,
            3,
            5e-3,
            0.01,
            0.008,
            0.004,
            b_dct.as_ptr(),
        );
        ppc_triad_run_stream_v61(
            &mut eng_ffi as *mut GaugeTriadEngineF32,
            train_x.as_ptr(),
            n_train,
            1,
            1e20,
            0,
            10.0,
            10.0,
            2.5,
            0,
            1.5,
            ffi_sh_tr.as_mut_ptr(),
            ffi_en_tr.as_mut_ptr(),
            ffi_lv_tr.as_mut_ptr(),
            ffi_al_tr.as_mut_ptr(),
        );
        ppc_triad_run_stream_v61(
            &mut eng_ffi as *mut GaugeTriadEngineF32,
            test_x.as_ptr(),
            n_test,
            0,
            25.0,
            1,
            10.0,
            10.0,
            2.5,
            0,
            1.5,
            ffi_sh.as_mut_ptr(),
            ffi_en.as_mut_ptr(),
            ffi_lv.as_mut_ptr(),
            ffi_al.as_mut_ptr(),
        );
    }

    assert_eq!(rs_sh, ffi_sh, "FFI s_shape diverged from direct Rust API");
    assert_eq!(rs_en, ffi_en, "FFI s_energy diverged from direct Rust API");
    assert_eq!(rs_lv, ffi_lv, "FFI s_level diverged from direct Rust API");
    assert_eq!(rs_al, ffi_al, "FFI alarm_dual diverged from direct Rust API");
}

fn run_online_stability_check(n_windows: usize, seed: u64) {
    let mut eng = init_default_engine(W, D, 4, 5e-3, 0.01, 0.008, 0.004);
    let mut rng = Rng64::new(seed);
    let mut win = [0.0f32; W * D];
    let mut max_sh = 0.0f32;
    let mut max_en = 0.0f32;
    let mut max_lv = 0.0f32;

    for step in 0..n_windows {
        // Multi-regime stream with slow drift + periodic mode switching
        let drift = ((step as f64) * 1e-4).sin() * 2.0;
        let phase = (step % 4) as f64 * 0.4;
        for t in 0..W {
            let tt = t as f64 * 0.5 + phase;
            win[t * D] = (tt.sin() + 0.15 * rng.next_normal() + drift) as f32;
            win[t * D + 1] = (tt.cos() + 0.15 * rng.next_normal() - 0.5 * drift) as f32;
            win[t * D + 2] = ((2.0 * tt).sin() * 0.7 + 0.15 * rng.next_normal()) as f32;
            win[t * D + 3] = ((0.5 * tt).cos() * 0.5 + 0.15 * rng.next_normal()) as f32;
        }
        let (sh, en, lv, _) = eng.step_v61(&win, 1, 500.0, 1, 10.0, 10.0, 2.5, 0, 1.5);
        assert!(sh.is_finite() && sh >= 0.0, "Non-finite s_shape at step {step}: {sh}");
        assert!(en.is_finite() && en >= 0.0, "Non-finite s_energy at step {step}: {en}");
        assert!(lv.is_finite() && lv >= 0.0, "Non-finite s_level at step {step}: {lv}");
        max_sh = max_sh.max(sh);
        max_en = max_en.max(en);
        max_lv = max_lv.max(lv);
    }

    // Verify exact symmetry of every active precision matrix (asym == 0.0) and boundedness
    let m = eng.m;
    let mut max_asym = 0.0f32;
    let mut max_prec_abs = 0.0f32;
    for k in 0..eng.k_active {
        for i in 0..m {
            assert!(eng.means[k][i].is_finite(), "Non-finite mean at k={k}, i={i}");
            for j in 0..m {
                let pij = eng.precs[k][i * m + j];
                let pji = eng.precs[k][j * m + i];
                assert!(pij.is_finite(), "Non-finite prec at k={k}, ({i},{j})");
                max_asym = max_asym.max((pij - pji).abs());
                max_prec_abs = max_prec_abs.max(pij.abs());
            }
        }
    }

    println!(
        "L1.9 ({n_windows} windows): 0 NaN/Inf, max_asym={max_asym:.1}, max_sh={max_sh:.2}, max_en={max_en:.2}, max_lv={max_lv:.2}"
    );
    assert_eq!(
        max_asym, 0.0,
        "L1.9 precision matrix symmetry violated: max_asym = {max_asym:.3e}"
    );
    assert!(
        max_prec_abs <= 1.0e5 + 1e-3,
        "L1.9 precision matrix exceeded clamp bound: {max_prec_abs}"
    );
    assert!(
        max_sh < 1.0e6 && max_en < 1.0e6 && max_lv < 1.0e6,
        "L1.9 unbounded scores: max_sh={max_sh}, max_en={max_en}, max_lv={max_lv}"
    );
}

#[test]
fn l1_9_online_stability_20k_windows() {
    // Fast everyday stability gate (runs in <1s even in debug builds)
    run_online_stability_check(20_000, 1109);
}

#[test]
#[ignore = "Full 1,000,000-window L1.9 stress test — run via `cargo test --release -- --ignored` (~3s)"]
fn l1_9_million_window_online_stability() {
    run_online_stability_check(1_000_000, 1109);
}

// ---------------------------------------------------------------------------
// Embedded Constraints & v0.6.1 Operational Mode Tests
// ---------------------------------------------------------------------------

#[test]
fn embedded_sram_footprint_under_16kb() {
    let bytes = ppc_triad_engine_bytes();
    assert_eq!(bytes, size_of::<GaugeTriadEngineF32>());
    assert!(
        bytes < 16 * 1024,
        "Embedded SRAM constraint violated: {bytes} B >= 16384 B"
    );
}

#[test]
fn v061_dual_stage_alarm_gate_rejects_single_moderate_spike() {
    let mut eng = init_default_engine(W, D, 3, 5e-3, 0.01, 0.20, 0.25);
    let mut rng = Rng64::new(2061);

    // Train on stationary baseline
    let mut base_win = [0.0f32; W * D];
    for _ in 0..100 {
        for t in 0..W {
            let tt = t as f32 * 0.6;
            base_win[t * D] = tt.sin() + 0.02 * rng.next_normal() as f32;
            base_win[t * D + 1] = tt.cos() + 0.02 * rng.next_normal() as f32;
            base_win[t * D + 2] = (2.0 * tt).sin() * 0.5 + 0.02 * rng.next_normal() as f32;
            base_win[t * D + 3] = (0.5 * tt).cos() * 0.5 + 0.02 * rng.next_normal() as f32;
        }
        eng.step_v61(&base_win, 1, 1e20, 0, 10.0, 10.0, 2.5, 0, 0.0);
    }

    // Create a moderate DC shift where lvl_thr < s_level < 2.5 * lvl_thr
    // After 100 low-noise windows, dc_var ~ 0.0, denom ~ 0.05, so dd = 0.85 => s_level ~ 14.4
    let mut mod_win = base_win;
    for t in 0..W {
        for j in 0..D {
            mod_win[t * D + j] += 0.85;
        }
    }
    let (_, _, lv1, al1) = eng.step_v61(&mod_win, 0, 100.0, 0, 100.0, 10.0, 2.5, 0, 0.0);
    assert!(lv1 > 10.0 && lv1 < 25.0, "Expected moderate s_level in (10, 25), got {lv1}");
    assert_eq!(al1, 0, "Single moderate 1-window spike must NOT trip dual-stage gate");

    // Second consecutive moderate spike MUST trip
    let (_, _, lv2, al2) = eng.step_v61(&mod_win, 0, 100.0, 0, 100.0, 10.0, 2.5, 0, 0.0);
    assert!(lv2 > 10.0 && lv2 < 25.0);
    assert_eq!(al2, 1, "Two consecutive moderate windows MUST trip dual-stage gate");

    // Reset streak with a normal window
    let (_, _, _, al_norm) = eng.step_v61(&base_win, 0, 100.0, 0, 100.0, 10.0, 2.5, 0, 0.0);
    assert_eq!(al_norm, 0, "Normal window should clear alarm");

    // Single severe spike (> 2.5x lvl_thr) MUST trip immediately on window 1
    let mut sev_win = base_win;
    for t in 0..W {
        for j in 0..D {
            sev_win[t * D + j] += 5.0;
        }
    }
    let (_, _, lv_sev, al_sev) = eng.step_v61(&sev_win, 0, 100.0, 0, 100.0, 10.0, 2.5, 0, 0.0);
    assert!(lv_sev > 25.0, "Expected severe s_level > 25, got {lv_sev}");
    assert_eq!(al_sev, 1, "Severe spike (> sev_mult * thr) must trip immediately");
}

#[test]
fn v061_huber_server_mode_vs_thermal_plateau_relock() {
    let mut base_win = [0.0f32; W * D];
    for t in 0..W {
        let tt = t as f32 * 0.6;
        base_win[t * D] = tt.sin();
        base_win[t * D + 1] = tt.cos();
        base_win[t * D + 2] = (2.0 * tt).sin() * 0.5;
        base_win[t * D + 3] = (0.5 * tt).cos() * 0.5;
    }
    let mut step_win = base_win;
    for t in 0..W {
        for j in 0..D {
            step_win[t * D + j] += 20.0;
        }
    }

    // 1. Server Telemetry mode (Huber-Clamped EMA): must NOT swallow a +20 sustained incident in 20 windows
    let mut eng_srv = init_default_engine(W, D, 3, 5e-3, 0.01, 0.002, 0.001);
    for _ in 0..80 {
        eng_srv.step_v61(&base_win, 1, 1e20, 0, 10.0, 10.0, 2.5, 0, 1.0);
    }
    let mut last_lv_srv = 0.0f32;
    for _ in 0..20 {
        let (_, _, lv, _) = eng_srv.step_v61(&step_win, 0, 100.0, 1, 10.0, 10.0, 2.5, 0, 1.0);
        last_lv_srv = lv;
    }
    assert!(
        last_lv_srv > 1000.0,
        "Server Huber-EMA must preserve sustained +20 incident signal, got s_level={last_lv_srv}"
    );

    // 2. Thermal / Sensor Chip mode (huber_clip=0, dc_relock=6): must re-lock onto stable +20°C plateau
    let mut eng_chip = init_default_engine(W, D, 3, 5e-3, 0.01, 0.20, 0.25);
    for _ in 0..80 {
        eng_chip.step_v61(&base_win, 1, 1e20, 0, 10.0, 10.0, 2.5, 6, 0.0);
    }
    // First few transition windows should alert, then after dc_relock=6 steady windows s_level drops to ~0
    let (_, _, lv_first, _) = eng_chip.step_v61(&step_win, 0, 100.0, 1, 10.0, 10.0, 2.5, 6, 0.0);
    assert!(lv_first > 100.0, "Initial +20°C step should trigger high s_level");
    for _ in 0..10 {
        eng_chip.step_v61(&step_win, 0, 100.0, 1, 10.0, 10.0, 2.5, 6, 0.0);
    }
    let (_, _, lv_settled, al_settled) =
        eng_chip.step_v61(&step_win, 0, 100.0, 1, 10.0, 10.0, 2.5, 6, 0.0);
    assert!(
        lv_settled < 1e-3 && al_settled == 0,
        "Thermal mode should re-lock on steady +20°C plateau, got s_level={lv_settled}, alarm={al_settled}"
    );
}
