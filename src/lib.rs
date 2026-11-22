#![cfg_attr(not(feature = "std"), no_std)]

//! # `ppc_gauge_triad_rs` v0.7.0 — Deterministic `#![no_std]` Streaming Anomaly Engine
//!
//! **PPC-Gauge-Triad** is a deterministic, zero-heap (`#![no_std]`), zero-matrix-inversion
//! `f32` streaming anomaly detection engine for multi-channel telemetry and embedded physical
//! sensors (`X_t ∈ ℝ^(w × d)`, default `w = 8`, `d = 4`).
//!
//! ## Orthogonal 3-Channel Gauge Decomposition
//!
//! 1. **`s_shape` (Canonical Shape):** Orthonormal temporal DCT-II projection followed by
//!    explicit `2 × 2` Givens-QR canonicalization on the proper similarity group `Sim+(d)`
//!    (`det = +1`, preserving chirality under improper reflections `det = -1`), scored via a
//!    symmetrized `K`-prototype rank-1 Sherman-Morrison precision bank (`K ≤ 4`, `0` matrix
//!    inversions, `0` stored raw training windows).
//! 2. **`s_energy` (Log-Fluctuation Energy):** Relative log-Frobenius energy deviation tracked
//!    via Huber-clamped EMA (Server mode) or fast ramp EMA + 3-window hysteresis (Thermal Chip mode).
//! 3. **`s_level` (Per-Channel DC Level):** Per-channel mean tracking via Huber-clamped EMA
//!    (tracks slow workload drift without swallowing sustained incidents) or stable-plateau
//!    re-lock (`dc_relock = 6` for abrupt physical regime steps).
//! 4. **Dual-Stage Alarm Gate (`step_v61`):** Immediate trip on severe faults (`> sev_mult * thr`)
//!    or 2-consecutive-window confirmation.
//! 5. **On-Chip Zero-Touch Auto-Warmup (`AutoCalBufF32` / `AutoCalEngineF32` — v0.7):**
//!    Fixed-size `#![no_std]` state machine (`Warmup -> Calibrate -> Monitor`, `MAX_CAL = 256`)
//!    computing split-conformal quantiles and `log1p`-MAD triad normalizers directly on chip.
//!
//! ## Embedded Resource Profile, `#![no_std]` Integration & Design Boundaries
//!
//! - **Engine Static Struct SRAM (`< 16 KB` compile-time assert):** [`GaugeTriadEngineF32`] =
//!   `11,624 B` (`11.35 KB`) on 64-bit / `11,604 B` (`11.33 KB`) on ARM Cortex-M4F; combined
//!   [`AutoCalEngineF32`] = `14,768 B` (`14.42 KB`) on 64-bit / `14,744 B` (`14.40 KB`) on
//!   Cortex-M4F. **Important:** this bounds only the engine's static struct `.bss` footprint and
//!   does *not* prove that a user's complete firmware fits inside `16 KB` RAM — final firmware
//!   RAM must additionally budget call stack (including a `1,024 B` stack scratch array during
//!   `finalize_calibration()`), interrupt frames, DMA/sensor buffers, logging buffers, and
//!   BSP/RTOS runtime overhead.
//! - **Heap Allocation:** `0 B` (`#![no_std]`, zero `alloc`). **Matrix Inversions:** `0`.
//! - **`#![no_std]` Library vs. Final Firmware (`#[panic_handler]`):** Compiling with
//!   `default-features = false` builds a pure `#![no_std]` library (`rlib`) **without** defining
//!   a `#[panic_handler]`, avoiding duplicate `panic_impl` conflicts with firmware panic crates
//!   (`panic-halt`, `panic-probe`, `panic-rtt-target`, etc.). Your final `#![no_std]` firmware
//!   binary must provide its own `#[panic_handler]` and startup runtime. Enable the opt-in
//!   `panic-handler` Cargo feature *only* if compiling a standalone `#![no_std]` `cdylib`.
//! - **Unsupervised Per-Device Warmup Required:** Pure zero-shot transfer across heterogeneous
//!   channel topologies is not claimed (`29.7%` mean AUROC across unseen servers). Zero-touch
//!   per-device unsupervised warmup (`100–400` normal windows, `0` labels) is a design
//!   requirement and recovers `75.3%–82.9%` mean AUROC.
//! - **Level 3 Hardware Status:** Physical silicon cycle counts, full `.elf`/`.map` flash/RAM
//!   footprints, and mW power profiling remain `PENDING` external hardware measurement (see
//!   `examples/level3_cortex_m4f_eval_template.rs` for the ARM Cortex-M4F DWT evaluation template).
//!
//! ## Quick Start: On-Chip Zero-Touch Auto-Warmup (`AutoCalEngineF32`)
//!
//! ```rust
//! use ppc_gauge_triad_rs::{AutoCalEngineF32, AutoCalState};
//!
//! // Build column-orthonormal temporal DCT-II basis for w = 8 (shape 8 x 7).
//! fn make_dct_w8() -> [f32; 56] {
//!     let mut b = [0.0f32; 56];
//!     for k in 0..7 {
//!         let mut norm_sq = 0.0f32;
//!         for t in 0..8 {
//!             let v = (core::f32::consts::PI * (t as f32 + 0.5) * ((k + 1) as f32) / 8.0).cos();
//!             b[t * 7 + k] = v;
//!             norm_sq += v * v;
//!         }
//!         let norm = norm_sq.sqrt();
//!         for t in 0..8 {
//!             b[t * 7 + k] /= norm;
//!         }
//!     }
//!     b
//! }
//!
//! let b_dct = make_dct_w8();
//! let mut engine = AutoCalEngineF32::zeroed();
//! // Locked Low-FAR server preset (e_alpha=0.008, dc_alpha=0.004, huber_clip=1.5),
//! // 75 warmup + 25 calibrate windows (100w zero-touch auto-warmup, alpha=0.05).
//! engine.init(8, 4, 3, 5e-3, 0.01, 0.008, 0.004, &b_dct, 75, 25, 0.05, 1, 2.5, 0, 1.5, 1);
//! assert_eq!(engine.state(), AutoCalState::Warmup);
//!
//! // Stream 100 normal windows through Warmup (0..75) -> Calibrate (75..100).
//! for step in 0..100usize {
//!     let mut win = [0.0f32; 32];
//!     for t in 0..8 {
//!         let phi = (t as f32) * 0.5 + (step as f32) * 0.1;
//!         win[t * 4] = phi.sin();
//!         win[t * 4 + 1] = phi.cos();
//!         win[t * 4 + 2] = (2.0 * phi).sin() * 0.6;
//!         win[t * 4 + 3] = (0.5 * phi).cos() * 0.5;
//!     }
//!     let res = engine.step(&win);
//!     assert_eq!(res.alarm_combined, 0);
//! }
//! assert_eq!(engine.state(), AutoCalState::Monitor);
//!
//! // Inject a severe multi-channel fault in Monitor mode: trips immediately.
//! let fault = [25.0f32; 32];
//! let res = engine.step(&fault);
//! assert_eq!(res.state, AutoCalState::Monitor as u8);
//! assert_eq!(res.alarm_combined, 1);
//! ```

// Only emit a fallback `#[panic_handler]` when explicitly opted in via `--features panic-handler`
// (for standalone `#![no_std]` cdylib checks). Library consumers linking `ppc_gauge_triad_rs`
// with `default-features = false` MUST supply their own firmware `#[panic_handler]`.
#[cfg(all(not(feature = "std"), feature = "panic-handler"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Maximum supported temporal window length `w`.
pub const MAX_W: usize = 8;
/// Maximum supported sensor/feature channel count `d`.
pub const MAX_D: usize = 4;
/// Maximum canonical shape vector dimension `m` (for `w = 8, d = 4`, `m = 26`).
pub const MAX_M: usize = 26;
/// Maximum number of online shape prototypes `k_max`.
pub const MAX_K: usize = 4;
/// Fixed on-chip calibration buffer capacity (`n_cal <= MAX_CAL`) for [`AutoCalBufF32`].
pub const MAX_CAL: usize = 256;

#[inline(always)]
fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.sqrt() }
    #[cfg(not(feature = "std"))]
    {
        if x <= 0.0 { return 0.0; }
        let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1fc0_0000);
        for _ in 0..4 { y = 0.5 * (y + x / y); }
        y
    }
}

#[inline(always)]
fn ln_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.ln() }
    #[cfg(not(feature = "std"))]
    {
        if x <= 1e-20 { return -46.0; }
        let bits = x.to_bits();
        let exp = (((bits >> 23) & 0xFF) as i32) - 127;
        let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
        let t = (m - 1.0) / (m + 1.0);
        let t2 = t * t;
        let ln_m = 2.0 * t * (1.0 + t2 * (0.33333334 + t2 * (0.2 + t2 * 0.14285715)));
        (exp as f32) * 0.69314718 + ln_m
    }
}

#[inline(always)]
fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { lo } else if x > hi { hi } else { x }
}

#[inline(always)]
fn abs_f32(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

#[inline(always)]
fn sanitize_nonneg_f32(x: f32) -> f32 {
    if x > 0.0 && x.is_finite() { x } else { 0.0 }
}

/// Deterministic `#![no_std]` in-place Shell sort (Knuth gaps, zero heap, zero recursion).
fn shell_sort_f32(arr: &mut [f32]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }
    let mut gap = 1usize;
    while gap < n / 3 {
        gap = gap * 3 + 1;
    }
    while gap >= 1 {
        for i in gap..n {
            let tmp = arr[i];
            let mut j = i;
            while j >= gap && arr[j - gap] > tmp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = tmp;
        }
        gap /= 3;
    }
}

#[inline]
fn median_sorted(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

#[inline]
fn ceil_pos_f64_to_usize(x: f64) -> usize {
    if x <= 0.0 {
        return 0;
    }
    let t = x as usize;
    if (t as f64) < x - 1e-12 {
        t + 1
    } else {
        t
    }
}

#[inline]
fn conformal_quantile_sorted(sorted: &[f32], alpha: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let a = clamp_f32(alpha, 1e-6, 0.999999) as f64;
    let c = ceil_pos_f64_to_usize((1.0 - a) * ((n + 1) as f64));
    let idx = if c == 0 {
        0
    } else if c - 1 >= n {
        n - 1
    } else {
        c - 1
    };
    sorted[idx]
}

fn fit_log1p_mad(raw: &[f32], scratch: &mut [f32; MAX_CAL]) -> (f32, f32) {
    let n = raw.len();
    if n == 0 {
        return (0.0, 0.1);
    }
    for i in 0..n {
        scratch[i] = ln_f32(1.0 + sanitize_nonneg_f32(raw[i]));
    }
    shell_sort_f32(&mut scratch[..n]);
    let med = median_sorted(&scratch[..n]);
    for i in 0..n {
        scratch[i] = abs_f32(scratch[i] - med);
    }
    shell_sort_f32(&mut scratch[..n]);
    let mad = median_sorted(&scratch[..n]) * 1.4826 + 0.1;
    (med, mad)
}

/// Core `#![no_std]` `f32` PPC-Gauge-Triad streaming anomaly engine (`~11.4 KB` static SRAM).
///
/// Maintains up to `MAX_K = 4` symmetrized Sherman-Morrison precision prototypes over
/// `Sim+(d)` canonical window shadows alongside dual-mode environmental EMA trackers
/// (`s_energy` and `s_level`) and dual-stage alarm streaks.
///
/// The C-compatible `#[repr(C)]` layout matches `include/ppc_gauge_triad.h` identically.
#[repr(C)]
pub struct GaugeTriadEngineF32 {
    /// Temporal window length `w` (`<= MAX_W`).
    pub w: usize,
    /// Number of channels `d` (`<= MAX_D`).
    pub d: usize,
    /// Canonical shape vector dimension `m` (`<= MAX_M`; `26` for `w = 8, d = 4`).
    pub m: usize,
    /// Maximum prototype bank capacity (`<= MAX_K`).
    pub k_max: usize,
    /// Currently active prototypes in `[0, k_max]`.
    pub k_active: usize,
    /// Initial isotropic covariance regularization `lam0`.
    pub lam0: f32,
    /// Minimum Sherman-Morrison learning rate floor `alpha_min`.
    pub alpha_min: f32,
    /// Log-energy EMA smoothing rate `e_alpha`.
    pub e_alpha: f32,
    /// Per-channel DC level EMA smoothing rate `dc_alpha`.
    pub dc_alpha: f32,
    /// Tracked log-fluctuation energy mean.
    pub e_ema: f32,
    /// Tracked log-fluctuation energy variance.
    pub e_var: f32,
    /// Per-channel tracked DC means.
    pub dc_ema: [f32; MAX_D],
    /// Per-channel tracked DC variances.
    pub dc_var: [f32; MAX_D],
    /// Previous window DC means (used for stable-plateau re-lock).
    pub prev_dc: [f32; MAX_D],
    /// Non-zero once environmental EMAs have been initialized.
    pub e_init: u8,
    /// Energy hysteresis streak counter (Thermal Chip mode).
    pub streak: u8,
    /// Stable-plateau DC re-lock streak counter (Thermal Chip mode).
    pub dc_streak: u8,
    /// Consecutive `s_shape > shape_gate` counter.
    pub sh_alarm_streak: u8,
    /// Consecutive `s_energy > imp_thr` counter.
    pub en_alarm_streak: u8,
    /// Consecutive `s_level > lvl_thr` counter.
    pub lv_alarm_streak: u8,
    /// Row-major orthonormal temporal DCT-II basis (`w × (w - 1)`).
    pub b_dct: [f32; MAX_W * MAX_W],
    /// Online canonical shape prototype means.
    pub means: [[f32; MAX_M]; MAX_K],
    /// Symmetrized rank-1 Sherman-Morrison precision matrices (`m × m` each).
    pub precs: [[f32; MAX_M * MAX_M]; MAX_K],
    /// Observation counts per prototype.
    pub counts: [u32; MAX_K],
}

impl GaugeTriadEngineF32 {
    /// Const all-zero initializer guaranteeing true `.bss` placement (`0 B` Flash `.data` image)
    /// on bare-metal MCUs. Call [`GaugeTriadEngineF32::init`] before processing windows.
    pub const fn zeroed() -> Self {
        Self {
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

    /// Initialize (or reset) the engine parameters and copy the orthonormal `w × (w - 1)` DCT-II basis.
    pub fn init(
        &mut self, w: usize, d: usize, k_max: usize,
        lam0: f32, alpha_min: f32, e_alpha: f32, dc_alpha: f32, b_dct: &[f32],
    ) {
        self.w = w;
        self.d = d;
        self.k_max = if k_max > MAX_K { MAX_K } else { k_max };
        self.k_active = 0;
        self.lam0 = lam0;
        self.alpha_min = alpha_min;
        self.e_alpha = e_alpha;
        self.dc_alpha = dc_alpha;
        self.e_ema = 0.0;
        self.e_var = 0.05;
        self.e_init = 0;
        self.streak = 0;
        self.dc_streak = 0;
        self.sh_alarm_streak = 0;
        self.en_alarm_streak = 0;
        self.lv_alarm_streak = 0;
        for j in 0..MAX_D {
            self.dc_ema[j] = 0.0;
            self.dc_var[j] = 0.05;
            self.prev_dc[j] = 0.0;
        }
        let wm1 = w - 1;
        let mut m = 0usize;
        for row_k in 0..wm1 { m += if row_k < d { row_k + 1 } else { d }; }
        self.m = m;
        for i in 0..(w * wm1) { self.b_dct[i] = b_dct[i]; }
        for k in 0..MAX_K { self.counts[k] = 0; }
    }

    /// Project a row-major `w × d` window `x_win` into its `Sim+(d)` canonical shadow `u_out`,
    /// per-channel DC means `dc_out`, and log-fluctuation energy `ln(||X_c||_F + 1e-6)`.
    ///
    /// Invariant under proper rotations `SO(d)` (`det = +1`), positive uniform scaling, and
    /// per-channel DC offsets, while remaining sensitive to chiral reflections (`det = -1`).
    #[inline]
    pub fn canonicalize(&self, x_win: &[f32], u_out: &mut [f32; MAX_M], dc_out: &mut [f32; MAX_D]) -> f32 {
        let w = self.w;
        let d = self.d;
        let wm1 = w - 1;
        let mut xc = [0.0f32; MAX_W * MAX_D];
        let mut norm_sq = 0.0f32;
        for j in 0..d {
            let mut mean = 0.0f32;
            for t in 0..w { mean += x_win[t * d + j]; }
            mean /= w as f32;
            dc_out[j] = mean;
            for t in 0..w {
                let v = x_win[t * d + j] - mean;
                xc[t * d + j] = v;
                norm_sq += v * v;
            }
        }
        let norm_raw = sqrt_f32(norm_sq);
        let log_e = ln_f32(norm_raw + 1e-6);
        if norm_raw < 1e-12 {
            for i in 0..self.m { u_out[i] = 0.0; }
            return log_e;
        }
        let inv_norm = 1.0 / norm_raw;

        let mut r_mat = [0.0f32; MAX_D * MAX_W];
        for j in 0..d {
            for k in 0..wm1 {
                let mut acc = 0.0f32;
                for t in 0..w { acc += self.b_dct[t * wm1 + k] * (xc[t * d + j] * inv_norm); }
                r_mat[j * wm1 + k] = acc;
            }
        }

        for col in 0..d {
            let mut row = d - 1;
            while row > col {
                let a = r_mat[(row - 1) * wm1 + col];
                let b = r_mat[row * wm1 + col];
                let rad = sqrt_f32(a * a + b * b);
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
        let mut col_sign = [1.0f32; MAX_D];
        for i in 0..(d - 1) {
            if r_mat[i * wm1 + i] < 0.0 {
                col_sign[i] = -1.0;
                det_sign = -det_sign;
            }
        }
        if det_sign < 0 { col_sign[d - 1] = -1.0; }

        let mut idx = 0usize;
        for row_k in 0..wm1 {
            let max_j = if row_k < d { row_k + 1 } else { d };
            for col_j in 0..max_j {
                u_out[idx] = r_mat[col_j * wm1 + row_k] * col_sign[col_j];
                idx += 1;
            }
        }
        log_e
    }

    #[inline]
    fn dist_to(&self, u: &[f32; MAX_M], k: usize) -> f32 {
        let m = self.m;
        let mu = &self.means[k];
        let p = &self.precs[k];
        let mut diff = [0.0f32; MAX_M];
        for i in 0..m { diff[i] = u[i] - mu[i]; }
        let mut acc = 0.0f32;
        for i in 0..m {
            let mut dot = 0.0f32;
            let row = i * m;
            for j in 0..m { dot += p[row + j] * diff[j]; }
            acc += diff[i] * dot;
        }
        if acc < 0.0 || !acc.is_finite() { 0.0 } else { acc }
    }

    #[inline]
    fn spawn_proto(&mut self, u: &[f32; MAX_M]) {
        let k = self.k_active;
        let m = self.m;
        let inv_lam = 1.0 / self.lam0;
        for i in 0..m {
            self.means[k][i] = u[i];
            for j in 0..m { self.precs[k][i * m + j] = if i == j { inv_lam } else { 0.0 }; }
        }
        self.counts[k] = 1;
        self.k_active += 1;
    }

    #[inline]
    fn sm_update(&mut self, k: usize, u: &[f32; MAX_M]) {
        let m = self.m;
        self.counts[k] = self.counts[k].saturating_add(1);
        let a_cnt = 1.0 / ((self.counts[k] as f32) + 4.0);
        let a = if a_cnt > self.alpha_min { a_cnt } else { self.alpha_min };
        let inv_1ma = 1.0 / (1.0 - a);

        let mut diff = [0.0f32; MAX_M];
        for i in 0..m { diff[i] = u[i] - self.means[k][i]; }

        let mut pv = [0.0f32; MAX_M];
        let mut q = 0.0f32;
        for i in 0..m {
            let mut dot = 0.0f32;
            let row = i * m;
            for j in 0..m { dot += self.precs[k][row + j] * diff[j]; }
            pv[i] = dot;
            q += diff[i] * dot;
        }
        let denom = (1.0 - a) + a * q;
        if denom > 1e-6 && denom.is_finite() {
            let scale = a / denom;
            let max_p = 1.0e5f32;
            for i in 0..m {
                let row = i * m;
                let pvi = pv[i] * scale;
                for j in i..m {
                    let mut val = (self.precs[k][row + j] - pvi * pv[j]) * inv_1ma;
                    if val > max_p { val = max_p; } else if val < -max_p { val = -max_p; }
                    self.precs[k][row + j] = val;
                    self.precs[k][j * m + i] = val;
                }
            }
        }
        for i in 0..m { self.means[k][i] += a * diff[i]; }
    }

    /// Process a single `w × d` window and return `(s_shape, s_energy, s_level, alarm_dual)`.
    ///
    /// - If `huber_clip > 0.0` (Server Telemetry mode): uses Huber-clamped EMA on energy and DC
    ///   level to track slow normal drift without swallowing sustained incidents.
    /// - If `huber_clip <= 0.0` (Thermal / Sensor Chip mode): uses fast ramp EMA, 3-window energy
    ///   hysteresis, and `dc_relock` stable-plateau re-lock.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ppc_gauge_triad_rs::GaugeTriadEngineF32;
    ///
    /// let mut b = [0.0f32; 56];
    /// for k in 0..7 {
    ///     let mut s = 0.0f32;
    ///     for t in 0..8 {
    ///         let v = (core::f32::consts::PI * (t as f32 + 0.5) * ((k + 1) as f32) / 8.0).cos();
    ///         b[t * 7 + k] = v;
    ///         s += v * v;
    ///     }
    ///     for t in 0..8 { b[t * 7 + k] /= s.sqrt(); }
    /// }
    /// let mut eng = GaugeTriadEngineF32::zeroed();
    /// eng.init(8, 4, 3, 5e-3, 0.01, 0.008, 0.004, &b);
    ///
    /// let win = [0.1f32; 32];
    /// let (sh, en, lv, alarm) = eng.step_v61(&win, 1, 1e20, 1, 10.0, 10.0, 2.5, 0, 1.5);
    /// assert!(sh.is_finite() && en.is_finite() && lv.is_finite() && alarm == 0);
    /// ```
    #[inline]
    pub fn step_v61(
        &mut self, x_win: &[f32], learn: u8, shape_gate: f32, adapt_env: u8,
        imp_thr: f32, lvl_thr: f32, sev_mult: f32, dc_relock: u8, huber_clip: f32,
    ) -> (f32, f32, f32, u8) {
        let mut u = [0.0f32; MAX_M];
        let mut dc = [0.0f32; MAX_D];
        let log_e = self.canonicalize(x_win, &mut u, &mut dc);

        let (s_energy, s_level) = if self.e_init == 0 {
            (0.0, 0.0)
        } else {
            let de = log_e - self.e_ema;
            let se = (de * de) / (self.e_var + 0.05);
            let mut sl = 0.0f32;
            for j in 0..self.d {
                let dd = dc[j] - self.dc_ema[j];
                sl += (dd * dd) / (self.dc_var[j] + 0.05);
            }
            (se, sl / (self.d as f32))
        };

        let (best_k, s_shape) = if self.k_active == 0 {
            (0usize, 0.0)
        } else {
            let mut bk = 0usize;
            let mut bd = self.dist_to(&u, 0);
            for k in 1..self.k_active {
                let dk = self.dist_to(&u, k);
                if dk < bd { bd = dk; bk = k; }
            }
            (bk, bd)
        };

        let shape_ok = self.k_active == 0 || s_shape <= shape_gate;

        if learn != 0 && shape_ok {
            if self.k_active == 0 {
                self.spawn_proto(&u);
            } else if self.k_active < self.k_max && s_shape > (0.35 / self.lam0) {
                self.spawn_proto(&u);
            } else {
                self.sm_update(best_k, &u);
            }
        }

        if (learn != 0 || adapt_env != 0) && shape_ok {
            if self.e_init == 0 {
                self.e_ema = log_e;
                for j in 0..self.d {
                    self.dc_ema[j] = dc[j];
                    self.prev_dc[j] = dc[j];
                }
                self.e_init = 1;
                self.streak = 0;
                self.dc_streak = 0;
            } else if huber_clip > 0.0 && learn == 0 {
                // Robust Huber-Clamped EMA: tracks slow normal baseline drift without swallowing sustained anomalies
                let ce = huber_clip * sqrt_f32(self.e_var + 0.05);
                let de_c = clamp_f32(log_e - self.e_ema, -ce, ce);
                self.e_ema += self.e_alpha * de_c;
                self.e_var = (1.0 - self.e_alpha) * self.e_var + self.e_alpha * (de_c * de_c);

                for j in 0..self.d {
                    let cd = huber_clip * sqrt_f32(self.dc_var[j] + 0.05);
                    let dd_c = clamp_f32(dc[j] - self.dc_ema[j], -cd, cd);
                    self.dc_ema[j] += self.dc_alpha * dd_c;
                    self.dc_var[j] = (1.0 - self.dc_alpha) * self.dc_var[j] + self.dc_alpha * (dd_c * dd_c);
                }
            } else {
                // Fast thermal-ramp EMA + hysteresis re-lock
                if s_energy <= imp_thr || learn != 0 {
                    self.streak = 0;
                    let de = log_e - self.e_ema;
                    self.e_ema += self.e_alpha * de;
                    self.e_var = (1.0 - self.e_alpha) * self.e_var + self.e_alpha * (de * de);
                } else {
                    self.streak = self.streak.saturating_add(1);
                    if self.streak >= 3 {
                        self.e_ema = log_e;
                        self.streak = 0;
                    }
                }
                if s_level <= lvl_thr || learn != 0 {
                    self.dc_streak = 0;
                    for j in 0..self.d {
                        let dd = dc[j] - self.dc_ema[j];
                        self.dc_ema[j] += self.dc_alpha * dd;
                        self.dc_var[j] = (1.0 - self.dc_alpha) * self.dc_var[j] + self.dc_alpha * (dd * dd);
                    }
                } else if dc_relock > 0 {
                    let mut step_sq = 0.0f32;
                    for j in 0..self.d {
                        let dprev = dc[j] - self.prev_dc[j];
                        step_sq += dprev * dprev;
                    }
                    step_sq /= self.d as f32;
                    if step_sq < 1.0 {
                        self.dc_streak = self.dc_streak.saturating_add(1);
                        if self.dc_streak >= dc_relock {
                            for j in 0..self.d { self.dc_ema[j] = dc[j]; }
                            self.dc_streak = 0;
                        }
                    } else {
                        self.dc_streak = 0;
                    }
                }
            }
            for j in 0..self.d { self.prev_dc[j] = dc[j]; }
        }

        let mut alarm_dual: u8 = 0;
        if learn == 0 && shape_gate < 1e10 {
            self.sh_alarm_streak = if s_shape > shape_gate { self.sh_alarm_streak.saturating_add(1) } else { 0 };
            self.en_alarm_streak = if s_energy > imp_thr { self.en_alarm_streak.saturating_add(1) } else { 0 };
            self.lv_alarm_streak = if s_level > lvl_thr { self.lv_alarm_streak.saturating_add(1) } else { 0 };

            let trip_sh = (s_shape > shape_gate * sev_mult) || (self.sh_alarm_streak >= 2);
            let trip_en = (s_energy > imp_thr * sev_mult) || (self.en_alarm_streak >= 2);
            let trip_lv = (s_level > lvl_thr * sev_mult) || (self.lv_alarm_streak >= 2);
            if trip_sh || trip_en || trip_lv { alarm_dual = 1; }
        }

        (s_shape, s_energy, s_level, alarm_dual)
    }
}

// ---------------------------------------------------------------------------
// v0.7 On-Chip Zero-Touch Auto-Warmup State Machine (`AutoCalBufF32`)
// ---------------------------------------------------------------------------

/// Operating phase of the on-chip [`AutoCalBufF32`] state machine.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AutoCalState {
    /// Unsupervised online prototype & EMA warmup phase (`0`).
    Warmup = 0,
    /// Frozen-model split-conformal & `log1p`-MAD calibration collection phase (`1`).
    Calibrate = 1,
    /// Active monitoring phase with locked on-chip thresholds and dual-stage gating (`2`).
    Monitor = 2,
}

/// Per-window output returned by [`AutoCalBufF32::step`] and [`AutoCalEngineF32::step`].
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AutoCalStepResult {
    /// Phase in which this window was processed (`0 = Warmup`, `1 = Calibrate`, `2 = Monitor`).
    pub state: u8,
    /// Per-channel dual-stage alarm output from `step_v61` (`0` or `1`).
    pub alarm_channel: u8,
    /// Combined triad dual-stage alarm output (`0` or `1`; always `0` before `Monitor`).
    pub alarm_combined: u8,
    /// Explicit struct alignment padding byte.
    pub _pad: u8,
    /// Raw canonical shape Mahalanobis distance `s_shape`.
    pub s_shape: f32,
    /// Raw relative log-energy deviation `s_energy`.
    pub s_energy: f32,
    /// Raw per-channel DC level deviation `s_level`.
    pub s_level: f32,
    /// Normalized combined triad anomaly score (`0.0` prior to `Monitor`).
    pub s_combined: f32,
}

/// Fixed-size `#![no_std]` on-chip calibration buffer & state machine (`Warmup -> Calibrate -> Monitor`).
///
/// Supports per-device unsupervised auto-warmup (`100–400`+ windows; up to `MAX_CAL = 256`
/// calibration windows) with `0 B` heap allocation and `3,140 B` (`3.07 KB`) static SRAM.
#[repr(C)]
pub struct AutoCalBufF32 {
    /// Current state discriminant (`0 = Warmup`, `1 = Calibrate`, `2 = Monitor`).
    pub state: u8,
    /// Whether environmental EMA adaptation is enabled during `Monitor` (`0` or `1`).
    pub adapt_env: u8,
    /// Stable-plateau DC re-lock window threshold (`0` for Server mode, `6` for Thermal Chip mode).
    pub dc_relock: u8,
    /// Whether dual-stage gating is enabled on the combined score (`0` or `1`).
    pub use_dual: u8,
    /// Consecutive `s_combined > thr_combined` counter.
    pub comb_alarm_streak: u8,
    /// Explicit 3-byte alignment padding.
    pub _pad0: [u8; 3],
    /// Target number of unsupervised warmup windows (`>= 1`).
    pub n_warmup: u16,
    /// Target number of calibration windows (`1..=MAX_CAL`).
    pub n_cal: u16,
    /// Completed warmup windows.
    pub warmup_count: u16,
    /// Collected calibration windows.
    pub cal_count: u16,
    /// Split-conformal miscoverage rate `alpha` (e.g. `0.05`).
    pub alpha: f32,
    /// Dual-stage immediate severity multiplier (e.g. `2.5`).
    pub sev_mult: f32,
    /// Huber-clamp factor (`> 0.0` for Server mode, `0.0` for Thermal Chip mode).
    pub huber_clip: f32,
    /// Calibrated split-conformal shape gate threshold.
    pub thr_shape: f32,
    /// Calibrated split-conformal energy threshold.
    pub thr_energy: f32,
    /// Calibrated split-conformal DC level threshold.
    pub thr_level: f32,
    /// Calibrated split-conformal combined triad score threshold.
    pub thr_combined: f32,
    /// Calibration median of `ln(1 + s_shape)`.
    pub sh_med: f32,
    /// Calibration scale `1.4826 * MAD + 0.1` of `ln(1 + s_shape)`.
    pub sh_mad: f32,
    /// Calibration median of `ln(1 + s_energy)`.
    pub en_med: f32,
    /// Calibration scale `1.4826 * MAD + 0.1` of `ln(1 + s_energy)`.
    pub en_mad: f32,
    /// Calibration median of `ln(1 + s_level)`.
    pub lv_med: f32,
    /// Calibration scale `1.4826 * MAD + 0.1` of `ln(1 + s_level)`.
    pub lv_mad: f32,
    /// Fixed-size buffer storing calibration `s_shape` values.
    pub cal_sh: [f32; MAX_CAL],
    /// Fixed-size buffer storing calibration `s_energy` values.
    pub cal_en: [f32; MAX_CAL],
    /// Fixed-size buffer storing calibration `s_level` values.
    pub cal_lv: [f32; MAX_CAL],
}

impl AutoCalBufF32 {
    /// Const all-zero initializer guaranteeing true `.bss` placement (`0 B` Flash `.data` image).
    /// Call [`AutoCalBufF32::init`] before calling [`AutoCalBufF32::step`].
    pub const fn zeroed() -> Self {
        Self {
            state: 0,
            adapt_env: 0,
            dc_relock: 0,
            use_dual: 0,
            comb_alarm_streak: 0,
            _pad0: [0; 3],
            n_warmup: 0,
            n_cal: 0,
            warmup_count: 0,
            cal_count: 0,
            alpha: 0.0,
            sev_mult: 0.0,
            huber_clip: 0.0,
            thr_shape: 0.0,
            thr_energy: 0.0,
            thr_level: 0.0,
            thr_combined: 0.0,
            sh_med: 0.0,
            sh_mad: 0.0,
            en_med: 0.0,
            en_mad: 0.0,
            lv_med: 0.0,
            lv_mad: 0.0,
            cal_sh: [0.0; MAX_CAL],
            cal_en: [0.0; MAX_CAL],
            cal_lv: [0.0; MAX_CAL],
        }
    }

    /// Initialize (or reset) the calibration state machine with locked a priori parameters.
    pub fn init(
        &mut self,
        n_warmup: u16,
        n_cal: u16,
        alpha: f32,
        adapt_env: u8,
        sev_mult: f32,
        dc_relock: u8,
        huber_clip: f32,
        use_dual: u8,
    ) {
        self.state = AutoCalState::Warmup as u8;
        self.adapt_env = adapt_env;
        self.dc_relock = dc_relock;
        self.use_dual = use_dual;
        self.comb_alarm_streak = 0;
        self._pad0 = [0; 3];
        self.n_warmup = if n_warmup == 0 { 1 } else { n_warmup };
        self.n_cal = if n_cal == 0 {
            1
        } else if (n_cal as usize) > MAX_CAL {
            MAX_CAL as u16
        } else {
            n_cal
        };
        self.warmup_count = 0;
        self.cal_count = 0;
        self.alpha = clamp_f32(alpha, 1e-4, 0.5);
        self.sev_mult = if sev_mult > 1.0 { sev_mult } else { 1.0 };
        self.huber_clip = if huber_clip > 0.0 { huber_clip } else { 0.0 };
        self.thr_shape = 1e20;
        self.thr_energy = 10.0;
        self.thr_level = 10.0;
        self.thr_combined = 1e20;
        self.sh_med = 0.0;
        self.sh_mad = 0.1;
        self.en_med = 0.0;
        self.en_mad = 0.1;
        self.lv_med = 0.0;
        self.lv_mad = 0.1;
    }

    /// Return the current [`AutoCalState`] phase.
    #[inline]
    pub fn state(&self) -> AutoCalState {
        match self.state {
            0 => AutoCalState::Warmup,
            1 => AutoCalState::Calibrate,
            _ => AutoCalState::Monitor,
        }
    }

    /// Compute the calibrated `log1p`-MAD normalized combined triad score.
    #[inline]
    pub fn combined_score(&self, s_shape: f32, s_energy: f32, s_level: f32) -> f32 {
        let rz_sh = (ln_f32(1.0 + sanitize_nonneg_f32(s_shape)) - self.sh_med) / self.sh_mad;
        let rz_en = (ln_f32(1.0 + sanitize_nonneg_f32(s_energy)) - self.en_med) / self.en_mad;
        let rz_lv = (ln_f32(1.0 + sanitize_nonneg_f32(s_level)) - self.lv_med) / self.lv_mad;
        let a = if rz_sh > 0.0 { rz_sh } else { 0.0 };
        let b = if rz_en > 0.0 { rz_en } else { 0.0 };
        let c = if rz_lv > 0.0 { rz_lv } else { 0.0 };
        a + b + c
    }

    fn finalize_calibration(&mut self, engine: &mut GaugeTriadEngineF32) {
        let n = (self.cal_count as usize).min(MAX_CAL);
        if n == 0 {
            self.state = AutoCalState::Monitor as u8;
            return;
        }
        let mut scratch = [0.0f32; MAX_CAL];

        // 1. Per-channel split-conformal thresholds
        for i in 0..n { scratch[i] = self.cal_sh[i]; }
        shell_sort_f32(&mut scratch[..n]);
        self.thr_shape = conformal_quantile_sorted(&scratch[..n], self.alpha);

        for i in 0..n { scratch[i] = self.cal_en[i]; }
        shell_sort_f32(&mut scratch[..n]);
        self.thr_energy = conformal_quantile_sorted(&scratch[..n], self.alpha);

        for i in 0..n { scratch[i] = self.cal_lv[i]; }
        shell_sort_f32(&mut scratch[..n]);
        self.thr_level = conformal_quantile_sorted(&scratch[..n], self.alpha);

        // 2. Robust log1p-MAD normalizers per channel
        let (sh_m, sh_d) = fit_log1p_mad(&self.cal_sh[..n], &mut scratch);
        self.sh_med = sh_m;
        self.sh_mad = sh_d;

        let (en_m, en_d) = fit_log1p_mad(&self.cal_en[..n], &mut scratch);
        self.en_med = en_m;
        self.en_mad = en_d;

        let (lv_m, lv_d) = fit_log1p_mad(&self.cal_lv[..n], &mut scratch);
        self.lv_med = lv_m;
        self.lv_mad = lv_d;

        // 3. Split-conformal threshold on the normalized combined triad score
        for i in 0..n {
            scratch[i] = self.combined_score(self.cal_sh[i], self.cal_en[i], self.cal_lv[i]);
        }
        shell_sort_f32(&mut scratch[..n]);
        self.thr_combined = conformal_quantile_sorted(&scratch[..n], self.alpha);

        // Reset alarm streaks prior to entering Monitor mode
        engine.sh_alarm_streak = 0;
        engine.en_alarm_streak = 0;
        engine.lv_alarm_streak = 0;
        self.comb_alarm_streak = 0;
        self.state = AutoCalState::Monitor as u8;
    }

    /// Execute one streaming window through the on-chip `Warmup -> Calibrate -> Monitor` state machine.
    #[inline]
    pub fn step(&mut self, engine: &mut GaugeTriadEngineF32, x_win: &[f32]) -> AutoCalStepResult {
        match self.state() {
            AutoCalState::Warmup => {
                let (sh, en, lv, _) = engine.step_v61(
                    x_win, 1, 1e20, 1, 10.0, 10.0, self.sev_mult, 0, 0.0,
                );
                self.warmup_count = self.warmup_count.saturating_add(1);
                if self.warmup_count >= self.n_warmup {
                    self.state = AutoCalState::Calibrate as u8;
                }
                AutoCalStepResult {
                    state: AutoCalState::Warmup as u8,
                    alarm_channel: 0,
                    alarm_combined: 0,
                    _pad: 0,
                    s_shape: sh,
                    s_energy: en,
                    s_level: lv,
                    s_combined: 0.0,
                }
            }
            AutoCalState::Calibrate => {
                let (sh, en, lv, _) = engine.step_v61(
                    x_win, 0, 1e20, 0, 10.0, 10.0, self.sev_mult, 0, 0.0,
                );
                let idx = self.cal_count as usize;
                if idx < MAX_CAL {
                    self.cal_sh[idx] = sh;
                    self.cal_en[idx] = en;
                    self.cal_lv[idx] = lv;
                    self.cal_count = self.cal_count.saturating_add(1);
                }
                if self.cal_count >= self.n_cal {
                    self.finalize_calibration(engine);
                }
                AutoCalStepResult {
                    state: AutoCalState::Calibrate as u8,
                    alarm_channel: 0,
                    alarm_combined: 0,
                    _pad: 0,
                    s_shape: sh,
                    s_energy: en,
                    s_level: lv,
                    s_combined: 0.0,
                }
            }
            AutoCalState::Monitor => {
                let (sh, en, lv, al_ch) = engine.step_v61(
                    x_win,
                    0,
                    self.thr_shape,
                    self.adapt_env,
                    self.thr_energy,
                    self.thr_level,
                    self.sev_mult,
                    self.dc_relock,
                    self.huber_clip,
                );
                let s_comb = self.combined_score(sh, en, lv);
                self.comb_alarm_streak = if s_comb > self.thr_combined {
                    self.comb_alarm_streak.saturating_add(1)
                } else {
                    0
                };
                let al_comb: u8 = if self.use_dual != 0 {
                    if (s_comb > self.thr_combined * self.sev_mult) || (self.comb_alarm_streak >= 2) {
                        1
                    } else {
                        0
                    }
                } else if s_comb > self.thr_combined {
                    1
                } else {
                    0
                };
                AutoCalStepResult {
                    state: AutoCalState::Monitor as u8,
                    alarm_channel: al_ch,
                    alarm_combined: al_comb,
                    _pad: 0,
                    s_shape: sh,
                    s_energy: en,
                    s_level: lv,
                    s_combined: s_comb,
                }
            }
        }
    }
}

/// Complete single-struct Zero-Touch Auto-Warmup Engine ([`GaugeTriadEngineF32`] + [`AutoCalBufF32`]).
///
/// Total static SRAM is enforced to remain `< 16 KB` at compile time on every target
/// (`14,768 B` on 64-bit, `14,744 B` on 32-bit ARM Cortex-M4F).
#[repr(C)]
pub struct AutoCalEngineF32 {
    /// Core PPC-Gauge-Triad engine (`~11.4 KB`).
    pub engine: GaugeTriadEngineF32,
    /// Fixed-size on-chip calibration buffer and state machine (`~3.1 KB`).
    pub cal: AutoCalBufF32,
}

impl AutoCalEngineF32 {
    /// Const zero-initializer suitable for static `.bss` storage on bare-metal MCUs.
    pub const fn zeroed() -> Self {
        Self {
            engine: GaugeTriadEngineF32::zeroed(),
            cal: AutoCalBufF32::zeroed(),
        }
    }

    /// Initialize both the core engine and the on-chip `AutoCalBufF32` state machine.
    pub fn init(
        &mut self,
        w: usize,
        d: usize,
        k_max: usize,
        lam0: f32,
        alpha_min: f32,
        e_alpha: f32,
        dc_alpha: f32,
        b_dct: &[f32],
        n_warmup: u16,
        n_cal: u16,
        alpha: f32,
        adapt_env: u8,
        sev_mult: f32,
        dc_relock: u8,
        huber_clip: f32,
        use_dual: u8,
    ) {
        self.engine.init(w, d, k_max, lam0, alpha_min, e_alpha, dc_alpha, b_dct);
        self.cal.init(n_warmup, n_cal, alpha, adapt_env, sev_mult, dc_relock, huber_clip, use_dual);
    }

    /// Advance the combined engine by one `w × d` row-major window.
    #[inline]
    pub fn step(&mut self, x_win: &[f32]) -> AutoCalStepResult {
        self.cal.step(&mut self.engine, x_win)
    }

    /// Return the current [`AutoCalState`] phase.
    #[inline]
    pub fn state(&self) -> AutoCalState {
        self.cal.state()
    }
}

// Compile-time embedded SRAM guarantees (AGENTS.md Rule 3: strictly < 16 KB)
const _: () = assert!(core::mem::size_of::<GaugeTriadEngineF32>() < 16 * 1024);
const _: () = assert!(core::mem::size_of::<AutoCalEngineF32>() < 16 * 1024);

// ---------------------------------------------------------------------------
// C FFI Exports (v0.6.1 compatible + v0.7 AutoCal)
// ---------------------------------------------------------------------------

/// Return `sizeof(GaugeTriadEngineF32)` in bytes for ABI validation and static buffer sizing.
#[no_mangle]
pub extern "C" fn ppc_triad_engine_bytes() -> usize {
    core::mem::size_of::<GaugeTriadEngineF32>()
}

/// Initialize a caller-allocated [`GaugeTriadEngineF32`] instance via C FFI.
///
/// # Safety
/// - `eng` must be a valid, aligned, writable pointer to a `GaugeTriadEngineF32`.
/// - `b_dct` must point to at least `w * (w - 1)` readable `f32` values (`2 <= w <= MAX_W`, `1 <= d <= MAX_D`).
#[no_mangle]
pub unsafe extern "C" fn ppc_triad_init(
    eng: *mut GaugeTriadEngineF32, w: usize, d: usize, k_max: usize,
    lam0: f32, alpha_min: f32, e_alpha: f32, dc_alpha: f32, b_dct: *const f32,
) {
    let b_s = core::slice::from_raw_parts(b_dct, w * (w - 1));
    (*eng).init(w, d, k_max, lam0, alpha_min, e_alpha, dc_alpha, b_s);
}

/// Process a batch of `n_samples` row-major `w × d` windows through [`GaugeTriadEngineF32::step_v61`].
///
/// # Safety
/// - `eng` must point to an initialized [`GaugeTriadEngineF32`].
/// - `x_batch` must point to at least `n_samples * w * d` readable `f32` values.
/// - `out_shape`, `out_energy`, `out_level`, and `out_alarm` must each point to at least
///   `n_samples` writable elements and must not alias `x_batch`.
#[no_mangle]
pub unsafe extern "C" fn ppc_triad_run_stream_v61(
    eng: *mut GaugeTriadEngineF32, x_batch: *const f32, n_samples: usize,
    learn: u8, shape_gate: f32, adapt_env: u8, imp_thr: f32, lvl_thr: f32,
    sev_mult: f32, dc_relock: u8, huber_clip: f32,
    out_shape: *mut f32, out_energy: *mut f32, out_level: *mut f32, out_alarm: *mut u8,
) {
    let e = &mut *eng;
    let stride = e.w * e.d;
    let x_all = core::slice::from_raw_parts(x_batch, n_samples * stride);
    let sh_s = core::slice::from_raw_parts_mut(out_shape, n_samples);
    let en_s = core::slice::from_raw_parts_mut(out_energy, n_samples);
    let lv_s = core::slice::from_raw_parts_mut(out_level, n_samples);
    let al_s = core::slice::from_raw_parts_mut(out_alarm, n_samples);
    for i in 0..n_samples {
        let win = &x_all[i * stride..(i + 1) * stride];
        let (sh, en, lv, al) = e.step_v61(win, learn, shape_gate, adapt_env, imp_thr, lvl_thr, sev_mult, dc_relock, huber_clip);
        sh_s[i] = sh; en_s[i] = en; lv_s[i] = lv; al_s[i] = al;
    }
}

/// Return `sizeof(AutoCalBufF32)` in bytes for ABI validation.
#[no_mangle]
pub extern "C" fn ppc_autocal_buf_bytes() -> usize {
    core::mem::size_of::<AutoCalBufF32>()
}

/// Return `sizeof(AutoCalEngineF32)` in bytes for ABI validation and static buffer sizing.
#[no_mangle]
pub extern "C" fn ppc_autocal_engine_bytes() -> usize {
    core::mem::size_of::<AutoCalEngineF32>()
}

/// Initialize a caller-allocated [`AutoCalEngineF32`] instance via C FFI.
///
/// # Safety
/// - `eng` must be a valid, aligned, writable pointer to an [`AutoCalEngineF32`].
/// - `b_dct` must point to at least `w * (w - 1)` readable `f32` values (`2 <= w <= MAX_W`, `1 <= d <= MAX_D`).
#[no_mangle]
pub unsafe extern "C" fn ppc_autocal_init(
    eng: *mut AutoCalEngineF32,
    w: usize,
    d: usize,
    k_max: usize,
    lam0: f32,
    alpha_min: f32,
    e_alpha: f32,
    dc_alpha: f32,
    b_dct: *const f32,
    n_warmup: u16,
    n_cal: u16,
    alpha: f32,
    adapt_env: u8,
    sev_mult: f32,
    dc_relock: u8,
    huber_clip: f32,
    use_dual: u8,
) {
    let b_s = core::slice::from_raw_parts(b_dct, w * (w - 1));
    (*eng).init(
        w, d, k_max, lam0, alpha_min, e_alpha, dc_alpha, b_s,
        n_warmup, n_cal, alpha, adapt_env, sev_mult, dc_relock, huber_clip, use_dual,
    );
}

/// Query the current state discriminant (`0 = Warmup`, `1 = Calibrate`, `2 = Monitor`).
///
/// # Safety
/// - `eng` must point to an initialized [`AutoCalEngineF32`].
#[no_mangle]
pub unsafe extern "C" fn ppc_autocal_state(eng: *const AutoCalEngineF32) -> u8 {
    (*eng).cal.state
}

/// Process a batch of `n_samples` windows through the on-chip [`AutoCalEngineF32::step`] pipeline.
///
/// # Safety
/// - `eng` must point to an initialized [`AutoCalEngineF32`].
/// - `x_batch` must point to at least `n_samples * w * d` readable `f32` values.
/// - All output buffers (`out_state`, `out_shape`, `out_energy`, `out_level`, `out_combined`,
///   `out_alarm_combined`) must each point to at least `n_samples` writable elements and must not
///   alias `x_batch`.
#[no_mangle]
pub unsafe extern "C" fn ppc_autocal_run_stream(
    eng: *mut AutoCalEngineF32,
    x_batch: *const f32,
    n_samples: usize,
    out_state: *mut u8,
    out_shape: *mut f32,
    out_energy: *mut f32,
    out_level: *mut f32,
    out_combined: *mut f32,
    out_alarm_combined: *mut u8,
) {
    let e = &mut *eng;
    let stride = e.engine.w * e.engine.d;
    let x_all = core::slice::from_raw_parts(x_batch, n_samples * stride);
    let st_s = core::slice::from_raw_parts_mut(out_state, n_samples);
    let sh_s = core::slice::from_raw_parts_mut(out_shape, n_samples);
    let en_s = core::slice::from_raw_parts_mut(out_energy, n_samples);
    let lv_s = core::slice::from_raw_parts_mut(out_level, n_samples);
    let cb_s = core::slice::from_raw_parts_mut(out_combined, n_samples);
    let al_s = core::slice::from_raw_parts_mut(out_alarm_combined, n_samples);
    for i in 0..n_samples {
        let win = &x_all[i * stride..(i + 1) * stride];
        let r = e.step(win);
        st_s[i] = r.state;
        sh_s[i] = r.s_shape;
        en_s[i] = r.s_energy;
        lv_s[i] = r.s_level;
        cb_s[i] = r.s_combined;
        al_s[i] = r.alarm_combined;
    }
}
