#![cfg_attr(not(feature = "std"), no_std)]

//! PPC-Gauge-Triad v0.6.1 (`#![no_std]` f32):
//! Orthogonal 3-channel gauge decomposition for streaming telemetry & physical sensors:
//!   1. `s_shape`  : Exact Sim+(d) Givens-QR canonical shadow (det=+1).
//!   2. `s_energy` : Relative log-fluctuation energy (Huber-clamped EMA for server drift OR 3-window plateau hysteresis for thermal chips).
//!   3. `s_level`  : Per-channel DC mean drift (Huber-clamped EMA to track drift WITHOUT swallowing sustained anomalies, OR stable-plateau re-lock for abrupt chip steps).
//!   4. Dual-Stage Alarm Gate (`step_v61`): Immediate trip (`> sev_mult * thr`) OR 2-consecutive-window confirmation.
//!
//! Hardware profile: ~11.4 KB static struct SRAM, 0 B heap, 0 matrix inversions.

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

const MAX_W: usize = 8;
const MAX_D: usize = 4;
const MAX_M: usize = 26;
const MAX_K: usize = 4;

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

#[repr(C)]
pub struct GaugeTriadEngineF32 {
    pub w: usize,
    pub d: usize,
    pub m: usize,
    pub k_max: usize,
    pub k_active: usize,
    pub lam0: f32,
    pub alpha_min: f32,
    pub e_alpha: f32,
    pub dc_alpha: f32,
    pub e_ema: f32,
    pub e_var: f32,
    pub dc_ema: [f32; MAX_D],
    pub dc_var: [f32; MAX_D],
    pub prev_dc: [f32; MAX_D],
    pub e_init: u8,
    pub streak: u8,
    pub dc_streak: u8,
    pub sh_alarm_streak: u8,
    pub en_alarm_streak: u8,
    pub lv_alarm_streak: u8,
    pub b_dct: [f32; MAX_W * MAX_W],
    pub means: [[f32; MAX_M]; MAX_K],
    pub precs: [[f32; MAX_M * MAX_M]; MAX_K],
    pub counts: [u32; MAX_K],
}

impl GaugeTriadEngineF32 {
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

    /// Returns `(s_shape, s_energy, s_level, alarm_dual)`.
    /// - If `huber_clip > 0.0` (Server Telemetry mode): uses Huber-clamped EMA on energy & DC level -> tracks slow normal drift WITHOUT swallowing sustained incidents!
    /// - If `huber_clip <= 0.0` (Thermal Chip mode): uses fast ramp EMA + 3-win energy hysteresis + `dc_relock` stable-plateau re-lock.
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

#[no_mangle]
pub extern "C" fn ppc_triad_engine_bytes() -> usize {
    core::mem::size_of::<GaugeTriadEngineF32>()
}

#[no_mangle]
pub unsafe extern "C" fn ppc_triad_init(
    eng: *mut GaugeTriadEngineF32, w: usize, d: usize, k_max: usize,
    lam0: f32, alpha_min: f32, e_alpha: f32, dc_alpha: f32, b_dct: *const f32,
) {
    let b_s = core::slice::from_raw_parts(b_dct, w * (w - 1));
    (*eng).init(w, d, k_max, lam0, alpha_min, e_alpha, dc_alpha, b_s);
}

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
