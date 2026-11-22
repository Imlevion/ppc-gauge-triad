//! v0.7 On-Chip `AutoCalBufF32` / `AutoCalEngineF32` Verification Suite.
//!
//! Verifies:
//!   - Strict `< 16 KB` combined static SRAM budget (`GaugeTriadEngineF32` + `AutoCalBufF32`)
//!   - Deterministic `Warmup -> Calibrate -> Monitor` state machine transitions
//!   - Exact numerical agreement with split-conformal + log1p-MAD reference (`< 1e-4`)
//!   - Anomaly detection & dual-stage gating after zero-touch on-chip auto-warmup
//!   - Bit-exact C FFI (`ppc_autocal_*`) parity with Rust API
//!   - Edge cases (clamped bounds, all-zero inputs, 0 NaN/Inf)

use core::mem::size_of;
use ppc_gauge_triad_rs::{
    ppc_autocal_buf_bytes, ppc_autocal_engine_bytes, ppc_autocal_init, ppc_autocal_run_stream,
    ppc_autocal_state, ppc_triad_engine_bytes, AutoCalBufF32, AutoCalEngineF32, AutoCalState,
    GaugeTriadEngineF32, MAX_CAL,
};

const W: usize = 8;
const D: usize = 4;

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
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        if u <= 1e-15 { 1e-15 } else { u }
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

fn gen_normal_stream(seed: u64, n_win: usize) -> Vec<f32> {
    let mut rng = Rng64::new(seed);
    let mut out = vec![0.0f32; n_win * W * D];
    for i in 0..n_win {
        let phase = (i as f64) * 0.15;
        for t in 0..W {
            let tt = t as f64 * 0.55 + phase;
            let base = (i * W + t) * D;
            out[base] = (tt.sin() + 0.12 * rng.next_normal()) as f32;
            out[base + 1] = (tt.cos() + 0.12 * rng.next_normal()) as f32;
            out[base + 2] = ((2.0 * tt).sin() * 0.6 + 0.12 * rng.next_normal()) as f32;
            out[base + 3] = ((0.5 * tt).cos() * 0.5 + 0.12 * rng.next_normal()) as f32;
        }
    }
    out
}

fn conformal_f64(vals: &[f32], alpha: f64) -> f64 {
    let mut s: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let idx = (((1.0 - alpha) * ((n + 1) as f64)).ceil() as isize - 1).clamp(0, (n - 1) as isize)
        as usize;
    s[idx]
}

fn median_f64(vals: &[f64]) -> f64 {
    let mut s = vals.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

fn triad_norm_f64(
    sh_c: &[f32],
    en_c: &[f32],
    lv_c: &[f32],
    sh_t: &[f32],
    en_t: &[f32],
    lv_t: &[f32],
) -> Vec<f64> {
    let rz = |c: &[f32], t: &[f32]| -> Vec<f64> {
        let lc: Vec<f64> = c.iter().map(|&v| (1.0 + v.max(0.0) as f64).ln()).collect();
        let med = median_f64(&lc);
        let dev: Vec<f64> = lc.iter().map(|&x| (x - med).abs()).collect();
        let mad = median_f64(&dev) * 1.4826 + 0.1;
        t.iter()
            .map(|&v| (((1.0 + v.max(0.0) as f64).ln() - med) / mad).max(0.0))
            .collect()
    };
    let a = rz(sh_c, sh_t);
    let b = rz(en_c, en_t);
    let c = rz(lv_c, lv_t);
    (0..sh_t.len()).map(|i| a[i] + b[i] + c[i]).collect()
}

#[test]
fn v07_sram_budget_strictly_under_16kb() {
    let core_b = ppc_triad_engine_bytes();
    let buf_b = ppc_autocal_buf_bytes();
    let total_b = ppc_autocal_engine_bytes();

    assert_eq!(core_b, size_of::<GaugeTriadEngineF32>());
    assert_eq!(buf_b, size_of::<AutoCalBufF32>());
    assert_eq!(total_b, size_of::<AutoCalEngineF32>());

    println!(
        "v0.7 SRAM: GaugeTriadEngineF32={core_b} B, AutoCalBufF32={buf_b} B, AutoCalEngineF32={total_b} B ({:.2} KB)",
        total_b as f64 / 1024.0
    );
    assert!(
        total_b < 16 * 1024,
        "Combined v0.7 SRAM exceeded 16 KB limit: {total_b} B >= 16384 B"
    );
}

#[test]
fn v07_state_machine_transitions_warmup_calibrate_monitor() {
    let b_dct = make_dct_f32(W);
    let mut eng = AutoCalEngineF32::zeroed();
    eng.init(W, D, 3, 5e-3, 0.01, 0.008, 0.004, &b_dct, 75, 25, 0.05, 1, 2.5, 0, 1.5, 1);

    assert_eq!(eng.state(), AutoCalState::Warmup);
    let data = gen_normal_stream(701, 150);

    for i in 0..150 {
        let win = &data[i * W * D..(i + 1) * W * D];
        let res = eng.step(win);
        if i < 75 {
            assert_eq!(res.state, AutoCalState::Warmup as u8);
            assert_eq!(res.alarm_combined, 0);
            assert_eq!(res.alarm_channel, 0);
        } else if i < 100 {
            assert_eq!(res.state, AutoCalState::Calibrate as u8);
            assert_eq!(res.alarm_combined, 0);
            assert_eq!(res.alarm_channel, 0);
        } else {
            assert_eq!(res.state, AutoCalState::Monitor as u8);
            assert!(res.s_combined.is_finite() && res.s_combined >= 0.0);
        }
    }

    assert_eq!(eng.state(), AutoCalState::Monitor);
    assert_eq!(eng.cal.warmup_count, 75);
    assert_eq!(eng.cal.cal_count, 25);
    assert!(eng.cal.thr_shape.is_finite() && eng.cal.thr_shape > 0.0);
    assert!(eng.cal.thr_combined.is_finite() && eng.cal.thr_combined > 0.0);
    assert!(eng.cal.sh_mad >= 0.1 && eng.cal.en_mad >= 0.1 && eng.cal.lv_mad >= 0.1);
}

#[test]
fn v07_on_chip_autocal_matches_manual_v061_split_conformal() {
    let b_dct = make_dct_f32(W);
    let n_warm = 150usize;
    let n_cal = 50usize;
    let n_mon = 80usize;
    let data = gen_normal_stream(702, n_warm + n_cal + n_mon);

    // 1. On-chip v0.7 AutoCalEngineF32 (zero external threshold/MAD logic)
    let mut auto = AutoCalEngineF32::zeroed();
    auto.init(
        W, D, 3, 5e-3, 0.01, 0.008, 0.004, &b_dct, n_warm as u16, n_cal as u16, 0.05, 1, 2.5, 0,
        1.5, 1,
    );
    let mut auto_comb = Vec::with_capacity(n_mon);
    let mut auto_alarm = Vec::with_capacity(n_mon);
    for i in 0..(n_warm + n_cal + n_mon) {
        let win = &data[i * W * D..(i + 1) * W * D];
        let r = auto.step(win);
        if i >= n_warm + n_cal {
            auto_comb.push(r.s_combined as f64);
            auto_alarm.push(r.alarm_combined != 0);
        }
    }

    // 2. Manual v0.6.1 GaugeTriadEngineF32 + f64 reference split-conformal & log1p-MAD
    let mut manual = GaugeTriadEngineF32::zeroed();
    manual.init(W, D, 3, 5e-3, 0.01, 0.008, 0.004, &b_dct);
    for i in 0..n_warm {
        let win = &data[i * W * D..(i + 1) * W * D];
        manual.step_v61(win, 1, 1e20, 1, 10.0, 10.0, 2.5, 0, 0.0);
    }
    let mut sh_c = Vec::with_capacity(n_cal);
    let mut en_c = Vec::with_capacity(n_cal);
    let mut lv_c = Vec::with_capacity(n_cal);
    for i in n_warm..(n_warm + n_cal) {
        let win = &data[i * W * D..(i + 1) * W * D];
        let (sh, en, lv, _) = manual.step_v61(win, 0, 1e20, 0, 10.0, 10.0, 2.5, 0, 0.0);
        sh_c.push(sh);
        en_c.push(en);
        lv_c.push(lv);
    }
    let thr_sh = conformal_f64(&sh_c, 0.05) as f32;
    let cal_comb = triad_norm_f64(&sh_c, &en_c, &lv_c, &sh_c, &en_c, &lv_c);
    let cal_comb_f32: Vec<f32> = cal_comb.iter().map(|&x| x as f32).collect();
    let thr_comb = conformal_f64(&cal_comb_f32, 0.05);

    let mut sh_t = Vec::with_capacity(n_mon);
    let mut en_t = Vec::with_capacity(n_mon);
    let mut lv_t = Vec::with_capacity(n_mon);
    for i in (n_warm + n_cal)..(n_warm + n_cal + n_mon) {
        let win = &data[i * W * D..(i + 1) * W * D];
        let (sh, en, lv, _) = manual.step_v61(win, 0, thr_sh, 1, 10.0, 10.0, 2.5, 0, 1.5);
        sh_t.push(sh);
        en_t.push(en);
        lv_t.push(lv);
    }
    let ref_comb = triad_norm_f64(&sh_c, &en_c, &lv_c, &sh_t, &en_t, &lv_t);

    assert!(
        (auto.cal.thr_shape - thr_sh).abs() < 1e-6,
        "thr_shape mismatch: on-chip={} vs ref={}",
        auto.cal.thr_shape,
        thr_sh
    );
    assert!(
        (auto.cal.thr_combined as f64 - thr_comb).abs() < 1e-4,
        "thr_combined mismatch: on-chip={} vs ref={}",
        auto.cal.thr_combined,
        thr_comb
    );

    let mut max_comb_diff = 0.0f64;
    let mut st = 0usize;
    let mut ref_alarm = Vec::with_capacity(n_mon);
    for i in 0..n_mon {
        max_comb_diff = max_comb_diff.max((auto_comb[i] - ref_comb[i]).abs());
        st = if ref_comb[i] > thr_comb { st + 1 } else { 0 };
        ref_alarm.push((ref_comb[i] > thr_comb * 2.5) || (st >= 2));
    }

    assert!(
        max_comb_diff < 1e-4,
        "On-chip combined score diverged from reference: max_diff={max_comb_diff:.3e}"
    );
    assert_eq!(auto_alarm, ref_alarm, "On-chip alarm decisions diverged from reference");
}

#[test]
fn v07_autocal_detects_faults_with_low_false_alarm_rate() {
    let b_dct = make_dct_f32(W);
    let mut eng = AutoCalEngineF32::zeroed();
    // 400w zero-touch auto-warmup (300 warmup + 100 calibrate), High-Recall preset
    eng.init(W, D, 3, 5e-3, 0.01, 0.002, 0.001, &b_dct, 300, 100, 0.05, 1, 2.5, 0, 1.0, 1);

    let normal = gen_normal_stream(703, 600);
    // Warmup + Calibrate on first 400 windows
    for i in 0..400 {
        eng.step(&normal[i * W * D..(i + 1) * W * D]);
    }
    assert_eq!(eng.state(), AutoCalState::Monitor);

    // Held-out 200 normal windows: expect low FAR with dual-stage gate
    let mut false_alarms = 0usize;
    for i in 400..600 {
        let r = eng.step(&normal[i * W * D..(i + 1) * W * D]);
        if r.alarm_combined != 0 {
            false_alarms += 1;
        }
    }
    let far = false_alarms as f64 / 200.0;
    assert!(far < 0.05, "Held-out normal FAR too high after 400w auto-warmup: {far:.3}");

    // Inject 20 sustained anomalous windows (DC + amplitude + shape distortion)
    let mut detected = 0usize;
    for i in 0..20 {
        let mut anom = normal[(400 + i) * W * D..(401 + i) * W * D].to_vec();
        for t in 0..W {
            anom[t * D] = anom[t * D] * 3.5 + 8.0;
            anom[t * D + 1] = -anom[t * D + 1] * 2.0 - 6.0;
        }
        let r = eng.step(&anom);
        if r.alarm_combined != 0 {
            detected += 1;
        }
    }
    assert_eq!(detected, 20, "Expected 100% detection on severe sustained fault, got {detected}/20");
}

#[test]
fn v07_autocal_ffi_exact_parity_with_rust_api() {
    let b_dct = make_dct_f32(W);
    let data = gen_normal_stream(704, 160);

    let mut rs = AutoCalEngineF32::zeroed();
    rs.init(W, D, 3, 5e-3, 0.01, 0.008, 0.004, &b_dct, 75, 25, 0.05, 1, 2.5, 0, 1.5, 1);

    let mut rs_st = vec![0u8; 160];
    let mut rs_sh = vec![0.0f32; 160];
    let mut rs_en = vec![0.0f32; 160];
    let mut rs_lv = vec![0.0f32; 160];
    let mut rs_cb = vec![0.0f32; 160];
    let mut rs_al = vec![0u8; 160];
    for i in 0..160 {
        let r = rs.step(&data[i * W * D..(i + 1) * W * D]);
        rs_st[i] = r.state;
        rs_sh[i] = r.s_shape;
        rs_en[i] = r.s_energy;
        rs_lv[i] = r.s_level;
        rs_cb[i] = r.s_combined;
        rs_al[i] = r.alarm_combined;
    }

    let mut ffi = AutoCalEngineF32::zeroed();
    let mut ffi_st = vec![0u8; 160];
    let mut ffi_sh = vec![0.0f32; 160];
    let mut ffi_en = vec![0.0f32; 160];
    let mut ffi_lv = vec![0.0f32; 160];
    let mut ffi_cb = vec![0.0f32; 160];
    let mut ffi_al = vec![0u8; 160];
    let final_state: u8;

    unsafe {
        ppc_autocal_init(
            &mut ffi as *mut AutoCalEngineF32,
            W, D, 3, 5e-3, 0.01, 0.008, 0.004, b_dct.as_ptr(),
            75, 25, 0.05, 1, 2.5, 0, 1.5, 1,
        );
        ppc_autocal_run_stream(
            &mut ffi as *mut AutoCalEngineF32,
            data.as_ptr(),
            160,
            ffi_st.as_mut_ptr(),
            ffi_sh.as_mut_ptr(),
            ffi_en.as_mut_ptr(),
            ffi_lv.as_mut_ptr(),
            ffi_cb.as_mut_ptr(),
            ffi_al.as_mut_ptr(),
        );
        final_state = ppc_autocal_state(&ffi as *const AutoCalEngineF32);
    }

    assert_eq!(final_state, AutoCalState::Monitor as u8);
    assert_eq!(rs_st, ffi_st);
    assert_eq!(rs_sh, ffi_sh);
    assert_eq!(rs_en, ffi_en);
    assert_eq!(rs_lv, ffi_lv);
    assert_eq!(rs_cb, ffi_cb);
    assert_eq!(rs_al, ffi_al);
}

#[test]
fn v07_autocal_edge_cases_clamping_and_zero_stream() {
    let b_dct = make_dct_f32(W);
    let mut eng = AutoCalEngineF32::zeroed();
    // Out-of-range bounds must safely clamp (n_warmup=0 -> 1, n_cal=999 -> MAX_CAL)
    eng.init(W, D, 3, 5e-3, 0.01, 0.008, 0.004, &b_dct, 0, 999, 0.05, 1, 2.5, 0, 1.5, 1);
    assert_eq!(eng.cal.n_warmup, 1);
    assert_eq!(eng.cal.n_cal, MAX_CAL as u16);

    // Feed all-zero stream through Warmup -> Calibrate -> Monitor: must never produce NaN/Inf
    let zeros = [0.0f32; W * D];
    for _ in 0..(1 + MAX_CAL + 10) {
        let r = eng.step(&zeros);
        assert!(r.s_shape.is_finite() && r.s_shape >= 0.0);
        assert!(r.s_energy.is_finite() && r.s_energy >= 0.0);
        assert!(r.s_level.is_finite() && r.s_level >= 0.0);
        assert!(r.s_combined.is_finite() && r.s_combined >= 0.0);
    }
    assert_eq!(eng.state(), AutoCalState::Monitor);
}
