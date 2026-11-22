//! Turnkey Bare-Metal ARM Cortex-M4F (`thumbv7em-none-eabihf`) Evaluation Firmware
//! for `ppc_gauge_triad_rs` v0.7.1 (`#![no_std]`, `0 B` heap, `0` matrix inversions).
//!
//! ## Verification Scope (`AGENTS.md` Rule 4)
//! - **Level 2.5 (No Physical Hardware Required):**
//!   - Default build (`cargo build --release`): produces a real linked ARMv7E-M `.elf` using
//!     `cortex-m-rt` + `panic-halt` + `ppc_gauge_triad_rs` (`default-features = false`), proving
//!     zero duplicate `#[panic_handler]` conflict and enabling exact `.text`/`.rodata`/`.data`/`.bss`
//!     measurement via `arm-none-eabi-size` or `benchmarks/inspect_elf.py`.
//!   - QEMU build (`cargo build --release --features qemu`): boots and executes the compiled
//!     ARMv7E-M hard-float (`thumbv7em-none-eabihf`) binary on `qemu-system-arm -machine mps2-an386`,
//!     verifying functional execution of `Warmup -> Calibrate -> Monitor` + severe fault detection
//!     with zero HardFaults. QEMU never produces Level 3 silicon timing.
//! - **Level 3 (Physical Silicon Verification — PENDING):**
//!   Flash the default `.elf` onto STM32F411 / STM32F446 / STM32F407 / nRF52840 hardware and
//!   read `LEVEL3_DWT_REPORT` via `probe-rs` / GDB after `LEVEL3_DONE == 0xD04E_0700`.

#![no_std]
#![no_main]

use core::mem::size_of;
use core::ptr::{addr_of_mut, read_volatile, write_volatile};

use cortex_m::asm;
use cortex_m_rt::entry;
use panic_halt as _;
#[cfg(feature = "qemu")]
use ppc_gauge_triad_rs::AutoCalState;
use ppc_gauge_triad_rs::{AutoCalEngineF32, GaugeTriadEngineF32, MAX_D, MAX_M};

#[cfg(feature = "qemu")]
use cortex_m_semihosting::{debug, hprintln};

const W: usize = 8;
const D: usize = 4;

/// Precomputed orthonormal temporal DCT-II basis for `w = 8` (`8 × 7`, stored in `.rodata` Flash).
static DCT_W8: [f32; 56] = [
    0.49039264, 0.46193977, 0.41573481, 0.35355339, 0.27778512, 0.19134172, 0.09754516,
    0.41573481, 0.19134172, -0.09754516, -0.35355339, -0.49039264, -0.46193977, -0.27778512,
    0.27778512, -0.19134172, -0.49039264, -0.35355339, 0.09754516, 0.46193977, 0.41573481,
    0.09754516, -0.46193977, -0.27778512, 0.35355339, 0.41573481, -0.19134172, -0.49039264,
    -0.09754516, -0.46193977, 0.27778512, 0.35355339, -0.41573481, -0.19134172, 0.49039264,
    -0.27778512, -0.19134172, 0.49039264, -0.35355339, -0.09754516, 0.46193977, -0.41573481,
    -0.41573481, 0.19134172, 0.09754516, -0.35355339, 0.49039264, -0.46193977, 0.27778512,
    -0.49039264, 0.46193977, -0.41573481, 0.35355339, -0.27778512, 0.19134172, -0.09754516,
];

/// Structured Level 3 silicon measurement record inspected via `probe-rs` / GDB.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Level3DwtReport {
    pub core_struct_bytes: u32,
    pub autocal_engine_bytes: u32,
    pub cycles_canonicalize: u32,
    pub cycles_step_score_only: u32,
    pub cycles_step_live_learn: u32,
    pub cycles_autocal_monitor: u32,
    pub cycles_autocal_400w_total: u32,
    pub final_state: u8,
    pub fault_alarm_combined: u8,
    pub all_finite: u8,
    pub _pad: u8,
}

impl Level3DwtReport {
    const fn zeroed() -> Self {
        Self {
            core_struct_bytes: 0,
            autocal_engine_bytes: 0,
            cycles_canonicalize: 0,
            cycles_step_score_only: 0,
            cycles_step_live_learn: 0,
            cycles_autocal_monitor: 0,
            cycles_autocal_400w_total: 0,
            final_state: 0,
            fault_alarm_combined: 0,
            all_finite: 0,
            _pad: 0,
        }
    }
}

/// Engine static `.bss` storage (`14,744 B` on `thumbv7em-none-eabihf`, `0 B` heap).
static mut STATIC_ENGINE: AutoCalEngineF32 = AutoCalEngineF32::zeroed();

/// Exported report symbol for GDB / `probe-rs` inspection on physical hardware.
#[no_mangle]
pub static mut LEVEL3_DWT_REPORT: Level3DwtReport = Level3DwtReport::zeroed();

/// Completion sentinel (`0xD04E_0700` once benchmark loop finishes).
#[no_mangle]
pub static mut LEVEL3_DONE: u32 = 0;

mod dwt {
    use core::ptr::{read_volatile, write_volatile};

    const DEMCR: *mut u32 = 0xE000_EDFC as *mut u32;
    const DWT_CTRL: *mut u32 = 0xE000_1000 as *mut u32;
    const DWT_CYCCNT: *mut u32 = 0xE000_1004 as *mut u32;

    const DEMCR_TRCENA: u32 = 1 << 24;
    const DWT_CTRL_CYCCNTENA: u32 = 1 << 0;

    #[inline(always)]
    pub unsafe fn enable_cycle_counter() {
        let demcr = read_volatile(DEMCR);
        write_volatile(DEMCR, demcr | DEMCR_TRCENA);
        write_volatile(DWT_CYCCNT, 0);
        let ctrl = read_volatile(DWT_CTRL);
        write_volatile(DWT_CTRL, ctrl | DWT_CTRL_CYCCNTENA);
    }

    #[inline(always)]
    pub unsafe fn read_cycles() -> u32 {
        read_volatile(DWT_CYCCNT)
    }
}

#[inline]
fn fill_deterministic_window(step: u32, out: &mut [f32; W * D]) {
    let mut x = step.wrapping_mul(1664525).wrapping_add(1013904223);
    for t in 0..W {
        let base = t * D;
        for j in 0..D {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let frac = ((x >> 16) & 0xFFFF) as f32 * (1.0 / 32768.0) - 1.0;
            let wave = ((t + j) as f32 - 3.5) * 0.25;
            out[base + j] = wave + 0.15 * frac;
        }
    }
}

#[entry]
fn main() -> ! {
    // SAFETY: Single-threaded bare-metal entry; cortex-m-rt enables CP10/CP11 FPU before main.
    let report = unsafe {
        dwt::enable_cycle_counter();

        let eng = &mut *addr_of_mut!(STATIC_ENGINE);
        let mut win = [0.0f32; W * D];
        let mut all_finite: u8 = 1;

        // 1. 400-window on-chip Zero-Touch Auto-Warmup (300 Warmup + 100 Calibrate)
        //    using locked Thermal / Sensor Chip preset (e_alpha=0.20, dc_alpha=0.25, dc_relock=6).
        eng.init(W, D, 3, 5e-3, 0.01, 0.20, 0.25, &DCT_W8, 300, 100, 0.05, 1, 2.5, 6, 0.0, 1);
        let t0 = dwt::read_cycles();
        for step in 0..400u32 {
            fill_deterministic_window(step, &mut win);
            let r = core::hint::black_box(eng.step(&win));
            if !r.s_shape.is_finite() || !r.s_energy.is_finite() || !r.s_level.is_finite() {
                all_finite = 0;
            }
        }
        let cyc_400w = dwt::read_cycles().wrapping_sub(t0);

        // 2. canonicalize() latency (64-window average)
        let mut u = [0.0f32; MAX_M];
        let mut dc = [0.0f32; MAX_D];
        let t1 = dwt::read_cycles();
        for step in 400..464u32 {
            fill_deterministic_window(step, &mut win);
            let log_e = core::hint::black_box(eng.engine.canonicalize(&win, &mut u, &mut dc));
            if !log_e.is_finite() {
                all_finite = 0;
            }
        }
        let cyc_canon = dwt::read_cycles().wrapping_sub(t1) / 64;

        // 3. AutoCalEngineF32::step() in Monitor mode (64-window average)
        let t2 = dwt::read_cycles();
        for step in 464..528u32 {
            fill_deterministic_window(step, &mut win);
            let r = core::hint::black_box(eng.step(&win));
            if !r.s_combined.is_finite() {
                all_finite = 0;
            }
        }
        let cyc_mon = dwt::read_cycles().wrapping_sub(t2) / 64;

        // 4. Severe multi-channel fault injection check (must trip dual-stage alarm immediately)
        let fault = [25.0f32; W * D];
        let fault_res = eng.step(&fault);

        // 5. Core step_v61() score-only & live-learn (64-window averages)
        let t3 = dwt::read_cycles();
        for step in 528..592u32 {
            fill_deterministic_window(step, &mut win);
            let _ = core::hint::black_box(eng.engine.step_v61(
                &win,
                0,
                eng.cal.thr_shape,
                1,
                eng.cal.thr_energy,
                eng.cal.thr_level,
                2.5,
                6,
                0.0,
            ));
        }
        let cyc_score = dwt::read_cycles().wrapping_sub(t3) / 64;

        let t4 = dwt::read_cycles();
        for step in 592..656u32 {
            fill_deterministic_window(step, &mut win);
            let _ = core::hint::black_box(eng.engine.step_v61(
                &win,
                1,
                500.0,
                1,
                eng.cal.thr_energy,
                eng.cal.thr_level,
                2.5,
                6,
                0.0,
            ));
        }
        let cyc_learn = dwt::read_cycles().wrapping_sub(t4) / 64;

        let rep = Level3DwtReport {
            core_struct_bytes: size_of::<GaugeTriadEngineF32>() as u32,
            autocal_engine_bytes: size_of::<AutoCalEngineF32>() as u32,
            cycles_canonicalize: cyc_canon,
            cycles_step_score_only: cyc_score,
            cycles_step_live_learn: cyc_learn,
            cycles_autocal_monitor: cyc_mon,
            cycles_autocal_400w_total: cyc_400w,
            final_state: eng.state() as u8,
            fault_alarm_combined: fault_res.alarm_combined,
            all_finite,
            _pad: 0,
        };

        write_volatile(addr_of_mut!(LEVEL3_DWT_REPORT), rep);
        write_volatile(addr_of_mut!(LEVEL3_DONE), 0xD04E_0700);
        read_volatile(addr_of_mut!(LEVEL3_DWT_REPORT))
    };

    #[cfg(feature = "qemu")]
    {
        let ok = report.core_struct_bytes < 16 * 1024
            && report.autocal_engine_bytes < 16 * 1024
            && report.final_state == AutoCalState::Monitor as u8
            && report.fault_alarm_combined == 1
            && report.all_finite == 1;

        let _ = hprintln!("=== PPC-Gauge-Triad v0.7.1 — Level 2.5 ARM Cortex-M4F QEMU Proof ===");
        let _ = hprintln!("Target                  : thumbv7em-none-eabihf (mps2-an386, FPU enabled)");
        let _ = hprintln!("Core struct (.bss)      : {} B", report.core_struct_bytes);
        let _ = hprintln!("AutoCalEngineF32 (.bss) : {} B (< 16 KB)", report.autocal_engine_bytes);
        let _ = hprintln!("State after 400w warmup : {} (2 = Monitor)", report.final_state);
        let _ = hprintln!("Injected fault alarm    : {} (expected 1)", report.fault_alarm_combined);
        let _ = hprintln!("All outputs finite      : {} (expected 1)", report.all_finite);
        let _ = hprintln!("Level 2.5 Emulation     : {}", if ok { "PASS" } else { "FAIL" });
        let _ = hprintln!("Level 3 Silicon Cycles  : PENDING (QEMU timing is NOT physical silicon)");

        if ok {
            debug::exit(debug::EXIT_SUCCESS);
        } else {
            debug::exit(debug::EXIT_FAILURE);
        }
    }

    #[cfg(not(feature = "qemu"))]
    {
        let _ = report;
    }

    loop {
        asm::wfi();
    }
}
