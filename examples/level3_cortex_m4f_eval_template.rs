//! # Level 3 Hardware Evaluation Template — ARM Cortex-M4F (`thumbv7em-none-eabihf`)
//!
//! **IMPORTANT (AGENTS.md Rule 4 — No Unverified Hardware Claims):**
//! This file is an **evaluation template for external hardware teams** with access to physical
//! ARM Cortex-M4F silicon (e.g., STM32F407 / STM32F446 / STM32H7 / nRF52840). Level 3 physical
//! silicon claims (DWT cycle count, µs latency at board `SYSCLK`, mW power consumption, and
//! `.elf`/`.map` Flash & SRAM footprints) are **PENDING** until measured on real hardware.
//!
//! ## How External Hardware Engineers Use This Template
//!
//! 1. Drop `run_cortex_m4f_dwt_eval()` into your board's `cortex-m-rt` `#[entry]` firmware
//!    (linking `ppc_gauge_triad_rs` with `default-features = false` for `#![no_std]`).
//!    Note: `ppc_gauge_triad_rs` intentionally does **not** define a `#[panic_handler]` by
//!    default, so your firmware binary must supply one (e.g. `panic-halt` or `panic-probe`)
//!    without risking duplicate `panic_impl` linker conflicts.
//! 2. Enable the CoreSight DWT cycle counter (provided below via raw ARMv7E-M MMIO addresses
//!    `DEMCR = 0xE000_EDFC`, `DWT_CTRL = 0xE000_1000`, `DWT_CYCCNT = 0xE000_1004` — zero extra
//!    peripheral crates required).
//! 3. Flash and capture output via `probe-rs` / `defmt` / RTT or SWO UART.
//! 4. Measure full firmware `.text` / `.rodata` / `.bss` / `.data` via `arm-none-eabi-size` or
//!    `cargo bloat --release --target thumbv7em-none-eabihf` (remember: the `< 16 KB` compile-time
//!    assertion covers only the engine struct `.bss` size, NOT stack, interrupts, DMA/sensor
//!    buffers, or RTOS/BSP runtime RAM).
//! 5. Measure active mW power with an inline power profiler (e.g., Nordic PPK2 / Otii Arc / oscilloscope shunt)
//!    while looping `AutoCalEngineF32::step()` on real IMU / thermal ADC windows.
//!
//! Running this file on a host PC (`cargo run --example level3_cortex_m4f_eval_template`) performs
//! only a **structural dry-run** (SRAM size + state-machine check) and never reports simulated MCU cycles.

#![cfg_attr(target_arch = "arm", no_std)]
#![cfg_attr(target_arch = "arm", no_main)]

use core::mem::size_of;
#[cfg(not(target_arch = "arm"))]
use ppc_gauge_triad_rs::AutoCalState;
use ppc_gauge_triad_rs::{AutoCalEngineF32, GaugeTriadEngineF32, MAX_D, MAX_M};

const W: usize = 8;
const D: usize = 4;

/// Precomputed orthonormal temporal DCT-II basis for `w = 8` (`8 × 7`, stored in `.rodata` Flash).
/// Eliminates any runtime `cos()`/`libm` dependency on bare-metal MCUs.
pub const DCT_W8: [f32; 56] = [
    0.49039264, 0.46193977, 0.41573481, 0.35355339, 0.27778512, 0.19134172, 0.09754516,
    0.41573481, 0.19134172, -0.09754516, -0.35355339, -0.49039264, -0.46193977, -0.27778512,
    0.27778512, -0.19134172, -0.49039264, -0.35355339, 0.09754516, 0.46193977, 0.41573481,
    0.09754516, -0.46193977, -0.27778512, 0.35355339, 0.41573481, -0.19134172, -0.49039264,
    -0.09754516, -0.46193977, 0.27778512, 0.35355339, -0.41573481, -0.19134172, 0.49039264,
    -0.27778512, -0.19134172, 0.49039264, -0.35355339, -0.09754516, 0.46193977, -0.41573481,
    -0.41573481, 0.19134172, 0.09754516, -0.35355339, 0.49039264, -0.46193977, 0.27778512,
    -0.49039264, 0.46193977, -0.41573481, 0.35355339, -0.27778512, 0.19134172, -0.09754516,
];

/// Structured Level 3 silicon measurement record to be logged via `defmt` / RTT / UART.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Level3DwtReport {
    pub core_struct_bytes: u32,
    pub autocal_engine_bytes: u32,
    pub cycles_canonicalize: u32,
    pub cycles_step_score_only: u32,
    pub cycles_step_live_learn: u32,
    pub cycles_autocal_monitor: u32,
    pub cycles_autocal_400w_total: u32,
    pub final_state: u8,
}

/// Static `.bss` engine instance (`0 B` heap, `< 16 KB` static SRAM).
static mut STATIC_ENGINE: AutoCalEngineF32 = AutoCalEngineF32::zeroed();

#[inline]
fn fill_deterministic_window(step: u32, out: &mut [f32; W * D]) {
    // Pure integer-to-f32 waveform generator (zero libm calls on MCU)
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

// ---------------------------------------------------------------------------
// Firmware-Side Panic Handler (active on bare-metal ARM targets)
// Replace with `use panic_halt as _;` or `use panic_probe as _;` in your BSP binary.
// Compiling this alongside `ppc_gauge_triad_rs` (`default-features = false`) also
// verifies zero duplicate `panic_impl` lang-item conflict from the library crate.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "arm")]
#[panic_handler]
fn firmware_panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

// ---------------------------------------------------------------------------
// ARM Cortex-M4F CoreSight DWT Cycle Counter Helpers (active on ARM targets)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "arm")]
mod dwt {
    const DEMCR: *mut u32 = 0xE000_EDFC as *mut u32;
    const DWT_CTRL: *mut u32 = 0xE000_1000 as *mut u32;
    const DWT_CYCCNT: *mut u32 = 0xE000_1004 as *mut u32;

    const DEMCR_TRCENA: u32 = 1 << 24;
    const DWT_CTRL_CYCCNTENA: u32 = 1 << 0;

    #[inline(always)]
    pub unsafe fn enable_cycle_counter() {
        let demcr = core::ptr::read_volatile(DEMCR);
        core::ptr::write_volatile(DEMCR, demcr | DEMCR_TRCENA);
        core::ptr::write_volatile(DWT_CYCCNT, 0);
        let ctrl = core::ptr::read_volatile(DWT_CTRL);
        core::ptr::write_volatile(DWT_CTRL, ctrl | DWT_CTRL_CYCCNTENA);
    }

    #[inline(always)]
    pub unsafe fn read_cycles() -> u32 {
        core::ptr::read_volatile(DWT_CYCCNT)
    }
}

/// Bare-metal ARM Cortex-M4F DWT evaluation entry function.
///
/// Call this from your board's `#[entry]` function after configuring clocks and FPU (CP10/CP11).
#[cfg(target_arch = "arm")]
#[no_mangle]
pub unsafe extern "C" fn run_cortex_m4f_dwt_eval() -> Level3DwtReport {
    dwt::enable_cycle_counter();

    let eng = &mut *core::ptr::addr_of_mut!(STATIC_ENGINE);
    let mut win = [0.0f32; W * D];

    // 1. Measure full 400-window on-chip Zero-Touch Auto-Warmup (300 warmup + 100 calibrate)
    //    using locked Thermal / Sensor Chip preset (e_alpha=0.20, dc_alpha=0.25, dc_relock=6).
    eng.init(W, D, 3, 5e-3, 0.01, 0.20, 0.25, &DCT_W8, 300, 100, 0.05, 1, 2.5, 6, 0.0, 1);
    let t0 = dwt::read_cycles();
    for step in 0..400u32 {
        fill_deterministic_window(step, &mut win);
        core::hint::black_box(eng.step(&win));
    }
    let cyc_400w = dwt::read_cycles().wrapping_sub(t0);

    // 2. Measure canonicalize() latency (average over 64 windows)
    let mut u = [0.0f32; MAX_M];
    let mut dc = [0.0f32; MAX_D];
    let t1 = dwt::read_cycles();
    for step in 400..464u32 {
        fill_deterministic_window(step, &mut win);
        core::hint::black_box(eng.engine.canonicalize(&win, &mut u, &mut dc));
    }
    let cyc_canon = dwt::read_cycles().wrapping_sub(t1) / 64;

    // 3. Measure AutoCalEngineF32::step() in Monitor mode (average over 64 windows)
    let t2 = dwt::read_cycles();
    for step in 464..528u32 {
        fill_deterministic_window(step, &mut win);
        core::hint::black_box(eng.step(&win));
    }
    let cyc_mon = dwt::read_cycles().wrapping_sub(t2) / 64;

    // 4. Measure core step_v61() score-only & live-learn (average over 64 windows)
    let t3 = dwt::read_cycles();
    for step in 528..592u32 {
        fill_deterministic_window(step, &mut win);
        core::hint::black_box(eng.engine.step_v61(&win, 0, eng.cal.thr_shape, 1, eng.cal.thr_energy, eng.cal.thr_level, 2.5, 6, 0.0));
    }
    let cyc_score = dwt::read_cycles().wrapping_sub(t3) / 64;

    let t4 = dwt::read_cycles();
    for step in 592..656u32 {
        fill_deterministic_window(step, &mut win);
        core::hint::black_box(eng.engine.step_v61(&win, 1, 500.0, 1, eng.cal.thr_energy, eng.cal.thr_level, 2.5, 6, 0.0));
    }
    let cyc_learn = dwt::read_cycles().wrapping_sub(t4) / 64;

    Level3DwtReport {
        core_struct_bytes: size_of::<GaugeTriadEngineF32>() as u32,
        autocal_engine_bytes: size_of::<AutoCalEngineF32>() as u32,
        cycles_canonicalize: cyc_canon,
        cycles_step_score_only: cyc_score,
        cycles_step_live_learn: cyc_learn,
        cycles_autocal_monitor: cyc_mon,
        cycles_autocal_400w_total: cyc_400w,
        final_state: eng.state() as u8,
    }
}

// ---------------------------------------------------------------------------
// Host Structural Dry-Run (compiles & checks template without claiming MCU cycles)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "arm"))]
fn main() {
    let mut win = [0.0f32; W * D];
    let mut u = [0.0f32; MAX_M];
    let mut dc = [0.0f32; MAX_D];

    // SAFETY: Single-threaded host dry-run verifying static `.bss` engine initialization & execution.
    let (final_state, thr_comb) = unsafe {
        let eng = &mut *core::ptr::addr_of_mut!(STATIC_ENGINE);
        eng.init(W, D, 3, 5e-3, 0.01, 0.20, 0.25, &DCT_W8, 300, 100, 0.05, 1, 2.5, 6, 0.0, 1);
        for step in 0..420u32 {
            fill_deterministic_window(step, &mut win);
            let r = eng.step(&win);
            assert!(r.s_shape.is_finite() && r.s_combined.is_finite());
        }
        let log_e = eng.engine.canonicalize(&win, &mut u, &mut dc);
        assert!(log_e.is_finite());
        (eng.state(), eng.cal.thr_combined)
    };

    assert_eq!(final_state, AutoCalState::Monitor);
    assert!(size_of::<AutoCalEngineF32>() < 16 * 1024);

    println!("Level 3 Cortex-M4F Evaluation Template — Host Structural Dry-Run PASS");
    println!("  Core GaugeTriadEngineF32 : {} B", size_of::<GaugeTriadEngineF32>());
    println!("  AutoCalEngineF32 (.bss)  : {} B (< 16 KB)", size_of::<AutoCalEngineF32>());
    println!("  State after 420 windows  : {:?} (thr_combined={:.3})", final_state, thr_comb);
    println!("  Level 3 Physical Silicon : PENDING (flash on thumbv7em-none-eabihf to read DWT CYCCNT)");
}
