/*
 * PPC-Gauge-Triad v0.7.1 — Official C99 / C++ ABI Header
 *
 * Matches `ppc_gauge_triad_rs` v0.7.1 (`#![no_std]`, f32, 0 B heap, 0 matrix inversions).
 * Embedded static SRAM footprint:
 *   - GaugeTriadEngineF32 : 11,624 B (64-bit) / 11,604 B (ARM Cortex-M4F)
 *   - AutoCalBufF32       :  3,140 B (all targets)
 *   - AutoCalEngineF32    : 14,768 B (64-bit) / 14,744 B (ARM Cortex-M4F) (< 16 KB)
 *
 * Dual-licensed under MIT OR Apache-2.0.
 */

#ifndef PPC_GAUGE_TRIAD_H
#define PPC_GAUGE_TRIAD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PPC_TRIAD_VERSION_MAJOR 0
#define PPC_TRIAD_VERSION_MINOR 7
#define PPC_TRIAD_VERSION_PATCH 1

#define PPC_TRIAD_MAX_W   8u
#define PPC_TRIAD_MAX_D   4u
#define PPC_TRIAD_MAX_M   26u
#define PPC_TRIAD_MAX_K   4u
#define PPC_TRIAD_MAX_CAL 256u

#define PPC_AUTOCAL_STATE_WARMUP    0u
#define PPC_AUTOCAL_STATE_CALIBRATE 1u
#define PPC_AUTOCAL_STATE_MONITOR   2u

/**
 * Core `#![no_std]` f32 PPC-Gauge-Triad engine (matches Rust `#[repr(C)] GaugeTriadEngineF32`).
 */
typedef struct GaugeTriadEngineF32 {
    size_t w;
    size_t d;
    size_t m;
    size_t k_max;
    size_t k_active;
    float lam0;
    float alpha_min;
    float e_alpha;
    float dc_alpha;
    float e_ema;
    float e_var;
    float dc_ema[PPC_TRIAD_MAX_D];
    float dc_var[PPC_TRIAD_MAX_D];
    float prev_dc[PPC_TRIAD_MAX_D];
    uint8_t e_init;
    uint8_t streak;
    uint8_t dc_streak;
    uint8_t sh_alarm_streak;
    uint8_t en_alarm_streak;
    uint8_t lv_alarm_streak;
    float b_dct[PPC_TRIAD_MAX_W * PPC_TRIAD_MAX_W];
    float means[PPC_TRIAD_MAX_K][PPC_TRIAD_MAX_M];
    float precs[PPC_TRIAD_MAX_K][PPC_TRIAD_MAX_M * PPC_TRIAD_MAX_M];
    uint32_t counts[PPC_TRIAD_MAX_K];
} GaugeTriadEngineF32;

/**
 * Per-window step output (matches Rust `#[repr(C)] AutoCalStepResult`).
 */
typedef struct AutoCalStepResult {
    uint8_t state;
    uint8_t alarm_channel;
    uint8_t alarm_combined;
    uint8_t _pad;
    float s_shape;
    float s_energy;
    float s_level;
    float s_combined;
} AutoCalStepResult;

/**
 * Fixed-size on-chip calibration buffer & state machine (matches Rust `#[repr(C)] AutoCalBufF32`).
 */
typedef struct AutoCalBufF32 {
    uint8_t state;
    uint8_t adapt_env;
    uint8_t dc_relock;
    uint8_t use_dual;
    uint8_t comb_alarm_streak;
    uint8_t _pad0[3];
    uint16_t n_warmup;
    uint16_t n_cal;
    uint16_t warmup_count;
    uint16_t cal_count;
    float alpha;
    float sev_mult;
    float huber_clip;
    float thr_shape;
    float thr_energy;
    float thr_level;
    float thr_combined;
    float sh_med;
    float sh_mad;
    float en_med;
    float en_mad;
    float lv_med;
    float lv_mad;
    float cal_sh[PPC_TRIAD_MAX_CAL];
    float cal_en[PPC_TRIAD_MAX_CAL];
    float cal_lv[PPC_TRIAD_MAX_CAL];
} AutoCalBufF32;

/**
 * Complete on-chip Zero-Touch Auto-Warmup engine (matches Rust `#[repr(C)] AutoCalEngineF32`).
 */
typedef struct AutoCalEngineF32 {
    GaugeTriadEngineF32 engine;
    AutoCalBufF32 cal;
} AutoCalEngineF32;

/* --------------------------------------------------------------------------
 * Core Engine FFI (v0.6.1 compatible)
 * -------------------------------------------------------------------------- */

size_t ppc_triad_engine_bytes(void);

void ppc_triad_init(
    GaugeTriadEngineF32 *eng,
    size_t w,
    size_t d,
    size_t k_max,
    float lam0,
    float alpha_min,
    float e_alpha,
    float dc_alpha,
    const float *b_dct
);

void ppc_triad_run_stream_v61(
    GaugeTriadEngineF32 *eng,
    const float *x_batch,
    size_t n_samples,
    uint8_t learn,
    float shape_gate,
    uint8_t adapt_env,
    float imp_thr,
    float lvl_thr,
    float sev_mult,
    uint8_t dc_relock,
    float huber_clip,
    float *out_shape,
    float *out_energy,
    float *out_level,
    uint8_t *out_alarm
);

/* --------------------------------------------------------------------------
 * v0.7 On-Chip Auto-Warmup State Machine FFI (`Warmup -> Calibrate -> Monitor`)
 * -------------------------------------------------------------------------- */

size_t ppc_autocal_buf_bytes(void);
size_t ppc_autocal_engine_bytes(void);

void ppc_autocal_init(
    AutoCalEngineF32 *eng,
    size_t w,
    size_t d,
    size_t k_max,
    float lam0,
    float alpha_min,
    float e_alpha,
    float dc_alpha,
    const float *b_dct,
    uint16_t n_warmup,
    uint16_t n_cal,
    float alpha,
    uint8_t adapt_env,
    float sev_mult,
    uint8_t dc_relock,
    float huber_clip,
    uint8_t use_dual
);

uint8_t ppc_autocal_state(const AutoCalEngineF32 *eng);

void ppc_autocal_run_stream(
    AutoCalEngineF32 *eng,
    const float *x_batch,
    size_t n_samples,
    uint8_t *out_state,
    float *out_shape,
    float *out_energy,
    float *out_level,
    float *out_combined,
    uint8_t *out_alarm_combined
);

#ifdef __cplusplus
}
#endif

#endif /* PPC_GAUGE_TRIAD_H */
