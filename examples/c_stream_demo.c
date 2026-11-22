/*
 * Minimal zero-heap C99 streaming demo for PPC-Gauge-Triad v0.7.0.
 *
 * Build & run on Linux after `cargo build --release`:
 *   gcc -Wall -Wextra -Werror -std=c99 -O2 -Iinclude examples/c_stream_demo.c \
 *       -Ltarget/release -lppc_gauge_triad_rs -Wl,-rpath,target/release -lm -o c_stream_demo
 *   ./c_stream_demo
 */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ppc_gauge_triad.h"

#define DEMO_W      8u
#define DEMO_D      4u
#define N_WARMUP    150u
#define N_CAL       50u
#define N_NORM_MON  40u
#define N_FAULT_MON 20u
#define N_TOTAL     (N_WARMUP + N_CAL + N_NORM_MON + N_FAULT_MON)

/* Static .bss allocation: 0 B heap */
static AutoCalEngineF32 g_autocal;
static GaugeTriadEngineF32 g_core;
static float g_stream[N_TOTAL * DEMO_W * DEMO_D];
static uint8_t g_states[N_TOTAL];
static float g_sh[N_TOTAL];
static float g_en[N_TOTAL];
static float g_lv[N_TOTAL];
static float g_comb[N_TOTAL];
static uint8_t g_alarm[N_TOTAL];

static uint64_t g_rng = 0x5EEDCAFE12345678ULL;

static float next_noise(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    /* Uniform in [-0.15, 0.15] */
    float u = (float)((g_rng >> 40) & 0xFFFFFFu) / (float)0x1000000u;
    return (u - 0.5f) * 0.30f;
}

static void make_dct_basis(size_t w, float *b_out) {
    const double pi = 3.14159265358979323846;
    size_t wm1 = w - 1u;
    for (size_t k = 0; k < wm1; ++k) {
        double norm_sq = 0.0;
        for (size_t t = 0; t < w; ++t) {
            double v = cos(pi * ((double)t + 0.5) * (double)(k + 1u) / (double)w);
            b_out[t * wm1 + k] = (float)v;
            norm_sq += v * v;
        }
        float inv = (float)(1.0 / sqrt(norm_sq));
        for (size_t t = 0; t < w; ++t) {
            b_out[t * wm1 + k] *= inv;
        }
    }
}

int main(void) {
    /* 1. Strict C-vs-Rust ABI size verification */
    size_t rs_core = ppc_triad_engine_bytes();
    size_t rs_buf  = ppc_autocal_buf_bytes();
    size_t rs_auto = ppc_autocal_engine_bytes();

    printf("C ABI check: Core=%zu B (Rust=%zu B), AutoCalBuf=%zu B (Rust=%zu B), AutoCalEngine=%zu B (Rust=%zu B)\n",
           sizeof(GaugeTriadEngineF32), rs_core,
           sizeof(AutoCalBufF32), rs_buf,
           sizeof(AutoCalEngineF32), rs_auto);

    assert(sizeof(GaugeTriadEngineF32) == rs_core);
    assert(sizeof(AutoCalBufF32) == rs_buf);
    assert(sizeof(AutoCalEngineF32) == rs_auto);
    assert(sizeof(AutoCalEngineF32) < 16384u);

    /* 2. Build DCT-II basis and deterministic synthetic sensor stream */
    float b_dct[DEMO_W * (DEMO_W - 1u)];
    make_dct_basis(DEMO_W, b_dct);

    for (size_t i = 0; i < N_TOTAL; ++i) {
        int is_fault = (i >= (N_WARMUP + N_CAL + N_NORM_MON));
        for (size_t t = 0; t < DEMO_W; ++t) {
            float phi = (float)t * 0.55f + (float)i * 0.12f;
            size_t base = (i * DEMO_W + t) * DEMO_D;
            float c0 = sinf(phi) + next_noise();
            float c1 = cosf(phi) + next_noise();
            float c2 = 0.6f * sinf(2.0f * phi) + next_noise();
            float c3 = 0.5f * cosf(0.5f * phi) + next_noise();
            if (is_fault) {
                c0 = c0 * 3.5f + 8.0f;
                c1 = -c1 * 2.0f - 6.0f;
            }
            g_stream[base + 0u] = c0;
            g_stream[base + 1u] = c1;
            g_stream[base + 2u] = c2;
            g_stream[base + 3u] = c3;
        }
    }

    /* 3. Zero-touch on-chip AutoCalEngineF32 (Locked Low-FAR preset: 0.008 / 0.004 / 1.5) */
    memset(&g_autocal, 0, sizeof(g_autocal));
    ppc_autocal_init(
        &g_autocal, DEMO_W, DEMO_D, 3u,
        5e-3f, 0.01f, 0.008f, 0.004f, b_dct,
        (uint16_t)N_WARMUP, (uint16_t)N_CAL, 0.05f,
        1u, 2.5f, 0u, 1.5f, 1u
    );
    assert(ppc_autocal_state(&g_autocal) == PPC_AUTOCAL_STATE_WARMUP);

    ppc_autocal_run_stream(
        &g_autocal, g_stream, N_TOTAL,
        g_states, g_sh, g_en, g_lv, g_comb, g_alarm
    );
    assert(ppc_autocal_state(&g_autocal) == PPC_AUTOCAL_STATE_MONITOR);

    /* Verify phase transitions & zero alarms during Warmup + Calibrate */
    for (size_t i = 0; i < N_WARMUP; ++i) {
        assert(g_states[i] == PPC_AUTOCAL_STATE_WARMUP);
        assert(g_alarm[i] == 0u);
    }
    for (size_t i = N_WARMUP; i < N_WARMUP + N_CAL; ++i) {
        assert(g_states[i] == PPC_AUTOCAL_STATE_CALIBRATE);
        assert(g_alarm[i] == 0u);
    }

    size_t false_alarms = 0u;
    for (size_t i = N_WARMUP + N_CAL; i < N_WARMUP + N_CAL + N_NORM_MON; ++i) {
        assert(g_states[i] == PPC_AUTOCAL_STATE_MONITOR);
        false_alarms += (g_alarm[i] != 0u) ? 1u : 0u;
    }

    size_t detected = 0u;
    for (size_t i = N_WARMUP + N_CAL + N_NORM_MON; i < N_TOTAL; ++i) {
        assert(g_states[i] == PPC_AUTOCAL_STATE_MONITOR);
        detected += (g_alarm[i] != 0u) ? 1u : 0u;
    }

    printf("AutoCalEngineF32 demo: normal FA=%zu/%u, fault recall=%zu/%u, thr_comb=%.3f\n",
           false_alarms, N_NORM_MON, detected, N_FAULT_MON, g_autocal.cal.thr_combined);
    assert(false_alarms <= 3u);
    assert(detected == N_FAULT_MON);

    /* 4. Smoke-check v0.6.1 direct GaugeTriadEngineF32 FFI */
    memset(&g_core, 0, sizeof(g_core));
    ppc_triad_init(&g_core, DEMO_W, DEMO_D, 3u, 5e-3f, 0.01f, 0.008f, 0.004f, b_dct);
    ppc_triad_run_stream_v61(
        &g_core, g_stream, 16u, 1u, 1e20f, 1u, 10.0f, 10.0f, 2.5f, 0u, 1.5f,
        g_sh, g_en, g_lv, g_alarm
    );
    assert(g_core.k_active >= 1u);

    printf("c_stream_demo PASS (0 heap, ABI verified).\n");
    return 0;
}
