# Changelog

All notable changes to `ppc_gauge_triad_rs` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-11-22

### Added
- **On-Chip Zero-Touch Auto-Warmup (`AutoCalBufF32` & `AutoCalEngineF32`):**
  Fixed-size `#![no_std]` state machine (`Warmup -> Calibrate -> Monitor`, `MAX_CAL = 256`)
  computing split-conformal quantiles, log1p-MAD normalizers, combined triad scores,
  and dual-stage alarm gating directly on chip with `0 B` heap allocation and `< 16 KB`
  combined static SRAM (`14,768 B` x86-64 / `14,744 B` Cortex-M4F, compile-time enforced).
- **C / Embedded Adopter Ecosystem:**
  - Official C99/C++ ABI header [`include/ppc_gauge_triad.h`](include/ppc_gauge_triad.h).
  - Standalone C streaming demo [`examples/c_stream_demo.c`](examples/c_stream_demo.c) verifying
    byte-exact struct ABI parity and `Warmup -> Calibrate -> Monitor` execution.
  - External hardware evaluation template [`examples/level3_cortex_m4f_eval_template.rs`](examples/level3_cortex_m4f_eval_template.rs)
    for ARM Cortex-M4F DWT cycle-counter benchmarking (Level 3 physical silicon verification remains `PENDING`).
- **Rustdoc & `crates.io` Packaging:**
  - Complete `Cargo.toml` metadata (`MIT OR Apache-2.0`), `LICENSE-MIT`, `LICENSE-APACHE`,
    public API `///` docs, `# Safety` contracts, and runnable doctests.
- **Verification & Reproducible Preprint Assets:**
  - Rust unit/integration test suites (`tests/level1_invariants.rs`, `tests/autocal_v07.rs`).
  - Criterion host CPU micro-benchmarks (`benches/triad_bench.rs`).
  - Local Python harness with offline SMD/PSM `.npz` caching (`benchmarks/`), 300-DPI
    figure generator (`benchmarks/generate_paper_figures.py`), and arXiv/IEEE LaTeX preprint
    draft (`paper/ppc_gauge_triad.tex`).

## [0.6.1] - 2026-11-22

### Verified
- **Level 1 (Algorithmic Invariants):** 9/9 passed (`SO(4)` rotation, `O(4)` chirality,
  scale `[1e-3..1e4]`, DC offset `±10`/`±1000`, combined `Sim+(4)`, 5 rank-deficient/near-singular
  cases, `f32` vs `f64`, Python-vs-Rust parity, and 1,000,000-window online stability with `asym == 0.0`).
- **Level 2 (Software & Real-World Hardening):** Verified Huber-Clamped EMA (Server mode),
  Stable-Plateau DC Re-Lock (Thermal Chip mode), Dual-Stage Alarm Gate, Orthogonal `3PC+Resid`
  feature, 8-server SMD/PSM replay (`78.7%` High-Recall / `77.1%` Low-FAR), cross-device
  unsupervised auto-warmup curve (`29.7% -> 75.3% -> 77.8% -> 82.9%`), 50,000-window 5-seed
  streaming (`0.30% ± 0.13%` FAR, `100.0%` fault recall), OpenML physical benchmarks,
  and 5-target cross-compilation (`~11.4 KB` core SRAM, `0 B` heap, `0` matrix inversions).
- **Level 3 (Physical Hardware Verification):** Explicitly documented as `PENDING` physical
  MCU/FPGA silicon measurement.
