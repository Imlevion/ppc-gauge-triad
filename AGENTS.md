# AGENTS.md — PPC-Gauge-Triad v0.6.1 Engineering Context (Google Antigravity App)

## Project Identity
- **Crate:** `ppc_gauge_triad_rs` v0.6.1 (`#![no_std]` f32 Rust, `~11.4 KB` SRAM, `0 B` heap, `0` matrix inversions).
- **Environment:** Google Antigravity App (Windows, Local Checkout) + **GitHub Remote MCP** active (`47 tools enabled`). Do not assume Context7 or Docker is installed.
- **Mission:** Advance PPC-Gauge-Triad from verified Level 2 (algorithmic + real-world software/cross-compile) to **Level 3 (Physical Hardware Verification on MCU/FPGA + Production Crate + Reproducible Research Paper)**.

## Strict Engineering & Scientific Rules (NEVER VIOLATE)
1. **Zero Post-Hoc Threshold Hacking:** Never tune thresholds after observing test-set metrics without locking parameters a priori and re-running on an independent held-out dataset/seed.
2. **Evidence-Aligned Claims Only:** Never claim universal zero-shot cross-device transfer or universal superiority over all baselines. Always document per-device unsupervised warmup (`100–400` windows) as a design requirement.
3. **Embedded Constraints:** Preserve `#![no_std]`, `0 B` heap allocation, `0` matrix inversions, and `< 16 KB` static struct SRAM on `thumbv7em-none-eabihf` (ARM Cortex-M4F).
4. **No Unverified Hardware Claims:** Level 3 claims (cycle count, µs on real MCU, mW power, flash/RAM `.map`) must come strictly from real hardware probe output (`probe-rs` / `defmt` / oscilloscope / power profiler).

## Next Roadmap Milestones (Antigravity App + GitHub MCP Workflow)
1. **Local Automated Benchmark Harness:** Port Colab Python benchmarks into a reproducible local `cargo test` + `pytest`/Criterion suite with cached SMD/PSM/OpenML datasets.
2. **GitHub Repo & CI/CD:** Initialize GitHub repository via GitHub MCP and add GitHub Actions for automated Level 1 tests + 5-target cross-compilation.
3. **On-Chip Auto-Warmup State Machine (`v0.7`):** Add optional fixed-size `#![no_std]` `AutoCalBufF32` (`Warmup -> Calibrate -> Monitor`) with C/FFI + Python (`PyO3`/`maturin`) bindings.
4. **Level 3 Physical Silicon Verification:** Flash onto an STM32F4/STM32H7 or ESP32-S3/nRF52840 dev board, measure DWT cycle counter latency, `.elf` flash/SRAM footprint (`cargo-bloat`/`arm-none-eabi-size`), and real IMU/thermal sensor streams via `probe-rs`.
