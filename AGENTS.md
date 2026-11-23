# AGENTS.md — PPC-Gauge-Triad v0.7.1 Engineering Context (Google Antigravity App)

## Project Identity
- **Crate:** `ppc_gauge_triad_rs` v0.7.1 (`#![no_std]` `f32` Rust, `0 B` heap, `0` matrix inversions; live on GitHub [`Imlevion/ppc-gauge-triad`](https://github.com/Imlevion/ppc-gauge-triad), [`crates.io`](https://crates.io/crates/ppc_gauge_triad_rs), and [`docs.rs`](https://docs.rs/ppc_gauge_triad_rs)).
- **Engine Static Struct SRAM (compile-time verified `< 16 KB`):**
  - Core `GaugeTriadEngineF32`: `11,604 B` (`11.33 KB`) on `thumbv7em-none-eabihf` / `11,624 B` (`11.35 KB`) on 64-bit hosts.
  - Combined `AutoCalEngineF32` (core + `AutoCalBufF32`): `14,744 B` (`14.40 KB`) on `thumbv7em-none-eabihf` / `14,768 B` (`14.42 KB`) on 64-bit hosts.
- **Environment:** Google Antigravity App (Windows, Local Checkout) + **GitHub Remote MCP** active (`47 tools enabled`). Do not assume Context7 or Docker is installed.
- **Mission:** Maintain verified Level 1 & Level 2 foundations cleanly and advance PPC-Gauge-Triad to **Level 3 (Physical Hardware Verification on real MCU/FPGA silicon + Production Firmware Integration + Reproducible Preprint)**.

## Strict Engineering & Scientific Rules (NEVER VIOLATE)
1. **Zero Post-Hoc Threshold Hacking:** Never tune thresholds after observing test-set metrics without locking parameters *a priori* and re-running on an independent held-out dataset/seed.
2. **Evidence-Aligned Claims Only:** Never claim universal zero-shot cross-device transfer or universal superiority over all baselines. Always document per-device unsupervised warmup (`100–400` normal windows, `0` labels) as a design requirement.
3. **Embedded Constraints & Honest Memory Accounting:**
   - Preserve `#![no_std]` library purity (`default-features = false` must NOT emit a `#[panic_handler]`; the final firmware binary supplies its own `#[panic_handler]`, while `--features panic-handler` is opt-in only for standalone `#![no_std]` `cdylib` builds).
   - Preserve `0 B` heap allocation, `0` matrix inversions, and `< 16 KB` **engine static struct SRAM** on `thumbv7em-none-eabihf`.
   - Never conflate engine struct `.bss` size (`< 16 KB`) with total final firmware RAM: always account for call stack (including `1,024 B` temporary sort scratch during `finalize_calibration()`), interrupt frames, DMA/sensor buffers, and BSP/RTOS runtime overhead.
4. **No Unverified Hardware Claims:** Level 3 claims (DWT cycle count, µs on real MCU, mW power, full firmware `.elf`/`.map` Flash & RAM) must come strictly from real hardware probe output (`probe-rs` / `defmt` / `arm-none-eabi-size` / oscilloscope / power profiler).
5. **Separate CI Scope from Offline Research Replay:** Always distinguish what runs automatically on GitHub Actions CI (Level 1 Rust invariants, `AutoCalBufF32` unit tests, Rustdoc/doctests, C99 ABI demo, Python `L1.8` parity/cache tests, `cargo package`, 5-target cross-compilation) from full SMD/PSM/OpenML research benchmarks reproduced locally from offline caches (`benchmarks/run_benchmarks.py --all --offline`).

## Completed Milestones (v0.7.0 – v0.7.1)
- [x] **Local Automated Benchmark Harness:** Ported Colab benchmarks into `tests/level1_invariants.rs`, `tests/autocal_v07.rs`, `benches/triad_bench.rs` (Criterion), and local Python scripts with offline SMD/PSM `.npz` caching (`benchmarks/`).
- [x] **Public GitHub Repo, CI/CD & `crates.io` Releases:** Published `Imlevion/ppc-gauge-triad` (`v0.7.0` & `v0.7.1`), green CI across 5 cross-compilation targets + Level 2.5 ELF/QEMU job, `crates.io` + `docs.rs`, and `CITATION.cff`.
- [x] **On-Chip Zero-Touch Auto-Warmup (`v0.7.0`) & C99 Ecosystem:** Added `#![no_std]` `AutoCalBufF32` / `AutoCalEngineF32` (`Warmup -> Calibrate -> Monitor`), `include/ppc_gauge_triad.h`, verified `examples/c_stream_demo.c`, `examples/level3_cortex_m4f_eval_template.rs`, 300-DPI figure generator, and arXiv/IEEE LaTeX preprint draft (`paper/ppc_gauge_triad.tex`).
- [x] **Level 2.5 Turnkey Cortex-M4F `.elf`, QEMU Verification & `v0.7.1` Embedded Hygiene (`firmware/cortex-m4f-eval/`):** Gated `#[panic_handler]` behind opt-in `panic-handler` feature, fixed `zeroed()` true all-zero `.bss` placement (`.data = 0 B`, `Total Flash = 23,624 B / 23.07 KB`, `Static RAM .bss = 14,784 B / 14.44 KB`), and wired automated QEMU `mps2-an386` functional execution into CI (strictly separated from Level 3 physical silicon).

## Next Roadmap Milestones
1. **Level 3 Physical Silicon Verification (`Imlevion/ppc-gauge-triad#3`):** Flash `firmware/cortex-m4f-eval/` onto STM32F4/STM32H7, nRF52840, or ESP32-S3 hardware; measure real DWT cycle counts, full firmware `.elf`/`.map` Flash & RAM (incl. stack/interrupts via `arm-none-eabi-size`/`cargo-bloat`), mW power, and real IMU/thermal sensor streams via `probe-rs`.
2. **Optional Ecosystem Extensions:** Native Python (`PyO3`/`maturin`) wheels and formal arXiv/Zenodo preprint publication once ready.

## GitHub Workflow Rules
- Always do new work on a dedicated feature/chore branch (never commit directly to `main`).
- Keep commits small and descriptive, push the branch, and open a Pull Request citing real measured numbers from commands run in the current session.
- Merge PRs only after `cargo test` (plus relevant local tests) and all GitHub Actions CI checks are green.
- Never publish to `crates.io` (`cargo publish`) without explicit instruction from the repository owner.

## X MCP Rules
- Use X MCP strictly to keep public posts consistent with verified repository state.
- Every number in a post must match the exact output of a command run or verified file in the repository — never invent, estimate, or round numbers upward.
- Zero hype claims (no "SOTA", no "beats everything"). Preserve honest framing: no magic zero-shot transfer, Level 3 physical silicon remains `PENDING`, and QEMU is functional verification only.
- Always present the exact final post text to the repository owner and publish **only** after receiving an explicit `"OK"`.
- If the MCP tool supports editing posts, edit the pinned post. Otherwise, reply in the same thread prefixed with `Update (v<version>):`, at most 3 posts, each strictly under 280 characters, including the GitHub link and hashtags `#rustlang #embedded #opensource #anomalydetection`.
- Record every published final post text in `docs/x_update_log.md` (date, exact text, and post link).
- Never store or commit X API keys, tokens, or secrets in the repository, commits, files, or logs; credentials live strictly in local Antigravity MCP config (`~/.gemini/config/mcp_config.json`).
