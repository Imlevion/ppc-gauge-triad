# `ppc-m4f-eval` — Turnkey ARM Cortex-M4F (`thumbv7em-none-eabihf`) Evaluation Firmware

Standalone bare-metal `#![no_std]` `#![no_main]` firmware crate linking **`ppc_gauge_triad_rs`** (`default-features = false`, zero heap, zero `panic_impl` conflict) with `cortex-m-rt` and `panic-halt`.

---

## Verification Boundary (`AGENTS.md` Rule 4)

- **Level 2.5 (Verified without physical board):**
  1. **Real Linked `.elf` Footprint (`cargo build --release`, `opt-level = 3`, LTO):**
     | ELF Section | Region | Exact Size |
     | :--- | :---: | ---: |
     | `.vector_table` | Flash | `1,024 B` |
     | `.text` | Flash | `22,376 B` |
     | `.rodata` | Flash | `224 B` |
     | `.data` | Flash + RAM | **`0 B`** (true all-zero `.bss` init) |
     | `.bss` (`AutoCalEngineF32` + report) | Static RAM | `14,784 B` (`14,744 B` engine + `40 B` report/sentinel) |
     | **Total Flash** | **Flash** | **`23,624 B` (`23.07 KB`)** |
     | **Total Static RAM** | **RAM** | **`14,784 B` (`14.44 KB`)** |

     > **Memory Accounting Note:** `Total Static RAM` (`14.44 KB`) covers static `.bss`/`.data` only. Final firmware deployments must also budget call stack (including a `1,024 B` temporary sort scratch array during `finalize_calibration()`), interrupt frames, DMA/ADC/sensor buffers, and BSP/RTOS runtime overhead.
  2. **Functional ARMv7E-M Emulation on QEMU (`--features qemu`):** Boots and executes the compiled `thumbv7em-none-eabihf` FPU binary on `qemu-system-arm -machine mps2-an386`, running `400` normal windows (`Warmup -> Calibrate -> Monitor`) + `64` monitor windows + severe fault injection (`alarm_combined == 1`) with zero HardFaults.
- **Level 3 (Physical Silicon — `PENDING`):** QEMU timing is **not** hardware silicon timing. Real DWT cycle counts (`DWT_CYCCNT`), board-clock µs latency, and inline mW power remain strictly `PENDING` physical MCU measurement (see [Issue #3](https://github.com/Imlevion/ppc-gauge-triad/issues/3)).

---

## 1. Level 2.5 — Build `.elf`, Inspect Footprint & Run in QEMU (Rp 0 / No Hardware)

```bash
cd firmware/cortex-m4f-eval

# 1. Build bare-metal STM32F411 / Cortex-M4F release .elf (no semihosting)
cargo build --release

# 2. Inspect exact Flash & Static RAM sections (cross-platform Python or arm-none-eabi-size)
python ../../benchmarks/inspect_elf.py
arm-none-eabi-size -A target/thumbv7em-none-eabihf/release/ppc-m4f-eval

# 3. Build & execute functional ARMv7E-M FPU proof in QEMU (mps2-an386)
cargo build --release --features qemu
qemu-system-arm -cpu cortex-m4 -machine mps2-an386 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel target/thumbv7em-none-eabihf/release/ppc-m4f-eval
```

---

## 2. Level 3 — Flash Onto Physical Cortex-M4F Silicon (STM32F4 / nRF52840)

Default `memory_stm32f411.x` targets **STM32F411CEU6 ("Black Pill", 512 KB Flash / 128 KB RAM)**. Edit `memory_stm32f411.x` for STM32F446RE, STM32F407VG, or nRF52840 as commented inside the file.

```bash
cd firmware/cortex-m4f-eval

# Build clean hardware .elf (WITHOUT --features qemu)
cargo build --release

# Flash via probe-rs (example: STM32F411CEUx or STM32F446RETx)
probe-rs download --chip STM32F411CEUx target/thumbv7em-none-eabihf/release/ppc-m4f-eval
probe-rs reset --chip STM32F411CEUx
```

Attach via `probe-rs gdb` (or `arm-none-eabi-gdb`) once `LEVEL3_DONE == 0xD04E0700` and inspect:

```gdb
print /x LEVEL3_DONE
print LEVEL3_DWT_REPORT
```

Please share the raw `LEVEL3_DWT_REPORT` output, MCU part number, exact `SYSCLK` MHz, and `arm-none-eabi-size` output on [Issue #3](https://github.com/Imlevion/ppc-gauge-triad/issues/3)!
