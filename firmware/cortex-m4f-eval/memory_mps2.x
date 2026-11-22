/*
 * QEMU ARM Cortex-M4F (`mps2-an386`) memory map for Level 2.5 functional emulation ONLY.
 * Selected automatically by build.rs when compiling with `--features qemu`.
 */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   : ORIGIN = 0x20000000, LENGTH = 4M
}
