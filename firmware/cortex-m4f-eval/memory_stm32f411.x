/*
 * Default bare-metal ARM Cortex-M4F memory map: STM32F411CEU6 ("Black Pill", 100 MHz Cortex-M4F)
 *
 * Adapt ORIGIN / LENGTH below for your target board before flashing:
 *   - STM32F411CEU6 (Black Pill) : FLASH = 512K @ 0x08000000, RAM = 128K @ 0x20000000
 *   - STM32F446RE   (Nucleo-64)  : FLASH = 512K @ 0x08000000, RAM = 128K @ 0x20000000
 *   - STM32F407VG   (Discovery)  : FLASH = 1024K @ 0x08000000, RAM = 128K @ 0x20000000
 *   - nRF52840      (DK / Dongle): FLASH = 1024K @ 0x00000000, RAM = 256K @ 0x20000000
 */
MEMORY
{
  FLASH : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   : ORIGIN = 0x20000000, LENGTH = 128K
}
