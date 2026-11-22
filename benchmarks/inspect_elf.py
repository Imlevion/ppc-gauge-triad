#!/usr/bin/env python3
"""Pure-Python 32-bit ELF section footprint inspector (zero external tools required).

Inspects a linked bare-metal ARM Cortex-M4F (`thumbv7em-none-eabihf`) `.elf` binary
and reports exact Flash (`.vector_table` + `.text` + `.rodata` + `.data` LMA) and
Static RAM (`.data` + `.bss` + `.uninit`) totals alongside the engine struct budget.
"""

from __future__ import annotations

import json
import os
import struct
import sys


SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4
SHT_NOBITS = 8


def inspect_elf32(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 52 or data[:4] != b"\x7fELF":
        raise ValueError(f"Not a valid ELF binary: {path}")
    ei_class = data[4]
    ei_data = data[5]
    if ei_class != 1 or ei_data != 1:
        raise ValueError(f"Expected ELF32 little-endian (ARM Cortex-M), got class={ei_class}, data={ei_data}")

    e_machine = struct.unpack_from("<H", data, 18)[0]
    e_shoff = struct.unpack_from("<I", data, 32)[0]
    e_shentsize = struct.unpack_from("<H", data, 46)[0]
    e_shnum = struct.unpack_from("<H", data, 48)[0]
    e_shstrndx = struct.unpack_from("<H", data, 50)[0]

    def read_shdr(idx: int) -> tuple:
        off = e_shoff + idx * e_shentsize
        return struct.unpack_from("<IIIIIIIIII", data, off)

    str_sh = read_shdr(e_shstrndx)
    str_off, str_size = str_sh[4], str_sh[5]
    shstrtab = data[str_off : str_off + str_size]

    def sec_name(name_off: int) -> str:
        end = shstrtab.find(b"\x00", name_off)
        return shstrtab[name_off:end].decode("utf-8", errors="replace")

    alloc_sections = []
    flash_bytes = 0
    ram_bytes = 0
    breakdown: dict[str, int] = {}

    for i in range(e_shnum):
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, *_ = read_shdr(i)
        if not (sh_flags & SHF_ALLOC) or sh_size == 0:
            continue
        name = sec_name(sh_name)
        is_nobits = sh_type == SHT_NOBITS
        is_write = bool(sh_flags & SHF_WRITE)

        # Standard Cortex-M Flash vs RAM accounting (mirrors `arm-none-eabi-size` / `cargo-size`):
        # - Non-NOBITS sections occupy Flash (including `.data` init image in Flash LMA).
        # - Writable ALLOC sections (`.data`, `.bss`, `.uninit`) occupy static RAM.
        if not is_nobits:
            flash_bytes += sh_size
        if is_write:
            ram_bytes += sh_size

        breakdown[name] = sh_size
        alloc_sections.append(
            {
                "name": name,
                "addr": f"0x{sh_addr:08x}",
                "size": sh_size,
                "in_flash": not is_nobits,
                "in_ram": is_write,
            }
        )

    return {
        "elf_path": path,
        "elf_file_bytes": os.path.getsize(path),
        "e_machine": e_machine,
        "is_arm": e_machine == 40,
        "sections": alloc_sections,
        "breakdown": breakdown,
        "total_flash_bytes": flash_bytes,
        "total_static_ram_bytes": ram_bytes,
    }


def main() -> int:
    default_elf = os.path.join(
        "firmware",
        "cortex-m4f-eval",
        "target",
        "thumbv7em-none-eabihf",
        "release",
        "ppc-m4f-eval",
    )
    pos_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    path = pos_args[0] if pos_args else default_elf
    as_json = "--json" in sys.argv

    info = inspect_elf32(path)
    if as_json:
        print(json.dumps(info, indent=2))
        return 0

    print("=== Level 2.5 Linked ARM Cortex-M4F (.elf) Footprint ===")
    print(f"Binary          : {info['elf_path']} (EM_ARM={info['is_arm']})")
    for sec in info["sections"]:
        region = []
        if sec["in_flash"]:
            region.append("FLASH")
        if sec["in_ram"]:
            region.append("RAM")
        print(f"  {sec['name']:<16} {sec['addr']}  {sec['size']:>7} B  [{'+'.join(region)}]")
    print("-" * 56)
    print(f"Total Flash     : {info['total_flash_bytes']:>7} B ({info['total_flash_bytes']/1024:.2f} KB)")
    print(f"Total Static RAM: {info['total_static_ram_bytes']:>7} B ({info['total_static_ram_bytes']/1024:.2f} KB)")
    print("Note: Static RAM excludes runtime call stack (incl. 1,024 B sort scratch in")
    print("      finalize_calibration()), ISR frames, and application DMA/sensor buffers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
