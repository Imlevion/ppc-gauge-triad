//! Pure-Rust 32-bit ELF section footprint inspector (zero external tools required).
//!
//! Replaces `benchmarks/inspect_elf.py` so Level 2.5 Cortex-M4F ELF inspection
//! runs natively in Rust across Linux, Windows, and macOS with zero Python
//! dependency.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const SHF_WRITE: u32 = 0x1;
const SHF_ALLOC: u32 = 0x2;
const SHT_NOBITS: u32 = 8;

#[derive(Debug, Clone)]
pub struct ElfSectionInfo {
    pub name: String,
    pub addr: u32,
    pub size: u32,
    pub in_flash: bool,
    pub in_ram: bool,
}

#[derive(Debug, Clone)]
pub struct ElfReport {
    pub elf_path: String,
    pub elf_file_bytes: usize,
    pub e_machine: u16,
    pub is_arm: bool,
    pub sections: Vec<ElfSectionInfo>,
    pub breakdown: BTreeMap<String, u32>,
    pub total_flash_bytes: u32,
    pub total_static_ram_bytes: u32,
}

fn read_u16_le(data: &[u8], off: usize) -> Result<u16, String> {
    let bytes: [u8; 2] = data
        .get(off..off + 2)
        .ok_or_else(|| format!("Unexpected EOF reading u16 at offset {off}"))?
        .try_into()
        .unwrap();
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32_le(data: &[u8], off: usize) -> Result<u32, String> {
    let bytes: [u8; 4] = data
        .get(off..off + 4)
        .ok_or_else(|| format!("Unexpected EOF reading u32 at offset {off}"))?
        .try_into()
        .unwrap();
    Ok(u32::from_le_bytes(bytes))
}

pub fn inspect_elf32_bytes(path_label: &str, data: &[u8]) -> Result<ElfReport, String> {
    if data.len() < 52 || &data[0..4] != b"\x7fELF" {
        return Err(format!("Not a valid ELF binary: {path_label}"));
    }
    let ei_class = data[4];
    let ei_data = data[5];
    if ei_class != 1 || ei_data != 1 {
        return Err(format!(
            "Expected ELF32 little-endian (ARM Cortex-M), got class={ei_class}, data={ei_data}"
        ));
    }

    let e_machine = read_u16_le(data, 18)?;
    let e_shoff = read_u32_le(data, 32)? as usize;
    let e_shentsize = read_u16_le(data, 46)? as usize;
    let e_shnum = read_u16_le(data, 48)? as usize;
    let e_shstrndx = read_u16_le(data, 50)? as usize;

    if e_shentsize < 40 || e_shstrndx >= e_shnum {
        return Err(format!(
            "Invalid ELF section header table (shentsize={e_shentsize}, shnum={e_shnum}, shstrndx={e_shstrndx})"
        ));
    }

    let str_off_hdr = e_shoff + e_shstrndx * e_shentsize;
    let str_file_off = read_u32_le(data, str_off_hdr + 16)? as usize;
    let str_size = read_u32_le(data, str_off_hdr + 20)? as usize;
    let shstrtab = data
        .get(str_file_off..str_file_off + str_size)
        .ok_or_else(|| "Section header string table out of bounds".to_string())?;

    let sec_name = |name_off: usize| -> String {
        if name_off >= shstrtab.len() {
            return String::new();
        }
        let tail = &shstrtab[name_off..];
        let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
        String::from_utf8_lossy(&tail[..end]).into_owned()
    };

    let mut sections = Vec::new();
    let mut breakdown = BTreeMap::new();
    let mut total_flash_bytes: u32 = 0;
    let mut total_static_ram_bytes: u32 = 0;

    for i in 0..e_shnum {
        let off = e_shoff + i * e_shentsize;
        let sh_name = read_u32_le(data, off)? as usize;
        let sh_type = read_u32_le(data, off + 4)?;
        let sh_flags = read_u32_le(data, off + 8)?;
        let sh_addr = read_u32_le(data, off + 12)?;
        let sh_size = read_u32_le(data, off + 20)?;

        if (sh_flags & SHF_ALLOC) == 0 || sh_size == 0 {
            continue;
        }
        let name = sec_name(sh_name);
        let is_nobits = sh_type == SHT_NOBITS;
        let is_write = (sh_flags & SHF_WRITE) != 0;

        // Standard Cortex-M Flash vs RAM accounting (mirrors `arm-none-eabi-size` / `cargo-size`):
        // - Non-NOBITS ALLOC sections occupy Flash (including `.data` init image in Flash LMA).
        // - Writable ALLOC sections (`.data`, `.bss`, `.uninit`) occupy static RAM.
        if !is_nobits {
            total_flash_bytes += sh_size;
        }
        if is_write {
            total_static_ram_bytes += sh_size;
        }
        breakdown.insert(name.clone(), sh_size);
        sections.push(ElfSectionInfo {
            name,
            addr: sh_addr,
            size: sh_size,
            in_flash: !is_nobits,
            in_ram: is_write,
        });
    }

    Ok(ElfReport {
        elf_path: path_label.to_string(),
        elf_file_bytes: data.len(),
        e_machine,
        is_arm: e_machine == 40,
        sections,
        breakdown,
        total_flash_bytes,
        total_static_ram_bytes,
    })
}

pub fn inspect_elf32(path: &Path) -> Result<ElfReport, String> {
    let data = fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    inspect_elf32_bytes(&path.display().to_string(), &data)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl ElfReport {
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"elf_path\": {},\n", json_escape(&self.elf_path)));
        s.push_str(&format!("  \"elf_file_bytes\": {},\n", self.elf_file_bytes));
        s.push_str(&format!("  \"e_machine\": {},\n", self.e_machine));
        s.push_str(&format!("  \"is_arm\": {},\n", self.is_arm));
        s.push_str("  \"sections\": [\n");
        for (idx, sec) in self.sections.iter().enumerate() {
            let comma = if idx + 1 < self.sections.len() { "," } else { "" };
            s.push_str(&format!(
                "    {{\"name\": {}, \"addr\": \"0x{:08x}\", \"size\": {}, \"in_flash\": {}, \"in_ram\": {}}}{}\n",
                json_escape(&sec.name),
                sec.addr,
                sec.size,
                sec.in_flash,
                sec.in_ram,
                comma
            ));
        }
        s.push_str("  ],\n");
        s.push_str("  \"breakdown\": {\n");
        for (idx, (k, v)) in self.breakdown.iter().enumerate() {
            let comma = if idx + 1 < self.breakdown.len() { "," } else { "" };
            s.push_str(&format!("    {}: {}{}\n", json_escape(k), v, comma));
        }
        s.push_str("  },\n");
        s.push_str(&format!("  \"total_flash_bytes\": {},\n", self.total_flash_bytes));
        s.push_str(&format!(
            "  \"total_static_ram_bytes\": {}\n",
            self.total_static_ram_bytes
        ));
        s.push('}');
        s
    }

    pub fn print_table(&self) {
        println!("=== Level 2.5 Linked ARM Cortex-M4F (.elf) Footprint (Rust ELF Inspector) ===");
        println!("Binary          : {} (EM_ARM={})", self.elf_path, self.is_arm);
        for sec in &self.sections {
            let region = match (sec.in_flash, sec.in_ram) {
                (true, true) => "FLASH+RAM",
                (true, false) => "FLASH",
                (false, true) => "RAM",
                (false, false) => "",
            };
            println!(
                "  {:<16} 0x{:08x}  {:>7} B  [{}]",
                sec.name, sec.addr, sec.size, region
            );
        }
        println!("{}", "-".repeat(56));
        println!(
            "Total Flash     : {:>7} B ({:.2} KB)",
            self.total_flash_bytes,
            self.total_flash_bytes as f64 / 1024.0
        );
        println!(
            "Total Static RAM: {:>7} B ({:.2} KB)",
            self.total_static_ram_bytes,
            self.total_static_ram_bytes as f64 / 1024.0
        );
        println!("Note: Static RAM excludes runtime call stack (incl. 1,024 B sort scratch in");
        println!("      finalize_calibration()), ISR frames, and application DMA/sensor buffers.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_synthetic_elf32(text_sz: u32, data_sz: u32, bss_sz: u32) -> Vec<u8> {
        let shstrtab: Vec<u8> = b"\0.text\0.data\0.bss\0.shstrtab\0".to_vec();
        let str_off = 52u32;
        let shoff = str_off + shstrtab.len() as u32;
        let mut buf = vec![0u8; shoff as usize + 5 * 40];

        // ELF32 header
        buf[0..4].copy_from_slice(b"\x7fELF");
        buf[4] = 1; // ELF32
        buf[5] = 1; // Little-endian
        buf[6] = 1; // Version
        buf[18..20].copy_from_slice(&40u16.to_le_bytes()); // EM_ARM
        buf[32..36].copy_from_slice(&shoff.to_le_bytes());
        buf[46..48].copy_from_slice(&40u16.to_le_bytes());
        buf[48..50].copy_from_slice(&5u16.to_le_bytes());
        buf[50..52].copy_from_slice(&4u16.to_le_bytes());

        buf[str_off as usize..str_off as usize + shstrtab.len()].copy_from_slice(&shstrtab);

        let write_shdr = |b: &mut [u8], idx: usize, name: u32, ty: u32, flags: u32, addr: u32, off: u32, sz: u32| {
            let base = shoff as usize + idx * 40;
            b[base..base + 4].copy_from_slice(&name.to_le_bytes());
            b[base + 4..base + 8].copy_from_slice(&ty.to_le_bytes());
            b[base + 8..base + 12].copy_from_slice(&flags.to_le_bytes());
            b[base + 12..base + 16].copy_from_slice(&addr.to_le_bytes());
            b[base + 16..base + 20].copy_from_slice(&off.to_le_bytes());
            b[base + 20..base + 24].copy_from_slice(&sz.to_le_bytes());
        };

        // 0: NULL, 1: .text, 2: .data, 3: .bss, 4: .shstrtab
        write_shdr(&mut buf, 1, 1, 1, SHF_ALLOC | 0x4, 0x0800_0000, 0, text_sz);
        write_shdr(&mut buf, 2, 7, 1, SHF_ALLOC | SHF_WRITE, 0x2000_0000, 0, data_sz);
        write_shdr(&mut buf, 3, 13, SHT_NOBITS, SHF_ALLOC | SHF_WRITE, 0x2000_0000 + data_sz, 0, bss_sz);
        write_shdr(&mut buf, 4, 18, 3, 0, 0, str_off, shstrtab.len() as u32);
        buf
    }

    #[test]
    fn parses_synthetic_cortex_m4f_elf32_sections() {
        let raw = build_synthetic_elf32(23_624, 0, 14_784);
        let rep = inspect_elf32_bytes("synthetic.elf", &raw).expect("valid synthetic ELF32");
        assert!(rep.is_arm);
        assert_eq!(rep.total_flash_bytes, 23_624);
        assert_eq!(rep.total_static_ram_bytes, 14_784);
        assert_eq!(rep.breakdown.get(".data").copied().unwrap_or(0), 0);
        assert_eq!(rep.breakdown.get(".bss").copied().unwrap_or(0), 14_784);
        assert!(rep.to_json().contains("\"total_flash_bytes\": 23624"));
    }
}
