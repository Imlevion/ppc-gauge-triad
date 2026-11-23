//! Pure-Rust `.npz` (ZIP + DEFLATE) and NumPy `.npy` v1/v2 reader.
//!
//! Reads cached SMD/PSM archives from `data_cache/*.npz` directly in Rust
//! without requiring Python or C compression libraries (`flate2` `rust_backend`).

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;

use flate2::read::DeflateDecoder;

#[derive(Debug, Clone)]
pub struct MatrixF64 {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl MatrixF64 {
    #[inline]
    pub fn row(&self, r: usize) -> &[f64] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }
}

#[derive(Debug, Clone)]
pub struct ServerDataset {
    pub train: MatrixF64,
    pub test: MatrixF64,
    pub test_label: Vec<i32>,
}

fn read_u16_le(buf: &[u8], off: usize) -> Result<u16, String> {
    let b: [u8; 2] = buf
        .get(off..off + 2)
        .ok_or_else(|| format!("EOF reading u16 at {off}"))?
        .try_into()
        .unwrap();
    Ok(u16::from_le_bytes(b))
}

fn read_u32_le(buf: &[u8], off: usize) -> Result<u32, String> {
    let b: [u8; 4] = buf
        .get(off..off + 4)
        .ok_or_else(|| format!("EOF reading u32 at {off}"))?
        .try_into()
        .unwrap();
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le(buf: &[u8], off: usize) -> Result<u64, String> {
    let b: [u8; 8] = buf
        .get(off..off + 8)
        .ok_or_else(|| format!("EOF reading u64 at {off}"))?
        .try_into()
        .unwrap();
    Ok(u64::from_le_bytes(b))
}

/// Extract all entries from a standard ZIP / `.npz` archive (supports STORE=0 and DEFLATE=8).
pub fn read_zip_entries(zip_bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, String> {
    // Locate End of Central Directory record (signature 0x06054b50) scanning backwards.
    if zip_bytes.len() < 22 {
        return Err("Archive too small to be a valid ZIP/NPZ".to_string());
    }
    let min_pos = zip_bytes.len().saturating_sub(65_557);
    let mut eocd_pos = None;
    for pos in (min_pos..=zip_bytes.len() - 22).rev() {
        if &zip_bytes[pos..pos + 4] == b"PK\x05\x06" {
            eocd_pos = Some(pos);
            break;
        }
    }
    let eocd = eocd_pos.ok_or_else(|| "ZIP End-of-Central-Directory signature not found".to_string())?;
    let total_entries = read_u16_le(zip_bytes, eocd + 10)? as usize;
    let cd_offset = read_u32_le(zip_bytes, eocd + 16)? as usize;

    let mut entries = BTreeMap::new();
    let mut cursor = cd_offset;

    for _ in 0..total_entries {
        if zip_bytes.get(cursor..cursor + 4) != Some(b"PK\x01\x02") {
            return Err(format!("Invalid ZIP Central Directory header at {cursor}"));
        }
        let method = read_u16_le(zip_bytes, cursor + 10)?;
        let mut comp_size = read_u32_le(zip_bytes, cursor + 20)? as usize;
        let mut uncomp_size = read_u32_le(zip_bytes, cursor + 24)? as usize;
        let name_len = read_u16_le(zip_bytes, cursor + 28)? as usize;
        let extra_len = read_u16_le(zip_bytes, cursor + 30)? as usize;
        let comment_len = read_u16_le(zip_bytes, cursor + 32)? as usize;
        let mut local_hdr_off = read_u32_le(zip_bytes, cursor + 42)? as usize;

        let name_start = cursor + 46;
        let name = String::from_utf8_lossy(
            zip_bytes
                .get(name_start..name_start + name_len)
                .ok_or_else(|| "ZIP entry name out of bounds".to_string())?,
        )
        .into_owned();

        // Handle ZIP64 extra field (header ID 0x0001) if any size/offset is 0xFFFF_FFFF
        let extra_start = name_start + name_len;
        let extra = zip_bytes
            .get(extra_start..extra_start + extra_len)
            .ok_or_else(|| "ZIP extra field out of bounds".to_string())?;
        if comp_size == 0xFFFF_FFFF || uncomp_size == 0xFFFF_FFFF || local_hdr_off == 0xFFFF_FFFF {
            let mut ex = 0usize;
            while ex + 4 <= extra.len() {
                let tag = read_u16_le(extra, ex)?;
                let sz = read_u16_le(extra, ex + 2)? as usize;
                ex += 4;
                if tag == 0x0001 && ex + sz <= extra.len() {
                    let mut z = ex;
                    if uncomp_size == 0xFFFF_FFFF && z + 8 <= ex + sz {
                        uncomp_size = read_u64_le(extra, z)? as usize;
                        z += 8;
                    }
                    if comp_size == 0xFFFF_FFFF && z + 8 <= ex + sz {
                        comp_size = read_u64_le(extra, z)? as usize;
                        z += 8;
                    }
                    if local_hdr_off == 0xFFFF_FFFF && z + 8 <= ex + sz {
                        local_hdr_off = read_u64_le(extra, z)? as usize;
                    }
                    break;
                }
                ex += sz;
            }
        }

        // Jump to local file header to find payload start
        if zip_bytes.get(local_hdr_off..local_hdr_off + 4) != Some(b"PK\x03\x04") {
            return Err(format!("Invalid ZIP local header for {name}"));
        }
        let loc_name_len = read_u16_le(zip_bytes, local_hdr_off + 26)? as usize;
        let loc_extra_len = read_u16_le(zip_bytes, local_hdr_off + 28)? as usize;
        let payload_start = local_hdr_off + 30 + loc_name_len + loc_extra_len;
        let payload = zip_bytes
            .get(payload_start..payload_start + comp_size)
            .ok_or_else(|| format!("ZIP payload out of bounds for {name}"))?;

        let decoded = match method {
            0 => payload.to_vec(),
            8 => {
                let mut dec = DeflateDecoder::new(payload);
                let mut out = Vec::with_capacity(uncomp_size);
                dec.read_to_end(&mut out)
                    .map_err(|e| format!("DEFLATE decode failed for {name}: {e}"))?;
                out
            }
            other => {
                return Err(format!("Unsupported ZIP compression method {other} for {name}"));
            }
        };

        entries.insert(name, decoded);
        cursor = extra_start + extra_len + comment_len;
    }

    Ok(entries)
}

struct NpyHeader {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
    payload_offset: usize,
}

fn parse_npy_header(buf: &[u8]) -> Result<NpyHeader, String> {
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        return Err("Invalid NPY magic bytes".to_string());
    }
    let major = buf[6];
    let (hdr_len, hdr_start) = match major {
        1 => (read_u16_le(buf, 8)? as usize, 10usize),
        2 | 3 => (read_u32_le(buf, 8)? as usize, 12usize),
        _ => return Err(format!("Unsupported NPY version {major}")),
    };
    let hdr_bytes = buf
        .get(hdr_start..hdr_start + hdr_len)
        .ok_or_else(|| "NPY header truncated".to_string())?;
    let hdr_str = std::str::from_utf8(hdr_bytes).map_err(|e| format!("Invalid UTF-8 NPY header: {e}"))?;

    // Extract 'descr': '...'
    let descr = extract_quoted_field(hdr_str, "'descr'")
        .or_else(|| extract_quoted_field(hdr_str, "\"descr\""))
        .ok_or_else(|| format!("Missing descr in NPY header: {hdr_str}"))?;

    let fortran_order = hdr_str.contains("'fortran_order': True")
        || hdr_str.contains("\"fortran_order\": True");

    // Extract 'shape': (d0, d1, ...)
    let shape_key = hdr_str
        .find("shape")
        .ok_or_else(|| format!("Missing shape in NPY header: {hdr_str}"))?;
    let tail = &hdr_str[shape_key..];
    let lparen = tail.find('(').ok_or_else(|| "Missing '(' in NPY shape".to_string())?;
    let rparen = tail.find(')').ok_or_else(|| "Missing ')' in NPY shape".to_string())?;
    let inner = &tail[lparen + 1..rparen];
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let dim = trimmed
                .parse::<usize>()
                .map_err(|e| format!("Bad shape dim '{trimmed}': {e}"))?;
            shape.push(dim);
        }
    }

    Ok(NpyHeader {
        descr,
        fortran_order,
        shape,
        payload_offset: hdr_start + hdr_len,
    })
}

fn extract_quoted_field(hdr: &str, key: &str) -> Option<String> {
    let pos = hdr.find(key)?;
    let rest = &hdr[pos + key.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let quote = after.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let body = &after[1..];
    let end = body.find(quote)?;
    Some(body[..end].to_string())
}

pub fn parse_npy_matrix_f64(buf: &[u8]) -> Result<MatrixF64, String> {
    let hdr = parse_npy_header(buf)?;
    if hdr.fortran_order {
        return Err("Fortran-order NPY arrays are not supported".to_string());
    }
    if hdr.shape.len() != 2 {
        return Err(format!("Expected 2D matrix, got shape {:?}", hdr.shape));
    }
    let (rows, cols) = (hdr.shape[0], hdr.shape[1]);
    let n = rows * cols;
    let payload = &buf[hdr.payload_offset..];
    let mut data = Vec::with_capacity(n);

    match hdr.descr.as_str() {
        "<f8" | "f8" => {
            if payload.len() < n * 8 {
                return Err("Truncated f64 NPY payload".to_string());
            }
            for chunk in payload[..n * 8].chunks_exact(8) {
                data.push(f64::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        "<f4" | "f4" => {
            if payload.len() < n * 4 {
                return Err("Truncated f32 NPY payload".to_string());
            }
            for chunk in payload[..n * 4].chunks_exact(4) {
                data.push(f32::from_le_bytes(chunk.try_into().unwrap()) as f64);
            }
        }
        other => return Err(format!("Unsupported float NPY dtype: {other}")),
    }
    Ok(MatrixF64 { rows, cols, data })
}

pub fn parse_npy_vec_i32(buf: &[u8]) -> Result<Vec<i32>, String> {
    let hdr = parse_npy_header(buf)?;
    let n: usize = hdr.shape.iter().product();
    let payload = &buf[hdr.payload_offset..];
    let mut out = Vec::with_capacity(n);
    match hdr.descr.as_str() {
        "<i4" | "i4" => {
            if payload.len() < n * 4 {
                return Err("Truncated i32 NPY payload".to_string());
            }
            for chunk in payload[..n * 4].chunks_exact(4) {
                out.push(i32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        "<i8" | "i8" => {
            if payload.len() < n * 8 {
                return Err("Truncated i64 NPY payload".to_string());
            }
            for chunk in payload[..n * 8].chunks_exact(8) {
                out.push(i64::from_le_bytes(chunk.try_into().unwrap()) as i32);
            }
        }
        "<f8" | "f8" => {
            if payload.len() < n * 8 {
                return Err("Truncated f64 label payload".to_string());
            }
            for chunk in payload[..n * 8].chunks_exact(8) {
                out.push(f64::from_le_bytes(chunk.try_into().unwrap()) as i32);
            }
        }
        other => return Err(format!("Unsupported integer NPY dtype: {other}")),
    }
    Ok(out)
}

pub fn load_server_npz(path: &Path) -> Result<ServerDataset, String> {
    let raw = fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let entries = read_zip_entries(&raw)?;
    let tr_buf = entries
        .get("train.npy")
        .ok_or_else(|| format!("Missing train.npy in {}", path.display()))?;
    let te_buf = entries
        .get("test.npy")
        .ok_or_else(|| format!("Missing test.npy in {}", path.display()))?;
    let yl_buf = entries
        .get("test_label.npy")
        .ok_or_else(|| format!("Missing test_label.npy in {}", path.display()))?;

    Ok(ServerDataset {
        train: parse_npy_matrix_f64(tr_buf)?,
        test: parse_npy_matrix_f64(te_buf)?,
        test_label: parse_npy_vec_i32(yl_buf)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_npy_f64(rows: usize, cols: usize, vals: &[f64]) -> Vec<u8> {
        let mut hdr = format!(
            "{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
        );
        while (10 + hdr.len() + 1) % 16 != 0 {
            hdr.push(' ');
        }
        hdr.push('\n');
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
        out.extend_from_slice(hdr.as_bytes());
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn roundtrip_npy_f64_parsing() {
        let npy = make_npy_f64(2, 3, &[1.5, -2.0, 3.25, 4.0, 5.5, -6.75]);
        let mat = parse_npy_matrix_f64(&npy).expect("parse 2x3 matrix");
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
        assert_eq!(mat.row(0), &[1.5, -2.0, 3.25]);
        assert_eq!(mat.row(1), &[4.0, 5.5, -6.75]);
    }
}
