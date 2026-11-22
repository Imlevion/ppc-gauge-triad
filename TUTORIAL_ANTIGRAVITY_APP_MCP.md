# Panduan Setup: PPC-Gauge-Triad di Google Antigravity App (Windows) + GitHub MCP

> **Status Setup Aktif:** Google Antigravity App (Windows) dengan **GitHub Remote MCP Server** (`https://api.githubcopilot.com/mcp/`, tanpa Docker). Terminal lokal (`cargo`, `python`, `pytest`) dan akses file lokal sudah bawaan Antigravity App.

## 1. Konfigurasi MCP Aktif (`mcp_config.json`)
File konfigurasi global di Windows dapat ditemukan lewat **`Customizations` → `Installed` → `MCP Servers` → ikon Copy (`Copy Config File Path`)**.

```json
{
  "mcpServers": {
    "github": {
      "serverUrl": "https://api.githubcopilot.com/mcp/",
      "headers": {
        "Authorization": "Bearer YOUR_GITHUB_TOKEN"
      }
    }
  }
}
```
Pastikan status **`github`** di bagian `MCP Servers` bertitik **hijau** (`47 tools enabled`).

---

## 2. Prompt Pertama untuk Memulai di Antigravity App
Pastikan di kotak chat bawah sudah terpilih proyek **`ppc_gauge_triad_v061`** (`Local Checkout`), lalu kirim:

> "Baca `AGENTS.md` dan `VALIDATION_REPORT_v061.md`. Buatkan `cargo test` lokal untuk memverifikasi invariant Level 1 dan skrip Python lokal dengan cache dataset SMD/PSM agar saya tidak perlu pakai Colab lagi, lalu bantu siapkan repositori GitHub untuk proyek ini."

---

## 3. MCP Opsional untuk Tahap Berikutnya (Pasang Hanya Saat Dibutuhkan)
- **Tahap Riset / Paper:** [arXiv MCP Server](https://github.com/blazickjp/arxiv-mcp-server) untuk pencarian paper & sitasi BibTeX.
- **Tahap Level 3 (Silikon Fisik MCU):** [Embedded Debugger MCP](https://github.com/adancurusul/embedded-debugger-mcp) (`probe-rs` / OpenOCD) saat papan dev ARM Cortex-M4F sudah terhubung ke laptop.
- **Opsional Dokumentasi Live:** [Context7 Remote MCP](https://github.com/upstash/context7) (`"serverUrl": "https://mcp.context7.com/mcp"`) jika suatu saat butuh penarikan dokumentasi library otomatis.
