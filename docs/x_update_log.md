# X (`@Levizyaa`) Public Update Log

Tracks all verified public posts/updates for **PPC-Gauge-Triad** to ensure strict consistency with repository state per [`AGENTS.md`](../AGENTS.md).

---

## 2026-11-23 — `Update (v0.7.1)` (Benchmark Hardening, Fair Baselines & Level 2.5 ELF/QEMU)

- **Account:** `@Levizyaa`
- **Publication Method:** Manual reply thread to the pinned `v0.7.0` thread (X MCP server installed and verified locally via `DataWhisker/x-mcp-server`, but live API access requires X Pay-Per-Use billing, so published manually by repository owner).
- **Verified Reference PR:** [Imlevion/ppc-gauge-triad#6](https://github.com/Imlevion/ppc-gauge-triad/pull/6) (merged `2026-11-23`, 4/4 CI checks passed; crate version unchanged at `v0.7.1`).
- **Thread Start Link (Reply 1):** https://x.com/Levizyaa/status/2124816099585957964
- **Attachment on Reply 1:** Regenerated `paper/figures/fig1_8server_ablation.png` (panel (b) y-axis starting at `50`, `mean ± std` error bars, live fair baselines).

### Exact Published Post Texts

**Reply 1 (235 chars):**
```text
Update (v0.7.1): Benchmark & claim hardening (window-level, NO point-adjust). On identical 8-server SMD/PSM windows, 3 heavier host-RAM baselines beat High-Recall (78.7±13.0%) on AUROC: PCA+Mahal Full-D 80.9%, RS-Hash 80.1%, kNN 79.0%.
```

**Reply 2 (238 chars):**
```text
Low-FAR (77.1±13.1% AUROC) keeps lowest Dual-Gate FAR: 7.6±7.2% vs HS-Trees 15.5%, RS-Hash 17.8%, batch 43.3–50.9%. Fixes: fig1 y-axis 60->50 (+std), honest preset provenance (Colab import), 29.7% zero-shot = inverted scores (not random).
```

**Reply 3 (259 raw / 239 X-weighted chars):**
```text
v0.7.1 adds opt-in panic-handler & Cortex-M4F .elf: 23.07 KB Flash, 0 B .data, 14.44 KB .bss (excl. stack/DMA) + QEMU functional proof only. Level 3 silicon PENDING.
https://github.com/Imlevion/ppc-gauge-triad
#rustlang #embedded #opensource #anomalydetection
```
