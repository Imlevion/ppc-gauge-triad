# Fair Baseline Comparison — 8-Server SMD + PSM Chronological Replay

## Evaluation Protocol (Identical Across All Methods)

- **Datasets:** 7 SMD servers (`machine-1-1`, `1-6`, `2-1`, `1-2`, `1-3`, `2-2`, `3-1`) + PSM eBay Server (`8` servers total, loaded from local offline `.npz` cache).
- **Windowing:** Non-overlapping windows of length `w = 8` (`to_windows_v6`, `3PC+Resid` `d = 4` -> flattened `32-D` window vectors; `max_tr_win = 2000` normal training windows). An additional `PCA + Mahalanobis (Full-D Raw)` row is evaluated on standardized full-channel `w * D` windows (`304-D` for SMD, `200-D` for PSM) as a fairness check against dimensionality compression.
- **Chronological Split:** First `75%` of `wtr` for fitting/warmup, held-out last `25%` of `wtr` for split-conformal calibration (`alpha = 0.05`). Test windows `wte` are evaluated strictly chronologically.
- **Labeling & Metrics:** **Window-level binary ground truth WITHOUT point-adjust** (`y_win = 1` iff any timestep inside the 8-sample window is labeled anomalous). **No point-adjustment (PA) is applied anywhere.**
- **Thresholding Fairness:** Every method's threshold is calibrated identically via finite-sample split-conformal quantile (`alpha = 0.05`) on the held-out `25%` calibration split. Both **Raw 1-Window FAR / Recall** and **Dual-Stage Gated FAR / Recall** (`> 2.5x thr` immediate OR 2-consecutive windows) are reported so the dual-stage alarm gate never biases cross-method FAR comparisons.
- **Variance & Seeds:** Randomized methods (`Half-Space Trees`, `RS-Hash`, `Isolation Forest`) are averaged over 5 fixed seeds `[0, 1, 2, 3, 4]` per server. Reported `mean ± std` is across the 8 servers (`ddof = 0`).
- **Locked Engine Presets:** `e_alpha`, `dc_alpha`, `huber_clip`, and `dc_relock` were **not** modified for this benchmark.

## 8-Server Summary Table (Mean ± Std Across 8 Servers, No Point-Adjust)

| Method | Category | AUROC (%) | Raw 1-Win FAR (%) | Dual-Gate FAR (%) | Dual-Gate Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| PPC-Gauge-Triad v0.7 High-Recall (`0.002/0.001/1.0`) | Streaming (Rust `f32`, 32-D) | **78.7 ± 13.0** | 37.2 ± 22.8 | 26.8 ± 24.9 | 68.1 ± 21.3 |
| PPC-Gauge-Triad v0.7 Low-FAR (`0.008/0.004/1.5`) | Streaming (Rust `f32`, 32-D) | **77.1 ± 13.1** | 14.9 ± 8.0 | 7.6 ± 7.2 | 48.2 ± 22.6 |
| PPC-Gauge-Triad v0.5 (`NoAdapt`) | Static (Rust `f32`, 32-D 4PC) | **76.1 ± 12.3** | 28.6 ± 24.4 | 19.0 ± 24.8 | 52.5 ± 23.6 |
| Half-Space Trees (Tan et al. 2011, 5 seeds) | Streaming (32-D, `psi=250`) | **74.1 ± 13.5** | 20.1 ± 19.0 | 15.5 ± 17.4 | 42.3 ± 25.1 |
| RS-Hash / RS-Stream (Sathe & Aggarwal 2016, 5 seeds) | Streaming (32-D, `W=1000`) | **80.1 ± 15.0** | 22.3 ± 23.0 | 17.8 ± 23.3 | 56.2 ± 29.5 |
| PCA + Mahalanobis (`0.95` var, matched 32-D) | Batch (32-D) | **74.9 ± 15.1** | 49.2 ± 34.2 | 43.8 ± 36.8 | 72.6 ± 24.6 |
| PCA + Mahalanobis (`0.95` var, full raw `w*D`) | Batch (Full `w*D`) | **80.9 ± 10.7** | 49.1 ± 33.7 | 43.3 ± 35.6 | 75.1 ± 26.8 |
| Isolation Forest (`n=100`, 5 seeds) | Batch (32-D) | **78.5 ± 11.0** | 57.1 ± 30.4 | 50.9 ± 32.4 | 75.0 ± 29.1 |
| kNN (`k=5`, Euclidean) | Batch (32-D) | **79.0 ± 11.0** | 55.4 ± 29.7 | 49.0 ± 33.0 | 77.6 ± 23.6 |
| OC-SVM (`RBF`, `nu=0.05`) | Batch (32-D) | **75.3 ± 12.8** | 55.1 ± 31.3 | 49.3 ± 34.9 | 73.6 ± 26.4 |

## Honest Findings & Where PPC-Gauge-Triad Wins or Loses

- **Highest 8-Server Mean AUROC:** PCA + Mahalanobis (`0.95` var, full raw `w*D`) at **80.9 ± 10.7%**.
- **Baselines outperforming PPC-Gauge-Triad v0.7 High-Recall (78.7%):** PCA + Mahalanobis (`0.95` var, full raw `w*D`) (80.9 ± 10.7%); RS-Hash / RS-Stream (Sathe & Aggarwal 2016, 5 seeds) (80.1 ± 15.0%); kNN (`k=5`, Euclidean) (79.0 ± 11.0%). Reported as-is without post-hoc preset changes.
- **Baselines outperforming PPC-Gauge-Triad v0.7 Low-FAR (77.1%):** PCA + Mahalanobis (`0.95` var, full raw `w*D`) (80.9 ± 10.7%); RS-Hash / RS-Stream (Sathe & Aggarwal 2016, 5 seeds) (80.1 ± 15.0%); kNN (`k=5`, Euclidean) (79.0 ± 11.0%); Isolation Forest (`n=100`, 5 seeds) (78.5 ± 11.0%).
- **Memory / Footprint Context:** Batch baselines (`kNN`, `OC-SVM`, `Isolation Forest`, `PCA+Mahalanobis`) require storing training buffers, trees, support vectors, or full `w*D` covariance/SVD decompositions in host RAM (`std` Python/heap). Streaming baselines (`Half-Space Trees`, `RS-Hash`) maintain multi-tree mass tables or `100 x 4 x 1024` Count-Min Sketches + FIFO ring buffers (~hundreds of KB to MBs). PPC-Gauge-Triad trades off raw offline modeling capacity for a fixed `< 16 KB` `#![no_std]` static struct footprint (`0 B` heap, `0` matrix inversions).
- **Deep Learning Baselines (USAD / TranAD) — `PENDING / Future Work`:** PyTorch is not installed in this lightweight offline verification environment (`numpy` + `scikit-learn` only). Deep temporal baselines (`USAD`, `TranAD`) are therefore explicitly marked as **future work** and are not estimated or copied from external papers whose preprocessing/point-adjust protocols differ.

## Per-Server AUROC (%) Breakdown (No Point-Adjust)

| Server | v0.7 High-Rec | v0.7 Low-FAR | HS-Trees | RS-Hash | PCA+Mahal (32-D) | PCA+Mahal (Full-D) | IsoForest | kNN | OC-SVM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SMD machine-1-1 | 92.4 | 88.5 | 76.7 | 92.7 | 86.0 | 85.0 | 87.8 | 86.4 | 85.3 |
| SMD machine-1-6 | 88.7 | 85.1 | 77.8 | 96.4 | 91.0 | 89.4 | 83.4 | 90.7 | 90.6 |
| SMD machine-2-1 | 77.1 | 77.7 | 73.4 | 75.7 | 71.3 | 76.0 | 69.9 | 73.3 | 64.9 |
| PSM eBay Server | 57.3 | 52.2 | 42.0 | 52.2 | 67.5 | 71.6 | 63.8 | 66.1 | 64.9 |
| SMD machine-1-2 | 88.0 | 88.0 | 87.8 | 90.0 | 88.2 | 89.6 | 90.4 | 91.4 | 85.7 |
| SMD machine-1-3 | 72.3 | 76.8 | 72.4 | 73.9 | 47.7 | 82.1 | 81.9 | 70.7 | 65.1 |
| SMD machine-2-2 | 61.5 | 59.6 | 73.8 | 64.9 | 58.6 | 59.3 | 61.5 | 63.7 | 56.7 |
| SMD machine-3-1 | 92.4 | 88.6 | 88.9 | 94.8 | 88.8 | 94.4 | 89.4 | 90.1 | 89.4 |

## Per-Server Dual-Gate FAR (%) Breakdown (`alpha = 0.05` Split-Conformal)

| Server | v0.7 High-Rec | v0.7 Low-FAR | HS-Trees | RS-Hash | PCA+Mahal (32-D) | PCA+Mahal (Full-D) | IsoForest | kNN | OC-SVM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SMD machine-1-1 | 23.3 | 3.9 | 11.2 | 18.4 | 54.0 | 56.9 | 70.3 | 64.5 | 63.0 |
| SMD machine-1-6 | 78.4 | 25.3 | 58.9 | 77.8 | 83.8 | 84.2 | 84.0 | 83.9 | 83.8 |
| SMD machine-2-1 | 8.2 | 3.5 | 19.0 | 10.0 | 4.6 | 2.2 | 31.4 | 7.6 | 5.7 |
| PSM eBay Server | 55.7 | 7.3 | 16.1 | 11.5 | 94.4 | 99.2 | 80.7 | 89.1 | 86.5 |
| SMD machine-1-2 | 1.9 | 2.2 | 1.0 | 0.8 | 3.1 | 0.9 | 0.8 | 1.0 | 0.6 |
| SMD machine-1-3 | 23.6 | 7.6 | 8.8 | 13.0 | 82.7 | 64.8 | 86.3 | 78.5 | 89.6 |
| SMD machine-2-2 | 10.3 | 9.5 | 8.1 | 8.5 | 19.4 | 19.7 | 7.5 | 21.7 | 15.8 |
| SMD machine-3-1 | 13.1 | 1.9 | 1.2 | 2.1 | 8.7 | 18.3 | 46.6 | 45.4 | 49.7 |

*Reproduce directly from local offline cache via `python benchmarks/run_benchmarks.py --baselines --offline`.*
