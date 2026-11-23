# Fair Baseline Comparison — 8-Server SMD + PSM Chronological Replay

## Evaluation Protocol (Rust-First Runner, Identical Across All Methods)

- **Execution Engine:** Core PPC-Gauge-Triad replay, **Half-Space Trees**, **RS-Hash**, **PCA + Mahalanobis** (matched `32-D` & full `w*D`), and **kNN (`k=5`)** run natively in Rust (`tools/ppc-bench/`). Only **Isolation Forest** and **OC-SVM** run via `scikit-learn` on the identical preprocessed windows.
- **Datasets & Windowing:** 7 SMD servers + PSM eBay Server (`8` servers, offline `data_cache/*.npz`), non-overlapping `w = 8` windows (`3PC+Resid`, `max_tr_win = 2000`), chronological `75% / 25%` fit/calibration split.
- **Labeling & AUROC:** **Window-level binary ground truth strictly WITHOUT point-adjust** (`y_win = 1` iff any sample in window is anomalous), scored with **exact `O(N log N)` Mann-Whitney U AUROC** (zero subsampling on PSM).
- **Thresholding Fairness:** Split-conformal `alpha = 0.05` on the held-out `25%` calibration split; both **Raw 1-Win FAR** and shift-invariant **Dual-Gate FAR / Recall** are reported identically for every method.
- **Locked Engine Presets:** `e_alpha`, `dc_alpha`, `huber_clip`, and `dc_relock` were **not** modified.

## 8-Server Summary Table (Mean ± Std Across 8 Servers, No Point-Adjust)

| Method | Category | AUROC (%) | Raw 1-Win FAR (%) | Dual-Gate FAR (%) | Dual-Gate Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| PPC-Gauge-Triad v0.7 High-Recall (`0.002/0.001/1.0`) | Streaming (Rust `f32`, 32-D) | **78.8 ± 13.0** | 37.2 ± 22.8 | 26.8 ± 24.9 | 68.1 ± 21.3 |
| PPC-Gauge-Triad v0.7 Low-FAR (`0.008/0.004/1.5`) | Streaming (Rust `f32`, 32-D) | **77.1 ± 13.0** | 14.9 ± 8.0 | 7.6 ± 7.2 | 48.2 ± 22.6 |
| PPC-Gauge-Triad v0.5 (`NoAdapt`) | Static (Rust `f32`, 4PC) | **76.0 ± 12.4** | 28.6 ± 24.4 | 19.0 ± 24.8 | 52.5 ± 23.6 |
| Half-Space Trees (Tan et al. 2011, 5 seeds) | Streaming (Rust, `psi=250`) | **73.9 ± 13.6** | 18.6 ± 17.7 | 14.5 ± 15.6 | 41.0 ± 25.4 |
| RS-Hash / RS-Stream (Sathe & Aggarwal 2016, 5 seeds) | Streaming (Rust, `W=1000`) | **80.3 ± 15.0** | 23.4 ± 23.4 | 18.8 ± 24.0 | 57.4 ± 29.7 |
| PCA + Mahalanobis (`0.95` var, matched 32-D) | Batch (Rust, 32-D) | **74.9 ± 15.1** | 49.2 ± 34.2 | 43.8 ± 36.8 | 72.6 ± 24.6 |
| PCA + Mahalanobis (`0.95` var, full raw `w*D`) | Batch (Rust, Full `w*D`) | **81.1 ± 10.7** | 49.1 ± 33.7 | 43.3 ± 35.6 | 75.1 ± 26.8 |
| kNN (`k=5`, Euclidean) | Batch (Rust, 32-D) | **79.1 ± 11.0** | 55.4 ± 29.7 | 49.0 ± 33.0 | 77.6 ± 23.6 |
| Isolation Forest (`n=100`, 5 seeds) | Batch (`sklearn`, 32-D) | **78.5 ± 10.9** | 57.1 ± 30.4 | 50.9 ± 32.4 | 75.0 ± 29.1 |
| OC-SVM (`RBF`, `nu=0.05`) | Batch (`sklearn`, 32-D) | **75.4 ± 12.8** | 55.1 ± 31.3 | 49.3 ± 34.9 | 73.6 ± 26.4 |

## Honest Findings & Where PPC-Gauge-Triad Wins or Loses

- **Highest 8-Server Mean AUROC:** PCA + Mahalanobis (`0.95` var, full raw `w*D`) at **81.1 ± 10.7%**.
- **Baselines outperforming PPC-Gauge-Triad v0.7 High-Recall (78.8%):** PCA + Mahalanobis (`0.95` var, full raw `w*D`) (81.1 ± 10.7%); RS-Hash / RS-Stream (Sathe & Aggarwal 2016, 5 seeds) (80.3 ± 15.0%); kNN (`k=5`, Euclidean) (79.1 ± 11.0%). Reported as-is without post-hoc preset changes.
- **Baselines outperforming PPC-Gauge-Triad v0.7 Low-FAR (77.1%):** PCA + Mahalanobis (`0.95` var, full raw `w*D`) (81.1 ± 10.7%); RS-Hash / RS-Stream (Sathe & Aggarwal 2016, 5 seeds) (80.3 ± 15.0%); kNN (`k=5`, Euclidean) (79.1 ± 11.0%); Isolation Forest (`n=100`, 5 seeds) (78.5 ± 10.9%).
- **Memory / Footprint Context:** Host-RAM batch and streaming baselines hold training buffers, trees, SVD covariances, or `100 x 4 x 1024` Count-Min Sketches + FIFO rings (~hundreds of KB to MBs). PPC-Gauge-Triad trades off raw offline AUROC for a fixed `< 16 KB` `#![no_std]` static struct footprint (`0 B` heap, `0` matrix inversions) and achieves the lowest Dual-Gate FAR (**`7.6 ± 7.2%`** with `v0.7 Low-FAR`).
- **Deep Learning Baselines (USAD / TranAD) — `PENDING / Future Work`:** PyTorch is not installed in this lightweight offline verification environment.

## Per-Server AUROC (%) Breakdown (Exact Mann-Whitney U, No Point-Adjust)

| Server | v0.7 High-Rec | v0.7 Low-FAR | HS-Trees | RS-Hash | PCA+Mahal (32-D) | PCA+Mahal (Full-D) | IsoForest | kNN | OC-SVM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SMD machine-1-1 | 92.4 | 88.5 | 74.8 | 92.9 | 86.0 | 85.0 | 87.8 | 86.4 | 85.3 |
| SMD machine-1-6 | 88.7 | 85.1 | 78.7 | 96.3 | 91.0 | 89.4 | 83.4 | 90.7 | 90.6 |
| SMD machine-2-1 | 77.1 | 77.7 | 73.3 | 76.3 | 71.3 | 76.0 | 69.9 | 73.3 | 64.9 |
| PSM eBay Server | 57.6 | 52.7 | 41.9 | 52.3 | 67.6 | 72.5 | 64.0 | 66.3 | 65.0 |
| SMD machine-1-2 | 88.0 | 88.0 | 87.2 | 90.2 | 88.2 | 89.6 | 90.4 | 91.4 | 85.7 |
| SMD machine-1-3 | 72.3 | 76.8 | 71.7 | 74.1 | 47.7 | 82.1 | 81.9 | 70.7 | 65.1 |
| SMD machine-2-2 | 61.5 | 59.6 | 73.9 | 65.3 | 58.6 | 59.3 | 61.5 | 63.7 | 56.7 |
| SMD machine-3-1 | 92.4 | 88.6 | 89.8 | 94.9 | 88.8 | 94.4 | 89.4 | 90.1 | 89.4 |

## Per-Server Dual-Gate FAR (%) Breakdown (`alpha = 0.05` Split-Conformal)

| Server | v0.7 High-Rec | v0.7 Low-FAR | HS-Trees | RS-Hash | PCA+Mahal (32-D) | PCA+Mahal (Full-D) | IsoForest | kNN | OC-SVM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SMD machine-1-1 | 23.3 | 3.9 | 5.6 | 20.3 | 54.0 | 56.9 | 70.3 | 64.5 | 63.0 |
| SMD machine-1-6 | 78.4 | 25.3 | 51.1 | 80.3 | 83.8 | 84.2 | 84.0 | 83.9 | 83.8 |
| SMD machine-2-1 | 8.2 | 3.5 | 24.6 | 11.8 | 4.6 | 2.2 | 31.4 | 7.6 | 5.7 |
| PSM eBay Server | 55.7 | 7.3 | 15.8 | 11.8 | 94.4 | 99.2 | 80.7 | 89.1 | 86.5 |
| SMD machine-1-2 | 1.9 | 2.2 | 1.0 | 0.7 | 3.1 | 0.9 | 0.8 | 1.0 | 0.6 |
| SMD machine-1-3 | 23.6 | 7.6 | 9.1 | 14.7 | 82.7 | 64.8 | 86.3 | 78.5 | 89.6 |
| SMD machine-2-2 | 10.3 | 9.5 | 7.2 | 8.6 | 19.4 | 19.7 | 7.5 | 21.7 | 15.8 |
| SMD machine-3-1 | 13.1 | 1.9 | 1.5 | 2.1 | 8.7 | 18.3 | 46.6 | 45.4 | 49.7 |

*Reproduce Rust-native core benchmarks directly via `cargo run --release --manifest-path tools/ppc-bench/Cargo.toml -- all`, or regenerate this full table including `sklearn` baselines via `python benchmarks/run_benchmarks.py --baselines --offline`.*
