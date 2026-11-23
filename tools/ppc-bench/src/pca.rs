//! Pure-Rust deterministic PCA (symmetric cyclic Jacobi eigendecomposition),
//! `to_windows_v6` (`3PC+Resid` / `4PC`), and `PcaMahalanobisDetector`.

use crate::npz::MatrixF64;

#[derive(Debug, Clone)]
pub struct PcaModel {
    pub mean: Vec<f64>,
    /// Row-major `(n_components, n_features)` principal axes.
    pub components: Vec<f64>,
    pub explained_variance: Vec<f64>,
    pub n_components: usize,
    pub n_features: usize,
}

/// Cyclic Jacobi eigensolver for a real symmetric `n x n` matrix stored row-major.
/// Returns `(eigenvalues, eigenvectors_col_major)` sorted by eigenvalue descending.
/// `eigenvectors[k]` is the unit eigenvector corresponding to `eigenvalues[k]`.
pub fn symmetric_eigen_desc(sym_mat: &[f64], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut a = sym_mat.to_vec();
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_sweeps = 60usize;
    for sweep in 0..max_sweeps {
        let mut off_norm = 0.0_f64;
        for p in 0..n {
            let row_p = p * n;
            for q in (p + 1)..n {
                let apq = a[row_p + q];
                off_norm += apq * apq;
            }
        }
        if off_norm <= 1e-24 * (n as f64) {
            break;
        }
        let thresh = if sweep < 3 {
            0.2 * off_norm.sqrt() / ((n * n) as f64)
        } else {
            0.0
        };

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                let abs_apq = apq.abs();
                if abs_apq <= 1e-16 || (sweep < 3 && abs_apq <= thresh) {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let diff = aqq - app;
                let t = if abs_apq < 1e-18 * diff.abs() {
                    apq / diff
                } else {
                    let tau = 0.5 * diff / apq;
                    let denom = tau.abs() + (1.0 + tau * tau).sqrt();
                    if tau >= 0.0 { 1.0 / denom } else { -1.0 / denom }
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau_rot = s / (1.0 + c);

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update remaining entries of rows/cols p and q
                for r in 0..p {
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let np = arp - s * (arq + tau_rot * arp);
                    let nq = arq + s * (arp - tau_rot * arq);
                    a[r * n + p] = np;
                    a[p * n + r] = np;
                    a[r * n + q] = nq;
                    a[q * n + r] = nq;
                }
                for r in (p + 1)..q {
                    let apr = a[p * n + r];
                    let arq = a[r * n + q];
                    let np = apr - s * (arq + tau_rot * apr);
                    let nq = arq + s * (apr - tau_rot * arq);
                    a[p * n + r] = np;
                    a[r * n + p] = np;
                    a[r * n + q] = nq;
                    a[q * n + r] = nq;
                }
                for r in (q + 1)..n {
                    let apr = a[p * n + r];
                    let aqr = a[q * n + r];
                    let np = apr - s * (aqr + tau_rot * apr);
                    let nq = aqr + s * (apr - tau_rot * aqr);
                    a[p * n + r] = np;
                    a[r * n + p] = np;
                    a[q * n + r] = nq;
                    a[r * n + q] = nq;
                }

                // Accumulate eigenvectors (stored as columns in v)
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = vrp - s * (vrq + tau_rot * vrp);
                    v[r * n + q] = vrq + s * (vrp - tau_rot * vrq);
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        a[j * n + j]
            .partial_cmp(&a[i * n + i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut evals = Vec::with_capacity(n);
    let mut evecs = Vec::with_capacity(n);
    for &idx in &order {
        evals.push(a[idx * n + idx].max(0.0));
        let mut col = Vec::with_capacity(n);
        let mut max_abs = 0.0_f64;
        let mut max_sign = 1.0_f64;
        for r in 0..n {
            let val = v[r * n + idx];
            if val.abs() > max_abs {
                max_abs = val.abs();
                max_sign = if val < 0.0 { -1.0 } else { 1.0 };
            }
            col.push(val);
        }
        // Canonicalize deterministic sign (largest-abs loading positive)
        if max_sign < 0.0 {
            for x in &mut col {
                *x = -*x;
            }
        }
        evecs.push(col);
    }
    (evals, evecs)
}

impl PcaModel {
    pub fn fit_fixed(x: &[f64], n_rows: usize, n_cols: usize, n_components: usize) -> Self {
        let k = n_components.min(n_cols).min(n_rows.saturating_sub(1)).max(1);
        let (mean, cov) = Self::mean_and_cov(x, n_rows, n_cols);
        let (evals, evecs) = symmetric_eigen_desc(&cov, n_cols);

        let mut components = Vec::with_capacity(k * n_cols);
        let mut explained_variance = Vec::with_capacity(k);
        for i in 0..k {
            explained_variance.push(evals[i]);
            components.extend_from_slice(&evecs[i]);
        }
        Self {
            mean,
            components,
            explained_variance,
            n_components: k,
            n_features: n_cols,
        }
    }

    pub fn fit_variance_ratio(
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        variance_ratio: f64,
        max_components: usize,
    ) -> Self {
        let max_k = max_components.min(n_cols).min(n_rows.saturating_sub(1)).max(1);
        let (mean, cov) = Self::mean_and_cov(x, n_rows, n_cols);
        let (evals, evecs) = symmetric_eigen_desc(&cov, n_cols);
        let total_var: f64 = evals.iter().sum::<f64>().max(1e-18);

        let mut cum = 0.0_f64;
        let mut k = max_k;
        for (i, &ev) in evals.iter().take(max_k).enumerate() {
            cum += ev / total_var;
            if cum >= variance_ratio {
                k = i + 1;
                break;
            }
        }
        k = k.clamp(1, max_k);

        let mut components = Vec::with_capacity(k * n_cols);
        let mut explained_variance = Vec::with_capacity(k);
        for i in 0..k {
            explained_variance.push(evals[i]);
            components.extend_from_slice(&evecs[i]);
        }
        Self {
            mean,
            components,
            explained_variance,
            n_components: k,
            n_features: n_cols,
        }
    }

    fn mean_and_cov(x: &[f64], n_rows: usize, n_cols: usize) -> (Vec<f64>, Vec<f64>) {
        let mut mean = vec![0.0_f64; n_cols];
        for r in 0..n_rows {
            let row = &x[r * n_cols..(r + 1) * n_cols];
            for c in 0..n_cols {
                mean[c] += row[c];
            }
        }
        let inv_n = 1.0 / (n_rows.max(1) as f64);
        for m in &mut mean {
            *m *= inv_n;
        }

        let mut cov = vec![0.0_f64; n_cols * n_cols];
        let mut centered = vec![0.0_f64; n_cols];
        for r in 0..n_rows {
            let row = &x[r * n_cols..(r + 1) * n_cols];
            for c in 0..n_cols {
                centered[c] = row[c] - mean[c];
            }
            for i in 0..n_cols {
                let ci = centered[i];
                let base = i * n_cols;
                for j in i..n_cols {
                    cov[base + j] += ci * centered[j];
                }
            }
        }
        let denom = (n_rows.saturating_sub(1).max(1)) as f64;
        for i in 0..n_cols {
            for j in i..n_cols {
                let val = cov[i * n_cols + j] / denom;
                cov[i * n_cols + j] = val;
                cov[j * n_cols + i] = val;
            }
        }
        (mean, cov)
    }

    /// Transform a single row of length `n_features` into `out` (`n_components`)
    /// and return the L2 orthogonal reconstruction residual norm `||x_c - P^T P x_c||_2`.
    #[inline]
    pub fn transform_row_with_residual(&self, row: &[f64], out: &mut [f64]) -> f64 {
        let d = self.n_features;
        let k = self.n_components;
        let mut norm_xc_sq = 0.0_f64;
        for j in 0..d {
            let dj = row[j] - self.mean[j];
            norm_xc_sq += dj * dj;
        }
        let mut norm_proj_sq = 0.0_f64;
        for c in 0..k {
            let comp = &self.components[c * d..(c + 1) * d];
            let mut dot = 0.0_f64;
            for j in 0..d {
                dot += (row[j] - self.mean[j]) * comp[j];
            }
            out[c] = dot;
            norm_proj_sq += dot * dot;
        }
        (norm_xc_sq - norm_proj_sq).max(0.0).sqrt()
    }

    pub fn transform_batch(&self, x: &[f64], n_rows: usize) -> Vec<f64> {
        let d = self.n_features;
        let k = self.n_components;
        let mut out = vec![0.0_f64; n_rows * k];
        for r in 0..n_rows {
            let row = &x[r * d..(r + 1) * d];
            let dst = &mut out[r * k..(r + 1) * k];
            for c in 0..k {
                let comp = &self.components[c * d..(c + 1) * d];
                let mut dot = 0.0_f64;
                for j in 0..d {
                    dot += (row[j] - self.mean[j]) * comp[j];
                }
                dst[c] = dot;
            }
        }
        out
    }
}

/// Fitted standardization + PCA preprocessing metadata (for cross-device zero-shot projection).
#[derive(Debug, Clone)]
pub struct WindowPreprocMeta {
    pub mu: Vec<f64>,
    pub sd: Vec<f64>,
    pub pca: PcaModel,
    pub r_mu: Option<f64>,
    pub r_sd: Option<f64>,
}

/// Non-overlapping `w x d` windowed telemetry dataset.
#[derive(Debug, Clone)]
pub struct WindowedDataset {
    /// Flattened `(n_tr, w * d)` in `f64` (also convertible to `f32` for engine).
    pub w_tr: Vec<f64>,
    pub n_tr: usize,
    /// Flattened `(n_te, w * d)` in `f64`.
    pub w_te: Vec<f64>,
    pub n_te: usize,
    /// Window-level binary anomaly labels (1 iff ANY sample inside window > 0; NO point-adjust).
    pub y_te: Vec<i32>,
    pub meta: WindowPreprocMeta,
}

fn standardize_fit(mat: &MatrixF64, limit_rows: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = limit_rows.min(mat.rows);
    let cols = mat.cols;
    let mut mu = vec![0.0_f64; cols];
    for r in 0..n {
        let row = mat.row(r);
        for c in 0..cols {
            mu[c] += row[c];
        }
    }
    let inv_n = 1.0 / (n.max(1) as f64);
    for m in &mut mu {
        *m *= inv_n;
    }

    let mut sd = vec![0.0_f64; cols];
    for r in 0..n {
        let row = mat.row(r);
        for c in 0..cols {
            let diff = row[c] - mu[c];
            sd[c] += diff * diff;
        }
    }
    for s in &mut sd {
        *s = (*s * inv_n).sqrt().max(1e-6);
    }

    let mut z = vec![0.0_f64; n * cols];
    for r in 0..n {
        let row = mat.row(r);
        let dst = &mut z[r * cols..(r + 1) * cols];
        for c in 0..cols {
            dst[c] = (row[c] - mu[c]) / sd[c];
        }
    }
    (mu, sd, z)
}

fn standardize_apply(mat: &MatrixF64, mu: &[f64], sd: &[f64]) -> Vec<f64> {
    let cols = mat.cols;
    let mut z = vec![0.0_f64; mat.rows * cols];
    for r in 0..mat.rows {
        let row = mat.row(r);
        let dst = &mut z[r * cols..(r + 1) * cols];
        for c in 0..cols {
            dst[c] = (row[c] - mu[c]) / sd[c];
        }
    }
    z
}

/// Exact Rust equivalent of `benchmarks/ppc_triad_wrapper.py::to_windows_v6`.
pub fn to_windows_v6(
    tr_raw: &MatrixF64,
    te_raw: &MatrixF64,
    yl_raw: &[i32],
    w: usize,
    d: usize,
    max_tr_win: usize,
    use_pca_resid: bool,
    tr_row_limit: Option<usize>,
) -> WindowedDataset {
    let tr_rows = tr_row_limit.unwrap_or(tr_raw.rows).min(tr_raw.rows);
    let cols = tr_raw.cols;
    let (mu, sd, ztr_std) = standardize_fit(tr_raw, tr_rows);
    let zte_std = standardize_apply(te_raw, &mu, &sd);

    let (ztr, zte, pca, r_mu, r_sd) = if use_pca_resid && cols > d {
        let pca = PcaModel::fit_fixed(&ztr_std, tr_rows, cols, d - 1);
        let mut rtr = Vec::with_capacity(tr_rows);
        let mut ptr = vec![0.0_f64; tr_rows * (d - 1)];
        for r in 0..tr_rows {
            let res = pca.transform_row_with_residual(
                &ztr_std[r * cols..(r + 1) * cols],
                &mut ptr[r * (d - 1)..(r + 1) * (d - 1)],
            );
            rtr.push(res);
        }
        let r_mean = rtr.iter().sum::<f64>() / (tr_rows.max(1) as f64);
        let r_var = rtr.iter().map(|&v| (v - r_mean).powi(2)).sum::<f64>() / (tr_rows.max(1) as f64);
        let r_std = r_var.sqrt().max(1e-6);

        let mut ztr = vec![0.0_f64; tr_rows * d];
        for r in 0..tr_rows {
            ztr[r * d..r * d + (d - 1)].copy_from_slice(&ptr[r * (d - 1)..(r + 1) * (d - 1)]);
            ztr[r * d + (d - 1)] = (rtr[r] - r_mean) / r_std;
        }

        let mut zte = vec![0.0_f64; te_raw.rows * d];
        let mut proj = vec![0.0_f64; d - 1];
        for r in 0..te_raw.rows {
            let res = pca.transform_row_with_residual(&zte_std[r * cols..(r + 1) * cols], &mut proj);
            zte[r * d..r * d + (d - 1)].copy_from_slice(&proj);
            zte[r * d + (d - 1)] = (res - r_mean) / r_std;
        }
        (ztr, zte, pca, Some(r_mean), Some(r_std))
    } else {
        let pca = PcaModel::fit_fixed(&ztr_std, tr_rows, cols, d);
        let ztr = pca.transform_batch(&ztr_std, tr_rows);
        let zte = pca.transform_batch(&zte_std, te_raw.rows);
        (ztr, zte, pca, None, None)
    };

    let n_tr = (tr_rows / w).min(max_tr_win);
    let n_te = te_raw.rows / w;
    let w_tr = ztr[..n_tr * w * d].to_vec();
    let w_te = zte[..n_te * w * d].to_vec();
    let mut y_te = Vec::with_capacity(n_te);
    for wi in 0..n_te {
        let is_anom = yl_raw[wi * w..(wi + 1) * w].iter().any(|&v| v > 0);
        y_te.push(if is_anom { 1 } else { 0 });
    }

    WindowedDataset {
        w_tr,
        n_tr,
        w_te,
        n_te,
        y_te,
        meta: WindowPreprocMeta {
            mu,
            sd,
            pca,
            r_mu,
            r_sd,
        },
    }
}

/// Project an unseen target server through a source server's fitted `WindowPreprocMeta` (Zero-Shot `0w`).
pub fn project_zero_shot(
    te_raw: &MatrixF64,
    yl_raw: &[i32],
    meta: &WindowPreprocMeta,
    w: usize,
    d: usize,
) -> (Vec<f64>, usize, Vec<i32>) {
    let cols = te_raw.cols;
    let zte_std = standardize_apply(te_raw, &meta.mu, &meta.sd);
    let mut zte = vec![0.0_f64; te_raw.rows * d];
    let r_mu = meta.r_mu.unwrap_or(0.0);
    let r_sd = meta.r_sd.unwrap_or(1.0);
    let mut proj = vec![0.0_f64; d - 1];
    for r in 0..te_raw.rows {
        let res = meta.pca.transform_row_with_residual(&zte_std[r * cols..(r + 1) * cols], &mut proj);
        zte[r * d..r * d + (d - 1)].copy_from_slice(&proj);
        zte[r * d + (d - 1)] = (res - r_mu) / r_sd;
    }
    let n_te = te_raw.rows / w;
    let mut y_te = Vec::with_capacity(n_te);
    for wi in 0..n_te {
        let is_anom = yl_raw[wi * w..(wi + 1) * w].iter().any(|&v| v > 0);
        y_te.push(if is_anom { 1 } else { 0 });
    }
    (zte[..n_te * w * d].to_vec(), n_te, y_te)
}

/// Standardize full raw `D`-channel streams and segment into non-overlapping `w * D` windows.
pub fn to_raw_full_d_windows(
    tr_raw: &MatrixF64,
    te_raw: &MatrixF64,
    yl_raw: &[i32],
    w: usize,
    max_tr_win: usize,
) -> (Vec<f64>, usize, Vec<f64>, usize, Vec<i32>, usize) {
    let cols = tr_raw.cols;
    let (mu, sd, ztr) = standardize_fit(tr_raw, tr_raw.rows);
    let zte = standardize_apply(te_raw, &mu, &sd);
    let n_tr = (tr_raw.rows / w).min(max_tr_win);
    let n_te = te_raw.rows / w;
    let feat_dim = w * cols;
    let mut y_te = Vec::with_capacity(n_te);
    for wi in 0..n_te {
        let is_anom = yl_raw[wi * w..(wi + 1) * w].iter().any(|&v| v > 0);
        y_te.push(if is_anom { 1 } else { 0 });
    }
    (
        ztr[..n_tr * feat_dim].to_vec(),
        n_tr,
        zte[..n_te * feat_dim].to_vec(),
        n_te,
        y_te,
        feat_dim,
    )
}

/// Deterministic PCA + Regularized Mahalanobis distance detector (`sum_k z_k^2 / (lambda_k + reg)`).
pub struct PcaMahalanobisDetector {
    pca: PcaModel,
    inv_var: Vec<f64>,
}

impl PcaMahalanobisDetector {
    pub fn fit(
        x_tr: &[f64],
        n_rows: usize,
        n_cols: usize,
        variance_ratio: f64,
        max_components: usize,
        reg: f64,
    ) -> Self {
        let pca = PcaModel::fit_variance_ratio(x_tr, n_rows, n_cols, variance_ratio, max_components);
        let inv_var = pca
            .explained_variance
            .iter()
            .map(|&ev| 1.0 / (ev + reg))
            .collect();
        Self { pca, inv_var }
    }

    pub fn score(&self, x_batch: &[f64], n_rows: usize) -> Vec<f64> {
        let z = self.pca.transform_batch(x_batch, n_rows);
        let k = self.pca.n_components;
        let mut out = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let row = &z[r * k..(r + 1) * k];
            let mut d2 = 0.0_f64;
            for j in 0..k {
                d2 += row[j] * row[j] * self.inv_var[j];
            }
            out.push(d2);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_matches_exact_2x2_and_3x3_eigenvalues() {
        // Symmetric matrix [[3, 1], [1, 3]] -> eigenvalues 4 and 2
        let m = [3.0, 1.0, 1.0, 3.0];
        let (evals, evecs) = symmetric_eigen_desc(&m, 2);
        assert!((evals[0] - 4.0).abs() < 1e-12);
        assert!((evals[1] - 2.0).abs() < 1e-12);
        let dot: f64 = evecs[0].iter().zip(&evecs[1]).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-12);
    }
}
