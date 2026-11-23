//! Pure-Rust deterministic streaming & window anomaly baselines and evaluation metrics.
//!
//! All detectors consume identical non-overlapping chronological `w=8` windows,
//! chronological 75%/25% fit/calibration splits, split-conformal `alpha=0.05`
//! thresholds, shift-invariant dual-stage alarm gating, and window-level binary
//! labels strictly WITHOUT point-adjust.

use crate::pca::{to_raw_full_d_windows, to_windows_v6, PcaMahalanobisDetector};
use crate::npz::ServerDataset;

pub const DEFAULT_SEEDS: [u64; 5] = [0, 1, 2, 3, 4];

/// Fast deterministic SplitMix64 PRNG (zero external crate dependency).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Uniform in [0, 1) using top 53 bits
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    #[inline]
    pub fn gen_range_usize(&mut self, lo: usize, hi_excl: usize) -> usize {
        if hi_excl <= lo {
            return lo;
        }
        lo + ((self.next_u64() as usize) % (hi_excl - lo))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EvalMetrics {
    pub auroc: f64,
    pub far_raw: f64,
    pub recall_raw: f64,
    pub far_dual: f64,
    pub recall_dual: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct AggregatedMetrics {
    pub auroc: f64,
    pub auroc_seed_std: f64,
    pub far_raw: f64,
    pub recall_raw: f64,
    pub far_dual: f64,
    pub recall_dual: f64,
}

/// Exact rank-based Mann-Whitney U AUROC in `O(N log N)` (handles ties via average rank).
pub fn compute_auroc(y_true: &[i32], scores: &[f64]) -> f64 {
    let n = y_true.len().min(scores.len());
    let mut n_pos: u64 = 0;
    let mut n_neg: u64 = 0;
    let mut pairs: Vec<(f64, i32)> = Vec::with_capacity(n);
    for i in 0..n {
        let y = if y_true[i] > 0 { 1 } else { 0 };
        if y == 1 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
        let s = if scores[i].is_finite() { scores[i] } else { 0.0 };
        pairs.push((s, y));
    }
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut sum_pos_ranks = 0.0_f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        // Average 1-based rank for tie group [i..j)
        let avg_rank = 0.5 * ((i + 1) as f64 + (j as f64));
        let mut pos_in_group = 0u64;
        for item in &pairs[i..j] {
            if item.1 == 1 {
                pos_in_group += 1;
            }
        }
        sum_pos_ranks += avg_rank * (pos_in_group as f64);
        i = j;
    }

    let u = sum_pos_ranks - (n_pos as f64) * ((n_pos + 1) as f64) * 0.5;
    u / ((n_pos as f64) * (n_neg as f64))
}

pub fn median_f64(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut s = vals.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let m = s.len();
    if m % 2 == 1 {
        s[m / 2]
    } else {
        0.5 * (s[m / 2 - 1] + s[m / 2])
    }
}

/// Finite-sample split-conformal `(1 - alpha)` quantile threshold.
pub fn conformal_threshold(scores_cal: &[f64], alpha: f64) -> f64 {
    if scores_cal.is_empty() {
        return 0.0;
    }
    let mut s = scores_cal.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let idx_f = ((1.0 - alpha) * ((n + 1) as f64)).ceil() - 1.0;
    let idx = (idx_f.max(0.0) as usize).min(n - 1);
    s[idx]
}

/// Exact Rust equivalent of `triad_norm_score` (`log1p`-MAD robust z-score sum).
pub fn triad_norm_score(
    sh_c: &[f64],
    en_c: &[f64],
    lv_c: &[f64],
    sh_t: &[f64],
    en_t: &[f64],
    lv_t: &[f64],
) -> Vec<f64> {
    fn rz(c: &[f64], t: &[f64]) -> Vec<f64> {
        let lc: Vec<f64> = c.iter().map(|&v| v.max(0.0).ln_1p()).collect();
        let med = median_f64(&lc);
        let abs_dev: Vec<f64> = lc.iter().map(|&v| (v - med).abs()).collect();
        let mad = median_f64(&abs_dev) * 1.4826 + 0.1;
        t.iter()
            .map(|&v| ((v.max(0.0).ln_1p() - med) / mad).max(0.0))
            .collect()
    }
    let r_sh = rz(sh_c, sh_t);
    let r_en = rz(en_c, en_t);
    let r_lv = rz(lv_c, lv_t);
    (0..sh_t.len()).map(|i| r_sh[i] + r_en[i] + r_lv[i]).collect()
}

pub fn dual_gate_scalar(scores: &[f64], thr: f64, sev_mult: f64) -> Vec<bool> {
    let mut al = vec![false; scores.len()];
    let mut streak = 0u32;
    for (i, &s) in scores.iter().enumerate() {
        streak = if s > thr { streak + 1 } else { 0 };
        if s > thr * sev_mult || streak >= 2 {
            al[i] = true;
        }
    }
    al
}

/// Shift-invariant dual-stage alarm gate (handles negative/offset scores fairly).
pub fn shift_invariant_dual_gate(
    scores_cal: &[f64],
    scores_te: &[f64],
    thr: f64,
    sev_mult: f64,
) -> Vec<bool> {
    let med = median_f64(scores_cal);
    let margin = (thr - med).max(0.0);
    let severe_thr = med + sev_mult * margin;
    let mut al = vec![false; scores_te.len()];
    let mut streak = 0u32;
    for (i, &s) in scores_te.iter().enumerate() {
        streak = if s > thr { streak + 1 } else { 0 };
        if s > severe_thr || streak >= 2 {
            al[i] = true;
        }
    }
    al
}

pub fn evaluate_score_stream(
    scores_cal: &[f64],
    scores_te: &[f64],
    y_te: &[i32],
    alpha: f64,
    sev_mult: f64,
) -> EvalMetrics {
    let thr = conformal_threshold(scores_cal, alpha);
    let al_raw: Vec<bool> = scores_te.iter().map(|&s| s > thr).collect();
    let al_dual = shift_invariant_dual_gate(scores_cal, scores_te, thr, sev_mult);
    let auroc = compute_auroc(y_te, scores_te) * 100.0;

    let mut n_neg = 0usize;
    let mut n_pos = 0usize;
    let mut fp_raw = 0usize;
    let mut tp_raw = 0usize;
    let mut fp_dual = 0usize;
    let mut tp_dual = 0usize;
    for i in 0..y_te.len() {
        if y_te[i] == 0 {
            n_neg += 1;
            if al_raw[i] { fp_raw += 1; }
            if al_dual[i] { fp_dual += 1; }
        } else {
            n_pos += 1;
            if al_raw[i] { tp_raw += 1; }
            if al_dual[i] { tp_dual += 1; }
        }
    }
    EvalMetrics {
        auroc,
        far_raw: (fp_raw as f64 / n_neg.max(1) as f64) * 100.0,
        recall_raw: (tp_raw as f64 / n_pos.max(1) as f64) * 100.0,
        far_dual: (fp_dual as f64 / n_neg.max(1) as f64) * 100.0,
        recall_dual: (tp_dual as f64 / n_pos.max(1) as f64) * 100.0,
    }
}

pub fn aggregate_runs(runs: &[EvalMetrics]) -> AggregatedMetrics {
    let n = runs.len().max(1) as f64;
    let auroc = runs.iter().map(|r| r.auroc).sum::<f64>() / n;
    let var_au = runs.iter().map(|r| (r.auroc - auroc).powi(2)).sum::<f64>() / n;
    AggregatedMetrics {
        auroc,
        auroc_seed_std: var_au.sqrt(),
        far_raw: runs.iter().map(|r| r.far_raw).sum::<f64>() / n,
        recall_raw: runs.iter().map(|r| r.recall_raw).sum::<f64>() / n,
        far_dual: runs.iter().map(|r| r.far_dual).sum::<f64>() / n,
        recall_dual: runs.iter().map(|r| r.recall_dual).sum::<f64>() / n,
    }
}

/// Streaming Half-Space Trees (Tan, Ting & Liu, IJCAI 2011) in pure Rust.
pub struct HalfSpaceTrees {
    n_trees: usize,
    max_depth: usize,
    window_size: usize,
    size_limit: i32,
    n_internal: usize,
    n_nodes: usize,
    split_feat: Vec<usize>,
    split_val: Vec<f64>,
    mass_r: Vec<i32>,
    mass_l: Vec<i32>,
    stream_count: usize,
    depth_scale: Vec<f64>,
}

impl HalfSpaceTrees {
    pub fn new(
        n_trees: usize,
        max_depth: usize,
        window_size: usize,
        size_limit: i32,
    ) -> Self {
        let n_internal = (1usize << max_depth) - 1;
        let n_nodes = (1usize << (max_depth + 1)) - 1;
        let depth_scale = (0..=max_depth).map(|d| (1usize << d) as f64).collect();
        Self {
            n_trees,
            max_depth,
            window_size,
            size_limit,
            n_internal,
            n_nodes,
            split_feat: vec![0; n_trees * n_internal],
            split_val: vec![0.0; n_trees * n_internal],
            mass_r: vec![0; n_trees * n_nodes],
            mass_l: vec![0; n_trees * n_nodes],
            stream_count: 0,
            depth_scale,
        }
    }

    pub fn fit(&mut self, x_tr: &[f64], n_rows: usize, dim: usize, seed: u64) {
        let mut lo = vec![f64::INFINITY; dim];
        let mut hi = vec![f64::NEG_INFINITY; dim];
        for r in 0..n_rows {
            let row = &x_tr[r * dim..(r + 1) * dim];
            for j in 0..dim {
                if row[j] < lo[j] { lo[j] = row[j]; }
                if row[j] > hi[j] { hi[j] = row[j]; }
            }
        }
        for j in 0..dim {
            if hi[j] - lo[j] < 1e-6 {
                hi[j] = lo[j] + 1e-6;
            }
        }

        let mut rng = SplitMix64::new(seed);
        let mut node_lo = vec![0.0_f64; self.n_internal * dim];
        let mut node_hi = vec![0.0_f64; self.n_internal * dim];

        for t in 0..self.n_trees {
            for j in 0..dim {
                let s = rng.uniform(lo[j], hi[j]);
                let half_w = 2.0 * (s - lo[j]).max(hi[j] - s);
                node_lo[j] = s - half_w;
                node_hi[j] = s + half_w;
            }
            for node in 0..self.n_internal {
                let feat = rng.gen_range_usize(0, dim);
                let nlo = node_lo[node * dim + feat];
                let nhi = node_hi[node * dim + feat];
                let mid = 0.5 * (nlo + nhi);
                self.split_feat[t * self.n_internal + node] = feat;
                self.split_val[t * self.n_internal + node] = mid;

                let left = 2 * node + 1;
                let right = left + 1;
                if left < self.n_internal {
                    for j in 0..dim {
                        node_lo[left * dim + j] = node_lo[node * dim + j];
                        node_hi[left * dim + j] = node_hi[node * dim + j];
                        node_lo[right * dim + j] = node_lo[node * dim + j];
                        node_hi[right * dim + j] = node_hi[node * dim + j];
                    }
                    node_hi[left * dim + feat] = mid;
                    node_lo[right * dim + feat] = mid;
                }
            }
        }

        self.mass_r.fill(0);
        self.mass_l.fill(0);
        self.stream_count = 0;

        for r in 0..n_rows {
            let row = &x_tr[r * dim..(r + 1) * dim];
            self.update_latest(row);
        }
        if self.mass_r.iter().all(|&m| m == 0) {
            self.mass_r.copy_from_slice(&self.mass_l);
            self.mass_l.fill(0);
            self.stream_count = 0;
        }
    }

    #[inline]
    fn update_latest(&mut self, row: &[f64]) {
        for t in 0..self.n_trees {
            let f_base = t * self.n_internal;
            let m_base = t * self.n_nodes;
            let mut node = 0usize;
            self.mass_l[m_base] += 1;
            for _ in 0..self.max_depth {
                let feat = self.split_feat[f_base + node];
                let val = self.split_val[f_base + node];
                node = 2 * node + 1 + usize::from(row[feat] >= val);
                self.mass_l[m_base + node] += 1;
            }
        }
        self.stream_count += 1;
        if self.stream_count >= self.window_size {
            self.mass_r.copy_from_slice(&self.mass_l);
            self.mass_l.fill(0);
            self.stream_count = 0;
        }
    }

    pub fn score_stream(&mut self, x_stream: &[f64], n_rows: usize, dim: usize, adapt: bool) -> Vec<f64> {
        let mut scores = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let row = &x_stream[r * dim..(r + 1) * dim];
            let mut mass_score = 0.0_f64;
            for t in 0..self.n_trees {
                let f_base = t * self.n_internal;
                let m_base = t * self.n_nodes;
                let mut node = 0usize;
                let mut chosen = self.mass_r[m_base];
                let mut chosen_d = 0usize;
                if chosen >= self.size_limit {
                    for d in 1..=self.max_depth {
                        let feat = self.split_feat[f_base + node];
                        let val = self.split_val[f_base + node];
                        node = 2 * node + 1 + usize::from(row[feat] >= val);
                        let m = self.mass_r[m_base + node];
                        chosen = m;
                        chosen_d = d;
                        if m < self.size_limit {
                            break;
                        }
                    }
                }
                mass_score += (chosen as f64) * self.depth_scale[chosen_d];
            }
            scores.push(-mass_score);
            if adapt {
                self.update_latest(row);
            }
        }
        scores
    }
}

/// Streaming RS-Hash / RS-Stream (Sathe & Aggarwal, ICDM 2016) in pure Rust.
pub struct RsHash {
    n_estimators: usize,
    window_size: usize,
    cms_width: usize,
    cms_depth: usize,
    lo: Vec<f64>,
    inv_scale: Vec<f64>,
    subspaces: Vec<Vec<usize>>,
    f_vals: Vec<f64>,
    shifts: Vec<Vec<f64>>,
    hash_coefs: Vec<Vec<i64>>,
    hash_bias: Vec<i64>,
    cms: Vec<i32>,
    ring: Vec<u16>,
    ring_pos: usize,
    ring_len: usize,
}

impl RsHash {
    pub fn new(n_estimators: usize, window_size: usize, cms_width: usize, cms_depth: usize) -> Self {
        Self {
            n_estimators,
            window_size,
            cms_width,
            cms_depth,
            lo: Vec::new(),
            inv_scale: Vec::new(),
            subspaces: Vec::new(),
            f_vals: Vec::new(),
            shifts: Vec::new(),
            hash_coefs: Vec::new(),
            hash_bias: Vec::new(),
            cms: vec![0; n_estimators * cms_depth * cms_width],
            ring: vec![0; window_size * n_estimators * cms_depth],
            ring_pos: 0,
            ring_len: 0,
        }
    }

    pub fn fit(&mut self, x_tr: &[f64], n_rows: usize, dim: usize, seed: u64) {
        self.lo = vec![f64::INFINITY; dim];
        let mut hi = vec![f64::NEG_INFINITY; dim];
        for r in 0..n_rows {
            let row = &x_tr[r * dim..(r + 1) * dim];
            for j in 0..dim {
                if row[j] < self.lo[j] { self.lo[j] = row[j]; }
                if row[j] > hi[j] { hi[j] = row[j]; }
            }
        }
        self.inv_scale = (0..dim)
            .map(|j| 1.0 / (hi[j] - self.lo[j]).max(1e-6))
            .collect();

        let mut rng = SplitMix64::new(seed);
        let s = self.window_size as f64;
        let f_lo = 1.0 / s.sqrt();
        let f_hi = 1.0 - f_lo;

        self.subspaces.clear();
        self.f_vals.clear();
        self.shifts.clear();
        self.hash_coefs.clear();
        self.hash_bias = vec![0; self.n_estimators * self.cms_depth];

        let mut perm: Vec<usize> = (0..dim).collect();
        for e in 0..self.n_estimators {
            let f = rng.uniform(f_lo, f_hi);
            self.f_vals.push(f);
            let log_s = s.ln() / (1.0 / f).ln();
            let r_cap = dim.min(8);
            let r_lo = ((1.0 + 0.5 * log_s).floor().max(1.0) as usize).min(r_cap);
            let r_hi = (log_s.floor().max(r_lo as f64) as usize).min(r_cap);
            let r = rng.gen_range_usize(r_lo, r_hi + 1);

            // Partial Fisher-Yates for r distinct feature indices
            for i in 0..r {
                let j = rng.gen_range_usize(i, dim);
                perm.swap(i, j);
            }
            self.subspaces.push(perm[..r].to_vec());

            let shifts: Vec<f64> = (0..r).map(|_| rng.uniform(0.0, f)).collect();
            self.shifts.push(shifts);

            let mut coefs = Vec::with_capacity(self.cms_depth * r);
            for d_i in 0..self.cms_depth {
                self.hash_bias[e * self.cms_depth + d_i] =
                    rng.gen_range_usize(0, self.cms_width) as i64;
                for _ in 0..r {
                    coefs.push(rng.gen_range_usize(1, 1_000_003) as i64);
                }
            }
            self.hash_coefs.push(coefs);
        }

        self.cms.fill(0);
        self.ring.fill(0);
        self.ring_pos = 0;
        self.ring_len = 0;

        let mut scratch = vec![0u16; self.n_estimators * self.cms_depth];
        for r in 0..n_rows {
            let row = &x_tr[r * dim..(r + 1) * dim];
            self.hash_row(row, &mut scratch);
            self.push_hashes(&scratch);
        }
    }

    #[inline]
    fn hash_row(&self, row: &[f64], out_buckets: &mut [u16]) {
        let w_mod = self.cms_width as i64;
        for e in 0..self.n_estimators {
            let cols = &self.subspaces[e];
            let shifts = &self.shifts[e];
            let inv_f = 1.0 / self.f_vals[e];
            let coefs = &self.hash_coefs[e];
            let r = cols.len();

            // Stack-allocated coordinates (r <= 8)
            let mut coords = [0i64; 8];
            for j in 0..r {
                let col = cols[j];
                let xn = (row[col] - self.lo[col]) * self.inv_scale[col];
                let c = ((xn + shifts[j]) * inv_f).floor().clamp(-1_000_000.0, 1_000_000.0);
                coords[j] = c as i64;
            }
            let base = e * self.cms_depth;
            for d_i in 0..self.cms_depth {
                let mut acc = self.hash_bias[base + d_i];
                let c_off = d_i * r;
                for j in 0..r {
                    acc = acc.wrapping_add(coords[j].wrapping_mul(coefs[c_off + j]));
                }
                let bucket = acc.rem_euclid(w_mod) as u16;
                out_buckets[base + d_i] = bucket;
            }
        }
    }

    #[inline]
    fn push_hashes(&mut self, buckets: &[u16]) {
        let stride = self.n_estimators * self.cms_depth;
        let ring_off = self.ring_pos * stride;
        if self.ring_len == self.window_size {
            for idx in 0..stride {
                let old_b = self.ring[ring_off + idx] as usize;
                self.cms[idx * self.cms_width + old_b] -= 1;
            }
        } else {
            self.ring_len += 1;
        }
        for idx in 0..stride {
            let b = buckets[idx];
            self.ring[ring_off + idx] = b;
            self.cms[idx * self.cms_width + (b as usize)] += 1;
        }
        self.ring_pos = (self.ring_pos + 1) % self.window_size;
    }

    pub fn score_stream(&mut self, x_stream: &[f64], n_rows: usize, dim: usize, adapt: bool) -> Vec<f64> {
        let mut scores = Vec::with_capacity(n_rows);
        let mut buckets = vec![0u16; self.n_estimators * self.cms_depth];
        let inv_est = 1.0 / (self.n_estimators as f64);

        for r in 0..n_rows {
            let row = &x_stream[r * dim..(r + 1) * dim];
            self.hash_row(row, &mut buckets);

            let mut sum_log = 0.0_f64;
            for e in 0..self.n_estimators {
                let base = e * self.cms_depth;
                let mut min_c = i32::MAX;
                for d_i in 0..self.cms_depth {
                    let idx = base + d_i;
                    let b = buckets[idx] as usize;
                    let c = self.cms[idx * self.cms_width + b];
                    if c < min_c {
                        min_c = c;
                    }
                }
                let mc = min_c.max(0) as f64;
                sum_log += (mc + 1.0).log2();
            }
            scores.push(-(sum_log * inv_est));
            if adapt {
                self.push_hashes(&buckets);
            }
        }
        scores
    }
}

/// Deterministic Euclidean kNN (`k = 5`) anomaly detector in pure Rust.
pub struct KnnDetector {
    x_tr: Vec<f64>,
    n_tr: usize,
    dim: usize,
    k: usize,
}

impl KnnDetector {
    pub fn fit(x_tr: &[f64], n_tr: usize, dim: usize, k: usize) -> Self {
        Self {
            x_tr: x_tr.to_vec(),
            n_tr,
            dim,
            k: k.min(n_tr).max(1),
        }
    }

    pub fn score(&self, x_query: &[f64], n_q: usize) -> Vec<f64> {
        let mut scores = Vec::with_capacity(n_q);
        let k = self.k;
        let dim = self.dim;
        let mut top_k = vec![f64::INFINITY; k];

        for q in 0..n_q {
            let qrow = &x_query[q * dim..(q + 1) * dim];
            top_k.fill(f64::INFINITY);
            let mut worst = f64::INFINITY;
            let mut worst_idx = 0usize;

            for t in 0..self.n_tr {
                let trow = &self.x_tr[t * dim..(t + 1) * dim];
                let mut d2 = 0.0_f64;
                for j in 0..dim {
                    let diff = qrow[j] - trow[j];
                    d2 += diff * diff;
                }
                if d2 < worst {
                    top_k[worst_idx] = d2;
                    worst = top_k[0];
                    worst_idx = 0;
                    for (idx, &val) in top_k.iter().enumerate().skip(1) {
                        if val > worst {
                            worst = val;
                            worst_idx = idx;
                        }
                    }
                }
            }
            let mean_dist = top_k.iter().map(|&d2| d2.sqrt()).sum::<f64>() / (k as f64);
            scores.push(mean_dist);
        }
        scores
    }
}

#[derive(Debug, Clone)]
pub struct RustBaselineServerResult {
    pub hs_trees: AggregatedMetrics,
    pub rs_hash: AggregatedMetrics,
    pub pca_mahalanobis_32d: AggregatedMetrics,
    pub pca_mahalanobis_fulld: AggregatedMetrics,
    pub raw_knn: AggregatedMetrics,
}

pub fn eval_rust_baselines_on_server(ds: &ServerDataset) -> RustBaselineServerResult {
    let win = to_windows_v6(&ds.train, &ds.test, &ds.test_label, 8, 4, 2000, true, None);
    let dim = 32usize;
    let kf = ((win.n_tr as f64) * 0.75) as usize;
    let n_cal = win.n_tr - kf;
    let x_tr = &win.w_tr[..kf * dim];
    let x_cal = &win.w_tr[kf * dim..];
    let x_te = &win.w_te;

    // 1. Half-Space Trees (5 seeds)
    let mut hst_runs = Vec::with_capacity(DEFAULT_SEEDS.len());
    for &seed in &DEFAULT_SEEDS {
        let mut hst = HalfSpaceTrees::new(25, 8, 250, 25);
        hst.fit(x_tr, kf, dim, seed);
        let sc_cal = hst.score_stream(x_cal, n_cal, dim, true);
        let sc_te = hst.score_stream(x_te, win.n_te, dim, true);
        hst_runs.push(evaluate_score_stream(&sc_cal, &sc_te, &win.y_te, 0.05, 2.5));
    }

    // 2. RS-Hash / RS-Stream (5 seeds)
    let mut rsh_runs = Vec::with_capacity(DEFAULT_SEEDS.len());
    for &seed in &DEFAULT_SEEDS {
        let mut rsh = RsHash::new(100, 1000, 1024, 4);
        rsh.fit(x_tr, kf, dim, seed);
        let sc_cal = rsh.score_stream(x_cal, n_cal, dim, true);
        let sc_te = rsh.score_stream(x_te, win.n_te, dim, true);
        rsh_runs.push(evaluate_score_stream(&sc_cal, &sc_te, &win.y_te, 0.05, 2.5));
    }

    // 3. PCA + Mahalanobis (Matched 32-D)
    let pca32 = PcaMahalanobisDetector::fit(x_tr, kf, dim, 0.95, 32, 1e-3);
    let m32 = evaluate_score_stream(
        &pca32.score(x_cal, n_cal),
        &pca32.score(x_te, win.n_te),
        &win.y_te,
        0.05,
        2.5,
    );

    // 4. PCA + Mahalanobis (Full-D Raw w*D)
    let (xtr_f, ntr_f, xte_f, nte_f, yte_f, fdim) =
        to_raw_full_d_windows(&ds.train, &ds.test, &ds.test_label, 8, 2000);
    assert_eq!(win.y_te, yte_f);
    let kf_f = ((ntr_f as f64) * 0.75) as usize;
    let ncal_f = ntr_f - kf_f;
    let pcaf = PcaMahalanobisDetector::fit(&xtr_f[..kf_f * fdim], kf_f, fdim, 0.95, 32, 1e-3);
    let mf = evaluate_score_stream(
        &pcaf.score(&xtr_f[kf_f * fdim..], ncal_f),
        &pcaf.score(&xte_f, nte_f),
        &yte_f,
        0.05,
        2.5,
    );

    // 5. kNN (k=5, Euclidean, 32-D)
    let knn = KnnDetector::fit(x_tr, kf, dim, 5);
    let m_knn = evaluate_score_stream(
        &knn.score(x_cal, n_cal),
        &knn.score(x_te, win.n_te),
        &win.y_te,
        0.05,
        2.5,
    );

    RustBaselineServerResult {
        hs_trees: aggregate_runs(&hst_runs),
        rs_hash: aggregate_runs(&rsh_runs),
        pca_mahalanobis_32d: aggregate_runs(&[m32]),
        pca_mahalanobis_fulld: aggregate_runs(&[mf]),
        raw_knn: aggregate_runs(&[m_knn]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_gaussian_matrix(rows: usize, dim: usize, seed: u64) -> Vec<f64> {
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::with_capacity(rows * dim);
        while out.len() < rows * dim {
            let u1 = rng.next_f64().max(1e-12);
            let u2 = rng.next_f64();
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
            let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();
            out.push(z0);
            if out.len() < rows * dim {
                out.push(z1);
            }
        }
        out
    }

    #[test]
    fn streaming_pca_and_knn_baselines_separate_anomalies_deterministically() {
        let dim = 16usize;
        let x_tr = synthetic_gaussian_matrix(400, dim, 10);
        let x_cal = synthetic_gaussian_matrix(120, dim, 20);
        let mut x_te = synthetic_gaussian_matrix(120, dim, 30);
        let mut y_te = vec![0i32; 120];
        for r in 90..120 {
            y_te[r] = 1;
            for c in 0..dim {
                x_te[r * dim + c] += 6.0;
            }
        }

        // Half-Space Trees
        let mut h1 = HalfSpaceTrees::new(15, 6, 100, 10);
        let mut h2 = HalfSpaceTrees::new(15, 6, 100, 10);
        h1.fit(&x_tr, 400, dim, 0);
        h2.fit(&x_tr, 400, dim, 0);
        let c1 = h1.score_stream(&x_cal, 120, dim, true);
        let t1 = h1.score_stream(&x_te, 120, dim, true);
        let _c2 = h2.score_stream(&x_cal, 120, dim, true);
        let t2 = h2.score_stream(&x_te, 120, dim, true);
        assert_eq!(t1, t2);
        let mh = evaluate_score_stream(&c1, &t1, &y_te, 0.05, 2.5);
        assert!(mh.auroc > 85.0, "HST AUROC too low: {}", mh.auroc);
        assert!(mh.far_dual <= mh.far_raw + 1e-9);

        // RS-Hash
        let mut r1 = RsHash::new(40, 300, 512, 4);
        let mut r2 = RsHash::new(40, 300, 512, 4);
        r1.fit(&x_tr, 400, dim, 0);
        r2.fit(&x_tr, 400, dim, 0);
        let rc1 = r1.score_stream(&x_cal, 120, dim, true);
        let rt1 = r1.score_stream(&x_te, 120, dim, true);
        let _rc2 = r2.score_stream(&x_cal, 120, dim, true);
        let rt2 = r2.score_stream(&x_te, 120, dim, true);
        assert_eq!(rt1, rt2);
        let mr = evaluate_score_stream(&rc1, &rt1, &y_te, 0.05, 2.5);
        assert!(mr.auroc > 85.0, "RSH AUROC too low: {}", mr.auroc);
        assert!(mr.far_dual <= mr.far_raw + 1e-9);

        // PCA + Mahalanobis
        let pm = PcaMahalanobisDetector::fit(&x_tr, 400, dim, 0.95, 16, 1e-3);
        let mp = evaluate_score_stream(&pm.score(&x_cal, 120), &pm.score(&x_te, 120), &y_te, 0.05, 2.5);
        assert!(mp.auroc > 95.0, "PCA+Mahal AUROC too low: {}", mp.auroc);
        assert!(mp.far_dual <= mp.far_raw + 1e-9);

        // kNN
        let knn = KnnDetector::fit(&x_tr, 400, dim, 5);
        let mk = evaluate_score_stream(&knn.score(&x_cal, 120), &knn.score(&x_te, 120), &y_te, 0.05, 2.5);
        assert!(mk.auroc > 95.0, "kNN AUROC too low: {}", mk.auroc);
        assert!(mk.far_dual <= mk.far_raw + 1e-9);
    }
}
