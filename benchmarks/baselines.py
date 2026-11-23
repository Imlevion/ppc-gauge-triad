"""Fair, offline-reproducible streaming and window anomaly detection baselines.

All baselines in this module consume the exact same non-overlapping chronological
windows (`w=8`, `to_windows_v6` `3PC+Resid` 32-D flattened windows, plus an
optional full-channel raw-window variant for PCA+Mahalanobis), the exact same
chronological 75%/25% train/calibration split, the exact same split-conformal
`alpha=0.05` calibration threshold, and the exact same window-level ground-truth
labels (`yte`, strictly WITHOUT point-adjust) as PPC-Gauge-Triad.

Implemented baselines (pure NumPy + scikit-learn, 0 extra dependencies):
- Half-Space Trees (Tan, Ting & Liu, IJCAI 2011): streaming two-profile mass trees.
- RS-Hash / RS-Stream (Sathe & Aggarwal, ICDM 2016): streaming randomized subspace
  count-min hashing with FIFO sliding window eviction.
- PCA + Mahalanobis: ridge-regularized Mahalanobis distance in PCA subspace
  (both matched 32-D window input and full-D raw window input).
- Classical batch baselines re-evaluated live on identical windows:
  Raw Isolation Forest, Raw kNN (k=5), and Raw OC-SVM (RBF).

Note on Deep Baselines (USAD / TranAD):
PyTorch is not installed in this lightweight offline verification environment, so
deep neural baselines (USAD / TranAD) are documented as PENDING / Future Work
rather than approximated or fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from benchmarks.ppc_triad_wrapper import (
    compute_auroc,
    conformal_threshold,
    to_windows_v6,
)

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class BaselineEvalMetrics:
    """Single-run window-level metrics (strictly WITHOUT point-adjust)."""

    auroc: float
    far_raw: float
    recall_raw: float
    far_dual: float
    recall_dual: float


def _shift_invariant_dual_gate(
    scores_cal: np.ndarray,
    scores_te: np.ndarray,
    thr: float,
    sev_mult: float = 2.5,
) -> np.ndarray:
    """Shift-invariant dual-stage alarm gate (handles negative or offset score scales fairly).

    Trips immediately if `s > med_cal + sev_mult * max(thr - med_cal, 0)` OR on
    2 consecutive windows with `s > thr`. When `med_cal == 0` (e.g. log1p-MAD
    triad scores), this reduces identically to `s > thr * sev_mult`.
    """
    med_cal = float(np.median(scores_cal))
    margin = max(thr - med_cal, 0.0)
    severe_thr = med_cal + sev_mult * margin
    al = np.zeros(len(scores_te), dtype=bool)
    streak = 0
    for i, s in enumerate(scores_te):
        streak = streak + 1 if s > thr else 0
        if (s > severe_thr) or (streak >= 2):
            al[i] = True
    return al


def evaluate_score_stream(
    scores_cal: np.ndarray,
    scores_te: np.ndarray,
    y_te: np.ndarray,
    alpha: float = 0.05,
    sev_mult: float = 2.5,
) -> BaselineEvalMetrics:
    """Compute AUROC plus both raw 1-window and dual-stage gated FAR/Recall."""
    thr = conformal_threshold(scores_cal, alpha=alpha)
    al_raw = scores_te > thr
    al_dual = _shift_invariant_dual_gate(scores_cal, scores_te, thr, sev_mult=sev_mult)
    auroc = compute_auroc(y_te, scores_te) * 100.0
    far_raw = float(np.mean(al_raw[y_te == 0])) * 100.0
    rec_raw = float(np.mean(al_raw[y_te == 1])) * 100.0
    far_dual = float(np.mean(al_dual[y_te == 0])) * 100.0
    rec_dual = float(np.mean(al_dual[y_te == 1])) * 100.0
    return BaselineEvalMetrics(
        auroc=auroc,
        far_raw=far_raw,
        recall_raw=rec_raw,
        far_dual=far_dual,
        recall_dual=rec_dual,
    )


class HalfSpaceTreesDetector:
    """Streaming Half-Space Trees (Tan, Ting & Liu, IJCAI 2011).

    Uses a full binary tree heap layout (`n_nodes = 2^(h+1) - 1`) per tree,
    two mass profiles (`r` reference, `l` latest) rotated every `window_size`
    windows, and exact `r[node] * 2^depth` anomaly mass scoring (negated so
    higher scores indicate higher anomaly likelihood).
    """

    def __init__(
        self,
        n_trees: int = 25,
        max_depth: int = 8,
        window_size: int = 250,
        size_limit: int = 25,
        seed: int = 0,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.window_size = window_size
        self.size_limit = size_limit
        self.seed = seed

        self.n_internal = (1 << max_depth) - 1
        self.n_nodes = (1 << (max_depth + 1)) - 1
        self.split_feat = np.zeros((n_trees, self.n_internal), dtype=np.int32)
        self.split_val = np.zeros((n_trees, self.n_internal), dtype=np.float64)
        self.mass_r = np.zeros((n_trees, self.n_nodes), dtype=np.int32)
        self.mass_l = np.zeros((n_trees, self.n_nodes), dtype=np.int32)
        self._stream_count = 0
        self._tree_idx = np.arange(n_trees, dtype=np.int32)
        self._depth_scale = np.array([float(1 << d) for d in range(max_depth + 1)], dtype=np.float64)

    def _build_trees(self, lo: np.ndarray, hi: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        dim = len(lo)
        span = np.maximum(hi - lo, 1e-6)
        hi_safe = lo + span

        for t in range(self.n_trees):
            # Algorithm 2 (Tan et al. 2011): randomized bounding workspace per tree
            s = rng.uniform(lo, hi_safe)
            half_w = 2.0 * np.maximum(s - lo, hi_safe - s)
            ws_lo = s - half_w
            ws_hi = s + half_w

            # Iterative BFS build of node intervals
            node_lo = np.empty((self.n_internal, dim), dtype=np.float64)
            node_hi = np.empty((self.n_internal, dim), dtype=np.float64)
            node_lo[0] = ws_lo
            node_hi[0] = ws_hi

            for node in range(self.n_internal):
                feat = int(rng.integers(0, dim))
                mid = 0.5 * (node_lo[node, feat] + node_hi[node, feat])
                self.split_feat[t, node] = feat
                self.split_val[t, node] = mid
                left = 2 * node + 1
                right = left + 1
                if left < self.n_internal:
                    node_lo[left] = node_lo[node]
                    node_hi[left] = node_hi[node]
                    node_hi[left, feat] = mid
                    node_lo[right] = node_lo[node]
                    node_hi[right] = node_hi[node]
                    node_lo[right, feat] = mid

    def _traverse_paths(self, x_batch: np.ndarray) -> np.ndarray:
        """Compute root-to-leaf node indices for every sample and tree: (N, T, h+1)."""
        n = len(x_batch)
        paths = np.empty((n, self.n_trees, self.max_depth + 1), dtype=np.int32)
        nodes = np.zeros((n, self.n_trees), dtype=np.int32)
        paths[:, :, 0] = 0
        t_idx = self._tree_idx[None, :]
        row_idx = np.arange(n, dtype=np.int32)[:, None]

        for depth in range(self.max_depth):
            feats = self.split_feat[t_idx, nodes]  # (N, T)
            vals = self.split_val[t_idx, nodes]  # (N, T)
            x_vals = x_batch[row_idx, feats]  # (N, T)
            go_right = (x_vals >= vals).astype(np.int32)
            nodes = 2 * nodes + 1 + go_right
            paths[:, :, depth + 1] = nodes
        return paths

    def fit(self, x_tr: np.ndarray) -> "HalfSpaceTreesDetector":
        x = np.asarray(x_tr, dtype=np.float64)
        lo = np.min(x, axis=0)
        hi = np.max(x, axis=0)
        self._build_trees(lo, hi)
        self.mass_r.fill(0)
        self.mass_l.fill(0)
        self._stream_count = 0

        paths = self._traverse_paths(x)
        t_idx = self._tree_idx[:, None]
        for i in range(len(x)):
            self.mass_l[t_idx, paths[i]] += 1
            self._stream_count += 1
            if self._stream_count >= self.window_size:
                self.mass_r[:, :] = self.mass_l
                self.mass_l.fill(0)
                self._stream_count = 0
        if int(self.mass_r.sum()) == 0:
            self.mass_r[:, :] = self.mass_l
            self.mass_l.fill(0)
            self._stream_count = 0
        return self

    def score_stream(self, x_stream: np.ndarray, adapt: bool = True) -> np.ndarray:
        x = np.asarray(x_stream, dtype=np.float64)
        n = len(x)
        paths = self._traverse_paths(x)  # (N, T, h+1)
        scores = np.empty(n, dtype=np.float64)
        t_idx = self._tree_idx[:, None]

        for i in range(n):
            p = paths[i]  # (T, h+1)
            r_vals = self.mass_r[t_idx, p]  # (T, h+1)
            # First depth where reference mass drops below size_limit, else leaf
            below = r_vals < self.size_limit
            any_below = np.any(below, axis=1)
            first_d = np.where(any_below, np.argmax(below, axis=1), self.max_depth)
            chosen_mass = r_vals[self._tree_idx, first_d]
            mass_score = float(np.sum(chosen_mass * self._depth_scale[first_d]))
            scores[i] = -mass_score

            if adapt:
                self.mass_l[t_idx, p] += 1
                self._stream_count += 1
                if self._stream_count >= self.window_size:
                    self.mass_r[:, :] = self.mass_l
                    self.mass_l.fill(0)
                    self._stream_count = 0
        return scores


class RSHashDetector:
    """Streaming RS-Hash / RS-Stream (Sathe & Aggarwal, ICDM 2016).

    Maintains `n_estimators` randomized subspace grids with independent
    Count-Min Sketches (`cms_depth` x `cms_width`) over an exact FIFO sliding
    window of size `window_size`. Scores are `-mean_e(log2(count_min + 1))` so
    higher scores indicate more anomalous windows.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        window_size: int = 1000,
        cms_width: int = 1024,
        cms_depth: int = 4,
        seed: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.cms_width = cms_width
        self.cms_depth = cms_depth
        self.seed = seed

        self._lo: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._subspaces: List[np.ndarray] = []
        self._f_vals: np.ndarray = np.empty(0, dtype=np.float64)
        self._shifts: List[np.ndarray] = []
        self._hash_coefs: List[np.ndarray] = []
        self._hash_bias: np.ndarray = np.empty(0, dtype=np.int64)
        self._cms = np.zeros((n_estimators, cms_depth, cms_width), dtype=np.int32)
        self._ring = np.zeros((window_size, n_estimators, cms_depth), dtype=np.int32)
        self._ring_pos = 0
        self._ring_len = 0
        self._est_idx = np.arange(n_estimators, dtype=np.int32)[:, None]
        self._dep_idx = np.arange(cms_depth, dtype=np.int32)[None, :]

    def _init_ensembles(self, dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        s = float(self.window_size)
        f_lo = 1.0 / np.sqrt(s)
        f_hi = 1.0 - f_lo
        self._f_vals = rng.uniform(f_lo, f_hi, size=self.n_estimators)
        self._subspaces = []
        self._shifts = []
        self._hash_coefs = []
        self._hash_bias = rng.integers(0, self.cms_width, size=(self.n_estimators, self.cms_depth), dtype=np.int64)

        for e in range(self.n_estimators):
            f = float(self._f_vals[e])
            log_s = np.log(s) / np.log(1.0 / f)
            r_lo = max(1, int(np.floor(1.0 + 0.5 * log_s)))
            r_hi = max(r_lo, int(np.floor(log_s)))
            r_cap = min(dim, 8)
            r_lo = min(r_lo, r_cap)
            r_hi = min(r_hi, r_cap)
            r = int(rng.integers(r_lo, r_hi + 1))
            cols = rng.choice(dim, size=r, replace=False).astype(np.int32)
            alpha = rng.uniform(0.0, f, size=r)
            # Odd large multipliers modulo cms_width for universal integer hashing
            coefs = rng.integers(1, 1_000_003, size=(self.cms_depth, r), dtype=np.int64)
            self._subspaces.append(cols)
            self._shifts.append(alpha)
            self._hash_coefs.append(coefs)

    def _compute_hashes(self, x_batch: np.ndarray) -> np.ndarray:
        """Vectorized hash table bucket indices for a batch: (N, n_estimators, cms_depth)."""
        assert self._lo is not None and self._scale is not None
        x_norm = (np.asarray(x_batch, dtype=np.float64) - self._lo) / self._scale
        n = len(x_norm)
        buckets = np.empty((n, self.n_estimators, self.cms_depth), dtype=np.int32)
        for e in range(self.n_estimators):
            cols = self._subspaces[e]
            # Clip bucket coordinates to safe int64 range while keeping out-of-range distinct
            coords = np.floor((x_norm[:, cols] + self._shifts[e]) / self._f_vals[e])
            coords = np.clip(coords, -1_000_000.0, 1_000_000.0).astype(np.int64)
            h = (coords @ self._hash_coefs[e].T + self._hash_bias[e]) % self.cms_width
            buckets[:, e, :] = h.astype(np.int32)
        return buckets

    def _push_one(self, b_row: np.ndarray) -> None:
        if self._ring_len == self.window_size:
            old = self._ring[self._ring_pos]
            self._cms[self._est_idx, self._dep_idx, old] -= 1
        else:
            self._ring_len += 1
        self._ring[self._ring_pos] = b_row
        self._cms[self._est_idx, self._dep_idx, b_row] += 1
        self._ring_pos = (self._ring_pos + 1) % self.window_size

    def fit(self, x_tr: np.ndarray) -> "RSHashDetector":
        x = np.asarray(x_tr, dtype=np.float64)
        self._lo = np.min(x, axis=0)
        self._scale = np.maximum(np.max(x, axis=0) - self._lo, 1e-6)
        self._init_ensembles(x.shape[1])
        self._cms.fill(0)
        self._ring.fill(0)
        self._ring_pos = 0
        self._ring_len = 0

        buckets = self._compute_hashes(x)
        for i in range(len(x)):
            self._push_one(buckets[i])
        return self

    def score_stream(self, x_stream: np.ndarray, adapt: bool = True) -> np.ndarray:
        buckets = self._compute_hashes(x_stream)
        n = len(buckets)
        scores = np.empty(n, dtype=np.float64)
        for i in range(n):
            b = buckets[i]
            counts = self._cms[self._est_idx, self._dep_idx, b]  # (E, D_cms)
            min_counts = np.maximum(0, np.min(counts, axis=1))
            scores[i] = -float(np.mean(np.log2(min_counts.astype(np.float64) + 1.0)))
            if adapt:
                self._push_one(b)
        return scores


class PCAMahalanobisDetector:
    """PCA + Regularized Mahalanobis distance detector (deterministic).

    Fits PCA retaining `variance_ratio` (capped at `max_components`) on `X_tr`
    and scores windows via ridge-regularized Mahalanobis squared distance in the
    PC subspace: `sum_k z_k^2 / (lambda_k + reg)`.
    """

    def __init__(
        self,
        variance_ratio: float = 0.95,
        max_components: int = 32,
        reg: float = 1e-3,
    ) -> None:
        self.variance_ratio = variance_ratio
        self.max_components = max_components
        self.reg = reg
        self._pca: PCA | None = None
        self._inv_var: np.ndarray | None = None

    def fit(self, x_tr: np.ndarray) -> "PCAMahalanobisDetector":
        x = np.asarray(x_tr, dtype=np.float64)
        max_k = min(self.max_components, x.shape[0] - 1, x.shape[1])
        pca_full = PCA(n_components=max_k, svd_solver="full").fit(x)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cum, self.variance_ratio) + 1)
        k = max(1, min(k, max_k))
        self._pca = PCA(n_components=k, svd_solver="full").fit(x)
        self._inv_var = 1.0 / (self._pca.explained_variance_ + self.reg)
        return self

    def score(self, x_batch: np.ndarray) -> np.ndarray:
        assert self._pca is not None and self._inv_var is not None
        z = self._pca.transform(np.asarray(x_batch, dtype=np.float64))
        return np.sum((z * z) * self._inv_var[None, :], axis=1)


def _to_raw_full_d_windows(
    tr_raw: np.ndarray,
    te_raw: np.ndarray,
    yl_raw: np.ndarray,
    w: int = 8,
    max_tr_win: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize full raw D channels on tr_raw and segment into non-overlapping w*D windows."""
    mu = tr_raw.mean(axis=0)
    sd = np.maximum(tr_raw.std(axis=0), 1e-6)
    ztr = (tr_raw - mu) / sd
    zte = (te_raw - mu) / sd
    ntr = min(max_tr_win, len(ztr) // w)
    nte = len(zte) // w
    xtr = ztr[: ntr * w].reshape(ntr, w * tr_raw.shape[1])
    xte = zte[: nte * w].reshape(nte, w * te_raw.shape[1])
    yte = (yl_raw[: nte * w].reshape(nte, w).max(axis=1) > 0).astype(np.int32)
    return xtr, xte, yte


def eval_all_baselines_on_server(
    tr_r: np.ndarray,
    te_r: np.ndarray,
    yl_r: np.ndarray,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all streaming and batch baselines on one server chronologically."""
    # Exact same windows & labels as PPC-Gauge-Triad (3PC+Resid, w=8, d=4 -> 32-D)
    wtr, wte, yte, _ = to_windows_v6(tr_r, te_r, yl_r, use_pca_resid=True)
    kf = int(len(wtr) * 0.75)
    x_tr = wtr[:kf].reshape(kf, -1)
    x_cal = wtr[kf:].reshape(len(wtr) - kf, -1)
    x_te = wte.reshape(len(wte), -1)

    out: Dict[str, Dict[str, float]] = {}

    def _agg_seeds(runs: List[BaselineEvalMetrics]) -> Dict[str, float]:
        return {
            "auroc": float(np.mean([r.auroc for r in runs])),
            "auroc_seed_std": float(np.std([r.auroc for r in runs], ddof=0)),
            "far_raw": float(np.mean([r.far_raw for r in runs])),
            "recall_raw": float(np.mean([r.recall_raw for r in runs])),
            "far_dual": float(np.mean([r.far_dual for r in runs])),
            "recall_dual": float(np.mean([r.recall_dual for r in runs])),
        }

    # 1. Half-Space Trees (Streaming, 5 seeds)
    hst_runs = []
    for seed in seeds:
        det = HalfSpaceTreesDetector(
            n_trees=25, max_depth=8, window_size=250, size_limit=25, seed=int(seed)
        ).fit(x_tr)
        sc_cal = det.score_stream(x_cal, adapt=True)
        sc_te = det.score_stream(x_te, adapt=True)
        hst_runs.append(evaluate_score_stream(sc_cal, sc_te, yte))
    out["hs_trees"] = _agg_seeds(hst_runs)

    # 2. RS-Hash / RS-Stream (Streaming, 5 seeds)
    rsh_runs = []
    for seed in seeds:
        det = RSHashDetector(
            n_estimators=100, window_size=1000, cms_width=1024, cms_depth=4, seed=int(seed)
        ).fit(x_tr)
        sc_cal = det.score_stream(x_cal, adapt=True)
        sc_te = det.score_stream(x_te, adapt=True)
        rsh_runs.append(evaluate_score_stream(sc_cal, sc_te, yte))
    out["rs_hash"] = _agg_seeds(rsh_runs)

    # 3. PCA + Mahalanobis (Matched 32-D windows, deterministic)
    pca_m32 = PCAMahalanobisDetector(variance_ratio=0.95, max_components=32, reg=1e-3).fit(x_tr)
    m32 = evaluate_score_stream(pca_m32.score(x_cal), pca_m32.score(x_te), yte)
    out["pca_mahalanobis_32d"] = _agg_seeds([m32])

    # 4. PCA + Mahalanobis (Full-D raw w*D windows, deterministic fairness check)
    xtr_full, xte_full, yte_full = _to_raw_full_d_windows(tr_r, te_r, yl_r)
    assert np.array_equal(yte, yte_full)
    kf_full = int(len(xtr_full) * 0.75)
    pca_mf = PCAMahalanobisDetector(variance_ratio=0.95, max_components=32, reg=1e-3).fit(
        xtr_full[:kf_full]
    )
    mf = evaluate_score_stream(
        pca_mf.score(xtr_full[kf_full:]), pca_mf.score(xte_full), yte_full
    )
    out["pca_mahalanobis_fulld"] = _agg_seeds([mf])

    # 5. Classical batch baselines re-evaluated live on identical 32-D windows
    iso_runs = []
    for seed in seeds:
        iso = IsolationForest(n_estimators=100, random_state=int(seed)).fit(x_tr)
        sc_cal = -iso.decision_function(x_cal)
        sc_te = -iso.decision_function(x_te)
        iso_runs.append(evaluate_score_stream(sc_cal, sc_te, yte))
    out["raw_isoforest"] = _agg_seeds(iso_runs)

    knn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(x_tr)
    knn_cal = knn.kneighbors(x_cal)[0].mean(axis=1)
    knn_te = knn.kneighbors(x_te)[0].mean(axis=1)
    out["raw_knn"] = _agg_seeds([evaluate_score_stream(knn_cal, knn_te, yte)])

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(x_tr)
    oc_cal = -ocsvm.decision_function(x_cal)
    oc_te = -ocsvm.decision_function(x_te)
    out["raw_ocsvm"] = _agg_seeds([evaluate_score_stream(oc_cal, oc_te, yte)])

    return out


BASELINE_DISPLAY_NAMES: Dict[str, str] = {
    "hs_trees": "Half-Space Trees (Streaming, 5 seeds)",
    "rs_hash": "RS-Hash / RS-Stream (Streaming, 5 seeds)",
    "pca_mahalanobis_32d": "PCA + Mahalanobis (Matched 32-D)",
    "pca_mahalanobis_fulld": "PCA + Mahalanobis (Full-D Raw)",
    "raw_isoforest": "Isolation Forest (Batch 32-D, 5 seeds)",
    "raw_knn": "kNN (k=5, Batch 32-D)",
    "raw_ocsvm": "OC-SVM (RBF, Batch 32-D)",
}


def summarize_across_servers(
    per_server: Dict[str, Dict[str, Dict[str, float]]],
    method_keys: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compute across-server mean and population std (ddof=0) for each method."""
    servers = list(per_server.keys())
    summary: Dict[str, Dict[str, float]] = {}
    for key in method_keys:
        au = np.array([per_server[s][key]["auroc"] for s in servers], dtype=np.float64)
        far_raw = np.array([per_server[s][key]["far_raw"] for s in servers], dtype=np.float64)
        rec_raw = np.array([per_server[s][key]["recall_raw"] for s in servers], dtype=np.float64)
        far_dual = np.array([per_server[s][key]["far_dual"] for s in servers], dtype=np.float64)
        rec_dual = np.array([per_server[s][key]["recall_dual"] for s in servers], dtype=np.float64)
        summary[key] = {
            "auroc_mean": float(np.mean(au)),
            "auroc_std": float(np.std(au, ddof=0)),
            "far_raw_mean": float(np.mean(far_raw)),
            "far_raw_std": float(np.std(far_raw, ddof=0)),
            "recall_raw_mean": float(np.mean(rec_raw)),
            "recall_raw_std": float(np.std(rec_raw, ddof=0)),
            "far_dual_mean": float(np.mean(far_dual)),
            "far_dual_std": float(np.std(far_dual, ddof=0)),
            "recall_dual_mean": float(np.mean(rec_dual)),
            "recall_dual_std": float(np.std(rec_dual, ddof=0)),
        }
    return summary
