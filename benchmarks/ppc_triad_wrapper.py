"""Cross-platform ctypes wrapper and pure-NumPy reference for PPC-Gauge-Triad v0.6.1.

Eliminates Google Colab dependency by loading the locally built Rust `cdylib`
(`.dll` on Windows, `.so` on Linux, `.dylib` on macOS) and exposing the locked
a priori operational presets documented in `VALIDATION_REPORT_v061.md`.
"""

from __future__ import annotations

import ctypes
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class OperationalPreset:
    """Locked a priori operational preset (no post-hoc tuning per AGENTS.md)."""

    name: str
    e_alpha: float
    dc_alpha: float
    huber_clip: float
    dc_relock: int
    use_pca_resid: bool
    adapt_env: bool
    use_dual: bool
    sev_mult: float = 2.5


# Locked a priori presets matching VALIDATION_REPORT_v061.md Section 3
PRESET_HIGH_RECALL = OperationalPreset(
    name="high_recall",
    e_alpha=0.002,
    dc_alpha=0.001,
    huber_clip=1.0,
    dc_relock=0,
    use_pca_resid=True,
    adapt_env=True,
    use_dual=True,
)

PRESET_LOW_FAR = OperationalPreset(
    name="low_far",
    e_alpha=0.008,
    dc_alpha=0.004,
    huber_clip=1.5,
    dc_relock=0,
    use_pca_resid=True,
    adapt_env=True,
    use_dual=True,
)

PRESET_LOW_FAR_PCA4 = OperationalPreset(
    name="low_far_pca4",
    e_alpha=0.008,
    dc_alpha=0.004,
    huber_clip=1.5,
    dc_relock=0,
    use_pca_resid=False,
    adapt_env=True,
    use_dual=True,
)

PRESET_THERMAL_CHIP = OperationalPreset(
    name="thermal_chip",
    e_alpha=0.20,
    dc_alpha=0.25,
    huber_clip=0.0,
    dc_relock=6,
    use_pca_resid=False,
    adapt_env=True,
    use_dual=True,
)

PRESET_V05_BASELINE = OperationalPreset(
    name="v05_baseline",
    e_alpha=0.10,
    dc_alpha=0.05,
    huber_clip=0.0,
    dc_relock=0,
    use_pca_resid=False,
    adapt_env=False,
    use_dual=False,
)

PRESETS: Dict[str, OperationalPreset] = {
    p.name: p
    for p in (
        PRESET_HIGH_RECALL,
        PRESET_LOW_FAR,
        PRESET_LOW_FAR_PCA4,
        PRESET_THERMAL_CHIP,
        PRESET_V05_BASELINE,
    )
}


def shared_lib_path(repo_root: Path = REPO_ROOT) -> Path:
    """Return platform-specific path to the compiled Rust release cdylib."""
    system = platform.system()
    if system == "Windows":
        filename = "ppc_gauge_triad_rs.dll"
    elif system == "Darwin":
        filename = "libppc_gauge_triad_rs.dylib"
    else:
        filename = "libppc_gauge_triad_rs.so"
    return repo_root / "target" / "release" / filename


_CACHED_LIB: Optional[ctypes.CDLL] = None


def load_rust_lib(repo_root: Path = REPO_ROOT, rebuild: bool = False) -> ctypes.CDLL:
    """Build (if needed) and load the native Rust `ppc_gauge_triad_rs` cdylib."""
    global _CACHED_LIB
    lib_path = shared_lib_path(repo_root)
    if rebuild or not lib_path.exists():
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(repo_root),
            check=True,
        )
    if not lib_path.exists():
        raise FileNotFoundError(f"Expected compiled Rust library at {lib_path}")

    if _CACHED_LIB is not None and not rebuild:
        return _CACHED_LIB

    lib = ctypes.CDLL(str(lib_path))
    lib.ppc_triad_engine_bytes.restype = ctypes.c_size_t
    lib.ppc_triad_init.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ppc_triad_run_stream_v61.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_uint8,
        ctypes.c_float,
        ctypes.c_uint8,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint8,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.ppc_autocal_buf_bytes.restype = ctypes.c_size_t
    lib.ppc_autocal_engine_bytes.restype = ctypes.c_size_t
    lib.ppc_autocal_state.argtypes = [ctypes.c_void_p]
    lib.ppc_autocal_state.restype = ctypes.c_uint8
    lib.ppc_autocal_init.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.c_float,
        ctypes.c_uint8,
        ctypes.c_float,
        ctypes.c_uint8,
        ctypes.c_float,
        ctypes.c_uint8,
    ]
    lib.ppc_autocal_run_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    _CACHED_LIB = lib
    return lib


def make_dct_f32(w: int) -> np.ndarray:
    """Create column-orthonormal temporal DCT-II basis B (w x (w-1)) in float32."""
    n = np.arange(w, dtype=np.float64)[:, None]
    k = np.arange(1, w, dtype=np.float64)[None, :]
    b = np.cos(np.pi * (n + 0.5) * k / w)
    b = b / np.linalg.norm(b, axis=0, keepdims=True)
    return np.ascontiguousarray(b, dtype=np.float32)


class RustTriadV061:
    """Thin ctypes wrapper around the `#![no_std]` `GaugeTriadEngineF32` struct."""

    def __init__(
        self,
        w: int = 8,
        d: int = 4,
        k_max: int = 3,
        lam0: float = 5e-3,
        alpha_min: float = 0.01,
        e_alpha: float = 0.008,
        dc_alpha: float = 0.004,
        lib: Optional[ctypes.CDLL] = None,
    ) -> None:
        self.lib = lib if lib is not None else load_rust_lib()
        self.w = w
        self.d = d
        self.sz = int(self.lib.ppc_triad_engine_bytes())
        self.buf = ctypes.create_string_buffer(self.sz)
        self.B = make_dct_f32(w)
        self.lib.ppc_triad_init(
            ctypes.byref(self.buf),
            w,
            d,
            k_max,
            ctypes.c_float(lam0),
            ctypes.c_float(alpha_min),
            ctypes.c_float(e_alpha),
            ctypes.c_float(dc_alpha),
            self.B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def run(
        self,
        x_wins: np.ndarray,
        learn: bool = False,
        shape_gate: float = 1e20,
        adapt_env: bool = False,
        imp_thr: float = 10.0,
        lvl_thr: float = 10.0,
        sev_mult: float = 2.5,
        dc_relock: int = 0,
        huber_clip: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xc = np.ascontiguousarray(x_wins, dtype=np.float32)
        n = len(xc)
        sh = np.empty(n, dtype=np.float32)
        en = np.empty(n, dtype=np.float32)
        lv = np.empty(n, dtype=np.float32)
        al = np.empty(n, dtype=np.uint8)
        self.lib.ppc_triad_run_stream_v61(
            ctypes.byref(self.buf),
            xc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            1 if learn else 0,
            ctypes.c_float(shape_gate),
            1 if adapt_env else 0,
            ctypes.c_float(imp_thr),
            ctypes.c_float(lvl_thr),
            ctypes.c_float(sev_mult),
            int(dc_relock),
            ctypes.c_float(huber_clip),
            sh.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            en.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            lv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            al.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        return sh.astype(np.float64), en.astype(np.float64), lv.astype(np.float64), al.astype(bool)


class RustAutoCalV07:
    """Thin ctypes wrapper around v0.7 on-chip `#![no_std]` `AutoCalEngineF32`.

    Executes the entire `Warmup -> Calibrate -> Monitor` state machine directly
    inside Rust (`~14.4 KB` static SRAM, `0 B` heap) with zero external Python
    threshold or MAD post-processing.
    """

    def __init__(
        self,
        w: int = 8,
        d: int = 4,
        k_max: int = 3,
        lam0: float = 5e-3,
        alpha_min: float = 0.01,
        e_alpha: float = 0.008,
        dc_alpha: float = 0.004,
        n_warmup: int = 300,
        n_cal: int = 100,
        alpha: float = 0.05,
        adapt_env: bool = True,
        sev_mult: float = 2.5,
        dc_relock: int = 0,
        huber_clip: float = 1.5,
        use_dual: bool = True,
        lib: Optional[ctypes.CDLL] = None,
    ) -> None:
        self.lib = lib if lib is not None else load_rust_lib()
        self.w = w
        self.d = d
        self.sz = int(self.lib.ppc_autocal_engine_bytes())
        self.buf = ctypes.create_string_buffer(self.sz)
        self.B = make_dct_f32(w)
        self.lib.ppc_autocal_init(
            ctypes.byref(self.buf),
            w,
            d,
            k_max,
            ctypes.c_float(lam0),
            ctypes.c_float(alpha_min),
            ctypes.c_float(e_alpha),
            ctypes.c_float(dc_alpha),
            self.B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(n_warmup),
            int(n_cal),
            ctypes.c_float(alpha),
            1 if adapt_env else 0,
            ctypes.c_float(sev_mult),
            int(dc_relock),
            ctypes.c_float(huber_clip),
            1 if use_dual else 0,
        )

    @property
    def state(self) -> int:
        return int(self.lib.ppc_autocal_state(ctypes.byref(self.buf)))

    def run(
        self, x_wins: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xc = np.ascontiguousarray(x_wins, dtype=np.float32)
        n = len(xc)
        st = np.empty(n, dtype=np.uint8)
        sh = np.empty(n, dtype=np.float32)
        en = np.empty(n, dtype=np.float32)
        lv = np.empty(n, dtype=np.float32)
        cb = np.empty(n, dtype=np.float32)
        al = np.empty(n, dtype=np.uint8)
        self.lib.ppc_autocal_run_stream(
            ctypes.byref(self.buf),
            xc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            st.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            sh.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            en.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            lv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            al.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        return (
            st,
            sh.astype(np.float64),
            en.astype(np.float64),
            lv.astype(np.float64),
            cb.astype(np.float64),
            al.astype(bool),
        )


# ---------------------------------------------------------------------------
# Pure-Python NumPy reference engine (for L1.8 Cross-Implementation Parity)
# ---------------------------------------------------------------------------


class NumpyTriadRefV061:
    """Pure-Python / NumPy float32 reference mirroring `src/lib.rs` for L1.8 parity."""

    def __init__(
        self,
        w: int = 8,
        d: int = 4,
        k_max: int = 3,
        lam0: float = 5e-3,
        alpha_min: float = 0.01,
        e_alpha: float = 0.008,
        dc_alpha: float = 0.004,
    ) -> None:
        self.w = w
        self.d = d
        self.wm1 = w - 1
        self.m = sum(row_k + 1 if row_k < d else d for row_k in range(self.wm1))
        self.k_max = min(k_max, 4)
        self.k_active = 0
        self.lam0 = np.float32(lam0)
        self.alpha_min = np.float32(alpha_min)
        self.e_alpha = np.float32(e_alpha)
        self.dc_alpha = np.float32(dc_alpha)
        self.e_ema = np.float32(0.0)
        self.e_var = np.float32(0.05)
        self.dc_ema = np.zeros(d, dtype=np.float32)
        self.dc_var = np.full(d, 0.05, dtype=np.float32)
        self.prev_dc = np.zeros(d, dtype=np.float32)
        self.e_init = 0
        self.streak = 0
        self.dc_streak = 0
        self.sh_alarm_streak = 0
        self.en_alarm_streak = 0
        self.lv_alarm_streak = 0
        self.B = make_dct_f32(w)
        self.means = np.zeros((4, self.m), dtype=np.float32)
        self.precs = np.zeros((4, self.m, self.m), dtype=np.float32)
        self.counts = np.zeros(4, dtype=np.uint32)

    def canonicalize(self, x_win: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        x = np.asarray(x_win, dtype=np.float32).reshape(self.w, self.d)
        dc = np.mean(x, axis=0, dtype=np.float32)
        xc = (x - dc[None, :]).astype(np.float32)
        norm_sq = np.float32(np.sum(xc * xc, dtype=np.float32))
        norm_raw = np.float32(np.sqrt(norm_sq))
        log_e = float(np.log(np.float32(norm_raw + np.float32(1e-6))))
        if norm_raw < 1e-12:
            return np.zeros(self.m, dtype=np.float32), dc, log_e

        inv_norm = np.float32(1.0) / norm_raw
        r_mat = (xc * inv_norm).T @ self.B  # (d, w-1)
        r_mat = r_mat.astype(np.float32, copy=True)

        for col in range(self.d):
            for row in range(self.d - 1, col, -1):
                a = np.float32(r_mat[row - 1, col])
                b = np.float32(r_mat[row, col])
                rad = np.float32(np.sqrt(a * a + b * b))
                if rad > 1e-7:
                    c = a / rad
                    s = -b / rad
                    r1 = r_mat[row - 1, col:].copy()
                    r2 = r_mat[row, col:].copy()
                    r_mat[row - 1, col:] = c * r1 - s * r2
                    r_mat[row, col:] = s * r1 + c * r2
                    r_mat[row, col] = np.float32(0.0)

        det_sign = 1
        col_sign = np.ones(self.d, dtype=np.float32)
        for i in range(self.d - 1):
            if r_mat[i, i] < 0.0:
                col_sign[i] = np.float32(-1.0)
                det_sign = -det_sign
        if det_sign < 0:
            col_sign[self.d - 1] = np.float32(-1.0)

        u = np.empty(self.m, dtype=np.float32)
        idx = 0
        for row_k in range(self.wm1):
            max_j = row_k + 1 if row_k < self.d else self.d
            for col_j in range(max_j):
                u[idx] = r_mat[col_j, row_k] * col_sign[col_j]
                idx += 1
        return u, dc, log_e

    def _dist_to(self, u: np.ndarray, k: int) -> float:
        diff = (u - self.means[k]).astype(np.float32)
        acc = float(diff @ (self.precs[k] @ diff))
        return 0.0 if (acc < 0.0 or not np.isfinite(acc)) else acc

    def _spawn_proto(self, u: np.ndarray) -> None:
        k = self.k_active
        self.means[k] = u
        self.precs[k] = np.eye(self.m, dtype=np.float32) * (np.float32(1.0) / self.lam0)
        self.counts[k] = 1
        self.k_active += 1

    def _sm_update(self, k: int, u: np.ndarray) -> None:
        self.counts[k] = np.uint32(int(self.counts[k]) + 1)
        a_cnt = np.float32(1.0 / (float(self.counts[k]) + 4.0))
        a = a_cnt if a_cnt > self.alpha_min else self.alpha_min
        inv_1ma = np.float32(1.0) / (np.float32(1.0) - a)

        diff = (u - self.means[k]).astype(np.float32)
        pv = (self.precs[k] @ diff).astype(np.float32)
        q = np.float32(np.dot(diff, pv))
        denom = (np.float32(1.0) - a) + a * q
        if denom > 1e-6 and np.isfinite(denom):
            scale = a / denom
            updated = (self.precs[k] - np.outer(pv * scale, pv)) * inv_1ma
            updated = np.clip(updated, -1.0e5, 1.0e5).astype(np.float32)
            # Enforce exact symmetry matching Rust upper-triangle mirror
            iu = np.triu_indices(self.m)
            sym = np.empty_like(updated)
            sym[iu] = updated[iu]
            sym[(iu[1], iu[0])] = updated[iu]
            self.precs[k] = sym
        self.means[k] = (self.means[k] + a * diff).astype(np.float32)

    def run(
        self,
        x_wins: np.ndarray,
        learn: bool = False,
        shape_gate: float = 1e20,
        adapt_env: bool = False,
        imp_thr: float = 10.0,
        lvl_thr: float = 10.0,
        sev_mult: float = 2.5,
        dc_relock: int = 0,
        huber_clip: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xc = np.asarray(x_wins, dtype=np.float32)
        n = len(xc)
        sh_out = np.empty(n, dtype=np.float64)
        en_out = np.empty(n, dtype=np.float64)
        lv_out = np.empty(n, dtype=np.float64)
        al_out = np.zeros(n, dtype=bool)

        for idx in range(n):
            u, dc, log_e = self.canonicalize(xc[idx])
            if self.e_init == 0:
                s_energy, s_level = 0.0, 0.0
            else:
                de = float(log_e - float(self.e_ema))
                s_energy = (de * de) / (float(self.e_var) + 0.05)
                dd = dc.astype(np.float64) - self.dc_ema.astype(np.float64)
                s_level = float(np.mean((dd * dd) / (self.dc_var.astype(np.float64) + 0.05)))

            if self.k_active == 0:
                best_k, s_shape = 0, 0.0
            else:
                dists = [self._dist_to(u, k) for k in range(self.k_active)]
                best_k = int(np.argmin(dists))
                s_shape = dists[best_k]

            shape_ok = (self.k_active == 0) or (s_shape <= shape_gate)
            if learn and shape_ok:
                if self.k_active == 0:
                    self._spawn_proto(u)
                elif self.k_active < self.k_max and s_shape > (0.35 / float(self.lam0)):
                    self._spawn_proto(u)
                else:
                    self._sm_update(best_k, u)

            if (learn or adapt_env) and shape_ok:
                if self.e_init == 0:
                    self.e_ema = np.float32(log_e)
                    self.dc_ema[:] = dc
                    self.prev_dc[:] = dc
                    self.e_init = 1
                    self.streak = 0
                    self.dc_streak = 0
                elif huber_clip > 0.0 and not learn:
                    ce = huber_clip * float(np.sqrt(float(self.e_var) + 0.05))
                    de_c = float(np.clip(log_e - float(self.e_ema), -ce, ce))
                    ea = float(self.e_alpha)
                    self.e_ema = np.float32(float(self.e_ema) + ea * de_c)
                    self.e_var = np.float32((1.0 - ea) * float(self.e_var) + ea * (de_c * de_c))
                    da = float(self.dc_alpha)
                    for j in range(self.d):
                        cd = huber_clip * float(np.sqrt(float(self.dc_var[j]) + 0.05))
                        dd_c = float(np.clip(float(dc[j]) - float(self.dc_ema[j]), -cd, cd))
                        self.dc_ema[j] = np.float32(float(self.dc_ema[j]) + da * dd_c)
                        self.dc_var[j] = np.float32((1.0 - da) * float(self.dc_var[j]) + da * (dd_c * dd_c))
                else:
                    ea = float(self.e_alpha)
                    if s_energy <= imp_thr or learn:
                        self.streak = 0
                        de = log_e - float(self.e_ema)
                        self.e_ema = np.float32(float(self.e_ema) + ea * de)
                        self.e_var = np.float32((1.0 - ea) * float(self.e_var) + ea * (de * de))
                    else:
                        self.streak = min(255, self.streak + 1)
                        if self.streak >= 3:
                            self.e_ema = np.float32(log_e)
                            self.streak = 0
                    da = float(self.dc_alpha)
                    if s_level <= lvl_thr or learn:
                        self.dc_streak = 0
                        for j in range(self.d):
                            dd = float(dc[j]) - float(self.dc_ema[j])
                            self.dc_ema[j] = np.float32(float(self.dc_ema[j]) + da * dd)
                            self.dc_var[j] = np.float32((1.0 - da) * float(self.dc_var[j]) + da * (dd * dd))
                    elif dc_relock > 0:
                        step_sq = float(np.mean((dc.astype(np.float64) - self.prev_dc.astype(np.float64)) ** 2))
                        if step_sq < 1.0:
                            self.dc_streak = min(255, self.dc_streak + 1)
                            if self.dc_streak >= dc_relock:
                                self.dc_ema[:] = dc
                                self.dc_streak = 0
                        else:
                            self.dc_streak = 0
                self.prev_dc[:] = dc

            alarm = False
            if (not learn) and (shape_gate < 1e10):
                self.sh_alarm_streak = self.sh_alarm_streak + 1 if s_shape > shape_gate else 0
                self.en_alarm_streak = self.en_alarm_streak + 1 if s_energy > imp_thr else 0
                self.lv_alarm_streak = self.lv_alarm_streak + 1 if s_level > lvl_thr else 0
                if (
                    (s_shape > shape_gate * sev_mult or self.sh_alarm_streak >= 2)
                    or (s_energy > imp_thr * sev_mult or self.en_alarm_streak >= 2)
                    or (s_level > lvl_thr * sev_mult or self.lv_alarm_streak >= 2)
                ):
                    alarm = True

            sh_out[idx] = s_shape
            en_out[idx] = s_energy
            lv_out[idx] = s_level
            al_out[idx] = alarm

        return sh_out, en_out, lv_out, al_out


# ---------------------------------------------------------------------------
# Scoring, Windowing, and Conformal Utilities (matching Colab v0.6.1)
# ---------------------------------------------------------------------------


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Exact (or deterministic 2000x2000 subsampled for huge sets) AUROC."""
    yt = np.asarray(y_true, dtype=np.int32)
    sc = np.asarray(scores, dtype=np.float64)
    pos, neg = sc[yt == 1], sc[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    if len(pos) * len(neg) > 4_000_000:
        rng = np.random.default_rng(0)
        pos = rng.choice(pos, min(len(pos), 2000), replace=False)
        neg = rng.choice(neg, min(len(neg), 2000), replace=False)
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def conformal_threshold(scores_cal: np.ndarray, alpha: float = 0.05) -> float:
    """Split-conformal finite-sample (1 - alpha) quantile threshold."""
    s = np.sort(np.asarray(scores_cal, dtype=np.float64))
    idx = max(0, min(len(s) - 1, int(np.ceil((1.0 - alpha) * (len(s) + 1))) - 1))
    return float(s[idx])


def to_windows_v6(
    tr_raw: np.ndarray,
    te_raw: np.ndarray,
    yl_raw: np.ndarray,
    w: int = 8,
    d: int = 4,
    max_tr_win: int = 2000,
    use_pca_resid: bool = True,
):
    """Standardize, project to d=4 (either 3PC+ResidNorm or 4PC), and segment into w x d windows."""
    mu = tr_raw.mean(axis=0)
    sd = np.maximum(tr_raw.std(axis=0), 1e-6)
    ztr_std = (tr_raw - mu) / sd
    zte_std = (te_raw - mu) / sd
    if use_pca_resid and tr_raw.shape[1] > d:
        pca = PCA(n_components=d - 1, random_state=0).fit(ztr_std)
        ptr, pte = pca.transform(ztr_std), pca.transform(zte_std)
        rtr = np.linalg.norm(ztr_std - pca.inverse_transform(ptr), axis=1, keepdims=True)
        rte = np.linalg.norm(zte_std - pca.inverse_transform(pte), axis=1, keepdims=True)
        r_mu = rtr.mean(axis=0, keepdims=True)
        r_sd = np.maximum(rtr.std(axis=0, keepdims=True), 1e-6)
        ztr = np.hstack([ptr, (rtr - r_mu) / r_sd])
        zte = np.hstack([pte, (rte - r_mu) / r_sd])
        meta = (mu, sd, pca, r_mu, r_sd)
    else:
        pca = PCA(n_components=d, random_state=0).fit(ztr_std)
        ztr, zte = pca.transform(ztr_std), pca.transform(zte_std)
        meta = (mu, sd, pca, None, None)

    ntr = min(max_tr_win, len(ztr) // w)
    nte = len(zte) // w
    wtr = ztr[: ntr * w].reshape(ntr, w, d)
    wte = zte[: nte * w].reshape(nte, w, d)
    yte = (yl_raw[: nte * w].reshape(nte, w).max(axis=1) > 0).astype(np.int32)
    return wtr, wte, yte, meta


def triad_norm_score(
    sh_c: np.ndarray,
    en_c: np.ndarray,
    lv_c: np.ndarray,
    sh_t: np.ndarray,
    en_t: np.ndarray,
    lv_t: np.ndarray,
) -> np.ndarray:
    """Combine log1p-MAD-normalized s_shape + s_energy + s_level scores."""

    def rz(c: np.ndarray, t: np.ndarray) -> np.ndarray:
        lc, lt = np.log1p(c), np.log1p(t)
        med = np.median(lc)
        mad = np.median(np.abs(lc - med)) * 1.4826 + 0.1
        return np.maximum(0.0, (lt - med) / mad)

    return rz(sh_c, sh_t) + rz(en_c, en_t) + rz(lv_c, lv_t)


def dual_gate_scalar(scores: np.ndarray, thr: float, sev_mult: float = 2.5) -> np.ndarray:
    """Dual-stage alarm gate on scalar combined score (immediate severe OR 2-consecutive)."""
    al = np.zeros(len(scores), dtype=bool)
    st = 0
    for i, s in enumerate(scores):
        st = st + 1 if s > thr else 0
        if (s > thr * sev_mult) or (st >= 2):
            al[i] = True
    return al
