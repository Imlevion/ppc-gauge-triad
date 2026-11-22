"""Automated Python verification suite (compatible with both `unittest` and `pytest`).

Run locally with either:
    python -m unittest discover -s benchmarks -p "test_*.py" -v
or (if pytest is installed):
    pytest benchmarks/test_local_suite.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.dataset_cache import _atomic_save_npz, load_psm_cached, load_smd_cached
from benchmarks.ppc_triad_wrapper import (
    PRESET_HIGH_RECALL,
    PRESET_LOW_FAR,
    PRESET_THERMAL_CHIP,
    NumpyTriadRefV061,
    RustTriadV061,
    load_rust_lib,
)


class TestPPCGaugeLocalSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = load_rust_lib()

    def test_embedded_sram_under_16kb(self) -> None:
        sram = int(self.lib.ppc_triad_engine_bytes())
        self.assertGreater(sram, 0)
        self.assertLess(sram, 16 * 1024, f"Struct SRAM {sram} B exceeds 16 KB limit")

    def test_l1_8_python_numpy_vs_rust_cdylib_parity(self) -> None:
        rng = np.random.default_rng(1108)
        x_tr = rng.standard_normal((120, 8, 4)).astype(np.float32)
        x_te = (rng.standard_normal((80, 8, 4)) * 1.2 + 0.3).astype(np.float32)

        rs = RustTriadV061(8, 4, k_max=3, lam0=5e-3, alpha_min=0.01, e_alpha=0.008, dc_alpha=0.004)
        py = NumpyTriadRefV061(8, 4, k_max=3, lam0=5e-3, alpha_min=0.01, e_alpha=0.008, dc_alpha=0.004)

        rs.run(x_tr, learn=True)
        py.run(x_tr, learn=True)

        r_sh, r_en, r_lv, r_al = rs.run(x_te, learn=False, shape_gate=50.0, adapt_env=True, huber_clip=1.5)
        p_sh, p_en, p_lv, p_al = py.run(x_te, learn=False, shape_gate=50.0, adapt_env=True, huber_clip=1.5)

        self.assertLess(float(np.max(np.abs(r_en - p_en))), 1e-4)
        self.assertLess(float(np.max(np.abs(r_lv - p_lv))), 1e-4)
        self.assertTrue(np.array_equal(r_al, p_al), "Alarm mismatch between Python ref and Rust cdylib")

        # Direct canonical shape parity check (< 1e-5 pre-registered L1.8 criterion)
        max_canon_diff = 0.0
        for i in range(40):
            pair = x_tr[2 * i : 2 * i + 2]
            r1 = RustTriadV061(8, 4, k_max=1, lam0=1.0, alpha_min=0.01)
            p1 = NumpyTriadRefV061(8, 4, k_max=1, lam0=1.0, alpha_min=0.01)
            rs_pair = r1.run(pair, learn=True)[0]
            py_pair = p1.run(pair, learn=True)[0]
            max_canon_diff = max(max_canon_diff, abs(float(rs_pair[1] - py_pair[1])))
        self.assertLess(max_canon_diff, 1e-5, f"L1.8 canonical parity exceeded 1e-5: {max_canon_diff:.3e}")

        # High correlation across streaming shape scores
        r_corr = float(np.corrcoef(r_sh, p_sh)[0, 1])
        self.assertGreater(r_corr, 0.9999)

    def test_l1_1_so4_rotation_invariance_ctypes(self) -> None:
        rng = np.random.default_rng(77)
        x_base = rng.standard_normal((100, 8, 4)).astype(np.float64)
        # Random Q in SO(4)
        q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1.0
        self.assertAlmostEqual(float(np.linalg.det(q)), 1.0, places=10)
        x_rot = x_base @ q

        eng1 = RustTriadV061(8, 4, k_max=3, lam0=5e-3, alpha_min=0.01)
        eng2 = RustTriadV061(8, 4, k_max=3, lam0=5e-3, alpha_min=0.01)
        sh1, en1, _, _ = eng1.run(x_base.astype(np.float32), learn=True)
        sh2, en2, _, _ = eng2.run(x_rot.astype(np.float32), learn=True)

        self.assertLess(float(np.max(np.abs(en1 - en2))), 1e-4)
        self.assertGreater(float(np.corrcoef(sh1, sh2)[0, 1]), 0.9999)

    def test_dataset_cache_roundtrip_and_offline_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            # Offline mode must raise cleanly when cache is absent
            with self.assertRaises(FileNotFoundError):
                load_smd_cached("machine-1-1", cache_dir=cache_dir, offline=True, verbose=False)
            with self.assertRaises(FileNotFoundError):
                load_psm_cached(cache_dir=cache_dir, offline=True, verbose=False)

            # Save synthetic cache and verify offline reload
            tr = np.ones((32, 38), dtype=np.float64)
            te = np.zeros((16, 38), dtype=np.float64)
            yl = np.array([0] * 12 + [1] * 4, dtype=np.int32)
            _atomic_save_npz(cache_dir / "smd_machine-1-1.npz", tr, te, yl)
            tr2, te2, yl2 = load_smd_cached("machine-1-1", cache_dir=cache_dir, offline=True, verbose=False)
            self.assertTrue(np.array_equal(tr, tr2))
            self.assertTrue(np.array_equal(te, te2))
            self.assertTrue(np.array_equal(yl, yl2))

    def test_locked_presets_match_validation_report(self) -> None:
        self.assertEqual((PRESET_HIGH_RECALL.e_alpha, PRESET_HIGH_RECALL.dc_alpha, PRESET_HIGH_RECALL.huber_clip), (0.002, 0.001, 1.0))
        self.assertEqual((PRESET_LOW_FAR.e_alpha, PRESET_LOW_FAR.dc_alpha, PRESET_LOW_FAR.huber_clip), (0.008, 0.004, 1.5))
        self.assertEqual((PRESET_THERMAL_CHIP.e_alpha, PRESET_THERMAL_CHIP.dc_alpha, PRESET_THERMAL_CHIP.huber_clip, PRESET_THERMAL_CHIP.dc_relock), (0.20, 0.25, 0.0, 6))


if __name__ == "__main__":
    unittest.main()
