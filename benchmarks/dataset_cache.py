"""Local disk cache for SMD (Server Machine Dataset) and PSM (eBay Server).

Downloads each dataset once from its upstream GitHub raw URL and stores parsed
NumPy arrays in `data_cache/*.npz` with atomic writes and a JSON manifest.
Requires zero external dependencies beyond NumPy and Python's standard library
(no `pandas` required).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = Path(os.environ.get("PPC_DATA_CACHE", str(REPO_ROOT / "data_cache")))

SMD_BASE_URL = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
)
PSM_BASE_URL = "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data"

# Canonical 8-server benchmark suite from VALIDATION_REPORT_v061.md Section 4.1
BENCHMARK_SERVERS: List[Tuple[str, str, Optional[str]]] = [
    ("SMD machine-1-1", "smd", "machine-1-1"),
    ("SMD machine-1-6", "smd", "machine-1-6"),
    ("SMD machine-2-1", "smd", "machine-2-1"),
    ("PSM eBay Server", "psm", None),
    ("SMD machine-1-2", "smd", "machine-1-2"),
    ("SMD machine-1-3", "smd", "machine-1-3"),
    ("SMD machine-2-2", "smd", "machine-2-2"),
    ("SMD machine-3-1", "smd", "machine-3-1"),
]


def _download_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ppc-gauge-triad-v061-cache/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_smd_txt(raw: bytes) -> np.ndarray:
    return np.loadtxt(io.BytesIO(raw), delimiter=",", dtype=np.float64)


def _parse_psm_features_csv(raw: bytes) -> np.ndarray:
    """Parse PSM train.csv / test.csv: drop header & timestamp col 0, fillna(0.0)."""
    text = raw.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        raise ValueError("Empty PSM CSV")
    ncols = len(header) - 1
    rows: List[List[float]] = []
    for row in reader:
        if not row:
            continue
        vals: List[float] = []
        for cell in row[1:]:
            s = cell.strip()
            if not s or s.lower() == "nan":
                vals.append(0.0)
            else:
                v = float(s)
                vals.append(0.0 if v != v else v)  # NaN -> 0.0
        if len(vals) == ncols:
            rows.append(vals)
    return np.asarray(rows, dtype=np.float64)


def _parse_psm_labels_csv(raw: bytes) -> np.ndarray:
    """Parse PSM test_label.csv: drop header, read column 1 as int32."""
    text = raw.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    labels: List[int] = []
    for row in reader:
        if len(row) >= 2 and row[1].strip():
            labels.append(int(float(row[1].strip())))
    return np.asarray(labels, dtype=np.int32)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_manifest(cache_dir: Path, key: str, meta: Dict) -> None:
    manifest_path = cache_dir / "cache_manifest.json"
    data: Dict = {}
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[key] = meta
    tmp = cache_dir / "cache_manifest.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, manifest_path)


def _atomic_save_npz(path: Path, train: np.ndarray, test: np.ndarray, test_label: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, train=train, test=test, test_label=test_label)
    os.replace(tmp, path)


def load_smd_cached(
    machine: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    force_refresh: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load `(train, test, test_label)` for an SMD machine from local `.npz` cache."""
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / f"smd_{machine}.npz"
    if npz_path.exists() and not force_refresh:
        with np.load(npz_path) as data:
            return data["train"], data["test"], data["test_label"]

    if offline:
        raise FileNotFoundError(
            f"Offline mode enabled and cache missing: {npz_path}. "
            f"Run without --offline first to populate {cache_dir}."
        )

    if verbose:
        print(f"[cache] Downloading SMD {machine} -> {npz_path.name} ...", flush=True)
    tr_raw = _parse_smd_txt(_download_bytes(f"{SMD_BASE_URL}/train/{machine}.txt"))
    te_raw = _parse_smd_txt(_download_bytes(f"{SMD_BASE_URL}/test/{machine}.txt"))
    yl_raw = _parse_smd_txt(_download_bytes(f"{SMD_BASE_URL}/test_label/{machine}.txt")).astype(
        np.int32
    )
    _atomic_save_npz(npz_path, tr_raw, te_raw, yl_raw)
    _update_manifest(
        cache_dir,
        f"smd_{machine}",
        {
            "file": npz_path.name,
            "sha256": _sha256_file(npz_path),
            "train_shape": list(tr_raw.shape),
            "test_shape": list(te_raw.shape),
            "label_shape": list(yl_raw.shape),
            "anomaly_rate": float(np.mean(yl_raw > 0)),
        },
    )
    return tr_raw, te_raw, yl_raw


def load_psm_cached(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    force_refresh: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load `(train, test, test_label)` for eBay PSM from local `.npz` cache."""
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / "psm_ebay.npz"
    if npz_path.exists() and not force_refresh:
        with np.load(npz_path) as data:
            return data["train"], data["test"], data["test_label"]

    if offline:
        raise FileNotFoundError(
            f"Offline mode enabled and cache missing: {npz_path}. "
            f"Run without --offline first to populate {cache_dir}."
        )

    if verbose:
        print(f"[cache] Downloading PSM eBay Server -> {npz_path.name} ...", flush=True)
    tr_raw = _parse_psm_features_csv(_download_bytes(f"{PSM_BASE_URL}/train.csv"))
    te_raw = _parse_psm_features_csv(_download_bytes(f"{PSM_BASE_URL}/test.csv"))
    yl_raw = _parse_psm_labels_csv(_download_bytes(f"{PSM_BASE_URL}/test_label.csv"))
    _atomic_save_npz(npz_path, tr_raw, te_raw, yl_raw)
    _update_manifest(
        cache_dir,
        "psm_ebay",
        {
            "file": npz_path.name,
            "sha256": _sha256_file(npz_path),
            "train_shape": list(tr_raw.shape),
            "test_shape": list(te_raw.shape),
            "label_shape": list(yl_raw.shape),
            "anomaly_rate": float(np.mean(yl_raw > 0)),
        },
    )
    return tr_raw, te_raw, yl_raw


def load_server(
    kind: str,
    machine: Optional[str] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    force_refresh: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == "smd":
        if not machine:
            raise ValueError("SMD requires a machine identifier (e.g. 'machine-1-1')")
        return load_smd_cached(machine, cache_dir, offline, force_refresh, verbose)
    if kind == "psm":
        return load_psm_cached(cache_dir, offline, force_refresh, verbose)
    raise ValueError(f"Unsupported dataset kind: {kind}")


def populate_all_caches(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    offline: bool = False,
    force_refresh: bool = False,
    verbose: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Ensure all 8 benchmark servers (7 SMD + 1 PSM) are cached and loaded."""
    loaded: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for display_name, kind, machine in BENCHMARK_SERVERS:
        loaded[display_name] = load_server(
            kind, machine, cache_dir=cache_dir, offline=offline, force_refresh=force_refresh, verbose=verbose
        )
    return loaded
