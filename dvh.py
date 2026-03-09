"""
SRS Analysis – DVH Computation
================================
Compute cumulative dose-volume histograms (DVHs) for structures.
Uses the same grid-sampling approach as metrics.py.  Results are cached
to disk as JSON files to avoid re-computation on repeated runs.
"""

import json
import logging
import os
from typing import Optional, Dict

import numpy as np

from dicom_io import Structure, DoseData
from structure_mapping import _point_in_structure_xy, _interpolate_dose

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_dvh(
    structure: Structure,
    dose: DoseData,
    rx_dose: float,
    step: float = 1.0,
    n_bins: int = 250,
) -> Optional[Dict]:
    """
    Compute cumulative DVH for *structure* sampled on a *step*-mm grid.

    Returns a dict with:
        dose_bins     : list of n_bins+1 dose values [Gy]
        vol_cc        : cumulative absolute volume ≥ each dose [cc]
        vol_pct       : cumulative relative volume ≥ each dose [%]
        total_vol_cc  : total structure volume estimated by grid
        rx_dose       : prescription dose [Gy]
    Returns None on failure.
    """
    if not structure.contours or dose is None:
        return None

    all_pts = np.vstack(structure.contours)
    min_xyz = all_pts.min(axis=0)
    max_xyz = all_pts.max(axis=0)

    xs = np.arange(min_xyz[0], max_xyz[0] + step * 0.5, step)
    ys = np.arange(min_xyz[1], max_xyz[1] + step * 0.5, step)
    z_contours = np.unique(np.round(all_pts[:, 2], 1))

    sample_vol = (step ** 3) / 1000.0   # mm³ → cc

    dose_vals = []
    for z in z_contours:
        for y in ys:
            for x in xs:
                if not _point_in_structure_xy((x, y, z), structure):
                    continue
                d = _interpolate_dose((x, y, z), dose)
                if d is not None:
                    dose_vals.append(d)

    if not dose_vals:
        logger.warning("DVH: no interior points for %s", structure.name)
        return None

    dose_arr = np.array(dose_vals, dtype=float)
    total_vol_cc = len(dose_arr) * sample_vol

    d_max = max(float(dose_arr.max()), rx_dose * 1.05, 0.01)
    bins = np.linspace(0.0, d_max, n_bins + 1)

    counts, _ = np.histogram(dose_arr, bins=bins)
    cum_counts  = np.cumsum(counts[::-1])[::-1]
    cum_vol_cc  = np.append(cum_counts * sample_vol, 0.0)
    cum_vol_pct = (100.0 * cum_vol_cc / total_vol_cc
                   if total_vol_cc > 0 else np.zeros_like(cum_vol_cc))

    logger.debug("DVH %s: %d pts, TV=%.3f cc", structure.name, len(dose_arr), total_vol_cc)

    return {
        "dose_bins":    [round(v, 4) for v in bins.tolist()],
        "vol_cc":       [round(v, 5) for v in cum_vol_cc.tolist()],
        "vol_pct":      [round(v, 3) for v in cum_vol_pct.tolist()],
        "total_vol_cc": round(total_vol_cc, 4),
        "rx_dose":      rx_dose,
    }


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, patient: str, plan_type: str, struct_name: str, rx: float) -> str:
    safe = (struct_name.replace(" ", "_").replace("/", "-")
            .replace("\\", "-").replace(":", "-").replace("*", "")
            .replace("?", ""))
    fname = f"{patient}__{plan_type}__{safe}__{rx:.1f}Gy.json"
    return os.path.join(cache_dir, fname)


def load_dvh_cache(cache_dir: str, patient: str, plan_type: str,
                   struct_name: str, rx: float) -> Optional[Dict]:
    p = _cache_path(cache_dir, patient, plan_type, struct_name, rx)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug("DVH cache read failed (%s): %s", p, exc)
        return None


def save_dvh_cache(cache_dir: str, patient: str, plan_type: str,
                   struct_name: str, rx: float, dvh: Dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    p = _cache_path(cache_dir, patient, plan_type, struct_name, rx)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dvh, f)
    except Exception as exc:
        logger.warning("DVH cache write failed (%s): %s", p, exc)


def get_dvh(
    structure: Structure,
    dose: DoseData,
    rx_dose: float,
    cache_dir: str,
    patient: str,
    plan_type: str,
    step: float = 1.0,
) -> Optional[Dict]:
    """Load DVH from cache if available, otherwise compute + cache."""
    dvh = load_dvh_cache(cache_dir, patient, plan_type, structure.name, rx_dose)
    if dvh is not None:
        logger.debug("DVH cache hit: %s/%s/%s", patient, plan_type, structure.name)
        return dvh
    logger.info("  DVH computing: %s/%s/%s  Rx=%.0f Gy",
                patient, plan_type, structure.name, rx_dose)
    dvh = compute_dvh(structure, dose, rx_dose, step=step)
    if dvh is not None:
        save_dvh_cache(cache_dir, patient, plan_type, structure.name, rx_dose, dvh)
    return dvh
