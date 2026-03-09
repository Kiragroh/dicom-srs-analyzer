"""
SRS Analysis Pipeline - Structure Mapping
==========================================
Detects PTV candidates, builds / updates the Excel mapping table, and
provides helper functions to resolve PTV naming and prescription dose.
"""

import logging
import re
from typing import List, Optional

import numpy as np

from config import PTV_DETECTION_MODE
from dicom_io import Structure, DoseData, PlanFiles

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PTV candidate detection
# ---------------------------------------------------------------------------

def is_ptv_candidate(structure: Structure, mode: str = PTV_DETECTION_MODE) -> bool:
    """
    Decide whether *structure* is a PTV candidate using *mode*.

    mode="name_startswith_ptv"  – name must start with "PTV" (case-insensitive).
    mode="dicom_type_ptv"       – DICOM RTROIInterpretedType == "PTV"
                                  (architecturally reserved, not yet active).
    """
    if mode == "name_startswith_ptv":
        return structure.name.strip().upper().startswith("PTV")
    elif mode == "dicom_type_ptv":
        # TODO: activate when needed; currently not used
        return structure.dicom_type.strip().upper() == "PTV"
    else:
        logger.warning("Unknown PTV_DETECTION_MODE '%s' – defaulting to name_startswith_ptv", mode)
        return structure.name.strip().upper().startswith("PTV")


# ---------------------------------------------------------------------------
# Prescription inference from DVH
# ---------------------------------------------------------------------------

def estimate_prescription_from_dvh(
    structure: Structure,
    dose: DoseData,
    percentile: float = 98.0,
) -> Optional[float]:
    """
    Estimate the prescription dose for *structure* by reading D98% from the
    dose grid and rounding to the nearest integer Gy.

    Returns None if insufficient data are available.
    """
    if dose is None or structure is None or not structure.contours:
        return None

    try:
        dose_values = _sample_dose_in_structure(structure, dose)
        if len(dose_values) == 0:
            return None
        d_percentile = float(np.percentile(dose_values, 100.0 - percentile))
        rounded = round(d_percentile)   # nearest integer Gy
        return float(rounded) if rounded > 0 else None
    except Exception as exc:
        logger.warning("Could not estimate prescription for %s: %s", structure.name, exc)
        return None


def _sample_dose_in_structure(
    structure: Structure,
    dose: DoseData,
    max_samples: int = 50_000,
) -> np.ndarray:
    """
    Quick point-in-polygon sampling inside the bounding box of *structure*
    to obtain a representative set of voxel doses.

    Uses a 2 mm grid for speed (sufficient for prescription estimation).
    """
    step = 2.0  # mm – coarse grid adequate for Rx estimation
    pts = _get_bounding_box_points(structure, step)
    if len(pts) == 0:
        return np.array([])

    # Downsample if too many points
    if len(pts) > max_samples:
        idx = np.random.choice(len(pts), max_samples, replace=False)
        pts = pts[idx]

    doses = []
    for p in pts:
        if _point_in_structure_xy(p, structure):
            d = _interpolate_dose(p, dose)
            if d is not None and d > 0:
                doses.append(d)

    return np.array(doses)


def _get_bounding_box_points(structure: Structure, step: float):
    """Return a list of (x,y,z) grid points within the structure bounding box."""
    if not structure.contours:
        return []

    all_pts = np.vstack(structure.contours)
    min_xyz = all_pts.min(axis=0)
    max_xyz = all_pts.max(axis=0)

    xs = np.arange(min_xyz[0], max_xyz[0] + step, step)
    ys = np.arange(min_xyz[1], max_xyz[1] + step, step)
    zs = np.unique(np.round(all_pts[:, 2], 1))   # only slices with contours

    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append((x, y, z))
    return points


def _point_in_structure_xy(point, structure: Structure) -> bool:
    """
    Fast 2-D point-in-polygon test (ray casting) on the nearest z-slice contour.
    """
    px, py, pz = point
    best_contour = None
    best_dz = float("inf")

    for contour in structure.contours:
        z = float(contour[0, 2])
        dz = abs(z - pz)
        if dz < best_dz:
            best_dz = dz
            best_contour = contour

    if best_contour is None or best_dz > 5.0:
        return False

    return _ray_cast_2d(px, py, best_contour[:, 0], best_contour[:, 1])


def _ray_cast_2d(px: float, py: float, xs, ys) -> bool:
    """Standard even-odd rule ray casting."""
    n = len(xs)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _interpolate_dose(point, dose) -> Optional[float]:
    """
    Trilinear interpolation of the dose value at an (x, y, z) point in mm.
    Returns None if the point is outside the dose grid.

    If *dose* has a ``transform_point`` method (i.e. it is a rotation-aware
    ShiftedDose), the point is inverse-transformed before the lookup so that
    the rotated dose is evaluated correctly.
    """
    if getattr(dose, "_has_rotation", False):
        point = dose.transform_point(np.asarray(point))
    x, y, z = point
    ox, oy, oz = dose.origin
    dx, dy, dz = dose.spacing
    nz, ny, nx = dose.dose_grid.shape

    # Convert from mm to voxel indices (continuous)
    ix = (x - ox) / dx
    iy = (y - oy) / dy
    iz = (z - oz) / dz

    if ix < 0 or iy < 0 or iz < 0 or ix >= nx - 1 or iy >= ny - 1 or iz >= nz - 1:
        return None

    x0, y0, z0 = int(ix), int(iy), int(iz)
    xd, yd, zd = ix - x0, iy - y0, iz - z0

    g = dose.dose_grid
    try:
        c000 = g[z0,     y0,     x0    ]
        c100 = g[z0,     y0,     x0 + 1]
        c010 = g[z0,     y0 + 1, x0    ]
        c110 = g[z0,     y0 + 1, x0 + 1]
        c001 = g[z0 + 1, y0,     x0    ]
        c101 = g[z0 + 1, y0,     x0 + 1]
        c011 = g[z0 + 1, y0 + 1, x0    ]
        c111 = g[z0 + 1, y0 + 1, x0 + 1]
    except IndexError:
        return None

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    return float(c0 * (1 - zd) + c1 * zd)


# ---------------------------------------------------------------------------
# PTV naming / ordering
# ---------------------------------------------------------------------------

# Ordering rule (documented for reproducibility):
#   Primary   – ascending StructureName_Original (alphabetical)
#   Secondary – descending Prescription_Gy_reference (higher dose first)
#   Tertiary  – ascending volume_cc (smaller volume first)

def build_new_structure_names(rows: list) -> list:
    """
    Assign NewStructureName values to a list of mapping-table row dicts
    (for a single patient) in a reproducible order.

    Modifies rows in-place and returns them.

    Rows that are excluded or have no dose reference are left with an
    empty NewStructureName or a placeholder.
    
    NOTE: Counter starts at 1 for each patient (not global).
    """
    active_rows = [r for r in rows if not _is_excluded(r)]
    active_rows.sort(key=_sort_key)

    counter = 1  # Restart at 1 per patient
    for row in active_rows:
        rx = _get_rx(row)
        if rx is not None:
            row["NewStructureName"] = f"PTV{counter:02d}_{int(rx)}Gy"
            counter += 1  # only advance when a name is actually assigned
        else:
            row["NewStructureName"] = ""
            if not row.get("Comment"):
                row["Comment"] = "TODO: prescription Gy could not be determined"

    return rows


def _sort_key(row: dict):
    rx = _get_rx(row)
    vol = float(row.get("Volume_cc", 0) or 0)
    name = str(row.get("StructureName_Original", ""))
    return (
        name,                               # alphabetical (primary)
        -(rx if rx is not None else -1),   # descending Rx (secondary)
        vol,                                # ascending volume (tertiary)
    )


def _get_rx(row: dict) -> Optional[float]:
    ref = row.get("Prescription_Gy_reference")
    det = row.get("Prescription_Gy_detected")
    if _valid_float(ref):
        return float(ref)
    if _valid_float(det):
        return float(det)
    return None


def _valid_float(v) -> bool:
    if v is None:
        return False
    try:
        f = float(v)
        return not (f != f) and f > 0  # NaN check + positive
    except (TypeError, ValueError):
        return False


def _is_excluded(row: dict) -> bool:
    """Robust interpretation of the ExcludeFromAnalysis cell value."""
    val = row.get("ExcludeFromAnalysis", False)
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"true", "1", "ja", "yes", "x"}


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def compute_centroid(structure: Structure) -> Optional[np.ndarray]:
    """
    Return the (x, y, z) centroid of all contour points as a 1-D numpy array.
    Returns None if the structure has no contour data.
    """
    if not structure.contours:
        return None
    try:
        all_pts = np.vstack(structure.contours)   # (N, 3)
        return all_pts.mean(axis=0)               # (3,)
    except Exception:
        return None
