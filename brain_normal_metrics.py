"""
Brain / normal-brain dose-volume metrics.

This module adds plan-level and lesion-local VxGy calculations for:
  - brain
  - brain minus PTV
  - brain minus GTV

It intentionally works directly on the RTDOSE grid and RTSTRUCT contours.
No CT image stack is required.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from matplotlib.path import Path as PolygonPath

from dicom_io import DoseData, Structure


@dataclass
class BrainSelection:
    structure: Structure
    source: str
    is_external_fallback: bool


NEGATIVE_BRAIN_NAME_TERMS = (
    "minus",
    "brainstem",
    "hippocampus",
    "optic",
    "chiasm",
    "pituitary",
    "eye",
    "lens",
    "cochlea",
)


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def select_brain_structure(
    structures: list[Structure],
    allow_external_fallback: bool = True,
) -> BrainSelection | None:
    """
    Select a brain contour.

    Preferred names are Whole Brain / Brain / Hirn. If none exists and
    allow_external_fallback is true, EXTERNAL/BODY is used as a pragmatic
    fallback for cranial SRS exports.
    """
    candidates: list[tuple[int, Structure, str, bool]] = []
    for structure in structures:
        name = normalize_name(structure.name)
        dicom_type = str(structure.dicom_type or "").strip().upper()
        if any(term in name for term in NEGATIVE_BRAIN_NAME_TERMS):
            continue

        score = -1
        source = ""
        fallback = False
        if name in {"whole brain_full", "whole brain full"}:
            score = 100
            source = "whole brain"
        elif "whole brain" in name:
            score = 90
            source = "whole brain"
        elif name in {"brain", "hirn"}:
            score = 80
            source = "brain"
        elif "brain" in name or "hirn" in name:
            score = 70
            source = "brain-like structure"
        elif allow_external_fallback and (dicom_type == "EXTERNAL" or name in {"external", "body"}):
            score = 10
            source = "external/body fallback"
            fallback = True

        if score >= 0:
            candidates.append((score, structure, source, fallback))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1].name))
    _, structure, source, fallback = candidates[0]
    return BrainSelection(structure=structure, source=source, is_external_fallback=fallback)


def dose_grid_is_axial_identity(dose: DoseData) -> bool:
    orientation = np.asarray(dose.image_orientation, dtype=float)
    return np.allclose(orientation, [1, 0, 0, 0, 1, 0], atol=1e-3)


def voxel_volume_cc(dose: DoseData) -> float:
    return float(np.prod(np.asarray(dose.spacing, dtype=float)) / 1000.0)


def mask_volume_cc(mask: np.ndarray, dose: DoseData) -> float:
    return float(np.count_nonzero(mask) * voxel_volume_cc(dose))


def structure_mask_on_dose_grid(structure: Structure, dose: DoseData) -> np.ndarray:
    """
    Rasterize RTSTRUCT contours on the RTDOSE grid.

    This follows the geometric assumptions used elsewhere in this repository:
    axial dose grid, identity row/column orientation, nearest dose slice for
    each contour. CT data are not used.
    """
    if not dose_grid_is_axial_identity(dose):
        raise ValueError("Only axial identity-oriented RTDOSE grids are supported.")

    nz, ny, nx = dose.dose_grid.shape
    mask = np.zeros((nz, ny, nx), dtype=bool)
    if not structure.contours:
        return mask

    ox, oy, oz = np.asarray(dose.origin, dtype=float)
    dx, dy, dz = np.asarray(dose.spacing, dtype=float)
    z_tol = max(abs(dz) / 2.0 + 1e-3, 0.75)

    for contour in structure.contours:
        if contour.shape[0] < 3:
            continue

        z_mean = float(np.mean(contour[:, 2]))
        iz = int(round((z_mean - oz) / dz))
        if iz < 0 or iz >= nz:
            continue
        if abs(z_mean - (oz + iz * dz)) > z_tol:
            continue

        min_x, min_y = contour[:, :2].min(axis=0)
        max_x, max_y = contour[:, :2].max(axis=0)
        ix0 = max(0, int(math.floor((min_x - ox) / dx)))
        ix1 = min(nx - 1, int(math.ceil((max_x - ox) / dx)))
        iy0 = max(0, int(math.floor((min_y - oy) / dy)))
        iy1 = min(ny - 1, int(math.ceil((max_y - oy) / dy)))
        if ix1 < ix0 or iy1 < iy0:
            continue

        xs = ox + np.arange(ix0, ix1 + 1) * dx
        ys = oy + np.arange(iy0, iy1 + 1) * dy
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        points = np.column_stack([xx.ravel(), yy.ravel()])
        inside = PolygonPath(contour[:, :2]).contains_points(points).reshape((len(ys), len(xs)))
        mask[iz, iy0 : iy1 + 1, ix0 : ix1 + 1] |= inside

    return mask


def volume_at_threshold_cc(mask: np.ndarray, dose: DoseData, threshold_gy: float) -> float:
    return float(np.count_nonzero(mask & (dose.dose_grid >= threshold_gy)) * voxel_volume_cc(dose))


def dose_at_volume_percent(dose_values: np.ndarray, pct: float) -> float:
    if dose_values.size == 0:
        return float("nan")
    return float(np.percentile(dose_values, 100.0 - pct))


def rounded_d99_prescription_gy(ptv_mask: np.ndarray, dose: DoseData) -> float:
    d99 = dose_at_volume_percent(dose.dose_grid[ptv_mask], 99.0)
    if not np.isfinite(d99) or d99 <= 0:
        return float("nan")
    return float(round(d99))


def local_region_mask(structure: Structure, dose: DoseData, margin_mm: float) -> np.ndarray:
    """
    Radial expansion of the structure bounding box on the RTDOSE grid.
    """
    nz, ny, nx = dose.dose_grid.shape
    out = np.zeros((nz, ny, nx), dtype=bool)
    if not structure.contours:
        return out

    pts = np.vstack(structure.contours)
    struct_min = pts.min(axis=0)
    struct_max = pts.max(axis=0)
    min_xyz = struct_min - margin_mm
    max_xyz = struct_max + margin_mm

    ox, oy, oz = np.asarray(dose.origin, dtype=float)
    dx, dy, dz = np.asarray(dose.spacing, dtype=float)
    ix0 = max(0, int(math.floor((min_xyz[0] - ox) / dx)))
    ix1 = min(nx - 1, int(math.ceil((max_xyz[0] - ox) / dx)))
    iy0 = max(0, int(math.floor((min_xyz[1] - oy) / dy)))
    iy1 = min(ny - 1, int(math.ceil((max_xyz[1] - oy) / dy)))
    iz0 = max(0, int(math.floor((min_xyz[2] - oz) / dz)))
    iz1 = min(nz - 1, int(math.ceil((max_xyz[2] - oz) / dz)))
    if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
        return out

    xs = ox + np.arange(ix0, ix1 + 1) * dx
    ys = oy + np.arange(iy0, iy1 + 1) * dy
    zs = oz + np.arange(iz0, iz1 + 1) * dz
    z3, y3, x3 = np.meshgrid(zs, ys, xs, indexing="ij")

    ddx = np.maximum(struct_min[0] - x3, np.maximum(0.0, x3 - struct_max[0]))
    ddy = np.maximum(struct_min[1] - y3, np.maximum(0.0, y3 - struct_max[1]))
    ddz = np.maximum(struct_min[2] - z3, np.maximum(0.0, z3 - struct_max[2]))
    valid = (ddx * ddx + ddy * ddy + ddz * ddz) <= (margin_mm + max(dose.spacing) * 0.5) ** 2
    out[iz0 : iz1 + 1, iy0 : iy1 + 1, ix0 : ix1 + 1] = valid
    return out


def threshold_metric_name(prefix: str, threshold_gy: float, scope: str) -> str:
    value = int(threshold_gy) if float(threshold_gy).is_integer() else threshold_gy
    return f"{prefix}-V{value}Gy_cc_{scope}"


def local_brain_dose_metrics(
    *,
    dose: DoseData,
    brain_mask: np.ndarray,
    ptv_mask: np.ndarray,
    gtv_mask: np.ndarray,
    ptv_structure: Structure,
    thresholds_gy: Iterable[float],
    margin_mm: float,
) -> dict[str, float]:
    region = local_region_mask(ptv_structure, dose, margin_mm) & brain_mask
    brain_minus_ptv = region & ~ptv_mask
    brain_minus_gtv = region & ~gtv_mask

    metrics: dict[str, float] = {"LocalMarginMm": float(margin_mm)}
    for threshold in thresholds_gy:
        metrics[threshold_metric_name("local", threshold, "Brain")] = volume_at_threshold_cc(region, dose, threshold)
        metrics[threshold_metric_name("local", threshold, "BrainMinusPTV")] = volume_at_threshold_cc(brain_minus_ptv, dose, threshold)
        metrics[threshold_metric_name("local", threshold, "BrainMinusGTV")] = volume_at_threshold_cc(brain_minus_gtv, dose, threshold)
    return metrics


def global_brain_dose_metrics(
    *,
    dose: DoseData,
    brain_mask: np.ndarray,
    ptv_union_mask: np.ndarray,
    gtv_union_mask: np.ndarray,
    thresholds_gy: Iterable[float],
) -> dict[str, float]:
    brain_minus_ptv = brain_mask & ~ptv_union_mask
    brain_minus_gtv = brain_mask & ~gtv_union_mask

    metrics = {
        "BrainVolumeCc": mask_volume_cc(brain_mask, dose),
        "PTVUnionVolumeCc": mask_volume_cc(ptv_union_mask, dose),
        "GTVUnionVolumeCc": mask_volume_cc(gtv_union_mask, dose),
    }
    for threshold in thresholds_gy:
        metrics[threshold_metric_name("global", threshold, "Brain")] = volume_at_threshold_cc(brain_mask, dose, threshold)
        metrics[threshold_metric_name("global", threshold, "BrainMinusPTV")] = volume_at_threshold_cc(brain_minus_ptv, dose, threshold)
        metrics[threshold_metric_name("global", threshold, "BrainMinusGTV")] = volume_at_threshold_cc(brain_minus_gtv, dose, threshold)
    return metrics


def numbers_after_prefix(name: str, prefix: str) -> list[int]:
    pattern = rf"\b{re.escape(prefix)}\s*_?\s*(\d+(?:\s*[+\-/]\s*\d+)*)"
    match = re.search(pattern, name, flags=re.IGNORECASE)
    if not match:
        return []
    return [int(v) for v in re.findall(r"\d+", match.group(1))]


def active_numbers_from_plan_label(plan_label: str) -> set[int]:
    text = str(plan_label or "")
    active: set[int] = set()
    for match in re.finditer(r"PTV\s*_?\s*(\d+)\s*-\s*(\d+)", text, flags=re.IGNORECASE):
        start, end = int(match.group(1)), int(match.group(2))
        active.update(range(min(start, end), max(start, end) + 1))
    for match in re.finditer(r"PTV\s*_?\s*(\d+(?:\s*\+\s*\d+)*)", text, flags=re.IGNORECASE):
        active.update(int(v) for v in re.findall(r"\d+", match.group(1)))
    return active

