"""
SRS Analysis Pipeline - Metrics
=================================
Implements the SRS quality metrics as defined in SRS_Metrics_v1.0.0RC1.html
(CalculationEngine class).

Metrics returned per structure / plan combination:
    TV          – target volume (cc) inside the structure contours
    coverage    – TV_at_Rx / TV  (fraction, i.e. V100%)
    paddickCI   – Paddick Conformity Index = TV_at_Rx² / (TV × PIV)
    rtogCI      – RTOG Conformity Index = PIV / TV
    HI          – Homogeneity Index = (D2 - D98) / D50
    GI          – Gradient Index = V_half_Rx / PIV
    Dmax        – Maximum dose in structure (Gy)
    PIV         – Prescription Isodose Volume (cc), volume ≥ Rx dose
    V12Gy       – Volume ≥ 12 Gy (cc)
    D2, D98, D50– Dose at 2 %, 98 %, 50 % of TV (Gy)
    effectiveDiameter – effective diameter from TV (mm)
    finalGrid   – sampling grid step used (mm)

TODO (verify against SRS_Metrics_v1.0.0RC1.html):
    - Clipping planes logic: the HTML app clips overlapping structures along a
      midplane to avoid double-counting.  In this batch pipeline each structure
      is treated independently (no clipping).  If inter-lesion dose overlap
      matters for multi-metastasis patients, this may need revisiting.
    - The HTML uses a 1 mm grid for targets ≥ 10 mm effective diameter and
      progressively refines down to 0.25 mm for smaller targets.  This code
      mirrors that strategy but with a fast 2 mm coarse pass for the
      prescription estimation step.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

from config import (
    GRID_STEP_SIZES, LARGE_TARGET_DIAMETER_MM,
    GRID_CONVERGENCE_TOL, V12GY_THRESHOLD,
    BOUNDARY_MARGIN_CI_MM, BOUNDARY_MARGIN_GI_MM,
    BOUNDARY_EXPAND_STEP_MM,
    BOUNDARY_CLOSURE_DELTA_MM, PIV_CLOSURE_TOL,
    BRIDGING_CI_THRESHOLD, BRIDGING_SMALL_MARGIN_MM,
    BRIDGING_LARGE_MARGIN_MM, BRIDGING_PIV_GROWTH_TOL,
)
from dicom_io import Structure, DoseData
from structure_mapping import _point_in_structure_xy, _interpolate_dose, compute_centroid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    patient_folder:    str = ""
    patient_id:        str = ""
    plan_type:         str = ""
    plan_label:        str = ""
    structure_name:    str = ""
    new_structure_name: str = ""
    scenario:          str = "nominal"

    TV:               float = float("nan")
    coverage:         float = float("nan")
    paddickCI:        float = float("nan")
    rtogCI:           float = float("nan")
    HI:               float = float("nan")
    GI:               float = float("nan")
    Dmax:             float = float("nan")
    PIV:              float = float("nan")
    V12Gy:            float = float("nan")
    D2:               float = float("nan")
    D98:              float = float("nan")
    D50:              float = float("nan")
    effectiveDiameter: float = float("nan")
    finalGrid:        float = float("nan")
    rx_dose:          float = float("nan")
    DistToIso_mm:     float = float("nan")
    ci_uncertain:     bool  = False   # True when 100%-Rx isodose hits sampling boundary
    gi_uncertain:     bool  = False   # True when 50%-Rx isodose hits sampling boundary
    bridging_suspected: bool = False  # True when PIV grows with small radius increase (adjacent dose)

    error_message:    str = ""


# ---------------------------------------------------------------------------
# Core sampling routine
# ---------------------------------------------------------------------------

def _calculate_metrics_for_step(
    structure: Structure,
    dose: DoseData,
    rx_dose: float,
    step: float,
    extra_margin: float = 0.0,
    radial_mask: bool = False,
    progress_cb=None,
) -> dict:
    """
    Monte-Carlo grid sampling.

    Points are on a regular grid with spacing *step* (mm).
    PIV and V12Gy use ALL grid points (regardless of structure membership).
    TV, Dmax, D2/D50/D98 use only points INSIDE the structure.

    *extra_margin* expands the bounding box on all sides (mm). Extra z-planes
    are added above and below the contour stack with *step* spacing so that
    PIV / V_half_Rx outside the PTV z-range are also captured.

    *radial_mask* – when True, only points within *extra_margin* mm from the
    SURFACE of the original (0-margin) bounding box are included in the
    expansion zone.  This prevents corners of the rectangular box from
    picking up dose from neighbouring structures.

    Returns a dict of raw metric values.
    """
    if not structure.contours:
        return _empty_metrics(step)

    # Bounding box of structure (+ optional expansion)
    all_pts = np.vstack(structure.contours)
    struct_min = all_pts.min(axis=0)   # original bbox (before expansion)
    struct_max = all_pts.max(axis=0)
    min_xyz = struct_min - extra_margin
    max_xyz = struct_max + extra_margin

    xs = np.arange(min_xyz[0], max_xyz[0] + step * 0.5, step)
    ys = np.arange(min_xyz[1], max_xyz[1] + step * 0.5, step)

    # z: regular step-spaced grid across the (optionally expanded) z-range.
    # sample_vol = step³ is only correct when ALL three axes use the same step.
    # _point_in_structure_xy finds the nearest contour (within 5 mm) for the 2-D
    # membership test, so intermediate z-planes are handled correctly.
    zs = np.arange(min_xyz[2], max_xyz[2] + step * 0.5, step)

    # Pre-compute radial mask: spherical dilation of original bbox.
    # Points in the expansion zone are only included if their Euclidean
    # distance to the nearest face/edge/corner of the struct bbox ≤ extra_margin.
    # This prevents rectangular corners from capturing neighbouring structure dose.
    if radial_mask and extra_margin > 0:
        X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing='ij')
        dx = np.maximum(struct_min[0] - X3, np.maximum(0.0, X3 - struct_max[0]))
        dy = np.maximum(struct_min[1] - Y3, np.maximum(0.0, Y3 - struct_max[1]))
        dz = np.maximum(struct_min[2] - Z3, np.maximum(0.0, Z3 - struct_max[2]))
        _valid = (dx * dx + dy * dy + dz * dz) <= (extra_margin + 0.5 * step) ** 2
        del X3, Y3, Z3, dx, dy, dz
    else:
        _valid = None

    sample_vol = (step ** 3) / 1000.0  # mm³ → cc

    histogram: dict = {}   # dose_bin (0.1 Gy) → volume (cc) inside structure
    PIV = 0.0
    V12 = 0.0
    V_half_Rx = 0.0
    Dmax = 0.0
    total_pts = len(xs) * len(ys) * len(zs)
    done = 0
    boundary_max_dose = 0.0   # max dose observed at the bounding-box faces

    ix_last = len(xs) - 1
    iy_last = len(ys) - 1
    iz_last = len(zs) - 1

    for iz, z in enumerate(zs):
        is_z_bnd = (iz == 0 or iz == iz_last)
        for iy, y in enumerate(ys):
            is_y_bnd = (iy == 0 or iy == iy_last)
            for ix, x in enumerate(xs):
                # Radial mask: skip points outside spherical dilation of original bbox
                if _valid is not None and not _valid[ix, iy, iz]:
                    done += 1
                    continue

                dose_val = _interpolate_dose((x, y, z), dose)
                if dose_val is None:
                    done += 1
                    continue

                # Track max dose at bounding-box boundary
                if is_z_bnd or is_y_bnd or ix == 0 or ix == ix_last:
                    if dose_val > boundary_max_dose:
                        boundary_max_dose = dose_val

                if dose_val >= rx_dose:
                    PIV += sample_vol
                if dose_val >= V12GY_THRESHOLD:
                    V12 += sample_vol
                if dose_val >= rx_dose * 0.5:
                    V_half_Rx += sample_vol

                # Inside-structure bookkeeping
                if _point_in_structure_xy((x, y, z), structure):
                    if dose_val > Dmax:
                        Dmax = dose_val
                    bin_key = round(dose_val * 10) / 10
                    histogram[bin_key] = histogram.get(bin_key, 0.0) + sample_vol

                done += 1

        if progress_cb and total_pts > 0:
            progress_cb(done / total_pts)

    # Derive DVH statistics
    sorted_bins = sorted(histogram.keys(), reverse=True)
    TV = sum(histogram[b] for b in sorted_bins)

    def dose_at_vol_pct(pct: float) -> float:
        if TV == 0:
            return 0.0
        target = TV * (pct / 100.0)
        acc = 0.0
        for b in sorted_bins:
            acc += histogram[b]
            if acc >= target:
                return b
        return 0.0

    def vol_at_dose(thr: float) -> float:
        acc = 0.0
        for b in sorted_bins:
            if b >= thr:
                acc += histogram[b]
            else:
                break
        return acc

    TV_at_Rx = vol_at_dose(rx_dose)
    D2  = dose_at_vol_pct(2.0)
    D98 = dose_at_vol_pct(98.0)
    D50 = dose_at_vol_pct(50.0)

    eff_diam = 2.0 * ((3.0 * TV * 1000.0) / (4.0 * np.pi)) ** (1.0 / 3.0) if TV > 0 else 0.0
    coverage  = TV_at_Rx / TV if TV > 0 else 0.0
    paddickCI = (TV_at_Rx ** 2) / (TV * PIV) if (TV > 0 and PIV > 0) else 0.0
    rtogCI    = PIV / TV if TV > 0 else 0.0
    HI        = (D2 - D98) / D50 if D50 > 0 else 0.0
    GI        = V_half_Rx / PIV if PIV > 0 else 0.0

    ci_uncertain = boundary_max_dose >= rx_dose
    gi_uncertain = boundary_max_dose >= rx_dose * 0.5
    boundary_max_dose_out = boundary_max_dose  # expose for callers

    return {
        "TV": TV, "Dmax": Dmax,
        "coverage": coverage,
        "paddickCI": paddickCI,
        "rtogCI": rtogCI,
        "HI": HI,
        "GI": GI,
        "PIV": PIV,
        "V12Gy": V12,
        "D2": D2, "D98": D98, "D50": D50,
        "effectiveDiameter": eff_diam,
        "finalGrid": step,
        "ci_uncertain": ci_uncertain,
        "gi_uncertain": gi_uncertain,
        "_boundary_max_dose": boundary_max_dose_out,
        "_V_half_Rx": V_half_Rx,
    }


def _empty_metrics(step: float) -> dict:
    keys = ["TV", "Dmax", "coverage", "paddickCI", "rtogCI", "HI", "GI",
            "PIV", "V12Gy", "D2", "D98", "D50", "effectiveDiameter", "finalGrid"]
    d = {k: float("nan") for k in keys}
    d["finalGrid"] = step
    d["ci_uncertain"] = False
    d["gi_uncertain"] = False
    d["bridging_suspected"] = False
    d["_boundary_max_dose"] = 0.0
    d["_V_half_Rx"] = 0.0
    return d



def _merge_metrics(fine: dict, expanded: dict) -> dict:
    """
    Merge fine-grid structural metrics with expanded-grid volume metrics.

    Structure-based metrics (TV, Dmax, D2/D50/D98, coverage, HI,
    effectiveDiameter, finalGrid) come from the fine-grid run.
    Global volume metrics (PIV, V_half_Rx, V12Gy) come from the
    expanded coarse-grid run so that isodoses outside the structure
    bounding box are properly captured.
    CI and GI are recomputed from the merged TV + expanded PIV/V_half_Rx.
    """
    TV       = fine["TV"]
    coverage = fine["coverage"]
    TV_at_Rx = coverage * TV if TV > 0 else 0.0

    PIV       = expanded["PIV"]
    V_half_Rx = expanded["_V_half_Rx"]

    new_paddickCI = (TV_at_Rx ** 2) / (TV * PIV) if (TV > 0 and PIV > 0) else 0.0
    new_rtogCI    = PIV / TV if TV > 0 else 0.0
    new_GI        = V_half_Rx / PIV if PIV > 0 else 0.0

    result = dict(fine)
    result["PIV"]       = PIV
    result["V12Gy"]     = expanded["V12Gy"]
    result["paddickCI"] = new_paddickCI
    result["rtogCI"]    = new_rtogCI
    result["GI"]        = new_GI
    result["ci_uncertain"] = expanded["ci_uncertain"]
    result["gi_uncertain"] = expanded["gi_uncertain"]
    result["bridging_suspected"] = expanded.get("bridging_suspected", False)
    return result


# ---------------------------------------------------------------------------
# Adaptive grid (mirrors calculateAllMetrics in the HTML)
# ---------------------------------------------------------------------------

def calculate_all_metrics(
    structure: Structure,
    dose: DoseData,
    rx_dose: float,
    progress_cb=None,
) -> dict:
    """
    Compute metrics with adaptive grid refinement + fixed-margin boundary scan.

    Strategy:
    1. Fine adaptive grid at 0 mm margin → TV, Dmax, D2/D50/D98, coverage.
    2. Coarse 1 mm grid at +BOUNDARY_FIXED_MARGIN_MM → PIV, V_half_Rx, V12Gy.
       ci_uncertain / gi_uncertain reflect whether the relevant isodose touches
       the boundary of the 15 mm-expanded box (not the structure edge).
    3. Merge: recompute CI and GI from fine TV and expanded PIV/V_half_Rx.
    """
    if dose is None:
        logger.warning("No dose data for structure %s – skipping.", structure.name)
        return _empty_metrics(float("nan"))

    # --- Pass 1: fine adaptive grid at 0 mm margin (structural metrics) ---
    initial = _calculate_metrics_for_step(structure, dose, rx_dose, 1.0)

    if initial["effectiveDiameter"] >= LARGE_TARGET_DIAMETER_MM:
        logger.debug(
            "    %s: diam=%.1f mm >= %.0f mm → 1.0 mm grid",
            structure.name, initial["effectiveDiameter"], LARGE_TARGET_DIAMETER_MM,
        )
        fine_metrics = initial
    else:
        prev_vol = -1.0
        fine_metrics = initial
        for step in GRID_STEP_SIZES:
            m = _calculate_metrics_for_step(structure, dose, rx_dose, step,
                                            progress_cb=progress_cb)
            cur_vol = m["TV"]
            if prev_vol >= 0:
                diff = abs(cur_vol - prev_vol) / prev_vol if prev_vol > 0 else 0.0
                if diff < GRID_CONVERGENCE_TOL:
                    logger.debug(
                        "    %s: converged at %.2f mm grid (\u0394Vol=%.1f%%) \u2192 keeping prev step",
                        structure.name, step, diff * 100,
                    )
                    break  # fine_metrics stays at previous-iteration result
            prev_vol = cur_vol
            fine_metrics = m

    # --- Pass 2a: CI expansion – radial 15 mm, 1 mm grid → PIV (100 % Rx) ---
    ci_metrics = _calculate_metrics_for_step(
        structure, dose, rx_dose,
        step=BOUNDARY_EXPAND_STEP_MM,
        extra_margin=BOUNDARY_MARGIN_CI_MM,
        radial_mask=True,
    )

    # --- Pass 2b: CI closure check (15 mm → 17 mm radial) ---
    ci_check = _calculate_metrics_for_step(
        structure, dose, rx_dose,
        step=BOUNDARY_EXPAND_STEP_MM,
        extra_margin=BOUNDARY_MARGIN_CI_MM + BOUNDARY_CLOSURE_DELTA_MM,
        radial_mask=True,
    )
    piv_a  = ci_metrics["PIV"]
    piv_b  = ci_check["PIV"]
    ci_unc = (abs(piv_b - piv_a) / piv_a > PIV_CLOSURE_TOL) if piv_a > 0 else False

    # --- Pass 3a: GI expansion – radial 20 mm, 1 mm grid → V_half_Rx (50 % Rx) ---
    gi_metrics = _calculate_metrics_for_step(
        structure, dose, rx_dose,
        step=BOUNDARY_EXPAND_STEP_MM,
        extra_margin=BOUNDARY_MARGIN_GI_MM,
        radial_mask=True,
    )

    # --- Pass 3b: GI closure check (20 mm → 22 mm radial) ---
    gi_check = _calculate_metrics_for_step(
        structure, dose, rx_dose,
        step=BOUNDARY_EXPAND_STEP_MM,
        extra_margin=BOUNDARY_MARGIN_GI_MM + BOUNDARY_CLOSURE_DELTA_MM,
        radial_mask=True,
    )
    vhalf_a = gi_metrics["_V_half_Rx"]
    vhalf_b = gi_check["_V_half_Rx"]
    gi_unc  = (abs(vhalf_b - vhalf_a) / vhalf_a > PIV_CLOSURE_TOL) if vhalf_a > 0 else False

    logger.debug(
        "    %s: PIV %.4f\u2192%.4f (%.1f%%) Vhalf %.4f\u2192%.4f (%.1f%%) \u2192 ci_unc=%s gi_unc=%s",
        structure.name,
        piv_a, piv_b, 100 * (piv_b - piv_a) / max(piv_a, 1e-9),
        vhalf_a, vhalf_b, 100 * (vhalf_b - vhalf_a) / max(vhalf_a, 1e-9),
        ci_unc, gi_unc,
    )

    # --- Pass 4: Bridging detection ---
    # Build a merged result using CI-margin PIV so we can compute a preliminary CI
    tv = fine_metrics.get("TV", 0.0) or 0.0
    tv_coverage = fine_metrics.get("coverage", 0.0) or 0.0
    tv_at_rx = tv_coverage * tv
    rough_ci = (tv_at_rx ** 2) / (tv * piv_a) if (tv > 0 and piv_a > 0) else 1.0

    bridging_suspected = False
    if rough_ci < BRIDGING_CI_THRESHOLD and piv_a > 0:
        m_small = _calculate_metrics_for_step(
            structure, dose, rx_dose,
            step=BOUNDARY_EXPAND_STEP_MM,
            extra_margin=BRIDGING_SMALL_MARGIN_MM,
            radial_mask=True,
        )
        m_large = _calculate_metrics_for_step(
            structure, dose, rx_dose,
            step=BOUNDARY_EXPAND_STEP_MM,
            extra_margin=BRIDGING_LARGE_MARGIN_MM,
            radial_mask=True,
        )
        piv_small = m_small["PIV"]
        piv_large = m_large["PIV"]
        if piv_small > 0 and (piv_large - piv_small) / piv_small > BRIDGING_PIV_GROWTH_TOL:
            bridging_suspected = True
            logger.info(
                "    %s: BRIDGING suspected – PIV %.4f (r=%gmm) → %.4f (r=%gmm) = +%.0f%%",
                structure.name, piv_small, BRIDGING_SMALL_MARGIN_MM,
                piv_large, BRIDGING_LARGE_MARGIN_MM,
                100 * (piv_large - piv_small) / piv_small,
            )

    # --- Merge: use CI-margin PIV for CI metrics, GI-margin V_half_Rx for GI ---
    # Build a combined vol_metrics dict: PIV from CI pass, V_half_Rx from GI pass
    vol_metrics = dict(ci_metrics)                           # base from CI pass
    vol_metrics["_V_half_Rx"] = gi_metrics["_V_half_Rx"]    # replace with GI pass value
    vol_metrics["V12Gy"] = max(ci_metrics["V12Gy"], gi_metrics["V12Gy"])  # take larger
    vol_metrics["ci_uncertain"] = ci_unc
    vol_metrics["gi_uncertain"] = gi_unc
    vol_metrics["bridging_suspected"] = bridging_suspected

    return _merge_metrics(fine_metrics, vol_metrics)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_metrics_for_plan(
    plan_files,         # PlanFiles
    structures: list,   # List[Structure]
    dose: DoseData,
    plan_meta,          # Optional[PlanMeta]
    active_rows,        # DataFrame rows with NewStructureName and rx info
    scenario: str = "nominal",
    dose_override: DoseData = None,
    isocenter: Optional[np.ndarray] = None,
) -> List[MetricResult]:
    """
    Compute metrics for all active (non-excluded) PTV structures of one plan.

    *active_rows* is a pandas DataFrame slice from the mapping table that
    contains the non-excluded PTV candidates for this patient / plan.
    *dose_override* allows passing a shifted dose grid for scenario analysis.
    """
    results = []
    used_dose = dose_override if dose_override is not None else dose

    if used_dose is None:
        logger.warning(
            "No dose available for %s/%s – skipping metric computation.",
            plan_files.patient_folder, plan_files.plan_type,
        )
        return results

    patient_id  = plan_meta.patient_id  if plan_meta else "Unknown"
    plan_label  = plan_meta.plan_label  if plan_meta else "Unknown"

    # Build look-up from original structure name → row info
    name_to_row = {}
    if active_rows is not None and not active_rows.empty:
        for _, row in active_rows.iterrows():
            orig = str(row.get("StructureName_Original", "")).strip()
            name_to_row[orig] = row

    # Match loaded structures to active rows
    struct_map = {s.name: s for s in structures}

    for orig_name, row in name_to_row.items():
        struct = struct_map.get(orig_name)
        if struct is None:
            logger.warning(
                "Structure '%s' not found in loaded RS for %s/%s – skipping.",
                orig_name, plan_files.patient_folder, plan_files.plan_type,
            )
            continue

        # Determine Rx dose
        rx = _resolve_rx(row)
        if rx is None or rx <= 0:
            logger.warning(
                "No valid Rx dose for %s/%s/%s – skipping metrics.",
                plan_files.patient_folder, plan_files.plan_type, orig_name,
            )
            result = MetricResult(
                patient_folder=plan_files.patient_folder,
                patient_id=patient_id,
                plan_type=plan_files.plan_type,
                plan_label=plan_label,
                structure_name=orig_name,
                new_structure_name=str(row.get("NewStructureName", "")),
                scenario=scenario,
                error_message="No valid Rx dose – metrics not computed",
            )
            results.append(result)
            continue

        logger.info(
            "  Computing metrics: %s / %s / %s  Rx=%.0f Gy  [%s]",
            plan_files.patient_folder, plan_files.plan_type, orig_name, rx, scenario,
        )

        try:
            m = calculate_all_metrics(struct, used_dose, rx)
            centroid = compute_centroid(struct)
            if centroid is not None and isocenter is not None:
                dist_iso = float(np.linalg.norm(centroid - isocenter))
            else:
                dist_iso = float("nan")
            result = MetricResult(
                patient_folder=plan_files.patient_folder,
                patient_id=patient_id,
                plan_type=plan_files.plan_type,
                plan_label=plan_label,
                structure_name=orig_name,
                new_structure_name=str(row.get("NewStructureName", "")),
                scenario=scenario,
                rx_dose=rx,
                DistToIso_mm=dist_iso,
                ci_uncertain=bool(m.get("ci_uncertain", False)),
                gi_uncertain=bool(m.get("gi_uncertain", False)),
                bridging_suspected=bool(m.get("bridging_suspected", False)),
                **{k: m[k] for k in [
                    "TV", "coverage", "paddickCI", "rtogCI", "HI", "GI",
                    "Dmax", "PIV", "V12Gy", "D2", "D98", "D50",
                    "effectiveDiameter", "finalGrid",
                ]},
            )
        except Exception as exc:
            logger.error(
                "Error computing metrics for %s/%s/%s: %s",
                plan_files.patient_folder, plan_files.plan_type, orig_name, exc,
            )
            result = MetricResult(
                patient_folder=plan_files.patient_folder,
                patient_id=patient_id,
                plan_type=plan_files.plan_type,
                plan_label=plan_label,
                structure_name=orig_name,
                new_structure_name=str(row.get("NewStructureName", "")),
                scenario=scenario,
                error_message=str(exc),
            )

        results.append(result)

    return results


def _resolve_rx(row) -> Optional[float]:
    """Prefer Prescription_Gy_reference, fall back to Prescription_Gy_detected."""
    def try_float(val):
        try:
            v = float(val)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    ref = try_float(row.get("Prescription_Gy_reference"))
    if ref:
        return ref
    return try_float(row.get("Prescription_Gy_detected"))


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def results_to_dataframe(results: List[MetricResult]):
    """Convert a list of MetricResult to a pandas DataFrame."""
    import pandas as pd
    from config import ANONYMIZE_OUTPUT
    from anonymization import anonymize_patient_id, anonymize_structure_name

    records = []
    for r in results:
        # Apply anonymization if enabled
        patient_folder = anonymize_patient_id(r.patient_folder) if ANONYMIZE_OUTPUT else r.patient_folder
        patient_id = anonymize_patient_id(r.patient_id) if ANONYMIZE_OUTPUT else r.patient_id
        struct_name = anonymize_structure_name(r.structure_name) if ANONYMIZE_OUTPUT else r.structure_name
        new_struct_name = anonymize_structure_name(r.new_structure_name) if ANONYMIZE_OUTPUT and r.new_structure_name else r.new_structure_name
        
        records.append({
            "PatientFolder":       patient_folder,
            "PatientID":           patient_id,
            "PlanType":            r.plan_type,
            "PlanLabel":           r.plan_label,
            "StructureName_Original": struct_name,
            "NewStructureName":    new_struct_name,
            "Scenario":            r.scenario,
            "Rx_Gy":               r.rx_dose,
            "TV_cc":               r.TV,
            "Coverage_pct":        r.coverage * 100 if not _isnan(r.coverage) else float("nan"),
            "PaddickCI":           r.paddickCI,
            "RTOG_CI":             r.rtogCI,
            "HI":                  r.HI,
            "GI":                  r.GI,
            "Dmax_Gy":             r.Dmax,
            "PIV_cc":              r.PIV,
            "V12Gy_cc":            r.V12Gy,
            "D2_Gy":               r.D2,
            "D98_Gy":              r.D98,
            "D50_Gy":              r.D50,
            "EffDiameter_mm":      r.effectiveDiameter,
            "FinalGrid_mm":        r.finalGrid,
            "DistToIso_mm":        r.DistToIso_mm,
            "CI_uncertain":        r.ci_uncertain,
            "GI_uncertain":        r.gi_uncertain,
            "Bridging_suspected":  r.bridging_suspected,
            "Error":               r.error_message,
        })
    return pd.DataFrame(records)


def _isnan(v) -> bool:
    try:
        import math
        return math.isnan(v)
    except (TypeError, ValueError):
        return True
