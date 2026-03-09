"""
SRS Analysis Pipeline - DICOM Export
======================================
Writes modified RD + RP files with new SOPInstanceUIDs into a separate folder
so they can be imported into a TPS alongside the original CT and structure set.

One RD+RP pair is written per scenario:
  - nominal:  <Patient>_<Plan>_nominal/
  - shifted:  <Patient>_<Plan>_<encoded_shift>/
    e.g.      zz_Paper_1_HA_dx1_dy0_dz1_rx0_ry1_rz0/

For translation-only scenarios the dose origin is shifted exactly.
Rotations are not yet re-sampled (TODO) – a warning is logged.

Guaranteed invariants:
  - FrameOfReferenceUID stays the same  → registers with original CT
  - ReferencedStructureSetSequence stays the same  → links to original RS
  - Cross-reference between the new RD and new RP is updated consistently

Enable / disable via ENABLE_DICOM_EXPORT in config.py.
"""

import logging
import os
from typing import Optional, List

import numpy as np
import pydicom
from pydicom.uid import generate_uid

from config import DICOM_EXPORT_DIR, ENABLE_DICOM_EXPORT
from dicom_io import PlanFiles
from shift_scenarios import _rotation_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation resampling
# ---------------------------------------------------------------------------

def _resample_dose_rotated(
    dose_grid: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    R_inv: np.ndarray,
    t: np.ndarray,
    isocenter: np.ndarray,
) -> np.ndarray:
    """
    Resample *dose_grid* for a rigid rotation around *isocenter*.
    For each voxel in the OUTPUT grid, compute the ORIGINAL position via
    the inverse transform and trilinear-interpolate.

    Output has the same shape, origin and spacing as the input.
    Pure numpy, no scipy dependency.
    """
    nz, ny, nx = dose_grid.shape
    ox, oy, oz = origin
    dx, dy, dz = spacing

    # Flat arrays of physical coordinates for every output voxel
    iz_arr = np.repeat(np.arange(nz), ny * nx).astype(float)
    iy_arr = np.tile(np.repeat(np.arange(ny), nx), nz).astype(float)
    ix_arr = np.tile(np.arange(nx), nz * ny).astype(float)

    X = ox + ix_arr * dx - isocenter[0] - t[0]
    Y = oy + iy_arr * dy - isocenter[1] - t[1]
    Z = oz + iz_arr * dz - isocenter[2] - t[2]

    # Apply inverse rotation: p_orig_centered = R^T @ p_new_centered
    pts = np.stack([X, Y, Z], axis=0)   # shape (3, N)
    pts_orig = R_inv @ pts              # shape (3, N)

    fx = (pts_orig[0] + isocenter[0] - ox) / dx
    fy = (pts_orig[1] + isocenter[1] - oy) / dy
    fz = (pts_orig[2] + isocenter[2] - oz) / dz

    # Valid mask (inside original grid)
    valid = (fx >= 0) & (fx < nx - 1) & (fy >= 0) & (fy < ny - 1) & (fz >= 0) & (fz < nz - 1)

    x0 = np.clip(fx.astype(int), 0, nx - 2)
    y0 = np.clip(fy.astype(int), 0, ny - 2)
    z0 = np.clip(fz.astype(int), 0, nz - 2)
    xd = fx - x0; yd = fy - y0; zd = fz - z0

    def g(dz_, dy_, dx_): return dose_grid[z0 + dz_, y0 + dy_, x0 + dx_]
    c00 = g(0,0,0)*(1-xd) + g(0,0,1)*xd
    c01 = g(1,0,0)*(1-xd) + g(1,0,1)*xd
    c10 = g(0,1,0)*(1-xd) + g(0,1,1)*xd
    c11 = g(1,1,0)*(1-xd) + g(1,1,1)*xd
    c0  = c00*(1-yd) + c10*yd
    c1  = c01*(1-yd) + c11*yd
    result = c0*(1-zd) + c1*zd
    result[~valid] = 0.0

    return result.reshape(nz, ny, nx)


# ---------------------------------------------------------------------------
# Filename encoding
# ---------------------------------------------------------------------------

def _scenario_suffix(scenario) -> str:
    """
    Build a compact suffix from the 6-DOF shift parameters.
    Values that are zero are still included so the pattern is always
    dx_dy_dz_rx_ry_rz, making it machine-readable.
    E.g.  dx1_dy0_dz1_rx0_ry1_rz0
    """
    def fmt(v: float) -> str:
        s = f"{v:g}"      # removes trailing zeros: 1.0 → '1', 0.5 → '0.5'
        return s.replace("-", "m").replace(".", "p")   # filesystem-safe

    return (
        f"dx{fmt(scenario.dx_mm)}"
        f"_dy{fmt(scenario.dy_mm)}"
        f"_dz{fmt(scenario.dz_mm)}"
        f"_rx{fmt(scenario.rx_deg)}"
        f"_ry{fmt(scenario.ry_deg)}"
        f"_rz{fmt(scenario.rz_deg)}"
    )


# ---------------------------------------------------------------------------
# Core export helper
# ---------------------------------------------------------------------------

def _export_single(
    pf: PlanFiles,
    out_dir: str,
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    dz_mm: float = 0.0,
    rx_deg: float = 0.0,
    ry_deg: float = 0.0,
    rz_deg: float = 0.0,
    isocenter: Optional[np.ndarray] = None,
    plan_label: Optional[str] = None,
) -> Optional[str]:
    """
    Write re-stamped RD + RP into *out_dir*.
    Translation is applied via ImagePositionPatient offset.
    Rotation (if any) is applied by resampling the dose pixel data.
    *plan_label* overrides RTPlanLabel / RTPlanName in the RP file.
    Returns out_dir on success, None on failure.
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        rd = pydicom.dcmread(pf.rd_path)
        rp = pydicom.dcmread(pf.rp_path)
    except Exception as exc:
        logger.error("DICOM export: read failed for %s/%s: %s",
                     pf.patient_folder, pf.plan_type, exc)
        return None

    old_rd_uid = str(rd.SOPInstanceUID)
    old_rp_uid = str(rp.SOPInstanceUID)
    new_rd_uid = generate_uid()
    new_rp_uid = generate_uid()

    # ── Patch RD ──────────────────────────────────────────────────────────────
    rd.SOPInstanceUID = new_rd_uid
    if hasattr(rd, "file_meta") and rd.file_meta:
        rd.file_meta.MediaStorageSOPInstanceUID = new_rd_uid

    has_rot = abs(rx_deg) + abs(ry_deg) + abs(rz_deg) > 1e-9

    # Apply translation: shift ImagePositionPatient
    if abs(dx_mm) + abs(dy_mm) + abs(dz_mm) > 1e-9:
        try:
            ipp = np.array(rd.ImagePositionPatient, dtype=float)
            ipp += np.array([dx_mm, dy_mm, dz_mm])
            rd.ImagePositionPatient = [f"{v:.6f}" for v in ipp]
        except Exception as exc:
            logger.warning("DICOM export: could not shift ImagePositionPatient: %s", exc)

    # Apply rotation: resample dose pixel data
    if has_rot and isocenter is not None:
        try:
            scaling = float(rd.DoseGridScaling)
            raw = rd.pixel_array.astype(float) * scaling  # Gy per voxel
            ipp_arr = np.array(rd.ImagePositionPatient, dtype=float)
            ps = [float(v) for v in rd.PixelSpacing]
            try:
                z_offs = [float(v) for v in rd.GridFrameOffsetVector]
                dz_val = abs(z_offs[1] - z_offs[0]) if len(z_offs) > 1 else float(rd.SliceThickness)
            except Exception:
                dz_val = float(rd.SliceThickness)
            origin  = np.array([ipp_arr[0], ipp_arr[1], ipp_arr[2]])
            spacing = np.array([ps[1], ps[0], dz_val])   # dx, dy, dz
            R = _rotation_matrix(rx_deg, ry_deg, rz_deg)
            t = np.array([dx_mm, dy_mm, dz_mm])
            resampled = _resample_dose_rotated(raw, origin, spacing, R.T, t, isocenter)
            new_pixel = np.round(resampled / scaling).astype(rd.pixel_array.dtype)
            rd.PixelData = new_pixel.tobytes()
            rd.NumberOfFrames = new_pixel.shape[0]
            logger.debug("DICOM export: rotation resampling applied (%s)", (rx_deg, ry_deg, rz_deg))
        except Exception as exc:
            logger.warning("DICOM export: rotation resampling failed: %s", exc)

    for seq_attr in ("ReferencedRTPlanSequence",):
        seq = getattr(rd, seq_attr, None)
        if seq:
            for item in seq:
                if str(getattr(item, "ReferencedSOPInstanceUID", "")) == old_rp_uid:
                    item.ReferencedSOPInstanceUID = new_rp_uid

    # ── Patch RP ──────────────────────────────────────────────────────────────
    rp.SOPInstanceUID = new_rp_uid
    if hasattr(rp, "file_meta") and rp.file_meta:
        rp.file_meta.MediaStorageSOPInstanceUID = new_rp_uid

    if plan_label:
        for attr in ("RTPlanLabel", "RTPlanName"):
            if hasattr(rp, attr):
                setattr(rp, attr, plan_label)
        logger.debug("DICOM export: RTPlanLabel/Name set to '%s'", plan_label)

    for seq_attr in ("ReferencedRTDoseSequence", "ReferencedDoseSequence"):
        seq = getattr(rp, seq_attr, None)
        if seq:
            for item in seq:
                if str(getattr(item, "ReferencedSOPInstanceUID", "")) == old_rd_uid:
                    item.ReferencedSOPInstanceUID = new_rd_uid

    # ── Save ──────────────────────────────────────────────────────────────────
    rd_out = os.path.join(out_dir, os.path.basename(pf.rd_path))
    rp_out = os.path.join(out_dir, os.path.basename(pf.rp_path))
    try:
        rd.save_as(rd_out, write_like_original=False)
        rp.save_as(rp_out, write_like_original=False)
    except Exception as exc:
        logger.error("DICOM export: save failed for %s/%s: %s",
                     pf.patient_folder, pf.plan_type, exc)
        return None

    logger.debug("  RD uid: ...%s  RP uid: ...%s", new_rd_uid[-12:], new_rp_uid[-12:])
    return out_dir


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_plan_dicom(
    pf: PlanFiles,
    scenarios: Optional[List] = None,
    isocenter: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Export one RD+RP pair per scenario into DICOM_EXPORT_DIR.

    *scenarios*  – list of ShiftScenario objects (including nominal).
                   If None, only the nominal plan is exported.
    *isocenter*  – 3-element (x,y,z) mm array; required for rotation resampling.

    Returns list of output directory paths (one per successfully exported scenario).
    """
    if not ENABLE_DICOM_EXPORT:
        return []

    if not pf.rd_path or not pf.rp_path:
        logger.warning("DICOM export: missing RD or RP for %s/%s – skipped.",
                       pf.patient_folder, pf.plan_type)
        return []

    if scenarios is None:
        # Nominal only
        from shift_scenarios import NOMINAL_SCENARIO
        scenarios = [NOMINAL_SCENARIO]

    base = f"{pf.patient_folder}_{pf.plan_type}"
    exported = []
    shift_idx = 0  # counter for non-nominal scenarios

    for sc in scenarios:
        is_nominal = (
            sc.dx_mm == 0 and sc.dy_mm == 0 and sc.dz_mm == 0
            and sc.rx_deg == 0 and sc.ry_deg == 0 and sc.rz_deg == 0
        )

        suffix = "nominal" if is_nominal else _scenario_suffix(sc)
        out_dir = os.path.join(DICOM_EXPORT_DIR, f"{base}_{suffix}")

        if is_nominal:
            plan_label = pf.plan_type               # e.g. "HA"
        else:
            shift_idx += 1
            plan_label = f"{pf.plan_type}.{shift_idx}"  # e.g. "HA.1"

        result = _export_single(
            pf, out_dir,
            dx_mm=sc.dx_mm, dy_mm=sc.dy_mm, dz_mm=sc.dz_mm,
            rx_deg=sc.rx_deg, ry_deg=sc.ry_deg, rz_deg=sc.rz_deg,
            isocenter=isocenter,
            plan_label=plan_label,
        )
        if result:
            logger.info("DICOM export OK: %s/%s [%s] label=%s \u2192 %s",
                        pf.patient_folder, pf.plan_type, sc.name, plan_label, out_dir)
            exported.append(result)

    return exported
