"""
SRS Analysis Pipeline - DICOM I/O
===================================
Handles scanning of patient folders, loading of DICOM files (RS, RD, RP, CT)
and extraction of relevant metadata.
"""

import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import pydicom

from config import (
    DATA_ROOT, DEBUG_MODE, DEBUG_MAX_PATIENTS, DEBUG_PLAN_TYPES,
    PATIENT_FOLDER_PREFIXES, PLAN_TYPE_HA, PLAN_TYPE_MM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PlanFiles:
    """All DICOM file paths belonging to one plan (HA or MM) of one patient."""
    patient_folder: str          # folder name, e.g. "zz_Paper_MM_UKE_1"
    patient_folder_path: str     # absolute path to folder
    plan_type: str               # "HA" or "MM"
    rp_path: Optional[str] = None
    rs_path: Optional[str] = None
    rd_path: Optional[str] = None


@dataclass
class Structure:
    """Single contour structure loaded from an RS DICOM."""
    name: str
    roi_number: int
    dicom_type: str              # e.g. "PTV", "OAR", "ORGAN", ""
    contours: list = field(default_factory=list)   # list of Nx3 arrays (x,y,z) in mm


@dataclass
class DoseData:
    """3-D dose grid loaded from an RD DICOM."""
    dose_grid: np.ndarray        # shape (Z, Y, X) in Gy
    origin: np.ndarray           # (x0, y0, z0) mm
    spacing: np.ndarray          # (dx, dy, dz) mm
    image_orientation: np.ndarray  # 6-element row/col cosines


@dataclass
class PlanMeta:
    """Metadata extracted from an RP DICOM."""
    patient_id: str
    plan_label: str
    isocenter: Optional[np.ndarray]   # (x, y, z) mm in DICOM patient coords


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

def scan_patient_folders(root: str = DATA_ROOT) -> List[str]:
    """
    Return sorted list of patient folder names found under *root*.
    Folders must start with one of PATIENT_FOLDER_PREFIXES.
    """
    folders = []
    try:
        entries = os.listdir(root)
    except OSError as exc:
        logger.error("Cannot list directory %s: %s", root, exc)
        return folders

    for entry in sorted(entries):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        if any(entry.startswith(pfx) for pfx in PATIENT_FOLDER_PREFIXES):
            folders.append(entry)

    logger.info("Found %d patient folder(s) under %s", len(folders), root)
    return folders


def find_plan_files(folder_name: str, root: str = DATA_ROOT) -> List[PlanFiles]:
    """
    Scan *folder_name* and return one PlanFiles per detected plan (HA / MM).
    File naming convention (case-insensitive):
        RD.<folder>.<plantype>.dcm
        RP.<folder>.<plantype>.dcm
        RS.<folder>.<plantype>.dcm
    """
    folder_path = os.path.join(root, folder_name)
    plan_map: Dict[str, PlanFiles] = {}

    try:
        filenames = os.listdir(folder_path)
    except OSError as exc:
        logger.error("Cannot list %s: %s", folder_path, exc)
        return []

    for fname in filenames:
        upper = fname.upper()
        if not upper.endswith(".DCM"):
            continue

        # Determine modality prefix
        if upper.startswith("RP."):
            modality = "rp"
        elif upper.startswith("RS."):
            modality = "rs"
        elif upper.startswith("RD."):
            modality = "rd"
        else:
            continue

        # Determine plan type from filename
        plan_type = _detect_plan_type(fname)
        if plan_type is None:
            logger.debug("Cannot determine plan type for %s – skipped", fname)
            continue

        key = plan_type
        if key not in plan_map:
            plan_map[key] = PlanFiles(
                patient_folder=folder_name,
                patient_folder_path=folder_path,
                plan_type=plan_type,
            )

        full_path = os.path.join(folder_path, fname)
        pf = plan_map[key]
        if modality == "rp":
            pf.rp_path = full_path
        elif modality == "rs":
            pf.rs_path = full_path
        elif modality == "rd":
            pf.rd_path = full_path

    result = list(plan_map.values())
    for pf in result:
        logger.debug(
            "  %s | %s | RS=%s RD=%s RP=%s",
            folder_name, pf.plan_type,
            "ok" if pf.rs_path else "MISSING",
            "ok" if pf.rd_path else "MISSING",
            "ok" if pf.rp_path else "MISSING",
        )
    return result


def _detect_plan_type(filename: str) -> Optional[str]:
    """
    Return "HA" or "MM" based on the plan-type token in *filename*.

    File naming convention: <MODALITY>.<PatientFolder>.<PlanType>.dcm
    e.g. RP.zz_Paper_MM_UKE_1.HA.dcm  →  "HA"
         RD.zz_Paper_MM_UKE_1.MM3.dcm  →  "MM"

    We inspect only the last dot-separated component before ".dcm" to avoid
    false positives from "_MM_UKE_" in the patient folder name portion.
    """
    # Strip extension and split by dots; plan-type token is the last component
    base = filename
    if base.upper().endswith(".DCM"):
        base = base[:-4]
    parts = base.split(".")
    if not parts:
        return None
    plan_token = parts[-1].upper()

    if PLAN_TYPE_HA.upper() in plan_token:
        return PLAN_TYPE_HA
    if PLAN_TYPE_MM.upper() in plan_token:
        return PLAN_TYPE_MM
    return None


# ---------------------------------------------------------------------------
# Debug-mode filtering
# ---------------------------------------------------------------------------

def apply_debug_filter(
    plan_files: List[PlanFiles],
    all_folder_names: List[str],
) -> List[PlanFiles]:
    """
    When DEBUG_MODE is active, restrict to the first DEBUG_MAX_PATIENTS folders
    and only plans whose plan_type is in DEBUG_PLAN_TYPES.
    Returns the filtered list.
    """
    if not DEBUG_MODE:
        return plan_files

    allowed_folders = set(all_folder_names[:DEBUG_MAX_PATIENTS])
    filtered = [
        pf for pf in plan_files
        if pf.patient_folder in allowed_folders
        and pf.plan_type in DEBUG_PLAN_TYPES
    ]
    logger.info(
        "[DEBUG MODE] Restricted to %d plan(s) (patients: %s, types: %s)",
        len(filtered),
        list(allowed_folders),
        DEBUG_PLAN_TYPES,
    )
    return filtered


# ---------------------------------------------------------------------------
# RS loading – structures
# ---------------------------------------------------------------------------

def load_structures(rs_path: str) -> List[Structure]:
    """
    Parse an RT Structure Set DICOM and return a list of Structure objects.
    Contour coordinates are returned in mm (patient coordinate system).
    """
    structures: List[Structure] = []
    if not rs_path or not os.path.isfile(rs_path):
        logger.warning("RS file not found: %s", rs_path)
        return structures

    try:
        ds = pydicom.dcmread(rs_path, force=True)
    except Exception as exc:
        logger.error("Failed to read RS %s: %s", rs_path, exc)
        return structures

    # Build ROI name + type lookup from StructureSetROISequence
    roi_info: Dict[int, dict] = {}
    for roi in getattr(ds, "StructureSetROISequence", []):
        roi_info[int(roi.ROINumber)] = {
            "name": getattr(roi, "ROIName", "Unknown"),
        }

    # Add interpreted type from RTROIObservationsSequence
    for obs in getattr(ds, "RTROIObservationsSequence", []):
        rn = int(getattr(obs, "ReferencedROINumber", -1))
        if rn in roi_info:
            roi_info[rn]["type"] = getattr(obs, "RTROIInterpretedType", "")

    # Parse contour data from ROIContourSequence
    for roi_contour in getattr(ds, "ROIContourSequence", []):
        rn = int(getattr(roi_contour, "ReferencedROINumber", -1))
        info = roi_info.get(rn, {"name": str(rn), "type": ""})
        name = info.get("name", "Unknown")
        dicom_type = info.get("type", "")

        contours = []
        for contour_seq in getattr(roi_contour, "ContourSequence", []):
            raw = list(contour_seq.ContourData)
            pts = np.array(raw, dtype=float).reshape(-1, 3)   # Nx3 (x,y,z)
            contours.append(pts)

        structures.append(Structure(
            name=name,
            roi_number=rn,
            dicom_type=dicom_type,
            contours=contours,
        ))

    logger.info("  Loaded %d structures from %s", len(structures), os.path.basename(rs_path))
    return structures


# ---------------------------------------------------------------------------
# RD loading – dose grid
# ---------------------------------------------------------------------------

def load_dose(rd_path: str) -> Optional[DoseData]:
    """
    Parse an RT Dose DICOM.  Returns a DoseData object with the 3-D dose
    array in Gy (float64).
    """
    if not rd_path or not os.path.isfile(rd_path):
        logger.warning("RD file not found: %s", rd_path)
        return None

    try:
        ds = pydicom.dcmread(rd_path, force=True)
    except Exception as exc:
        logger.error("Failed to read RD %s: %s", rd_path, exc)
        return None

    try:
        scaling = float(getattr(ds, "DoseGridScaling", 1.0))
        pixel_array = ds.pixel_array.astype(np.float64) * scaling  # Gy

        image_position = np.array(ds.ImagePositionPatient, dtype=float)  # (x0,y0,z0)
        pixel_spacing = np.array(ds.PixelSpacing, dtype=float)           # (row_spacing, col_spacing)

        # Build slice positions from GridFrameOffsetVector
        offsets = np.array(ds.GridFrameOffsetVector, dtype=float)
        # ImagePositionPatient gives the top-left corner of the first slice
        # The z-offsets are relative to that; spacing is ImageOrientationPatient-aware,
        # but for axial CTs/dose grids the z-component is dominant.
        image_orientation = np.array(getattr(ds, "ImageOrientationPatient", [1,0,0,0,1,0]), dtype=float)

        # spacing array: (col_spacing, row_spacing, z_spacing)
        # pixel_array shape: (frames, rows, cols) = (Z, Y, X)
        z_spacing = offsets[1] - offsets[0] if len(offsets) > 1 else 1.0
        spacing = np.array([pixel_spacing[1], pixel_spacing[0], abs(z_spacing)])  # dx, dy, dz

    except Exception as exc:
        logger.error("Failed to extract dose data from %s: %s", rd_path, exc)
        return None

    logger.info(
        "  Loaded dose grid %s from %s (scaling=%g Gy/unit)",
        pixel_array.shape, os.path.basename(rd_path), scaling,
    )
    return DoseData(
        dose_grid=pixel_array,
        origin=image_position,
        spacing=spacing,
        image_orientation=image_orientation,
    )


# ---------------------------------------------------------------------------
# RP loading – plan metadata
# ---------------------------------------------------------------------------

def load_plan_meta(rp_path: str) -> Optional[PlanMeta]:
    """
    Parse an RT Plan DICOM.  Returns PlanMeta with patient ID, plan label,
    and the isocenter coordinates (first beam) in mm.
    """
    if not rp_path or not os.path.isfile(rp_path):
        logger.warning("RP file not found: %s", rp_path)
        return None

    try:
        ds = pydicom.dcmread(rp_path, force=True)
    except Exception as exc:
        logger.error("Failed to read RP %s: %s", rp_path, exc)
        return None

    patient_id = str(getattr(ds, "PatientID", "Unknown"))
    plan_label = str(getattr(ds, "RTPlanLabel", "Unknown"))

    isocenter = None
    for beam in getattr(ds, "BeamSequence", []):
        for cp in getattr(beam, "ControlPointSequence", []):
            iso_raw = getattr(cp, "IsocenterPosition", None)
            if iso_raw is not None:
                isocenter = np.array(iso_raw, dtype=float)
                break
        if isocenter is not None:
            break

    logger.info(
        "  RP: patient_id=%s  plan_label=%s  isocenter=%s",
        patient_id, plan_label, isocenter,
    )
    return PlanMeta(
        patient_id=patient_id,
        plan_label=plan_label,
        isocenter=isocenter,
    )


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_plan(pf: PlanFiles) -> Tuple[List[Structure], Optional[DoseData], Optional[PlanMeta]]:
    """
    Load all DICOM data for a single plan.  Returns (structures, dose, meta).
    Any component that cannot be loaded is returned as None / empty list.
    """
    logger.info("Loading plan: %s / %s", pf.patient_folder, pf.plan_type)
    structures = load_structures(pf.rs_path)
    dose = load_dose(pf.rd_path)
    meta = load_plan_meta(pf.rp_path)
    return structures, dose, meta
