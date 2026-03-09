"""
SRS Analysis – MLC Viewer Data Extraction
==========================================
Reads RP DICOM BeamSequence and extracts MLC leaf positions, jaw positions,
and beam metadata as JSON-serializable structures for the interactive HTML viewer.

DICOM conventions used:
  MLCX leaves move in X.  Bank A = negative-X side (leaf tips reported as negative),
  Bank B = positive-X side.  LeafPositionBoundaries = Y edges of each leaf row.
"""

import logging
from typing import List, Dict, Optional

import pydicom

logger = logging.getLogger(__name__)


def load_mlc_data(rp_path: str) -> List[Dict]:
    """
    Parse an RP DICOM and return a list of beam dicts.

    Each beam dict::
        name              – beam name / description
        gantry_angle      – initial gantry angle (°)
        collimator_angle  – initial collimator angle (°)
        couch_angle       – initial patient-support angle (°)
        n_leaf_pairs      – number of MLC leaf pairs (0 if no MLC found)
        leaf_boundaries   – Y boundaries of leaf pairs [mm], length n_leaf_pairs+1
        control_points    – list of CP dicts (see below)

    Each control-point dict::
        meterset_weight  – cumulative MU weight
        gantry_angle     – gantry angle at this CP (°)
        leaf_a           – bank-A leaf tip positions [mm], length n_leaf_pairs
        leaf_b           – bank-B leaf tip positions [mm], length n_leaf_pairs
        jaw_x1, jaw_x2   – X jaw positions [mm] (None if not present)
        jaw_y1, jaw_y2   – Y jaw positions [mm] (None if not present)

    Returns [] if file is missing, unreadable or has no BeamSequence.
    """
    if not rp_path:
        return []

    try:
        ds = pydicom.dcmread(rp_path, force=True)
    except Exception as exc:
        logger.error("MLC: failed to read RP %s: %s", rp_path, exc)
        return []

    beams_out: List[Dict] = []

    for beam in getattr(ds, "BeamSequence", []):
        beam_name = str(
            getattr(beam, "BeamName", None) or
            getattr(beam, "BeamDescription", "Beam")
        )

        cp_seq: list = list(getattr(beam, "ControlPointSequence", []))
        first_cp = cp_seq[0] if cp_seq else None

        gantry0    = float(getattr(first_cp, "GantryAngle",             0)) if first_cp else 0.0
        collimator = float(getattr(first_cp, "BeamLimitingDeviceAngle", 0)) if first_cp else 0.0
        couch      = float(getattr(first_cp, "PatientSupportAngle",     0)) if first_cp else 0.0

        # ── Leaf geometry ────────────────────────────────────────────────────
        n_leaf_pairs   = 0
        leaf_boundaries: List[float] = []
        for bld in getattr(beam, "BeamLimitingDeviceSequence", []):
            dtype = str(getattr(bld, "RTBeamLimitingDeviceType", "")).upper()
            if "MLCX" in dtype or ("MLC" in dtype):
                raw = list(getattr(bld, "LeafPositionBoundaries", []))
                if raw:
                    leaf_boundaries = [float(v) for v in raw]
                    n_leaf_pairs    = len(leaf_boundaries) - 1
                break

        # ── Control points ───────────────────────────────────────────────────
        control_points: List[Dict] = []
        prev_gantry              = gantry0
        prev_leaf_a: Optional[List[float]] = None
        prev_leaf_b: Optional[List[float]] = None
        prev_jaw = {"x1": None, "x2": None, "y1": None, "y2": None}

        for cp in cp_seq:
            if hasattr(cp, "GantryAngle"):
                prev_gantry = float(cp.GantryAngle)

            leaf_a = prev_leaf_a
            leaf_b = prev_leaf_b
            jaw    = dict(prev_jaw)

            for bld_pos in getattr(cp, "BeamLimitingDevicePositionSequence", []):
                dtype     = str(getattr(bld_pos, "RTBeamLimitingDeviceType", "")).upper()
                positions = [float(v) for v in getattr(bld_pos, "LeafJawPositions", [])]

                if "MLCX" in dtype or ("MLC" in dtype and len(positions) > 4):
                    n = len(positions) // 2
                    leaf_a = positions[:n]
                    leaf_b = positions[n:]
                elif dtype in ("ASYMX", "X"):
                    if len(positions) >= 2:
                        jaw["x1"] = positions[0]
                        jaw["x2"] = positions[1]
                elif dtype in ("ASYMY", "Y"):
                    if len(positions) >= 2:
                        jaw["y1"] = positions[0]
                        jaw["y2"] = positions[1]

            if leaf_a is not None:
                prev_leaf_a = leaf_a
                prev_leaf_b = leaf_b
            if jaw["x1"] is not None:
                prev_jaw = dict(jaw)

            control_points.append({
                "meterset_weight": round(float(getattr(cp, "CumulativeMetersetWeight", 0)), 4),
                "gantry_angle":    round(prev_gantry, 2),
                "leaf_a":          [round(v, 2) for v in prev_leaf_a] if prev_leaf_a else [],
                "leaf_b":          [round(v, 2) for v in prev_leaf_b] if prev_leaf_b else [],
                "jaw_x1":          prev_jaw["x1"],
                "jaw_x2":          prev_jaw["x2"],
                "jaw_y1":          prev_jaw["y1"],
                "jaw_y2":          prev_jaw["y2"],
            })

        # Skip beams with <2 control points (static, not useful for viewer)
        if len(control_points) < 2:
            logger.debug("MLC: skipping beam '%s' (only %d control point)", beam_name, len(control_points))
            continue

        beams_out.append({
            "name":             beam_name,
            "gantry_angle":     round(gantry0, 2),
            "collimator_angle": round(collimator, 2),
            "couch_angle":      round(couch, 2),
            "n_leaf_pairs":     n_leaf_pairs,
            "leaf_boundaries":  [round(v, 2) for v in leaf_boundaries],
            "control_points":   control_points,
        })

    logger.info("MLC: %d beam(s) loaded from %s (filtered: ≥2 control points)", len(beams_out), rp_path)
    return beams_out
