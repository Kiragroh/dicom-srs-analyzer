"""
SRS Analysis Pipeline - 6D Dose Shift Scenarios
=================================================
Applies translational and rotational shifts around the plan isocenter
to generate perturbed dose grids for robustness analysis.

Technical approach
------------------
A 6-DOF rigid transformation (3 translations in mm + 3 rotations in degrees)
is applied to the **sampling coordinates** rather than resampling the dose
grid itself.  Concretely, to evaluate the dose at a point p after applying
shift S, we compute the inverse-transformed point p' = S⁻¹(p) and look up
the original dose there.  This is equivalent to shifting the dose distribution
by S relative to the fixed anatomy.

Rotation convention
-------------------
Rotations are extrinsic (fixed-axis) in the order Rx → Ry → Rz
(roll → pitch → yaw), applied around the plan isocenter.
This matches common clinical robustness conventions.
Units: translations in mm, rotations in degrees.

Limitations / TODOs
-------------------
- The approach implicitly assumes the CT/dose grid is large enough that
  shifted sampling points do not leave the volume boundary.  Out-of-bounds
  points return dose = 0, which may underestimate PIV for large shifts.
- True resampling (e.g. scipy.ndimage.affine_transform) would be more
  accurate for large shifts but is slower.  Swap _build_shifted_dose()
  implementation to enable.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import SHIFT_SCENARIOS_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class ShiftScenario:
    name:   str           = "nominal"
    dx_mm:  float         = 0.0
    dy_mm:  float         = 0.0
    dz_mm:  float         = 0.0
    rx_deg: float         = 0.0
    ry_deg: float         = 0.0
    rz_deg: float         = 0.0


NOMINAL_SCENARIO = ShiftScenario(name="nominal")


def scenario_label(sc: ShiftScenario) -> str:
    """Short display label: 'nominal' or 'dx/dy/dz//rx/ry/rz' (e.g. '1/0/0//0/0/0')."""
    if sc.name == "nominal":
        return "nominal"

    def fmt(v: float) -> str:
        f = float(v)
        return str(int(f)) if f == int(f) else str(round(f, 2))

    return (f"{fmt(sc.dx_mm)}/{fmt(sc.dy_mm)}/{fmt(sc.dz_mm)}/"
            f"/{fmt(sc.rx_deg)}/{fmt(sc.ry_deg)}/{fmt(sc.rz_deg)}")


# ---------------------------------------------------------------------------
# Default example scenarios table
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS = [
    ShiftScenario(name="nominal",          dx_mm=0,   dy_mm=0,   dz_mm=0,   rx_deg=0,   ry_deg=0,   rz_deg=0),
    ShiftScenario(name="shift_scenario_01", dx_mm=1.0, dy_mm=0,   dz_mm=0,   rx_deg=0,   ry_deg=0,   rz_deg=0),
    ShiftScenario(name="shift_scenario_02", dx_mm=0,   dy_mm=0,   dz_mm=1.0, rx_deg=0,   ry_deg=1.0, rz_deg=0),
]


def create_default_scenarios_table(path: str = SHIFT_SCENARIOS_PATH) -> None:
    """Write the default shift-scenarios Excel if it does not exist yet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        logger.info("Shift scenarios table already exists: %s", path)
        return
    df = pd.DataFrame([
        {
            "name":   s.name,
            "dx_mm":  s.dx_mm,
            "dy_mm":  s.dy_mm,
            "dz_mm":  s.dz_mm,
            "rx_deg": s.rx_deg,
            "ry_deg": s.ry_deg,
            "rz_deg": s.rz_deg,
        }
        for s in DEFAULT_SCENARIOS
    ])
    df.to_excel(path, index=False)
    logger.info("Created default shift scenarios table: %s", path)


def load_scenarios(path: str = SHIFT_SCENARIOS_PATH) -> List[ShiftScenario]:
    """Load shift scenarios from Excel.  Falls back to DEFAULT_SCENARIOS."""
    if not os.path.isfile(path):
        logger.warning("Scenarios file not found (%s) – using built-in defaults.", path)
        return DEFAULT_SCENARIOS

    try:
        df = pd.read_excel(path)
        scenarios = []
        for _, row in df.iterrows():
            scenarios.append(ShiftScenario(
                name=str(row.get("name", "unnamed")),
                dx_mm=float(row.get("dx_mm", 0)),
                dy_mm=float(row.get("dy_mm", 0)),
                dz_mm=float(row.get("dz_mm", 0)),
                rx_deg=float(row.get("rx_deg", 0)),
                ry_deg=float(row.get("ry_deg", 0)),
                rz_deg=float(row.get("rz_deg", 0)),
            ))
        logger.info("Loaded %d scenarios from %s", len(scenarios), path)
        return scenarios
    except Exception as exc:
        logger.error("Failed to read scenarios from %s: %s – using defaults.", path, exc)
        return DEFAULT_SCENARIOS


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    Build a 3×3 rotation matrix for extrinsic XYZ Euler angles (degrees).
    Order: Rx first, then Ry, then Rz (i.e. combined R = Rz @ Ry @ Rx).
    """
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    Rx = np.array([[1,           0,            0],
                   [0,  np.cos(rx), -np.sin(rx)],
                   [0,  np.sin(rx),  np.cos(rx)]])

    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [           0, 1,           0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [          0,           0, 1]])

    return Rz @ Ry @ Rx


def apply_shift_to_point(
    point: np.ndarray,
    scenario: ShiftScenario,
    isocenter: np.ndarray,
) -> np.ndarray:
    """
    Apply the inverse shift transformation to *point* so that the dose
    lookup in the original grid is equivalent to evaluating the shifted
    dose at *point*.

    Forward transform:  p_new = R @ (p - iso) + iso + t
    Inverse transform:  p_orig = Rᵀ @ (p_new - iso - t) + iso
    """
    t = np.array([scenario.dx_mm, scenario.dy_mm, scenario.dz_mm])
    R = _rotation_matrix(scenario.rx_deg, scenario.ry_deg, scenario.rz_deg)
    R_inv = R.T   # rotation matrices are orthogonal: R⁻¹ = Rᵀ

    p_centered = point - isocenter - t
    p_orig = R_inv @ p_centered + isocenter
    return p_orig


# ---------------------------------------------------------------------------
# Shifted dose wrapper
# ---------------------------------------------------------------------------

class ShiftedDose:
    """
    Wraps a DoseData object and applies a rigid shift on every dose lookup.
    This allows the metrics code to work unchanged – it calls _interpolate_dose
    via structure_mapping, which accepts a DoseData-like object.

    We achieve this by subclassing or duck-typing DoseData:
    the dose_grid, origin, spacing, and image_orientation attributes are
    forwarded from the underlying object, but we expose a modified `origin`
    that effectively shifts the grid.

    Simple translation approach:
        Shifting the dose by (dx, dy, dz) is equivalent to changing the
        origin of the dose grid by (-dx, -dy, -dz).  For pure translations
        this is exact.  Rotations require point-by-point inverse mapping,
        which we handle by overriding the lookup in a monkey-patch approach.

    For the initial implementation we support only pure translations via
    origin offset.  Rotation support is scaffolded but deactivated with a
    clear TODO marker.
    """

    def __init__(self, base_dose, scenario: ShiftScenario, isocenter: np.ndarray):
        self._base = base_dose
        self._scenario = scenario
        self._isocenter = np.asarray(isocenter, dtype=float)
        # Copy attributes for duck-typing compatibility
        self.dose_grid          = base_dose.dose_grid
        self.image_orientation  = base_dose.image_orientation
        self.spacing            = base_dose.spacing

        has_rot = (abs(scenario.rx_deg) > 1e-6 or abs(scenario.ry_deg) > 1e-6
                   or abs(scenario.rz_deg) > 1e-6)
        self._has_rotation = has_rot

        if has_rot:
            # Rotation present: keep original origin; dose lookup uses transform_point
            self.origin = base_dose.origin.copy()
            self._R_inv = _rotation_matrix(
                scenario.rx_deg, scenario.ry_deg, scenario.rz_deg).T
            self._t = np.array([scenario.dx_mm, scenario.dy_mm, scenario.dz_mm])
        else:
            # Pure translation: shift origin (exact, fast)
            t = np.array([scenario.dx_mm, scenario.dy_mm, scenario.dz_mm])
            self.origin = base_dose.origin - t

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Inverse-transform *point* so the original-grid lookup gives the shifted dose."""
        # Forward: p_new = R @ (p_orig - iso) + iso + t
        # Inverse: p_orig = R^T @ (p_new - iso - t) + iso
        p_centered = np.asarray(point) - self._isocenter - self._t
        return self._R_inv @ p_centered + self._isocenter


def build_shifted_dose(base_dose, scenario: ShiftScenario, isocenter: np.ndarray):
    """
    Return a DoseData-compatible object representing the dose distribution
    shifted by *scenario* around *isocenter*.

    For the nominal scenario (all zeros) the base dose is returned unchanged.
    """
    is_nominal = (
        scenario.dx_mm == 0 and scenario.dy_mm == 0 and scenario.dz_mm == 0
        and scenario.rx_deg == 0 and scenario.ry_deg == 0 and scenario.rz_deg == 0
    )
    if is_nominal:
        return base_dose

    return ShiftedDose(base_dose, scenario, isocenter)
