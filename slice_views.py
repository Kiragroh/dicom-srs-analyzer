"""
SRS Analysis Pipeline - Orthogonal Slice Views
================================================
For each active PTV, generates three PNG images showing the dose distribution
and structure contour in the axial, coronal, and sagittal planes through the
PTV centroid.  Visual style mirrors SRS_Metrics_v1.0.0RC1.html.

Output folder layout:
    VIEWS_OUTPUT_DIR/<PatientFolder>/<PlanType>/<SafeStructureName>/
        axial.png
        coronal.png
        sagittal.png

Configure via ENABLE_VIEWS, VIEW_HALF_MM, VIEW_DPI in config.py.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend – must precede pyplot
import matplotlib.pyplot as plt

from config import (ENABLE_VIEWS, VIEW_HALF_MM, VIEW_DPI, VIEWS_OUTPUT_DIR,
                    SHOW_COMPUTATION_BOUNDARY, BOUNDARY_FIXED_MARGIN_MM)
from dicom_io import Structure, DoseData
from structure_mapping import compute_centroid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dose-slice helpers
# ---------------------------------------------------------------------------

def _vox(val: float, origin: float, spacing: float, size: int) -> int:
    """Convert mm coordinate to nearest integer voxel index (clamped)."""
    return int(np.clip(round((val - origin) / spacing), 0, size - 1))


def _axial_slice(dose: DoseData, z_mm: float) -> tuple:
    """
    Extract a 2D axial (XY) slice at z=z_mm from the dose grid.
    Returns (slice_2d, x_coords_mm, y_coords_mm).
    """
    origin = dose.origin
    spacing = dose.spacing
    grid = dose.dose_grid

    # Check if this is a rotated ShiftedDose
    if hasattr(dose, '_has_rotation') and dose._has_rotation:
        # Rotation present: build slice via point-by-point transform
        ny, nx = grid.shape[1], grid.shape[2]
        x_coords = origin[0] + np.arange(nx) * spacing[0]
        y_coords = origin[1] + np.arange(ny) * spacing[1]
        slice_2d = np.zeros((ny, nx), dtype=grid.dtype)
        for iy in range(ny):
            for ix in range(nx):
                pt = np.array([x_coords[ix], y_coords[iy], z_mm])
                pt_orig = dose.transform_point(pt)
                # Interpolate from base grid
                from structure_mapping import _interpolate_dose
                slice_2d[iy, ix] = _interpolate_dose(pt_orig, dose._base)
        return slice_2d, x_coords, y_coords

    # Pure translation or nominal: direct grid access
    z_idx = int(round((z_mm - origin[2]) / spacing[2]))
    z_idx = max(0, min(z_idx, grid.shape[0] - 1))
    slice_2d = grid[z_idx, :, :]
    ny, nx = slice_2d.shape
    x_coords = origin[0] + np.arange(nx) * spacing[0]
    y_coords = origin[1] + np.arange(ny) * spacing[1]
    return slice_2d, x_coords, y_coords


def _coronal_slice(dose: DoseData, y_mm: float) -> tuple:
    """
    Extract a 2D coronal (XZ) slice at y=y_mm.
    Returns (slice_2d, x_coords_mm, z_coords_mm).
    """
    origin = dose.origin
    spacing = dose.spacing
    grid = dose.dose_grid

    if hasattr(dose, '_has_rotation') and dose._has_rotation:
        nz, nx = grid.shape[0], grid.shape[2]
        x_coords = origin[0] + np.arange(nx) * spacing[0]
        z_coords = origin[2] + np.arange(nz) * spacing[2]
        slice_2d = np.zeros((nz, nx), dtype=grid.dtype)
        for iz in range(nz):
            for ix in range(nx):
                pt = np.array([x_coords[ix], y_mm, z_coords[iz]])
                pt_orig = dose.transform_point(pt)
                from structure_mapping import _interpolate_dose
                slice_2d[iz, ix] = _interpolate_dose(pt_orig, dose._base)
        return slice_2d, x_coords, z_coords

    y_idx = int(round((y_mm - origin[1]) / spacing[1]))
    y_idx = max(0, min(y_idx, grid.shape[1] - 1))
    slice_2d = grid[:, y_idx, :]
    nz, nx = slice_2d.shape
    x_coords = origin[0] + np.arange(nx) * spacing[0]
    z_coords = origin[2] + np.arange(nz) * spacing[2]
    return slice_2d, x_coords, z_coords


def _sagittal_slice(dose: DoseData, x_mm: float) -> tuple:
    """
    Extract a 2D sagittal (YZ) slice at x=x_mm.
    Returns (slice_2d, y_coords_mm, z_coords_mm).
    """
    origin = dose.origin
    spacing = dose.spacing
    grid = dose.dose_grid

    if hasattr(dose, '_has_rotation') and dose._has_rotation:
        nz, ny = grid.shape[0], grid.shape[1]
        y_coords = origin[1] + np.arange(ny) * spacing[1]
        z_coords = origin[2] + np.arange(nz) * spacing[2]
        slice_2d = np.zeros((nz, ny), dtype=grid.dtype)
        for iz in range(nz):
            for iy in range(ny):
                pt = np.array([x_mm, y_coords[iy], z_coords[iz]])
                pt_orig = dose.transform_point(pt)
                from structure_mapping import _interpolate_dose
                slice_2d[iz, iy] = _interpolate_dose(pt_orig, dose._base)
        return slice_2d, y_coords, z_coords

    x_idx = int(round((x_mm - origin[0]) / spacing[0]))
    x_idx = max(0, min(x_idx, grid.shape[2] - 1))
    slice_2d = grid[:, :, x_idx]
    nz, ny = slice_2d.shape
    y_coords = origin[1] + np.arange(ny) * spacing[1]
    z_coords = origin[2] + np.arange(nz) * spacing[2]
    return slice_2d, y_coords, z_coords


# ---------------------------------------------------------------------------
# Contour extraction helpers
# ---------------------------------------------------------------------------

def _axial_contours(
    structure: Structure, z_mm: float, tol_mm: float = 1.5
) -> List[np.ndarray]:
    """
    Return list of (N, 2) xy-polygon arrays for the axial plane(s) nearest
    to *z_mm*.  Falls back to the single closest plane if none are within tol.
    """
    if not structure.contours:
        return []
    ranked = sorted(
        ((abs(float(c[0, 2]) - z_mm), c) for c in structure.contours if len(c) >= 3),
        key=lambda t: t[0],
    )
    if not ranked:
        return []
    min_dz = ranked[0][0]
    cutoff = max(tol_mm, min_dz + 0.1)
    return [c[:, :2] for dz, c in ranked if dz <= cutoff]


def _coronal_silhouette(
    structure: Structure, y_mm: float, tol_mm: float = 5.0
) -> Optional[np.ndarray]:
    """
    For each contour z-plane: if any of its points has y ≈ y_mm (within tol)
    collect the x-extent at that z-level; otherwise use the full x-extent if
    y_mm is inside the contour's y-bounding-box.
    Returns (N, 2) array of (x, z) scatter points representing the silhouette,
    or None if the structure doesn't cross this plane.
    """
    pts = []
    for c in structure.contours:
        z_level = float(c[0, 2])
        xs, ys = c[:, 0], c[:, 1]
        near = np.abs(ys - y_mm) <= tol_mm
        if near.any():
            pts.append((xs[near].min(), z_level))
            pts.append((xs[near].max(), z_level))
        elif ys.min() - tol_mm <= y_mm <= ys.max() + tol_mm:
            pts.append((xs.min(), z_level))
            pts.append((xs.max(), z_level))
    return np.array(pts, dtype=float) if pts else None


def _sagittal_silhouette(
    structure: Structure, x_mm: float, tol_mm: float = 5.0
) -> Optional[np.ndarray]:
    """
    Same idea as _coronal_silhouette but for the sagittal plane.
    Returns (N, 2) array of (y, z) scatter points.
    """
    pts = []
    for c in structure.contours:
        z_level = float(c[0, 2])
        xs, ys = c[:, 0], c[:, 1]
        near = np.abs(xs - x_mm) <= tol_mm
        if near.any():
            pts.append((ys[near].min(), z_level))
            pts.append((ys[near].max(), z_level))
        elif xs.min() - tol_mm <= x_mm <= xs.max() + tol_mm:
            pts.append((ys.min(), z_level))
            pts.append((ys.max(), z_level))
    return np.array(pts, dtype=float) if pts else None


# ---------------------------------------------------------------------------
# Single-view renderer
# ---------------------------------------------------------------------------

def _render_view(
    title: str,
    slice_2d: np.ndarray,
    h_coords: np.ndarray,
    v_coords: np.ndarray,
    h_label: str,
    v_label: str,
    cx: float,          # centroid horizontal position in mm
    cy: float,          # centroid vertical position in mm
    iso_h: float,       # isocenter horizontal
    iso_v: float,       # isocenter vertical
    rx_dose: float,
    half_mm: float,
    closed_contours: Optional[List[np.ndarray]] = None,
    scatter_pts: Optional[np.ndarray] = None,
    bbox_rect: Optional[tuple] = None,   # (h_min, h_max, v_min, v_max) in mm
    nominal_slice: Optional[np.ndarray] = None,  # for shift-scenario overlay
) -> plt.Figure:
    """
    Render one orthogonal dose-slice view and return a matplotlib Figure.

    Layers (back to front):
      1. Dose heatmap (inferno colormap)
      2. 100 % Rx isodose line (cyan, solid)
      3. 50 % Rx isodose line (yellow, dashed)
      2b. Nominal 100%/50% isodose (gray, dashed) if nominal_slice provided
      4. Structure contour – closed polygon (axial) or scatter silhouette
      5. Isocenter marker (red +)
      6. Centroid marker (cyan x)
      7. 10 mm scale bar
    """
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.set_facecolor("black")

    # Crop arrays to view window
    hm = (h_coords >= cx - half_mm) & (h_coords <= cx + half_mm)
    vm = (v_coords >= cy - half_mm) & (v_coords <= cy + half_mm)

    if not hm.any() or not vm.any():
        ax.text(0.5, 0.5, "outside dose grid", color="white",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(title, color="white", fontsize=8)
        return fig

    s = slice_2d[np.ix_(vm, hm)]
    hc, vc = h_coords[hm], v_coords[vm]

    # 1. Dose heatmap
    vmax = max(float(s.max()), rx_dose if rx_dose else 1.0, 1e-6)
    cmap = plt.cm.inferno.copy()
    cmap.set_under("black")
    ax.pcolormesh(hc, vc, s, cmap=cmap, vmin=0.01, vmax=vmax,
                  shading="auto", rasterized=True)

    # 2+3. Isodose lines
    if s.max() > 0 and rx_dose and rx_dose > 0:
        try:
            ax.contour(hc, vc, s, levels=[rx_dose],
                       colors=["cyan"], linewidths=[1.2], alpha=0.9)
            ax.contour(hc, vc, s, levels=[rx_dose * 0.5],
                       colors=["yellow"], linewidths=[0.7], alpha=0.7,
                       linestyles=["dashed"])
        except Exception:
            pass

    # 3b. Nominal isodoses (gray, semi-transparent) for shift scenarios
    if nominal_slice is not None:
        ax.contour(h_coords, v_coords, nominal_slice, levels=[rx_dose],
                   colors="gray", linewidths=0.6, linestyles="dashed", alpha=0.5, zorder=2)
        ax.contour(h_coords, v_coords, nominal_slice, levels=[rx_dose * 0.5],
                   colors="gray", linewidths=0.6, linestyles="dashed", alpha=0.4, zorder=2)

    # 4a. Closed contour polygons (axial view)
    if closed_contours:
        for poly in closed_contours:
            if len(poly) >= 2:
                closed = np.vstack([poly, poly[:1]])
                ax.plot(closed[:, 0], closed[:, 1],
                        color="lime", linewidth=1.2, alpha=0.9)

    # 4b. Silhouette scatter (coronal / sagittal views)
    if scatter_pts is not None and len(scatter_pts) > 0:
        ax.scatter(scatter_pts[:, 0], scatter_pts[:, 1],
                   color="lime", s=3, alpha=0.8, linewidths=0, zorder=3)

    # 5. Isocenter (clipped to view – not shown if outside window)
    ax.plot(iso_h, iso_v, "+", color="red", markersize=10,
            markeredgewidth=1.5, zorder=5, clip_on=True)

    # 6. Centroid
    ax.plot(cx, cy, "x", color="cyan", markersize=6,
            markeredgewidth=1.2, zorder=5)

    # 7b. Computation bounding box (dashed white rectangle)
    if bbox_rect is not None and SHOW_COMPUTATION_BOUNDARY:
        bh0, bh1, bv0, bv1 = bbox_rect
        from matplotlib.patches import Rectangle
        rect = Rectangle((bh0, bv0), bh1 - bh0, bv1 - bv0,
                         linewidth=0.8, edgecolor="white", facecolor="none",
                         linestyle="dashed", alpha=0.45, zorder=4)
        ax.add_patch(rect)

    # 7. 10 mm scale bar (bottom-right)
    sb_x1 = cx + half_mm * 0.55
    sb_x0 = sb_x1 - 10.0
    sb_y  = cy - half_mm * 0.86
    ax.plot([sb_x0, sb_x1], [sb_y, sb_y], "-", color="white", linewidth=2, zorder=6)
    ax.text((sb_x0 + sb_x1) / 2, sb_y + half_mm * 0.04, "10 mm",
            color="white", fontsize=5, ha="center", va="bottom", zorder=6)

    ax.set_xlim(cx - half_mm, cx + half_mm)
    ax.set_ylim(cy - half_mm, cy + half_mm)
    ax.set_aspect("equal", adjustable="box")   # 1 mm = 1 mm on both axes
    ax.set_xlabel(h_label, color="#aaa", fontsize=7)
    ax.set_ylabel(v_label, color="#aaa", fontsize=7)
    ax.tick_params(colors="#888", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    ax.set_title(title, color="white", fontsize=8, pad=3)
    plt.tight_layout(pad=0.3)
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_structure_views(
    patient_folder: str,
    plan_type: str,
    structure: Structure,
    dose: DoseData,
    isocenter: np.ndarray,
    rx_dose: float,
    scenario: str = "nominal",
    nominal_dose: Optional[DoseData] = None,
) -> Dict[str, str]:
    """
    Generate axial / coronal / sagittal PNG views for *structure* and save them
    under VIEWS_OUTPUT_DIR/<patient_folder>/<plan_type>/<safe_name>/<scenario>/.

    Returns a dict  {'axial': path, 'coronal': path, 'sagittal': path}
    (only keys for successfully generated files are included).
    Returns {} if ENABLE_VIEWS is False or on unrecoverable error.
    """
    if not ENABLE_VIEWS:
        return {}
    if dose is None:
        return {}

    centroid = compute_centroid(structure)
    if centroid is None:
        logger.warning("Views: no centroid for %s/%s/%s – skipped.",
                       patient_folder, plan_type, structure.name)
        return {}

    cx = float(centroid[0])
    cy = float(centroid[1])
    cz = float(centroid[2])
    ix = float(isocenter[0])
    iy = float(isocenter[1])
    iz = float(isocenter[2])

    # Computation bounding box (structure bbox + BOUNDARY_FIXED_MARGIN_MM)
    all_pts = np.vstack(structure.contours)
    bb_min = all_pts.min(axis=0) - BOUNDARY_FIXED_MARGIN_MM
    bb_max = all_pts.max(axis=0) + BOUNDARY_FIXED_MARGIN_MM
    # (x_min, x_max, y_min, y_max, z_min, z_max)
    bbox_axial   = (bb_min[0], bb_max[0], bb_min[1], bb_max[1])
    bbox_coronal = (bb_min[0], bb_max[0], bb_min[2], bb_max[2])
    bbox_sagittal= (bb_min[1], bb_max[1], bb_min[2], bb_max[2])

    # Build output directory (flattened: no scenario subfolder)
    safe_name = (structure.name
                 .replace(" ", "_").replace("/", "_").replace("\\", "_")
                 .replace(":", "_").replace("*", "_").replace("?", "_"))
    safe_sc = scenario.replace(" ", "_").replace("/", "-")
    out_dir = os.path.join(VIEWS_OUTPUT_DIR, patient_folder, plan_type, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    try:
        # Extract nominal slices for overlay (shift scenarios only)
        nom_axial = nom_coronal = nom_sagittal = None
        if nominal_dose is not None and scenario != "nominal":
            nom_axial, _, _ = _axial_slice(nominal_dose, cz)
            nom_coronal, _, _ = _coronal_slice(nominal_dose, cy)
            nom_sagittal, _, _ = _sagittal_slice(nominal_dose, cx)

        # ── Axial (Z-slice through centroid) ──────────────────────────────────
        sl, hc, vc = _axial_slice(dose, cz)
        contours = _axial_contours(structure, cz)
        fig = _render_view(
            f"Axial  z={cz:.0f} mm",
            sl, hc, vc, "X [mm]", "Y [mm]",
            cx, cy, ix, iy,
            rx_dose, VIEW_HALF_MM,
            closed_contours=contours,
            bbox_rect=bbox_axial,
            nominal_slice=nom_axial,
        )
        p = os.path.join(out_dir, f"axial_{safe_sc}.png")
        fig.savefig(p, dpi=VIEW_DPI, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        paths["axial"] = p

        # ── Coronal (Y-slice through centroid) ────────────────────────────────
        sl, hc, vc = _coronal_slice(dose, cy)
        sil = _coronal_silhouette(structure, cy)
        fig = _render_view(
            f"Coronal  y={cy:.0f} mm",
            sl, hc, vc, "X [mm]", "Z [mm]",
            cx, cz, ix, iz,
            rx_dose, VIEW_HALF_MM,
            scatter_pts=sil,
            bbox_rect=bbox_coronal,
            nominal_slice=nom_coronal,
        )
        p = os.path.join(out_dir, f"coronal_{safe_sc}.png")
        fig.savefig(p, dpi=VIEW_DPI, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        paths["coronal"] = p

        # ── Sagittal (X-slice through centroid) ───────────────────────────────
        sl, hc, vc = _sagittal_slice(dose, cx)
        sil = _sagittal_silhouette(structure, cx)
        fig = _render_view(
            f"Sagittal  x={cx:.0f} mm",
            sl, hc, vc, "Y [mm]", "Z [mm]",
            cy, cz, iy, iz,
            rx_dose, VIEW_HALF_MM,
            scatter_pts=sil,
            bbox_rect=bbox_sagittal,
            nominal_slice=nom_sagittal,
        )
        p = os.path.join(out_dir, f"sagittal_{safe_sc}.png")
        fig.savefig(p, dpi=VIEW_DPI, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        paths["sagittal"] = p

        logger.info("  Views saved: %s/%s/%s",
                    patient_folder, plan_type, structure.name)

    except Exception as exc:
        logger.error("Views render error for %s/%s/%s: %s",
                     patient_folder, plan_type, structure.name, exc)

    return paths
