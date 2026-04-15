"""
SRS Analysis Pipeline – 3D Structure vs Isodose Visualisation  (v4)
=====================================================================

One PNG per patient / plan-type / scenario (nominal + each shifted).

Layout
─────────────────────────────────────────────────────────────────
  Full width  – 1 × 2 grid of 3-D orthographic views
      [0] Front-Left   elev=25°  az=-55°
      [1] Front-Right  elev=25°  az=125°

Drawing layers (back → front, enables protrusion effect)
─────────────────────────────────────────────────────────────────
  Nominal plot  (struct vs isodose):
      1. Green  – nominal 100%-Rx isodose
      2. Gray   – PTV structure  (covers isodose inside)

  Shifted plot  (nominal vs shifted isodose):
      1. Magenta – shifted 100%-Rx isodose protrusion
      2. Green   – nominal 100%-Rx isodose  (covers magenta inside nominal)
      3. Gray    – PTV structure             (covers both inside structure)

  Isocenter  → ★ marker (drawn last, always visible)

Shading
─────────────────────────────────────────────────────────────────
  Each polygon is shaded by a light-source model:
    intensity = diffuse * |cos θ| + ambient
  Edge outlines are drawn at reduced alpha for depth cues.

Dose-shift convention  (matches shift_scenarios.py)
─────────────────────────────────────────────────────────────────
  Forward transform:  p' = R @ (p − iso) + iso + [dx, dy, dz]
  R = Rz(rz) @ Ry(ry) @ Rx(rx)   (extrinsic, degrees)
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Visual constants
# ─────────────────────────────────────────────────────────────────────────────

_STRUCT_COLOR     = "#b0b0b0"   # PTV structure – gray
_ISO_NOM_COLOR    = "#00cc44"   # nominal 100%-Rx isodose – green
_ISO_SHIFT_COLOR  = "#ee00cc"   # shifted isodose protrusion – magenta
_ISO_MARKER       = "#ffffff"   # isocenter ★
_BG               = "#0d1117"   # figure background
_LABEL_COLOR      = "#dddddd"   # structure name text
_GRID_COLOR       = "#1c1c2c"   # axes grid lines
_PANE_EDGE        = "#252535"   # 3-D pane edges

# Light-source direction (azimuth / altitude in degrees)
_LIGHT_AZ  = 225.0
_LIGHT_ALT = 55.0
_AMBIENT   = 0.35   # minimum brightness fraction
_DIFFUSE   = 0.65   # diffuse contribution

_MAX_PTS      = 40    # max points per structure contour (decimate for 2-D panels)
_ISO_MAX_PTS  = 35    # max points per isodose contour  (decimate for 2-D panels)

try:
    from config import PLOT_3D_MAX_FACES  as _MAX_FACES
    from config import PLOT_3D_ISO_STRIDE as _ISO_Z_STRIDE
except ImportError:
    _MAX_FACES    = 12000
    _ISO_Z_STRIDE = 1

# Two 3-D viewpoints (elevation°, azimuth°, label)
_VIEWS = [
    (25,  -55, "Front-Left"),
    (25,  125, "Front-Right"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """3×3 rotation matrix – extrinsic Rx → Ry → Rz (degrees)."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(rx), -np.sin(rx)],
                   [0,  np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

# keep _transform_contours for 2D projections


def _transform_contours(contours: list,
                         isocenter: np.ndarray,
                         scenario) -> list:
    """Forward-transform contour points by scenario's 6-DOF dose shift."""
    t = np.array([scenario.dx_mm, scenario.dy_mm, scenario.dz_mm], dtype=float)
    R = _rotation_matrix(scenario.rx_deg, scenario.ry_deg, scenario.rz_deg)
    out = []
    for c in contours:
        pts = np.asarray(c, dtype=float)
        if len(pts) == 0:
            continue
        out.append((R @ (pts - isocenter).T).T + isocenter + t)
    return out


def _transform_verts(verts_xyz: np.ndarray,
                     isocenter: np.ndarray,
                     scenario) -> np.ndarray:
    """Forward-transform mesh vertices by scenario's 6-DOF dose shift."""
    t = np.array([scenario.dx_mm, scenario.dy_mm, scenario.dz_mm], dtype=float)
    R = _rotation_matrix(scenario.rx_deg, scenario.ry_deg, scenario.rz_deg)
    return (R @ (verts_xyz - isocenter).T).T + isocenter + t


def _decimate(contours: list, max_pts: int = _MAX_PTS) -> list:
    """Uniformly subsample each contour to ≤ max_pts points."""
    result = []
    for c in contours:
        pts = np.asarray(c, dtype=float)
        if len(pts) == 0:
            continue
        if len(pts) > max_pts:
            idx = np.round(np.linspace(0, len(pts) - 1, max_pts)).astype(int)
            pts = pts[idx]
        result.append(pts)
    return result


def _centroid(contours: list) -> Optional[np.ndarray]:
    """Mean of all contour points; None if no points."""
    stacks = [np.asarray(c) for c in contours if len(c) > 0]
    return np.vstack(stacks).mean(axis=0) if stacks else None


# ─────────────────────────────────────────────────────────────────────────────
# 2-D isodose contour extraction  (used for 2-D projection panels only)
# ─────────────────────────────────────────────────────────────────────────────

def _find_contours_2d(x_coords: np.ndarray, y_coords: np.ndarray,
                       dose_slice: np.ndarray, level: float) -> list:
    """Contours at *level* in a dose slice → list of (N,2) in (x_mm, y_mm)."""
    try:
        from skimage.measure import find_contours as _skfind
        raw = _skfind(dose_slice, level)
        result = []
        for c in raw:
            xm = np.interp(c[:, 1], np.arange(len(x_coords)), x_coords)
            ym = np.interp(c[:, 0], np.arange(len(y_coords)), y_coords)
            result.append(np.column_stack([xm, ym]))
        return result
    except ImportError:
        pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_t, ax_t = plt.subplots()
    try:
        cs = ax_t.contour(x_coords, y_coords, dose_slice, levels=[level])
        result = []
        for col in cs.collections:
            for path in col.get_paths():
                if len(path.vertices) >= 3:
                    result.append(path.vertices.copy())
        return result
    finally:
        plt.close(fig_t)


def _extract_isodose_contours(dose, rx_dose: float) -> list:
    """Slice-by-slice 100%-Rx contours → list of (N,3) xyz arrays."""
    if dose is None or rx_dose is None or rx_dose <= 0:
        return []
    ox, oy, oz = dose.origin
    dx, dy, dz = dose.spacing
    nz, ny, nx = dose.dose_grid.shape
    x_coords = ox + np.arange(nx) * dx
    y_coords = oy + np.arange(ny) * dy
    contours = []
    for zi in range(0, nz, _ISO_Z_STRIDE):
        z_mm = oz + zi * dz
        dose_slice = dose.dose_grid[zi]
        if dose_slice.max() < rx_dose:
            continue
        raw = _find_contours_2d(x_coords, y_coords, dose_slice, rx_dose)
        for c in raw:
            if len(c) < 3:
                continue
            pts = np.hstack([c, np.full((len(c), 1), z_mm)])
            if len(pts) > _ISO_MAX_PTS:
                idx = np.round(
                    np.linspace(0, len(pts) - 1, _ISO_MAX_PTS)).astype(int)
                pts = pts[idx]
            contours.append(pts)
    return contours


# ─────────────────────────────────────────────────────────────────────────────
# Marching-cubes smooth 3-D surfaces
# ─────────────────────────────────────────────────────────────────────────────

def _mc_import():
    """Return skimage.measure.marching_cubes or None."""
    try:
        from skimage.measure import marching_cubes
        return marching_cubes
    except ImportError:
        pass
    try:
        from skimage.measure import marching_cubes_lewiner as marching_cubes
        return marching_cubes
    except ImportError:
        return None


def _verts_to_xyz(verts: np.ndarray, origin, spacing) -> np.ndarray:
    """
    Convert marching-cubes verts (axis-0=z, 1=y, 2=x, in mm from grid origin)
    to patient xyz coordinates.
    """
    ox, oy, oz = origin
    return np.column_stack([
        ox + verts[:, 2],
        oy + verts[:, 1],
        oz + verts[:, 0],
    ])


def _decimate_faces(faces: np.ndarray, max_faces: int = _MAX_FACES) -> np.ndarray:
    """Uniformly subsample face indices to ≤ max_faces."""
    if len(faces) <= max_faces:
        return faces
    idx = np.round(np.linspace(0, len(faces) - 1, max_faces)).astype(int)
    return faces[idx]


def _marching_cubes_dose(dose, rx_dose: float,
                          stride: Optional[int] = None,
                          max_faces: Optional[int] = None,
                          ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Smooth 100%-Rx isodose surface via marching cubes on the dose grid.
    stride/max_faces default to the module-level _ISO_Z_STRIDE / _MAX_FACES.
    """
    mc = _mc_import()
    if mc is None:
        return None, None

    if stride is None:
        stride = _ISO_Z_STRIDE
    if max_faces is None:
        max_faces = _MAX_FACES

    dz, dy, dx = dose.spacing[2], dose.spacing[1], dose.spacing[0]
    grid = dose.dose_grid[::stride, ::stride, ::stride]
    sp   = (dz * stride, dy * stride, dx * stride)

    sigma = max(0.3, 0.5 / stride)
    try:
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid.astype(float), sigma=sigma)
    except ImportError:
        grid = grid.astype(float)

    try:
        verts, faces, _, _ = mc(grid, level=rx_dose, spacing=sp,
                                 allow_degenerate=False)
    except Exception as exc:
        logger.warning("marching_cubes dose: %s", exc)
        return None, None

    if len(verts) == 0:
        return None, None

    verts_xyz = _verts_to_xyz(verts, dose.origin, dose.spacing)
    faces     = _decimate_faces(faces, max_faces=max_faces)
    return verts_xyz, faces


def _voxelize_and_mesh_structures(
    structures, active_names: set, dose,
    stride: int = 1,
    max_faces: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Rasterise active PTV contours onto the dose voxel grid and extract a
    smooth mesh via marching cubes.
    Returns (verts_xyz, faces) or (None, None).
    """
    mc = _mc_import()
    if mc is None:
        return None, None

    if max_faces is None:
        max_faces = _MAX_FACES

    try:
        from skimage.draw import polygon as sk_polygon
    except ImportError:
        return None, None

    ox, oy, oz = dose.origin
    dx, dy, dz = dose.spacing
    nz, ny, nx = dose.dose_grid.shape

    mask = np.zeros((nz, ny, nx), dtype=np.uint8)

    for struct in structures:
        if struct.name not in active_names:
            continue
        for c in struct.contours:
            pts = np.asarray(c)
            if len(pts) < 3:
                continue
            z_mm = float(pts[0, 2])
            zi   = int(round((z_mm - oz) / dz))
            if not (0 <= zi < nz):
                continue
            xi = (pts[:, 0] - ox) / dx
            yi = (pts[:, 1] - oy) / dy
            try:
                rr, cc = sk_polygon(yi, xi, shape=(ny, nx))
                mask[zi, rr, cc] = 1
            except Exception:
                continue

    if mask.sum() == 0:
        return None, None

    sigma = max(0.4, 0.8 / max(stride, 1))
    try:
        from scipy.ndimage import binary_fill_holes, gaussian_filter
        for zi in range(nz):
            if mask[zi].any():
                mask[zi] = binary_fill_holes(mask[zi]).astype(np.uint8)
        mask_f = gaussian_filter(mask.astype(float), sigma=sigma)
    except ImportError:
        mask_f = mask.astype(float)

    grid = mask_f[::stride, ::stride, ::stride]
    sp   = (dz * stride, dy * stride, dx * stride)

    try:
        verts, faces, _, _ = mc(grid, level=0.5, spacing=sp,
                                 allow_degenerate=False)
    except Exception as exc:
        logger.warning("marching_cubes struct: %s", exc)
        return None, None

    if len(verts) == 0:
        return None, None

    verts_xyz = _verts_to_xyz(verts, dose.origin, dose.spacing)
    faces     = _decimate_faces(faces, max_faces=max_faces)
    return verts_xyz, faces


def _voxelize_and_mesh_per_structure(
    structures, active_names: set, dose,
    stride: int = 1,
    max_faces: Optional[int] = None,
) -> List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Like _voxelize_and_mesh_structures but returns one (name, verts, faces)
    tuple per active structure instead of merging them all.
    Structures with zero voxels are skipped.
    """
    mc = _mc_import()
    if mc is None:
        return []

    if max_faces is None:
        max_faces = _MAX_FACES

    try:
        from skimage.draw import polygon as sk_polygon
    except ImportError:
        return []

    ox, oy, oz = dose.origin
    dx, dy, dz = dose.spacing
    nz, ny, nx = dose.dose_grid.shape

    results: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]] = []

    for struct in structures:
        if struct.name not in active_names:
            continue

        mask = np.zeros((nz, ny, nx), dtype=np.uint8)
        for c in struct.contours:
            pts = np.asarray(c)
            if len(pts) < 3:
                continue
            z_mm = float(pts[0, 2])
            zi   = int(round((z_mm - oz) / dz))
            if not (0 <= zi < nz):
                continue
            xi = (pts[:, 0] - ox) / dx
            yi = (pts[:, 1] - oy) / dy
            try:
                rr, cc = sk_polygon(yi, xi, shape=(ny, nx))
                mask[zi, rr, cc] = 1
            except Exception:
                continue

        if mask.sum() == 0:
            continue

        sigma = max(0.4, 0.8 / max(stride, 1))
        try:
            from scipy.ndimage import binary_fill_holes, gaussian_filter
            for zi in range(nz):
                if mask[zi].any():
                    mask[zi] = binary_fill_holes(mask[zi]).astype(np.uint8)
            mask_f = gaussian_filter(mask.astype(float), sigma=sigma)
        except ImportError:
            mask_f = mask.astype(float)

        grid = mask_f[::stride, ::stride, ::stride]
        sp   = (dz * stride, dy * stride, dx * stride)

        try:
            verts, faces, _, _ = mc(grid, level=0.5, spacing=sp,
                                     allow_degenerate=False)
        except Exception:
            continue

        if len(verts) == 0:
            continue

        verts_xyz = _verts_to_xyz(verts, dose.origin, dose.spacing)
        faces     = _decimate_faces(faces, max_faces=max_faces)
        results.append((struct.name, verts_xyz, faces))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Shading + 3-D mesh rendering
# ─────────────────────────────────────────────────────────────────────────────

_LIGHT_VEC: Optional[np.ndarray] = None


def _get_light_vec() -> np.ndarray:
    global _LIGHT_VEC
    if _LIGHT_VEC is None:
        az  = np.radians(_LIGHT_AZ)
        alt = np.radians(_LIGHT_ALT)
        _LIGHT_VEC = np.array([np.cos(alt) * np.cos(az),
                                np.cos(alt) * np.sin(az),
                                np.sin(alt)])
    return _LIGHT_VEC


def _shade_mesh(verts_xyz: np.ndarray, faces: np.ndarray,
                base_hex: str, alpha: float = 0.90) -> list:
    """Per-triangle RGBA facecolors using Lambertian shading on normals."""
    from matplotlib.colors import to_rgb
    lv   = _get_light_vec()
    base = np.array(to_rgb(base_hex))

    v0 = verts_xyz[faces[:, 0]]
    v1 = verts_xyz[faces[:, 1]]
    v2 = verts_xyz[faces[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(nrm, axis=1, keepdims=True)
    mag = np.where(mag < 1e-12, 1.0, mag)
    nrm = nrm / mag

    cos_t = np.abs(nrm @ lv)
    intensity = _AMBIENT + _DIFFUSE * cos_t

    rgb_all = np.clip(base[np.newaxis, :] * intensity[:, np.newaxis], 0, 1)
    return [np.append(rgb, alpha) for rgb in rgb_all]


def _plot_mesh_3d(ax, verts_xyz: Optional[np.ndarray],
                   faces: Optional[np.ndarray],
                   color: str, alpha: float = 0.90) -> None:
    """Render a triangular mesh with Lambertian shading."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if verts_xyz is None or faces is None or len(faces) == 0:
        return
    fc = _shade_mesh(verts_xyz, faces, color, alpha)
    from matplotlib.colors import to_rgb
    ec_rgb = np.clip(np.array(to_rgb(color)) * 0.5, 0, 1)
    ec = np.append(ec_rgb, 0.25)
    polys = verts_xyz[faces]
    coll  = Poly3DCollection(list(polys))
    coll.set_facecolor(fc)
    coll.set_edgecolor([ec] * len(faces))
    coll.set_linewidth(0.10)
    ax.add_collection3d(coll)


def _struct_labels_3d(ax, structures, active_names: set) -> None:
    """Text label at the centroid of each active PTV on first 3-D view."""
    for struct in structures:
        if struct.name not in active_names:
            continue
        c = _centroid(_decimate(struct.contours))
        if c is None:
            continue
        short = struct.name[:20] + ("…" if len(struct.name) > 20 else "")
        ax.text(c[0], c[1], c[2], short,
                color=_LABEL_COLOR, fontsize=5.5, ha="center", alpha=0.90,
                bbox=dict(boxstyle="round,pad=0.15", fc=_BG, ec="none",
                          alpha=0.55))


# ─────────────────────────────────────────────────────────────────────────────
# 2-D projection helpers  (outline-only, dashed/solid)
# ─────────────────────────────────────────────────────────────────────────────

def _proj2d(contours: list, dim_x: int, dim_y: int) -> list:
    return [np.asarray(c)[:, [dim_x, dim_y]] for c in contours if len(c) >= 2]


def _plot_outline_2d(ax2, contours_2d: list, color: str,
                     lw: float = 1.3, alpha: float = 0.90,
                     ls: str = "-") -> None:
    """Closed outline for each contour in 2-D (fill=none)."""
    for c in contours_2d:
        if len(c) >= 3:
            cl = np.vstack([c, c[0]])
            ax2.plot(cl[:, 0], cl[:, 1], color=color,
                     linewidth=lw, alpha=alpha, linestyle=ls)


# ─────────────────────────────────────────────────────────────────────────────
# Axis styling
# ─────────────────────────────────────────────────────────────────────────────

def _style_3d(ax, view_label: str = "") -> None:
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_facecolor(_BG)
    for axis_obj in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis_obj.pane.fill = False
        axis_obj.pane.set_edgecolor(_PANE_EDGE)
        axis_obj._axinfo["grid"]["color"] = _GRID_COLOR
    ax.set_xlabel("X", color="#555555", fontsize=6, labelpad=2)
    ax.set_ylabel("Y", color="#555555", fontsize=6, labelpad=2)
    ax.set_zlabel("Z", color="#555555", fontsize=6, labelpad=2)
    ax.tick_params(colors="#444444", labelsize=4)
    if view_label:
        ax.set_title(view_label, color="#888888", fontsize=7, pad=2)


def _style_2d(ax2, xlabel: str, ylabel: str, title: str) -> None:
    ax2.set_facecolor(_BG)
    ax2.set_xlabel(xlabel, color="#777777", fontsize=6, labelpad=2)
    ax2.set_ylabel(ylabel, color="#777777", fontsize=6, labelpad=2)
    ax2.set_title(title, color="#aaaaaa", fontsize=7, pad=2)
    ax2.tick_params(colors="#444444", labelsize=4)
    ax2.set_aspect("equal", adjustable="datalim")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333344")
    ax2.grid(True, color=_GRID_COLOR, linewidth=0.4, alpha=0.7)


def _apply_equal_axes(ax3d_list: list,
                       ref_verts_list: list) -> None:
    """
    Set identical equal-scaled limits on every 3-D axis so all four views
    show the same spatial region at the same scale.
    """
    all_pts = [v for v in ref_verts_list if v is not None and len(v) > 0]
    if not all_pts:
        return
    combined = np.vstack(all_pts)
    lo, hi   = combined.min(axis=0), combined.max(axis=0)
    center   = (lo + hi) * 0.5
    half     = float(np.max(hi - lo)) * 0.55  # 10% margin
    for ax in ax3d_list:
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)


# ─────────────────────────────────────────────────────────────────────────────
# Rx dose resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_rx_for_plan(mapping_df, patient_folder: str,
                          plan_type: str) -> Optional[float]:
    """Return the highest active prescription dose for this plan."""
    import pandas as pd
    mask = (
        (mapping_df["PatientFolder"] == patient_folder) &
        (mapping_df["PlanType"] == plan_type)
    )
    rows = mapping_df[mask]
    for col in ("Prescription_Gy_reference", "Prescription_Gy_detected"):
        if col in rows.columns:
            vals = pd.to_numeric(rows[col], errors="coerce").dropna()
            if len(vals) > 0:
                return float(vals.max())
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core figure renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_one_plot(
    mesh_layers: list,         # [(verts, faces, color, legend), ...] back→front
    isocenter: np.ndarray,
    structures,
    active_names: set,
    title: str,
    output_dir: str,
    file_stem: str,
    axis_limits: Optional[tuple] = None,  # (xmin, xmax, ymin, ymax, zmin, zmax)
) -> Optional[str]:
    """
    2-subplot figure: two side-by-side Lambertian-shaded 3-D views.
    axis_limits enforces exact same limits across all scenarios.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(18, 9), facecolor=_BG)
    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.02, right=0.98,
                           top=0.92, bottom=0.05,
                           wspace=0.08)

    ax3d_list = [fig.add_subplot(gs[i], projection="3d") for i in range(2)]

    for ax3d, (elev, azim, vlabel) in zip(ax3d_list, _VIEWS):
        _style_3d(ax3d, vlabel)
        ax3d.view_init(elev=elev, azim=azim)

    # ── 3-D: mesh layers back → front ────────────────────────────────────
    for verts, faces, color, _ in mesh_layers:
        for ax3d in ax3d_list:
            _plot_mesh_3d(ax3d, verts, faces, color)

    # ── Equal axis scaling ────────────────────────────────────────────────
    if axis_limits is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = axis_limits
        for ax in ax3d_list:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
    else:
        all_verts = [v for v, _, _, _ in mesh_layers if v is not None]
        _apply_equal_axes(ax3d_list, all_verts)

    # ── 3-D: isocenter ────────────────────────────────────────────────────
    for ax3d in ax3d_list:
        ax3d.scatter(*isocenter, color=_ISO_MARKER, s=140, marker="*",
                     zorder=20, depthshade=False)

    # ── Labels + legend (first 3-D view only) ────────────────────────────
    _struct_labels_3d(ax3d_list[0], structures, active_names)
    ax3d_list[0].scatter([], [], [], color=_ISO_MARKER, s=70,
                         marker="*", label="Isocenter")
    for _, _, color, label in mesh_layers:
        ax3d_list[0].plot([], [], [], color=color, linewidth=9,
                          alpha=0.85, label=label)
    handles, labels_ = ax3d_list[0].get_legend_handles_labels()
    if handles:
        ax3d_list[0].legend(handles, labels_,
                             loc="upper left", bbox_to_anchor=(0.0, 1.0),
                             fontsize=7.5, framealpha=0.35,
                             labelcolor="white",
                             facecolor="#1a1a2e", edgecolor="#444466")

    fig.suptitle(title, color="white", fontsize=11, y=0.975, x=0.50)

    out_path = os.path.join(output_dir, f"{file_stem}.png")
    try:
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=_BG)
        return out_path
    except Exception as exc:
        logger.error("3D plot save failed %s: %s", out_path, exc)
        return None
    finally:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_3d_plots(
    plan_files_list,
    mapping_df,
    scenarios,
    output_dir: str,
    render_mode: str = "isodose_protrusion",
) -> List[str]:
    """
    Generate one PNG per patient / plan-type / scenario.

    Nominal plot   – smooth gray PTV mesh + smooth green isodose mesh
    Shifted plots  – smooth green nominal isodose + smooth magenta shifted
                     isodose (no structure shown in shifted plots)

    Layout: 2 × 3-D Lambertian-shaded views (equal axes) + 3 × 2-D outlines.
    2-D outlines:
      nominal  – structure dashed gray, isodose dashed green
      shifted  – nominal isodose solid green, shifted isodose dashed magenta
    """
    from dicom_io import load_plan
    from excel_io import get_active_structures

    global _LIGHT_VEC
    _LIGHT_VEC = None

    os.makedirs(output_dir, exist_ok=True)
    active_df    = get_active_structures(mapping_df)
    paths: List[str] = []

    all_scenarios = list(scenarios)
    if not any(s.name == "nominal" for s in all_scenarios):
        from shift_scenarios import ShiftScenario
        all_scenarios = [ShiftScenario(name="nominal")] + all_scenarios

    for pf in plan_files_list:
        mask = (
            (active_df["PatientFolder"] == pf.patient_folder) &
            (active_df["PlanType"]      == pf.plan_type)
        )
        if active_df[mask].empty:
            continue
        active_names = set(active_df[mask]["StructureName_Original"].tolist())

        try:
            structures, dose, meta = load_plan(pf)
        except Exception as exc:
            logger.warning("3D plot: load failed %s/%s: %s",
                           pf.patient_folder, pf.plan_type, exc)
            continue

        if meta is None or meta.isocenter is None:
            logger.warning("3D plot: no isocenter %s/%s – skipped.",
                           pf.patient_folder, pf.plan_type)
            continue
        if dose is None:
            logger.warning("3D plot: no dose %s/%s – skipped.",
                           pf.patient_folder, pf.plan_type)
            continue

        isocenter = meta.isocenter
        rx_dose   = _resolve_rx_for_plan(mapping_df, pf.patient_folder,
                                          pf.plan_type)
        if rx_dose is None or rx_dose <= 0:
            logger.warning("3D plot: no Rx dose %s/%s – skipped.",
                           pf.patient_folder, pf.plan_type)
            continue

        # ── Build smooth meshes (once per plan) ───────────────────────────
        logger.info("3D plot: building isodose mesh %s/%s (%.0f Gy) …",
                    pf.patient_folder, pf.plan_type, rx_dose)
        iso_verts, iso_faces = _marching_cubes_dose(dose, rx_dose)
        if iso_verts is None:
            logger.warning("3D plot: isodose mesh empty %s/%s – skipped.",
                           pf.patient_folder, pf.plan_type)
            continue
        logger.info("3D plot:   isodose mesh: %d faces", len(iso_faces))

        logger.info("3D plot: building structure mesh %s/%s …",
                    pf.patient_folder, pf.plan_type)
        str_verts, str_faces = _voxelize_and_mesh_structures(
            structures, active_names, dose)
        if str_verts is not None:
            logger.info("3D plot:   structure mesh: %d faces", len(str_faces))
        else:
            logger.warning("3D plot:   structure mesh failed – nominal "
                           "plot will have no structure mesh.")

        # ── Compute axis limits once (from nominal meshes) ────────────────
        all_ref_verts = [iso_verts]
        if str_verts is not None:
            all_ref_verts.append(str_verts)
        
        combined = np.vstack([v for v in all_ref_verts if v is not None and len(v) > 0])
        lo, hi   = combined.min(axis=0), combined.max(axis=0)
        center   = (lo + hi) * 0.5
        half     = float(np.max(hi - lo)) * 0.55
        axis_limits = (
            center[0] - half, center[0] + half,
            center[1] - half, center[1] + half,
            center[2] - half, center[2] + half,
        )
        logger.info("3D plot: axis limits computed: x=[%.1f, %.1f], y=[%.1f, %.1f], z=[%.1f, %.1f]",
                    *axis_limits)

        folder_clean = (pf.patient_folder[3:]
                        if pf.patient_folder.startswith("zz_")
                        else pf.patient_folder)
        plan_label = f"{folder_clean} / {pf.plan_type}"
        safe_pat   = pf.patient_folder.replace(" ", "_")

        for sc in all_scenarios:
            is_nominal = (
                sc.name == "nominal" or (
                    sc.dx_mm == 0 and sc.dy_mm == 0 and sc.dz_mm == 0 and
                    sc.rx_deg == 0 and sc.ry_deg == 0 and sc.rz_deg == 0
                )
            )
            safe_sc   = sc.name.replace(" ", "_").replace("/", "-")
            file_stem = f"{safe_pat}__{pf.plan_type}__{safe_sc}"

            if is_nominal:
                # Nominal: green isodose (back) + gray struct (front)
                mesh_layers = [
                    (iso_verts, iso_faces, _ISO_NOM_COLOR,
                     f"Nominal {rx_dose:.0f} Gy isodose"),
                    (str_verts, str_faces, _STRUCT_COLOR,
                     "PTV structure"),
                ]
                sc_str = "Nominal  (structure vs isodose)"

            else:
                # Shifted: transform isodose mesh verts
                iso_sh_verts = _transform_verts(iso_verts, isocenter, sc)

                has_rot = bool(sc.rx_deg or sc.ry_deg or sc.rz_deg)
                sc_str = (
                    f"{sc.name}  "
                    f"Δ=({sc.dx_mm:+.1f},{sc.dy_mm:+.1f},"
                    f"{sc.dz_mm:+.1f}) mm"
                )
                if has_rot:
                    sc_str += (
                        f"  rot=({sc.rx_deg:.1f},"
                        f"{sc.ry_deg:.1f},{sc.rz_deg:.1f})°"
                    )
                # Shifted: magenta (back) + green nominal (front) – no struct
                mesh_layers = [
                    (iso_sh_verts, iso_faces, _ISO_SHIFT_COLOR,
                     "Shifted isodose protrusion"),
                    (iso_verts,    iso_faces, _ISO_NOM_COLOR,
                     f"Nominal {rx_dose:.0f} Gy isodose"),
                ]

            out = _render_one_plot(
                mesh_layers=mesh_layers,
                isocenter=isocenter,
                structures=structures,
                active_names=active_names,
                title=f"{plan_label}  ·  {sc_str}",
                output_dir=output_dir,
                file_stem=file_stem,
                axis_limits=axis_limits,
            )
            if out:
                paths.append(out)
                logger.info("3D plot saved: %s", out)

    logger.info("3D plots: %d PNG(s) written to %s", len(paths), output_dir)
    return paths
