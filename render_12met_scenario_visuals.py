from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
ANALYZER_DIR = ROOT
DEFAULT_DATA_ROOT = ROOT.parent
DEFAULT_OUTPUT_DIR = ROOT / "output" / "rigid_error_case"
if str(ANALYZER_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYZER_DIR))

from brain_normal_export import build_target_maps, clean_text  # noqa: E402
from brain_normal_metrics import active_numbers_from_plan_label  # noqa: E402
from dicom_io import load_dose, load_plan_meta, load_structures  # noqa: E402
import plot_3d as p3d  # noqa: E402
from rigid_error_impact import (  # noqa: E402
    discover_plan_sets,
    forward_transform_point,
    measured_translation_scenario,
    rotation_scenario,
    sample_dose_at_points,
)
from shift_scenarios import ShiftScenario, build_shifted_dose  # noqa: E402
from structure_mapping import compute_centroid  # noqa: E402


BG = "#070b12"
NOMINAL_COLOR = "#15d56f"
SHIFTED_COLOR = "#ff2aa3"
PTV_COLOR = "#dce6ee"
VECTOR_COLOR = "#f9d84a"
ISO_MARKER = "#ffffff"
ACTIVE_SCENARIO_COLOR = "#39e56d"
INACTIVE_SCENARIO_COLOR = "#4f5663"
DIFF_SHIFTED_COLOR = "#ff3aa8"
UNCOVERED_COLOR = "#00f5ff"
CAMERA_DISTANCE = 7.0
COMBINED_FIGSIZE = (8.4, 9.0)
ISO_LAYER_NAMES = {
    "Nominale Isodose",
    "Verschobene Isodose",
    "Szenario-Isodose",
    "Iso-Interaktion",
    "Abweichung zur Nominalisodose",
}

OAR_COLORS = {
    "Eye Left": "#38d7ff",
    "Eye Right": "#38d7ff",
    "Optic Nerve Left": "#ffd84a",
    "Optic Nerve Right": "#ffd84a",
    "Chiasm": "#b96cff",
    "Brainstem": "#62d46f",
}


@dataclass(frozen=True)
class MeshLayer:
    name: str
    verts: np.ndarray
    faces: np.ndarray
    color: str
    alpha: float
    facecolors: np.ndarray | None = None


def _sanitize(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return re.sub(r"_+", "_", text).strip("_")


def _centroid(structure) -> np.ndarray:
    c = compute_centroid(structure)
    if c is None:
        return np.vstack(structure.contours).mean(axis=0)
    return np.asarray(c, dtype=float)


def _scenario_title(sc: ShiftScenario) -> str:
    if sc.name.startswith("rot_total_1"):
        return "Rotation 1 deg"
    if sc.name.startswith("rot_total_2"):
        return "Rotation 2 deg"
    if sc.name == "measured_translation_only":
        return "Gemessene Translation"
    if sc.name == "sixd_total_0p5mm_0p5deg_xyz_equal":
        return "0.5 mm + 0.5 deg 6D"
    return sc.name.replace("_", " ")


def _scenario_short_label(sc: ShiftScenario) -> str:
    if sc.name.startswith("rot_total_1"):
        return "1 deg rot"
    if sc.name.startswith("rot_total_2"):
        return "2 deg rot"
    if sc.name == "measured_translation_only":
        return "0.8 mm shift"
    if sc.name == "sixd_total_0p5mm_0p5deg_xyz_equal":
        return "0.5 mm + 0.5 deg"
    return _scenario_title(sc)


def _scenario_compact_detail(sc: ShiftScenario) -> str:
    if sc.name == "measured_translation_only":
        return f"dx/dy/dz={sc.dx_mm:.1f}/{sc.dy_mm:.1f}/{sc.dz_mm:.1f} mm"
    if sc.name == "sixd_total_0p5mm_0p5deg_xyz_equal":
        return f"dxyz={sc.dx_mm:.2f} mm, rxyz={sc.rx_deg:.2f} deg"
    if sc.rx_deg or sc.ry_deg or sc.rz_deg:
        return f"rx/ry/rz={sc.rx_deg:.2f}/{sc.ry_deg:.2f}/{sc.rz_deg:.2f} deg"
    return _scenario_detail(sc)


def _scenario_detail(sc: ShiftScenario) -> str:
    return (
        f"shift=({sc.dx_mm:+.2f}, {sc.dy_mm:+.2f}, {sc.dz_mm:+.2f}) mm   "
        f"rot=({sc.rx_deg:.2f}, {sc.ry_deg:.2f}, {sc.rz_deg:.2f}) deg"
    )


def _sixd_equal_scenario(total_mm: float = 0.5, total_deg: float = 0.5) -> ShiftScenario:
    per_translation_axis = float(total_mm) / 3.0
    per_rotation_axis = float(total_deg) / 3.0
    return ShiftScenario(
        name="sixd_total_0p5mm_0p5deg_xyz_equal",
        dx_mm=per_translation_axis,
        dy_mm=per_translation_axis,
        dz_mm=per_translation_axis,
        rx_deg=per_rotation_axis,
        ry_deg=per_rotation_axis,
        rz_deg=per_rotation_axis,
    )


def _make_scenarios(args: argparse.Namespace) -> list[ShiftScenario]:
    return [
        rotation_scenario(1.0),
        rotation_scenario(2.0),
        measured_translation_scenario(
            vertical_cm=args.translation_vertical_cm,
            longitudinal_cm=args.translation_longitudinal_cm,
            lateral_cm=args.translation_lateral_cm,
        ),
        _sixd_equal_scenario(),
    ]


def _select_targets(structures, plan_label: str) -> list[tuple[int, object]]:
    ptv_by_number, gtv_by_number, _ = build_target_maps(structures)
    numbers = sorted(set(ptv_by_number).intersection(gtv_by_number))
    active_numbers = active_numbers_from_plan_label(clean_text(plan_label))
    if active_numbers:
        numbers = [number for number in numbers if number in active_numbers]
    return [(number, ptv_by_number[number]) for number in numbers]


def _mesh_structures(structures, active_names: set[str], dose, max_faces: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    meshes = {}
    for name, verts, faces in p3d._voxelize_and_mesh_per_structure(  # noqa: SLF001
        structures,
        active_names,
        dose,
        stride=1,
        max_faces=max_faces,
    ):
        if verts is not None and faces is not None and len(faces) > 0:
            meshes[name] = (verts, faces)
    return meshes


def _raw_ptv_volume_layers(
    targets: list[tuple[int, object]],
    dose,
    *,
    max_faces: int,
    alpha: float,
) -> list[MeshLayer]:
    layers: list[MeshLayer] = []
    for number, structure in targets:
        ptv_mask = _structure_mask_on_dose_grid(structure, dose)
        verts, faces = _mesh_binary_mask(ptv_mask, dose, max_faces=max_faces, smooth=False)
        if verts is None or faces is None or len(faces) == 0:
            continue
        layers.append(MeshLayer(f"PTV/Mets voxel PTV{number:02d}", verts, faces, PTV_COLOR, alpha))
    return layers


def _structure_mask_on_dose_grid(structure, dose) -> np.ndarray:
    try:
        from scipy.ndimage import binary_fill_holes
        from skimage.draw import polygon as sk_polygon
    except ImportError:
        return np.zeros(dose.dose_grid.shape, dtype=bool)

    ox, oy, oz = np.asarray(dose.origin, dtype=float)
    dx, dy, dz = np.asarray(dose.spacing, dtype=float)
    nz, ny, nx = dose.dose_grid.shape
    mask = np.zeros((nz, ny, nx), dtype=np.uint8)

    for contour in structure.contours:
        points = np.asarray(contour, dtype=float)
        if points.shape[0] < 3:
            continue
        z_mm = float(np.mean(points[:, 2]))
        zi = int(round((z_mm - oz) / dz))
        if not (0 <= zi < nz):
            continue
        xi = (points[:, 0] - ox) / dx
        yi = (points[:, 1] - oy) / dy
        try:
            rr, cc = sk_polygon(yi, xi, shape=(ny, nx))
        except Exception:
            continue
        mask[zi, rr, cc] = 1

    if not bool(mask.any()):
        return mask.astype(bool)
    for zi in range(nz):
        if mask[zi].any():
            mask[zi] = binary_fill_holes(mask[zi]).astype(np.uint8)
    return mask.astype(bool)


def _mask_voxel_points(mask: np.ndarray, dose) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return indices, np.empty((0, 3), dtype=float)
    origin = np.asarray(dose.origin, dtype=float)
    spacing = np.asarray(dose.spacing, dtype=float)
    points = np.empty((indices.shape[0], 3), dtype=float)
    points[:, 0] = origin[0] + indices[:, 2] * spacing[0]
    points[:, 1] = origin[1] + indices[:, 1] * spacing[1]
    points[:, 2] = origin[2] + indices[:, 0] * spacing[2]
    return indices, points


def _mesh_binary_mask(
    mask: np.ndarray,
    dose,
    *,
    max_faces: int,
    stride: int = 1,
    smooth: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mc = p3d._mc_import()  # noqa: SLF001
    if mc is None or not bool(mask.any()):
        return None, None

    dz, dy, dx = dose.spacing[2], dose.spacing[1], dose.spacing[0]
    if smooth:
        try:
            from scipy.ndimage import gaussian_filter

            grid = gaussian_filter(mask.astype(float), sigma=max(0.35, 0.70 / max(stride, 1)))
        except ImportError:
            grid = mask.astype(float)
    else:
        grid = mask.astype(float)

    grid = grid[::stride, ::stride, ::stride]
    if float(grid.max()) < 0.5:
        return None, None

    try:
        verts, faces, _, _ = mc(
            grid,
            level=0.5,
            spacing=(dz * stride, dy * stride, dx * stride),
            allow_degenerate=False,
        )
    except Exception:
        return None, None
    if len(verts) == 0 or len(faces) == 0:
        return None, None

    verts_xyz = p3d._verts_to_xyz(verts, dose.origin, dose.spacing)  # noqa: SLF001
    return verts_xyz, _decimated_faces(faces, max_faces)


def _decimated_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if faces is None:
        return faces
    if max_faces <= 0:
        return faces
    return p3d._decimate_faces(faces, max_faces=max_faces)  # noqa: SLF001


def _animation_face_limit(layer: MeshLayer, args: argparse.Namespace) -> int:
    if layer.name in ISO_LAYER_NAMES:
        return args.anim_iso_faces
    if layer.name.startswith("PTV/Mets"):
        return args.anim_ptv_faces
    if layer.name in OAR_COLORS:
        return args.anim_oar_faces
    return args.anim_context_faces


def _flat_facecolors(color: str, alpha: float, count: int) -> np.ndarray:
    from matplotlib.colors import to_rgba

    if count <= 0:
        return np.empty((0, 4), dtype=float)
    return np.tile(np.asarray(to_rgba(color, alpha), dtype=float), (count, 1))


def _face_centroids(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return verts[faces].mean(axis=1)


def _expand_face_mask(mask: np.ndarray, faces: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0 or mask.size == 0 or not bool(mask.any()):
        return mask

    face_by_vertex: list[list[int]] = [[] for _ in range(int(faces.max()) + 1)]
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            face_by_vertex[int(vertex_index)].append(face_index)

    expanded = mask.copy()
    for _ in range(iterations):
        next_mask = expanded.copy()
        for face_index in np.flatnonzero(expanded):
            for vertex_index in faces[face_index]:
                next_mask[face_by_vertex[int(vertex_index)]] = True
        if int(next_mask.sum()) == int(expanded.sum()):
            break
        expanded = next_mask
    return expanded


def _surface_difference_layers(
    nominal_verts: np.ndarray,
    shifted_verts: np.ndarray,
    faces: np.ndarray,
    dose,
    *,
    rx_gy: float,
    dose_margin_gy: float,
    threshold_mm: float,
    min_faces: int,
) -> list[MeshLayer]:
    from scipy.spatial import cKDTree

    shifted_centroids = _face_centroids(shifted_verts, faces)
    nominal_dose_at_shifted = sample_dose_at_points(shifted_centroids, dose)
    shifted_mask = (~np.isfinite(nominal_dose_at_shifted)) | (nominal_dose_at_shifted < rx_gy - dose_margin_gy)

    if int(shifted_mask.sum()) < min_faces:
        nominal_centroids = _face_centroids(nominal_verts, faces)
        nominal_tree = cKDTree(nominal_centroids)
        shifted_dist, _ = nominal_tree.query(shifted_centroids, k=1)
        shifted_mask = shifted_dist >= threshold_mm
        if int(shifted_mask.sum()) < min_faces:
            cutoff = max(float(np.quantile(shifted_dist, 0.70)), threshold_mm * 0.5)
            shifted_mask = shifted_dist >= cutoff

    shifted_mask = _expand_face_mask(shifted_mask, faces, iterations=1)

    return [
        MeshLayer(
            "Abweichung zur Nominalisodose",
            shifted_verts,
            faces[shifted_mask],
            DIFF_SHIFTED_COLOR,
            0.90,
        ),
    ]


def _surface_interaction_layers(
    nominal_verts: np.ndarray,
    shifted_verts: np.ndarray,
    faces: np.ndarray,
    *,
    nominal_dose,
    rx_gy: float,
    dose_margin_gy: float,
) -> list[MeshLayer]:
    shifted_centroids = _face_centroids(shifted_verts, faces)

    nominal_dose_at_shifted = sample_dose_at_points(shifted_centroids, nominal_dose)
    shifted_only = (~np.isfinite(nominal_dose_at_shifted)) | (nominal_dose_at_shifted < rx_gy - dose_margin_gy)

    shifted_only = _expand_face_mask(shifted_only, faces, iterations=1)
    shifted_faces = faces[shifted_only]
    nominal_facecolors = _flat_facecolors(NOMINAL_COLOR, 0.54, len(faces))
    shifted_facecolors = _flat_facecolors(DIFF_SHIFTED_COLOR, 0.72, len(shifted_faces))
    verts = np.vstack([nominal_verts, shifted_verts])
    combined_faces = np.vstack([faces, shifted_faces + len(nominal_verts)])
    facecolors = np.vstack([nominal_facecolors, shifted_facecolors])

    return [
        MeshLayer(
            "Iso-Interaktion",
            verts,
            combined_faces,
            NOMINAL_COLOR,
            1.0,
            facecolors=facecolors,
        ),
    ]


def _undercovered_ptv_volume_layers(
    targets: list[tuple[int, object]],
    dose,
    shifted_dose,
    *,
    rx_gy: float,
    dose_margin_gy: float,
    max_faces: int,
) -> list[MeshLayer]:
    layers: list[MeshLayer] = []
    for number, structure in targets:
        ptv_mask = _structure_mask_on_dose_grid(structure, dose)
        indices, points = _mask_voxel_points(ptv_mask, dose)
        if points.size == 0:
            continue
        shifted_dose_at_ptv = sample_dose_at_points(points, shifted_dose)
        undercovered_values = (~np.isfinite(shifted_dose_at_ptv)) | (shifted_dose_at_ptv < rx_gy - dose_margin_gy)
        if not bool(undercovered_values.any()):
            continue
        undercovered_mask = np.zeros_like(ptv_mask, dtype=bool)
        undercovered_indices = indices[undercovered_values]
        undercovered_mask[
            undercovered_indices[:, 0],
            undercovered_indices[:, 1],
            undercovered_indices[:, 2],
        ] = True
        verts, faces = _mesh_binary_mask(undercovered_mask, dose, max_faces=max_faces)
        if verts is None or faces is None or len(faces) == 0:
            continue
        layers.append(
            MeshLayer(
                f"PTV ohne 100% PTV{number:02d}",
                verts,
                faces,
                UNCOVERED_COLOR,
                0.86,
            )
        )
    return layers


def _ptv_surface_undercoverage_layers(
    target_meshes: dict[str, tuple[np.ndarray, np.ndarray]],
    shifted_dose,
    *,
    rx_gy: float,
    dose_margin_gy: float,
) -> list[MeshLayer]:
    layers: list[MeshLayer] = []
    for name, (verts, faces) in target_meshes.items():
        if faces is None or len(faces) == 0:
            continue
        centroids = _face_centroids(verts, faces)
        shifted_dose_at_ptv = sample_dose_at_points(centroids, shifted_dose)
        undercovered = (~np.isfinite(shifted_dose_at_ptv)) | (shifted_dose_at_ptv < rx_gy - dose_margin_gy)
        undercovered = _expand_face_mask(undercovered, faces, iterations=1)
        undercovered_faces = faces[undercovered]
        if len(undercovered_faces) == 0:
            continue
        layers.append(
            MeshLayer(
                f"PTV ohne 100% {name}",
                verts,
                undercovered_faces,
                UNCOVERED_COLOR,
                0.94,
            )
        )
    return layers


def _largest_undercovered_focus(
    layers: list[MeshLayer],
    targets: list[tuple[int, object]],
) -> tuple[str, tuple[int, object], MeshLayer] | None:
    miss_layers = [layer for layer in layers if layer.name.startswith("PTV ohne 100% ")]
    if not miss_layers:
        return None
    miss_layer = max(miss_layers, key=lambda layer: 0 if layer.faces is None else len(layer.faces))
    target_name = miss_layer.name.removeprefix("PTV ohne 100% ")
    ptv_layer = next((layer for layer in layers if layer.name == f"PTV/Mets {target_name}"), None)
    if ptv_layer is None:
        return None
    target = next((target for target in targets if getattr(target[1], "name", "") == target_name), None)
    if target is None:
        return None
    return target_name, target, ptv_layer


def _focus_center_radius(ptv_layer: MeshLayer) -> tuple[np.ndarray, float] | None:
    if ptv_layer.verts is None or ptv_layer.faces is None or len(ptv_layer.faces) == 0:
        return None
    points = ptv_layer.verts[ptv_layer.faces].reshape(-1, 3)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    radius = max(7.5, float(np.max(hi - lo)) * 0.72)
    return center, radius


def _focus_limits(center: np.ndarray, radius: float) -> tuple[float, ...]:
    return (
        center[0] - radius,
        center[0] + radius,
        center[1] - radius,
        center[1] + radius,
        center[2] - radius,
        center[2] + radius,
    )


def _focus_face_limit(layer: MeshLayer) -> int:
    if layer.name.startswith("PTV/Mets ") or layer.name.startswith("PTV ohne 100% "):
        return 2600
    if layer.name in ISO_LAYER_NAMES:
        return 5200
    if layer.name in OAR_COLORS:
        return 1800
    return 3200


def _clip_layer_to_focus(layer: MeshLayer, center: np.ndarray, radius: float) -> MeshLayer | None:
    if layer.verts is None or layer.faces is None or len(layer.faces) == 0:
        return None
    centroids = _face_centroids(layer.verts, layer.faces)
    mask = np.linalg.norm(centroids - center, axis=1) <= radius
    if not bool(mask.any()):
        return None
    faces = layer.faces[mask]
    facecolors = layer.facecolors[mask] if layer.facecolors is not None else None
    max_faces = _focus_face_limit(layer)
    if max_faces > 0 and len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
        faces = faces[indices]
        facecolors = facecolors[indices] if facecolors is not None else None
    return MeshLayer(layer.name, layer.verts, faces, layer.color, layer.alpha, facecolors=facecolors)


def _stage_focus_layers(
    layers: list[MeshLayer],
    *,
    target_name: str,
    center: np.ndarray,
    radius: float,
) -> list[MeshLayer]:
    focused: list[MeshLayer] = []
    for layer in layers:
        if layer.name.startswith("PTV/Mets ") and layer.name != f"PTV/Mets {target_name}":
            continue
        if layer.name.startswith("PTV ohne 100% ") and layer.name != f"PTV ohne 100% {target_name}":
            continue
        clipped = _clip_layer_to_focus(layer, center, radius)
        if clipped is not None:
            focused.append(clipped)
    return focused


def _add_stage_focus_zoom_inset(
    fig,
    parent_position: list[float],
    *,
    current_layers: list[MeshLayer],
    focus_reference_layers: list[MeshLayer],
    isocenter: np.ndarray,
    targets: list[tuple[int, object]],
    scenario: ShiftScenario,
    azim: float,
    elev: float,
    show_vectors: bool,
) -> None:
    focus = _largest_undercovered_focus(focus_reference_layers, targets)
    if focus is None:
        return
    target_name, target, ptv_layer = focus
    focus_geometry = _focus_center_radius(ptv_layer)
    if focus_geometry is None:
        return
    center, radius = focus_geometry
    focused_layers = _stage_focus_layers(current_layers, target_name=target_name, center=center, radius=radius * 1.65)
    if not focused_layers:
        return
    x, y, width, height = parent_position
    inset_position = [
        x + 0.030 * width,
        y + 0.030 * height,
        0.295 * width,
        0.295 * height,
    ]
    ax = fig.add_axes(inset_position, projection="3d")
    _render_axis(
        ax,
        layers=focused_layers,
        isocenter=isocenter,
        targets=[target],
        scenario=scenario,
        axis_limits=_focus_limits(center, radius),
        azim=azim,
        elev=elev,
        show_labels=False,
        label_points=[],
        show_vectors=show_vectors,
        camera_distance=4.2,
    )

    from matplotlib.patches import Rectangle

    border = Rectangle(
        (inset_position[0], inset_position[1]),
        inset_position[2],
        inset_position[3],
        transform=fig.transFigure,
        fill=False,
        edgecolor=UNCOVERED_COLOR,
        linewidth=1.15,
        alpha=0.92,
        zorder=40,
    )
    fig.add_artist(border)


def _plot_mesh_layer_3d(ax, layer: MeshLayer) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if layer.verts is None or layer.faces is None or len(layer.faces) == 0:
        return

    if layer.facecolors is None:
        facecolors = _flat_facecolors(layer.color, layer.alpha, len(layer.faces))
    else:
        facecolors = layer.facecolors
    collection = Poly3DCollection(list(layer.verts[layer.faces]), antialiaseds=True)
    collection.set_facecolor(facecolors)
    collection.set_edgecolor("none")
    collection.set_linewidth(0.0)
    ax.add_collection3d(collection)


def _axis_limits(layers: Iterable[MeshLayer], points: Iterable[np.ndarray], pad_fraction: float = 0.10) -> tuple[float, ...]:
    arrays = [layer.verts for layer in layers if layer.verts is not None and len(layer.verts) > 0]
    arrays.extend(np.asarray(point, dtype=float).reshape(1, 3) for point in points)
    combined = np.vstack(arrays)
    lo = combined.min(axis=0)
    hi = combined.max(axis=0)
    center = (lo + hi) * 0.5
    half = float(np.max(hi - lo)) * (0.5 + pad_fraction)
    return (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
        center[2] - half,
        center[2] + half,
    )


def _apply_axis_limits(ax, limits: tuple[float, ...]) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _scale_axis_limits(limits: tuple[float, ...], factor: float) -> tuple[float, ...]:
    xmin, xmax, ymin, ymax, zmin, zmax = limits
    center = np.array(
        [
            (xmin + xmax) * 0.5,
            (ymin + ymax) * 0.5,
            (zmin + zmax) * 0.5,
        ],
        dtype=float,
    )
    half = np.array(
        [
            (xmax - xmin) * 0.5,
            (ymax - ymin) * 0.5,
            (zmax - zmin) * 0.5,
        ],
        dtype=float,
    ) * float(factor)
    return (
        center[0] - half[0],
        center[0] + half[0],
        center[1] - half[1],
        center[1] + half[1],
        center[2] - half[2],
        center[2] + half[2],
    )


def _plot_vectors(ax, targets: list[tuple[int, object]], scenario: ShiftScenario, isocenter: np.ndarray) -> None:
    for number, structure in targets:
        start = _centroid(structure)
        end = forward_transform_point(start, scenario, isocenter)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 0.05:
            continue
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=VECTOR_COLOR,
            linewidth=1.1,
            alpha=0.90,
        )
        ax.scatter(
            [end[0]],
            [end[1]],
            [end[2]],
            color=VECTOR_COLOR,
            s=10,
            marker="o",
            depthshade=False,
            alpha=0.95,
        )

def _plot_labels(ax, label_points: list[tuple[str, np.ndarray, str]], fontsize: float) -> None:
    for label, point, color in label_points:
        ax.text(
            point[0],
            point[1],
            point[2],
            label,
            color=color,
            fontsize=fontsize,
            ha="center",
            alpha=0.92,
            bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.50),
        )


def _render_axis(
    ax,
    *,
    layers: list[MeshLayer],
    isocenter: np.ndarray,
    targets: list[tuple[int, object]],
    scenario: ShiftScenario,
    axis_limits: tuple[float, ...],
    azim: float,
    elev: float,
    show_labels: bool,
    label_points: list[tuple[str, np.ndarray, str]],
    show_vectors: bool = True,
    camera_distance: float = CAMERA_DISTANCE,
) -> None:
    p3d._style_3d(ax)  # noqa: SLF001
    ax.set_facecolor(BG)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.dist = camera_distance
    except Exception:
        pass
    for layer in layers:
        _plot_mesh_layer_3d(ax, layer)
    if show_vectors:
        _plot_vectors(ax, targets, scenario, isocenter)
    ax.scatter(
        [isocenter[0]],
        [isocenter[1]],
        [isocenter[2]],
        color=ISO_MARKER,
        s=90,
        marker="*",
        depthshade=False,
        zorder=20,
    )
    target_centers = np.vstack([_centroid(structure) for _, structure in targets])
    ax.scatter(
        target_centers[:, 0],
        target_centers[:, 1],
        target_centers[:, 2],
        color="#ffffff",
        s=7,
        marker="o",
        depthshade=False,
        alpha=0.70,
    )
    if show_labels:
        _plot_labels(ax, label_points, fontsize=6.2)
    _apply_axis_limits(ax, axis_limits)
    try:
        ax.margins(0)
    except Exception:
        pass


def _legend_handles(_layers: list[MeshLayer]):
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], color=PTV_COLOR, linewidth=8, alpha=0.45, label="PTV/Mets"),
        Line2D([0], [0], color=NOMINAL_COLOR, linewidth=8, alpha=0.75, label="Nominale 20 Gy Isodose"),
        Line2D([0], [0], color=DIFF_SHIFTED_COLOR, linewidth=8, alpha=0.85, label="Abweichung zur Nominalisodose"),
        Line2D([0], [0], color=UNCOVERED_COLOR, linewidth=8, alpha=0.90, label="PTV ohne 100% nach Shift"),
        Line2D([0], [0], color=VECTOR_COLOR, linewidth=2, alpha=0.95, label="Verschiebung PTV-Zentrum"),
        Line2D([0], [0], color=OAR_COLORS["Brainstem"], linewidth=6, alpha=0.75, label="Brainstem"),
        Line2D([0], [0], color=OAR_COLORS["Optic Nerve Left"], linewidth=6, alpha=0.75, label="Optic Nerves"),
        Line2D([0], [0], color=OAR_COLORS["Chiasm"], linewidth=6, alpha=0.75, label="Chiasma"),
        Line2D([0], [0], color=OAR_COLORS["Eye Left"], linewidth=6, alpha=0.75, label="Eyes"),
        Line2D(
            [0],
            [0],
            color=ISO_MARKER,
            marker="*",
            markerfacecolor=ISO_MARKER,
            markeredgecolor=ISO_MARKER,
            linestyle="None",
            markersize=9,
            label="Isozentrum",
        ),
    ]


def _place_data_legend(fig, layers: list[MeshLayer], *, y: float = 0.055, ncol: int = 5) -> None:
    legend = fig.legend(
        handles=_legend_handles(layers),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fontsize=7,
        framealpha=0.35,
        labelcolor="white",
        facecolor="#101722",
        edgecolor="#354052",
    )
    legend.set_in_layout(False)


def _place_scenario_legend(
    fig,
    scenarios: list[ShiftScenario],
    active_scenario: ShiftScenario,
    *,
    y: float = 0.016,
) -> None:
    from matplotlib.lines import Line2D

    handles = []
    for scenario in scenarios:
        active = scenario.name == active_scenario.name
        color = ACTIVE_SCENARIO_COLOR if active else INACTIVE_SCENARIO_COLOR
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                linestyle="None",
                markersize=8,
                label=_scenario_title(scenario),
            )
        )
    legend = fig.legend(
        handles=handles,
        title="Aktives Szenario",
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        fontsize=7,
        title_fontsize=7,
        framealpha=0.35,
        labelcolor="white",
        facecolor="#101722",
        edgecolor="#354052",
    )
    legend.get_title().set_color("white")
    legend.set_in_layout(False)


def render_static(
    output_dir: Path,
    *,
    scenario: ShiftScenario,
    layers: list[MeshLayer],
    isocenter: np.ndarray,
    targets: list[tuple[int, object]],
    axis_limits: tuple[float, ...],
    label_points: list[tuple[str, np.ndarray, str]],
    rx_gy: float,
    static_dpi: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 10.15), facecolor=BG)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    fig.subplots_adjust(left=0.010, right=0.995, top=0.890, bottom=0.145, wspace=0.015)
    for ax, azim, title in zip(axes, (-55, 125), ("Front-left", "Front-right")):
        _render_axis(
            ax,
            layers=layers,
            isocenter=isocenter,
            targets=targets,
            scenario=scenario,
            axis_limits=axis_limits,
            azim=azim,
            elev=23,
            show_labels=True,
            label_points=label_points,
            show_vectors=True,
        )
        ax.set_title(title, color="#b9c2d0", fontsize=9, pad=3)
    _place_data_legend(fig, layers, y=0.043, ncol=5)
    fig.suptitle(
        f"12-Met SRS: nominale vs. verschobene {rx_gy:g} Gy Isodose - {_scenario_title(scenario)}\n"
        f"{_scenario_detail(scenario)}",
        color="white",
        fontsize=13,
        y=0.985,
    )
    fig.text(
        0.5,
        0.014,
        "Gruen: geplante Isodose. Magenta: durch starre 6D-Szenario-Transformation verschobene Isodose. "
        "Strukturen dienen nur als Orientierungshilfe.",
        color="#9aa6b8",
        fontsize=8,
        ha="center",
    )
    out = output_dir / f"12met_iso_shift_{_sanitize(scenario.name)}.png"
    fig.savefig(out, dpi=static_dpi, facecolor=BG)
    plt.close(fig)
    return out


def render_frame(
    *,
    scenario: ShiftScenario,
    layers: list[MeshLayer],
    isocenter: np.ndarray,
    targets: list[tuple[int, object]],
    axis_limits: tuple[float, ...],
    label_points: list[tuple[str, np.ndarray, str]],
    rx_gy: float,
    azim: float,
    frame_dpi: int,
    all_scenarios: list[ShiftScenario] | None = None,
    show_scenario_status: bool = False,
    stage_title: str | None = None,
    show_vectors: bool = True,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9), dpi=frame_dpi, facecolor=BG)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.935, bottom=0.165)
    ax.set_position([0.02, 0.185, 0.96, 0.720])
    _render_axis(
        ax,
        layers=layers,
        isocenter=isocenter,
        targets=targets,
        scenario=scenario,
        axis_limits=axis_limits,
        azim=azim,
        elev=22,
        show_labels=False,
        label_points=label_points,
        show_vectors=show_vectors,
    )
    _place_data_legend(fig, layers, y=0.060, ncol=5)
    if show_scenario_status and all_scenarios:
        _place_scenario_legend(fig, all_scenarios, scenario, y=0.016)
    title = f"{_scenario_short_label(scenario)}"
    if stage_title:
        title += f" | {stage_title}"
    fig.suptitle(title, color="white", fontsize=13, y=0.985)
    fig.text(
        0.985,
        0.972,
        f"{rx_gy:g} Gy Isodose\n{_scenario_detail(scenario)}",
        color="#9aa6b8",
        fontsize=6.8,
        ha="right",
        va="top",
    )
    if not show_scenario_status:
        fig.text(
            0.5,
            0.016,
            "Strukturen: PTVs, Augen, Sehnerven, Chiasma, Hirnstamm",
            color="#9aa6b8",
            fontsize=8,
            ha="center",
        )
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


def render_combined_frame(
    *,
    scenarios: list[ShiftScenario],
    stage_layers: dict[str, dict[str, list[MeshLayer]]],
    stage_key: str,
    isocenter: np.ndarray,
    targets: list[tuple[int, object]],
    axis_limits: tuple[float, ...],
    label_points: list[tuple[str, np.ndarray, str]],
    rx_gy: float,
    azim: float,
    frame_dpi: int,
    stage_title: str,
    show_vectors: bool,
    camera_distance: float,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=COMBINED_FIGSIZE, dpi=frame_dpi, facecolor=BG)
    positions = [
        [0.000, 0.535, 0.500, 0.365],
        [0.500, 0.535, 0.500, 0.365],
        [0.000, 0.145, 0.500, 0.365],
        [0.500, 0.145, 0.500, 0.365],
    ]
    axes = [fig.add_axes(position, projection="3d") for position in positions]

    for ax, scenario, position in zip(axes, scenarios, positions):
        _render_axis(
            ax,
            layers=stage_layers[scenario.name][stage_key],
            isocenter=isocenter,
            targets=targets,
            scenario=scenario,
            axis_limits=axis_limits,
            azim=azim,
            elev=22,
            show_labels=False,
            label_points=label_points,
            show_vectors=show_vectors,
            camera_distance=camera_distance,
        )
        ax.text2D(
            0.025,
            0.935,
            _scenario_short_label(scenario),
            transform=ax.transAxes,
            color="white",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="#101722", ec="#354052", alpha=0.78),
        )
        ax.text2D(
            0.975,
            0.935,
            _scenario_compact_detail(scenario),
            transform=ax.transAxes,
            color="#b9c2d0",
            fontsize=7,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.20", fc="#101722", ec="none", alpha=0.55),
        )
        _add_stage_focus_zoom_inset(
            fig,
            position,
            current_layers=stage_layers[scenario.name][stage_key],
            focus_reference_layers=stage_layers[scenario.name].get("miss", []),
            isocenter=isocenter,
            targets=targets,
            scenario=scenario,
            azim=azim,
            elev=22,
            show_vectors=show_vectors,
        )

    reference_layers = stage_layers[scenarios[0].name][stage_key]
    _place_data_legend(fig, reference_layers, y=0.060, ncol=5)
    fig.suptitle(stage_title, color="white", fontsize=13, y=0.985)
    fig.text(
        0.985,
        0.972,
        f"{rx_gy:g} Gy Isodose\nsynchrone Kamera, gleiche Legende",
        color="#9aa6b8",
        fontsize=6.8,
        ha="right",
        va="top",
    )
    fig.text(
        0.5,
        0.016,
        "Alle Szenarien laufen synchron mit gleicher Kamera und gemeinsamer Legende.",
        color="#9aa6b8",
        fontsize=8,
        ha="center",
    )
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


def save_staged_animation(
    output_dir: Path,
    *,
    scenario: ShiftScenario,
    stage_layers: dict[str, list[MeshLayer]],
    isocenter: np.ndarray,
    targets: list[tuple[int, object]],
    axis_limits: tuple[float, ...],
    label_points: list[tuple[str, np.ndarray, str]],
    rx_gy: float,
    fps: int,
    build_seconds: float,
    rotation_seconds: float,
    frame_dpi: int,
    write_gif: bool,
) -> tuple[Path | None, Path | None]:
    import imageio.v2 as imageio

    stages = [
        ("ptv", "1. PTV + OAR", build_seconds, 55.0, False),
        ("nominal", "2. Nominale 20 Gy", build_seconds, 55.0, False),
        ("shifted", "3. Iso-Interaktion", rotation_seconds, 360.0, True),
        ("diff", "4. Rest ausserhalb nominal", rotation_seconds, 360.0, True),
        ("miss", "5. PTV ohne 100%", rotation_seconds, 360.0, True),
    ]
    output_stem = f"12met_iso_shift_{_sanitize(scenario.name)}_staged"
    gif_path = output_dir / f"{output_stem}.gif"
    mp4_path = output_dir / f"{output_stem}.mp4"

    gif_writer = imageio.get_writer(gif_path, mode="I", fps=fps, loop=0) if write_gif else None
    try:
        mp4_writer = imageio.get_writer(mp4_path, fps=fps, quality=8)
    except Exception:
        mp4_writer = None
        mp4_path = None

    current_azim = -55.0
    try:
        for key, title, seconds, azim_delta, show_vectors in stages:
            frame_count = max(1, int(round(fps * seconds)))
            for frame_index in range(frame_count):
                progress = frame_index / max(frame_count - 1, 1)
                azim = current_azim + azim_delta * progress
                layers = stage_layers[key]
                frame = render_frame(
                    scenario=scenario,
                    layers=layers,
                    isocenter=isocenter,
                    targets=targets,
                    axis_limits=axis_limits,
                    label_points=label_points,
                    rx_gy=rx_gy,
                    azim=azim,
                    frame_dpi=frame_dpi,
                    all_scenarios=None,
                    show_scenario_status=False,
                    stage_title=title,
                    show_vectors=show_vectors,
                )
                if gif_writer is not None:
                    gif_writer.append_data(frame)
                if mp4_writer is not None:
                    mp4_writer.append_data(frame)
            current_azim += azim_delta
    finally:
        if gif_writer is not None:
            gif_writer.close()
        if mp4_writer is not None:
            mp4_writer.close()
    return (gif_path if write_gif else None), mp4_path


def save_combined_staged_animation(
    output_dir: Path,
    *,
    scenarios: list[ShiftScenario],
    stage_layers: dict[str, dict[str, list[MeshLayer]]],
    isocenter: np.ndarray,
    targets: list[tuple[int, object]],
    axis_limits: tuple[float, ...],
    label_points: list[tuple[str, np.ndarray, str]],
    rx_gy: float,
    fps: int,
    build_seconds: float,
    rotation_seconds: float,
    frame_dpi: int,
    zoom_factor: float,
    camera_distance: float,
    write_gif: bool,
) -> tuple[Path | None, Path | None]:
    import imageio.v2 as imageio

    stages = [
        ("ptv", "1. PTV + OAR", build_seconds, 55.0, False),
        ("nominal", "2. Nominale 20 Gy", build_seconds, 55.0, False),
        ("shifted", "3. Iso-Interaktion", rotation_seconds, 360.0, True),
        ("diff", "4. Rest ausserhalb nominal", rotation_seconds, 360.0, True),
        ("miss", "5. PTV ohne 100%", rotation_seconds, 360.0, True),
    ]
    output_stem = "12met_iso_shift_all4_staged_grid"
    gif_path = output_dir / f"{output_stem}.gif"
    mp4_path = output_dir / f"{output_stem}.mp4"
    zoomed_axis_limits = _scale_axis_limits(axis_limits, zoom_factor)

    gif_writer = imageio.get_writer(gif_path, mode="I", fps=fps, loop=0) if write_gif else None
    try:
        mp4_writer = imageio.get_writer(mp4_path, fps=fps, quality=8)
    except Exception:
        mp4_writer = None
        mp4_path = None

    current_azim = -55.0
    try:
        for key, title, seconds, azim_delta, show_vectors in stages:
            frame_count = max(1, int(round(fps * seconds)))
            for frame_index in range(frame_count):
                progress = frame_index / max(frame_count - 1, 1)
                azim = current_azim + azim_delta * progress
                frame = render_combined_frame(
                    scenarios=scenarios,
                    stage_layers=stage_layers,
                    stage_key=key,
                    isocenter=isocenter,
                    targets=targets,
                    axis_limits=zoomed_axis_limits,
                    label_points=label_points,
                    rx_gy=rx_gy,
                    azim=azim,
                    frame_dpi=frame_dpi,
                    stage_title=title,
                    show_vectors=show_vectors,
                    camera_distance=camera_distance,
                )
                if gif_writer is not None:
                    gif_writer.append_data(frame)
                if mp4_writer is not None:
                    mp4_writer.append_data(frame)
            current_azim += azim_delta
    finally:
        if gif_writer is not None:
            gif_writer.close()
        if mp4_writer is not None:
            mp4_writer.close()
    return (gif_path if write_gif else None), mp4_path


def save_animation(
    output_dir: Path,
    *,
    scenarios: list[ShiftScenario],
    scenario_layers: dict[str, list[MeshLayer]],
    isocenter: np.ndarray,
    targets: list[tuple[int, object]],
    axis_limits: tuple[float, ...],
    label_points: list[tuple[str, np.ndarray, str]],
    rx_gy: float,
    fps: int,
    seconds_per_scenario: float,
    frame_dpi: int,
    output_stem: str,
    show_scenario_status: bool,
    write_gif: bool,
) -> tuple[Path | None, Path | None]:
    import imageio.v2 as imageio

    frames_per_scenario = max(1, int(round(fps * seconds_per_scenario)))
    total_frames = frames_per_scenario * len(scenarios)
    gif_path = output_dir / f"{output_stem}.gif"
    mp4_path = output_dir / f"{output_stem}.mp4"

    gif_writer = imageio.get_writer(gif_path, mode="I", fps=fps, loop=0) if write_gif else None
    try:
        mp4_writer = imageio.get_writer(mp4_path, fps=fps, quality=8)
    except Exception:
        mp4_writer = None
        mp4_path = None

    try:
        for sc_index, scenario in enumerate(scenarios):
            for local_frame in range(frames_per_scenario):
                global_frame = sc_index * frames_per_scenario + local_frame
                azim = -65.0 + 360.0 * global_frame / max(total_frames, 1)
                frame = render_frame(
                    scenario=scenario,
                    layers=scenario_layers[scenario.name],
                    isocenter=isocenter,
                    targets=targets,
                    axis_limits=axis_limits,
                    label_points=label_points,
                    rx_gy=rx_gy,
                    azim=azim,
                    frame_dpi=frame_dpi,
                    all_scenarios=scenarios,
                    show_scenario_status=show_scenario_status,
                )
                if gif_writer is not None:
                    gif_writer.append_data(frame)
                if mp4_writer is not None:
                    mp4_writer.append_data(frame)
    finally:
        if gif_writer is not None:
            gif_writer.close()
        if mp4_writer is not None:
            mp4_writer.close()
    return (gif_path if write_gif else None), mp4_path


def build_scene(args: argparse.Namespace):
    plan_sets = discover_plan_sets(args.data_root)
    if not plan_sets:
        raise RuntimeError(f"No RTSTRUCT/RTPLAN/RTDOSE set found below {args.data_root}")
    plan_set = plan_sets[0]
    structures = load_structures(str(plan_set.rtstruct_path))
    dose = load_dose(str(plan_set.rtdose_path))
    meta = load_plan_meta(str(plan_set.rtplan_path))
    if dose is None or meta is None or meta.isocenter is None:
        raise RuntimeError("Could not load RTDOSE/RTPLAN isocenter.")

    targets = _select_targets(structures, clean_text(meta.plan_label))
    if not targets:
        raise RuntimeError("No individual PTV/GTV target pairs found.")

    target_names = {structure.name for _, structure in targets}
    oar_names = {structure.name for structure in structures if structure.name in OAR_COLORS}
    target_meshes = _mesh_structures(structures, target_names, dose, args.ptv_faces)
    oar_meshes = _mesh_structures(structures, oar_names, dose, args.oar_faces)

    iso_verts, iso_faces = p3d._marching_cubes_dose(  # noqa: SLF001
        dose,
        args.rx_gy,
        stride=args.iso_stride,
        max_faces=args.iso_faces,
    )
    if iso_verts is None or iso_faces is None or len(iso_faces) == 0:
        raise RuntimeError(f"No {args.rx_gy:g} Gy isodose surface could be extracted.")

    oar_layers = [
        MeshLayer(name, verts, faces, OAR_COLORS[name], args.oar_alpha)
        for name, (verts, faces) in oar_meshes.items()
    ]
    ptv_layers = [
        MeshLayer(f"PTV/Mets {name}", verts, faces, PTV_COLOR, args.ptv_alpha)
        for name, (verts, faces) in target_meshes.items()
    ]
    scenarios = _make_scenarios(args)
    scenario_layers_static: dict[str, list[MeshLayer]] = {}
    scenario_layers_anim: dict[str, list[MeshLayer]] = {}
    scenario_stage_layers: dict[str, dict[str, list[MeshLayer]]] = {}

    for scenario in scenarios:
        shifted_verts = p3d._transform_verts(iso_verts, meta.isocenter, scenario)  # noqa: SLF001
        shifted_dose = build_shifted_dose(dose, scenario, meta.isocenter)
        static_layers = [
            *oar_layers,
            MeshLayer("Verschobene Isodose", shifted_verts, iso_faces, SHIFTED_COLOR, args.shifted_alpha),
            MeshLayer("Nominale Isodose", iso_verts, iso_faces, NOMINAL_COLOR, args.nominal_alpha),
            *ptv_layers,
        ]
        anim_layers = [
            MeshLayer(
                layer.name,
                layer.verts,
                _decimated_faces(layer.faces, _animation_face_limit(layer, args)),
                layer.color,
                layer.alpha,
            )
            for layer in static_layers
        ]
        nominal_layer = MeshLayer("Nominale Isodose", iso_verts, iso_faces, NOMINAL_COLOR, args.nominal_alpha)
        interaction_layers = _surface_interaction_layers(
            iso_verts,
            shifted_verts,
            iso_faces,
            nominal_dose=dose,
            rx_gy=args.rx_gy,
            dose_margin_gy=args.diff_dose_margin_gy,
        )
        miss_layers = _ptv_surface_undercoverage_layers(
            target_meshes,
            shifted_dose,
            rx_gy=args.rx_gy,
            dose_margin_gy=args.diff_dose_margin_gy,
        )
        diff_layers = _surface_difference_layers(
            iso_verts,
            shifted_verts,
            iso_faces,
            dose,
            rx_gy=args.rx_gy,
            dose_margin_gy=args.diff_dose_margin_gy,
            threshold_mm=args.diff_threshold_mm,
            min_faces=args.diff_min_faces,
        )
        scenario_stage_layers[scenario.name] = {
            "ptv": [*oar_layers, *ptv_layers],
            "nominal": [*oar_layers, nominal_layer, *ptv_layers],
            "shifted": [*oar_layers, *interaction_layers, *ptv_layers],
            "diff": [*oar_layers, *diff_layers, *ptv_layers],
            "miss": [*oar_layers, *ptv_layers, *miss_layers],
        }
        scenario_layers_static[scenario.name] = static_layers
        scenario_layers_anim[scenario.name] = anim_layers

    target_points = [_centroid(structure) for _, structure in targets]
    oar_label_points = []
    for name in sorted(oar_names):
        structure = next(structure for structure in structures if structure.name == name)
        label = {
            "Eye Left": "Eye L",
            "Eye Right": "Eye R",
            "Optic Nerve Left": "Optic N L",
            "Optic Nerve Right": "Optic N R",
            "Chiasm": "Chiasma",
            "Brainstem": "Brainstem",
        }.get(name, name)
        oar_label_points.append((label, _centroid(structure), OAR_COLORS[name]))

    label_points = oar_label_points

    all_axis_layers = [
        layer
        for scenario_layers in scenario_layers_static.values()
        for layer in scenario_layers
    ]
    axis_limits = _axis_limits(all_axis_layers, [meta.isocenter, *target_points], pad_fraction=args.axis_pad)
    return (
        scenarios,
        scenario_layers_static,
        scenario_layers_anim,
        scenario_stage_layers,
        meta.isocenter,
        targets,
        axis_limits,
        label_points,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render 12-met SRS scenario isodose visuals with anatomical context.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rx-gy", type=float, default=20.0)
    parser.add_argument("--translation-vertical-cm", type=float, default=0.08)
    parser.add_argument("--translation-longitudinal-cm", type=float, default=0.02)
    parser.add_argument("--translation-lateral-cm", type=float, default=0.02)
    parser.add_argument("--iso-faces", type=int, default=50000)
    parser.add_argument("--iso-stride", type=int, default=1)
    parser.add_argument("--ptv-faces", type=int, default=2600)
    parser.add_argument("--oar-faces", type=int, default=5200)
    parser.add_argument("--anim-iso-faces", type=int, default=50000)
    parser.add_argument("--anim-ptv-faces", type=int, default=1200)
    parser.add_argument("--anim-oar-faces", type=int, default=2200)
    parser.add_argument("--anim-context-faces", type=int, default=2200)
    parser.add_argument("--undercovered-faces", type=int, default=2600)
    parser.add_argument("--nominal-alpha", type=float, default=0.50)
    parser.add_argument("--shifted-alpha", type=float, default=0.70)
    parser.add_argument("--ptv-alpha", type=float, default=0.36)
    parser.add_argument("--oar-alpha", type=float, default=0.30)
    parser.add_argument("--axis-pad", type=float, default=-0.13)
    parser.add_argument("--static-dpi", type=int, default=240)
    parser.add_argument("--frame-dpi", type=int, default=160)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seconds-per-scenario", type=float, default=4.0)
    parser.add_argument("--fixed-rotation-seconds", type=float, default=12.0)
    parser.add_argument("--staged-build-seconds", type=float, default=2.5)
    parser.add_argument("--staged-rotation-seconds", type=float, default=12.0)
    parser.add_argument("--combined-zoom-factor", type=float, default=1.00)
    parser.add_argument("--combined-camera-distance", type=float, default=5.4)
    parser.add_argument("--diff-threshold-mm", type=float, default=0.20)
    parser.add_argument("--diff-dose-margin-gy", type=float, default=0.05)
    parser.add_argument("--diff-min-faces", type=int, default=250)
    parser.add_argument("--fixed-scenario", default="rot_total_1p0deg_xyz_equal")
    parser.add_argument("--staged-scenario", default="all", help="Scenario name for staged rendering, or 'all'.")
    parser.add_argument(
        "--animation-mode",
        choices=["fixed", "switching", "both", "staged", "combined", "all"],
        default="staged",
    )
    parser.add_argument("--write-gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-render", action="store_true")
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-animation", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.fast_render:
        args.write_gif = False
        args.fps = min(args.fps, 8)
        args.frame_dpi = min(args.frame_dpi, 100)
        args.iso_faces = min(args.iso_faces, 22000)
        args.ptv_faces = min(args.ptv_faces, 1400)
        args.oar_faces = min(args.oar_faces, 1800)
        args.undercovered_faces = min(args.undercovered_faces, 1200)
        args.staged_build_seconds = min(args.staged_build_seconds, 1.2)
        args.staged_rotation_seconds = min(args.staged_rotation_seconds, 5.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        scenarios,
        scenario_layers_static,
        scenario_layers_anim,
        scenario_stage_layers,
        isocenter,
        targets,
        axis_limits,
        label_points,
    ) = build_scene(args)

    written = []
    if not args.skip_static:
        for scenario in scenarios:
            written.append(
                render_static(
                    args.output_dir,
                    scenario=scenario,
                    layers=scenario_layers_static[scenario.name],
                    isocenter=isocenter,
                    targets=targets,
                    axis_limits=axis_limits,
                    label_points=label_points,
                    rx_gy=args.rx_gy,
                    static_dpi=args.static_dpi,
                )
            )

    if not args.skip_animation:
        if args.animation_mode in {"fixed", "both", "all"}:
            fixed_scenario = next((scenario for scenario in scenarios if scenario.name == args.fixed_scenario), scenarios[0])
            gif_path, mp4_path = save_animation(
                args.output_dir,
                scenarios=[fixed_scenario],
                scenario_layers=scenario_layers_anim,
                isocenter=isocenter,
                targets=targets,
                axis_limits=axis_limits,
                label_points=label_points,
                rx_gy=args.rx_gy,
                fps=args.fps,
                seconds_per_scenario=args.fixed_rotation_seconds,
                frame_dpi=args.frame_dpi,
                output_stem=f"12met_iso_shift_{_sanitize(fixed_scenario.name)}_rotation",
                show_scenario_status=False,
                write_gif=args.write_gif,
            )
            if gif_path is not None:
                written.append(gif_path)
            if mp4_path is not None:
                written.append(mp4_path)

        if args.animation_mode in {"switching", "both", "all"}:
            gif_path, mp4_path = save_animation(
                args.output_dir,
                scenarios=scenarios,
                scenario_layers=scenario_layers_anim,
                isocenter=isocenter,
                targets=targets,
                axis_limits=axis_limits,
                label_points=label_points,
                rx_gy=args.rx_gy,
                fps=args.fps,
                seconds_per_scenario=args.seconds_per_scenario,
                frame_dpi=args.frame_dpi,
                output_stem="12met_iso_shift_scenario_switching_rotation",
                show_scenario_status=True,
                write_gif=args.write_gif,
            )
            if gif_path is not None:
                written.append(gif_path)
            if mp4_path is not None:
                written.append(mp4_path)

        if args.animation_mode in {"staged", "all"}:
            staged_scenarios = [
                scenario for scenario in scenarios
                if args.staged_scenario == "all" or scenario.name == args.staged_scenario
            ]
            if not staged_scenarios:
                raise RuntimeError(f"No staged scenario matched {args.staged_scenario!r}.")
            for scenario in staged_scenarios:
                gif_path, mp4_path = save_staged_animation(
                    args.output_dir,
                    scenario=scenario,
                    stage_layers=scenario_stage_layers[scenario.name],
                    isocenter=isocenter,
                    targets=targets,
                    axis_limits=axis_limits,
                    label_points=label_points,
                    rx_gy=args.rx_gy,
                    fps=args.fps,
                    build_seconds=args.staged_build_seconds,
                    rotation_seconds=args.staged_rotation_seconds,
                    frame_dpi=args.frame_dpi,
                    write_gif=args.write_gif,
                )
                if gif_path is not None:
                    written.append(gif_path)
                if mp4_path is not None:
                    written.append(mp4_path)

        if args.animation_mode in {"combined", "all"}:
            gif_path, mp4_path = save_combined_staged_animation(
                args.output_dir,
                scenarios=scenarios,
                stage_layers=scenario_stage_layers,
                isocenter=isocenter,
                targets=targets,
                axis_limits=axis_limits,
                label_points=label_points,
                rx_gy=args.rx_gy,
                fps=args.fps,
                build_seconds=args.staged_build_seconds,
                rotation_seconds=args.staged_rotation_seconds,
                frame_dpi=args.frame_dpi,
                zoom_factor=args.combined_zoom_factor,
                camera_distance=args.combined_camera_distance,
                write_gif=args.write_gif,
            )
            if gif_path is not None:
                written.append(gif_path)
            if mp4_path is not None:
                written.append(mp4_path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
