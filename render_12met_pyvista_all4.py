from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_final_12met_visuals as final_export  # noqa: E402
import render_12met_scenario_visuals as scene3d  # noqa: E402


BG = "#070b12"
GRID = "#20283a"
TEXT = "#f4f7fb"
MUTED = "#a8b2c4"
CYAN = scene3d.UNCOVERED_COLOR
FOCUS_PTV_COLOR = "#fff1a8"
CANVAS_SIZE = (1520, 1632)
PANEL_SIZE = (690, 600)
INSET_SIZE = (260, 210)
PANEL_POSITIONS = [(45, 116), (785, 116), (45, 762), (785, 762)]
DEFAULT_FOCUS_TARGET_NUMBER = 8


@dataclass(frozen=True)
class PreparedLayer:
    name: str
    mesh: object
    color: str
    opacity: float
    rgba: np.ndarray | None
    culling: str | bool


SMOOTH_LAYER_NAMES = {
    "Nominale Isodose",
    "Verschobene Isodose",
    "Abweichung zur Nominalisodose",
    "Iso-Interaktion",
}


@dataclass(frozen=True)
class FocusScene:
    target_name: str
    target: tuple[int, object]
    center: np.ndarray
    radius: float
    layers: dict[str, list[PreparedLayer]]
    volume_cc: float | None = None
    distance_to_iso_mm: float | None = None


@dataclass(frozen=True)
class PreparedScene:
    scenarios: list
    stage_layers: dict[str, dict[str, list[PreparedLayer]]]
    focus: dict[str, FocusScene]
    isocenter: np.ndarray
    targets: list[tuple[int, object]]
    axis_limits: tuple[float, ...]
    rx_gy: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render smooth 12-met all4 staged views with PyVista/VTK.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "rigid_error_case",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--build-seconds", type=float, default=2.0)
    parser.add_argument("--rotation-seconds", type=float, default=20.0 / 3.0)
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--write-gif", action="store_true")
    parser.add_argument("--focus-target-number", type=int, default=DEFAULT_FOCUS_TARGET_NUMBER)
    return parser.parse_args()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(40)
FONT_PANEL = _font(30, bold=True)
FONT_DETAIL = _font(18)
FONT_SMALL = _font(18)
FONT_LEGEND = _font(16)

STAGES = [
    ("ptv", "1. PTV + OAR", 55.0, False),
    ("nominal", "2. Nominale 20 Gy", 55.0, False),
    ("shifted", "3. Iso-Interaktion", 360.0, True),
    ("diff", "4. Rest ausserhalb nominal", 360.0, True),
    ("miss", "5. PTV ohne 20 Gy", 360.0, True),
]


def _base_render_args(output_dir: Path) -> argparse.Namespace:
    return scene3d.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--skip-static",
            "--skip-animation",
            "--no-write-gif",
            "--iso-faces",
            "65000",
            "--ptv-faces",
            "8000",
            "--oar-faces",
            "65000",
            "--anim-iso-faces",
            "65000",
            "--anim-ptv-faces",
            "8000",
            "--anim-oar-faces",
            "65000",
            "--anim-context-faces",
            "65000",
            "--combined-zoom-factor",
            "1.00",
            "--combined-camera-distance",
            "5.4",
            "--diff-dose-margin-gy",
            "0.05",
        ]
    )


def _polydata_from_layer(layer: scene3d.MeshLayer):
    import pyvista as pv

    verts = np.asarray(layer.verts, dtype=float)
    faces = np.asarray(layer.faces, dtype=np.int64)
    cells = np.empty((len(faces), 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    mesh = pv.PolyData(verts, cells.ravel())
    mesh = mesh.clean(tolerance=1e-7)
    try:
        mesh = mesh.compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=False,
        )
    except Exception:
        pass
    is_oar = layer.name in scene3d.OAR_COLORS
    should_smooth = layer.name.startswith("PTV/Mets ") or layer.name.startswith("PTV ohne 100% ") or layer.name in SMOOTH_LAYER_NAMES
    if is_oar:
        try:
            mesh = mesh.smooth_taubin(n_iter=10, pass_band=0.16, boundary_smoothing=False)
            mesh = mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
        except Exception:
            pass
    elif should_smooth:
        try:
            if mesh.n_cells < 6000:
                mesh = mesh.subdivide(2, subfilter="loop")
            elif mesh.n_cells < 30000:
                mesh = mesh.subdivide(1, subfilter="loop")
            mesh = mesh.smooth_taubin(n_iter=60, pass_band=0.045, boundary_smoothing=True)
            mesh = mesh.smooth(n_iter=24, relaxation_factor=0.020, boundary_smoothing=True)
            mesh = mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
        except Exception:
            pass
    return mesh


def _is_focus_ptv_layer(name: str, focus_target_number: int) -> bool:
    return name.startswith(f"PTV/Mets PTV{focus_target_number:02d}")


def _layer_opacity(layer: scene3d.MeshLayer, focus_target_number: int = DEFAULT_FOCUS_TARGET_NUMBER) -> float:
    if _is_focus_ptv_layer(layer.name, focus_target_number):
        return 0.60
    if layer.name.startswith("PTV/Mets "):
        return 0.36
    if layer.name == "Nominale Isodose":
        return 0.44
    if layer.name == "Verschobene Isodose":
        return 0.48
    if layer.name in {"Abweichung zur Nominalisodose", "Iso-Interaktion"}:
        return 0.58
    if layer.name.startswith("PTV ohne 100% "):
        return 0.62
    if layer.name in scene3d.OAR_COLORS:
        return 0.34
    return float(layer.alpha)


def _prepare_layer(layer: scene3d.MeshLayer, focus_target_number: int = DEFAULT_FOCUS_TARGET_NUMBER) -> PreparedLayer:
    rgba = None
    if layer.facecolors is not None:
        facecolors = np.asarray(layer.facecolors)
        if facecolors.max(initial=1.0) <= 1.0:
            rgba = np.clip(facecolors * 255, 0, 255).astype(np.uint8)
        else:
            rgba = np.clip(facecolors, 0, 255).astype(np.uint8)
    return PreparedLayer(
        name=layer.name,
        mesh=_polydata_from_layer(layer),
        color=FOCUS_PTV_COLOR if _is_focus_ptv_layer(layer.name, focus_target_number) else layer.color,
        opacity=_layer_opacity(layer, focus_target_number),
        rgba=rgba,
        culling="back",
    )


def _prepare_layers(
    layers: list[scene3d.MeshLayer],
    focus_target_number: int = DEFAULT_FOCUS_TARGET_NUMBER,
) -> list[PreparedLayer]:
    return [
        _prepare_layer(layer, focus_target_number)
        for layer in layers
        if layer.faces is not None and len(layer.faces) > 0
    ]


def _full_interaction_layers(static_layers: list[scene3d.MeshLayer]) -> list[scene3d.MeshLayer]:
    oars = [layer for layer in static_layers if layer.name in scene3d.OAR_COLORS]
    nominal = [layer for layer in static_layers if layer.name == "Nominale Isodose"]
    shifted = [layer for layer in static_layers if layer.name == "Verschobene Isodose"]
    return [*oars, *nominal, *shifted]


def _preferred_focus(
    layers_by_stage: dict[str, list[scene3d.MeshLayer]],
    targets: list[tuple[int, object]],
    *,
    preferred_number: int = DEFAULT_FOCUS_TARGET_NUMBER,
) -> tuple[str, tuple[int, object], scene3d.MeshLayer] | None:
    target = next((item for item in targets if item[0] == preferred_number), None)
    if target is None:
        return scene3d._largest_undercovered_focus(layers_by_stage["miss"], targets)  # noqa: SLF001

    target_name = getattr(target[1], "name", "")
    for layer in layers_by_stage.get("ptv", []):
        if layer.name == f"PTV/Mets {target_name}":
            return target_name, target, layer

    needle = f"PTV{preferred_number:02d}"
    for stage_layers in layers_by_stage.values():
        for layer in stage_layers:
            if layer.name.startswith("PTV/Mets ") and needle in layer.name:
                return layer.name.removeprefix("PTV/Mets "), target, layer

    return scene3d._largest_undercovered_focus(layers_by_stage["miss"], targets)  # noqa: SLF001


def _target_metric_map(
    targets: list[tuple[int, object]],
    isocenter: np.ndarray,
    data_root: Path,
) -> dict[str, dict[str, float]]:
    try:
        plan_sets = scene3d.discover_plan_sets(data_root)
        dose = scene3d.load_dose(str(plan_sets[0].rtdose_path)) if plan_sets else None
    except Exception:
        dose = None
    if dose is None:
        return {}

    metrics: dict[str, dict[str, float]] = {}
    for number, structure in targets:
        try:
            mask = scene3d._structure_mask_on_dose_grid(structure, dose)  # noqa: SLF001
            volume_cc = float(np.count_nonzero(mask) * np.prod(dose.spacing) / 1000.0)
            centroid = scene3d._centroid(structure)  # noqa: SLF001
            distance_to_iso_mm = float(np.linalg.norm(centroid - isocenter))
        except Exception:
            continue
        values = {"volume_cc": volume_cc, "distance_to_iso_mm": distance_to_iso_mm}
        metrics[getattr(structure, "name", f"PTV{number:02d}")] = values
        metrics[f"PTV{number:02d}"] = values
    return metrics


def _build_scene(output_dir: Path, focus_target_number: int = DEFAULT_FOCUS_TARGET_NUMBER) -> PreparedScene:
    args = _base_render_args(output_dir)
    (
        scenarios,
        static_layers,
        _anim_layers,
        stage_layers,
        isocenter,
        targets,
        axis_limits,
        _label_points,
    ) = scene3d.build_scene(args)
    target_metrics = _target_metric_map(targets, isocenter, args.data_root)
    prepared_stage_layers: dict[str, dict[str, list[PreparedLayer]]] = {}
    prepared_focus: dict[str, FocusScene] = {}
    for scenario in scenarios:
        raw_stage_layers = dict(stage_layers[scenario.name])
        raw_stage_layers["shifted"] = _full_interaction_layers(static_layers[scenario.name])
        prepared_stage_layers[scenario.name] = {
            key: _prepare_layers(layers, focus_target_number)
            for key, layers in raw_stage_layers.items()
        }
        focus = _preferred_focus(raw_stage_layers, targets, preferred_number=focus_target_number)
        if focus is None:
            continue
        target_name, target, ptv_layer = focus
        focus_geometry = scene3d._focus_center_radius(ptv_layer)  # noqa: SLF001
        if focus_geometry is None:
            continue
        center, radius = focus_geometry
        focus_layers = {}
        for key, layers in raw_stage_layers.items():
            clipped = scene3d._stage_focus_layers(  # noqa: SLF001
                layers,
                target_name=target_name,
                center=center,
                radius=radius * 1.65,
            )
            focus_layers[key] = _prepare_layers(clipped, focus_target_number)
        metrics = target_metrics.get(target_name) or target_metrics.get(f"PTV{target[0]:02d}", {})
        prepared_focus[scenario.name] = FocusScene(
            target_name=target_name,
            target=target,
            center=center,
            radius=radius,
            layers=focus_layers,
            volume_cc=metrics.get("volume_cc"),
            distance_to_iso_mm=metrics.get("distance_to_iso_mm"),
        )
    return PreparedScene(
        scenarios=scenarios,
        stage_layers=prepared_stage_layers,
        focus=prepared_focus,
        isocenter=isocenter,
        targets=targets,
        axis_limits=scene3d._scale_axis_limits(axis_limits, 1.0),  # noqa: SLF001
        rx_gy=args.rx_gy,
    )


def _camera_position(bounds: tuple[float, ...], azim: float, elev: float, distance_factor: float = 3.4):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5])
    extent = max(xmax - xmin, ymax - ymin, zmax - zmin)
    a = math.radians(azim)
    e = math.radians(elev)
    direction = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    position = center + direction * extent * distance_factor
    return position, center, extent


def _configure_camera(plotter, bounds: tuple[float, ...], azim: float, elev: float, scale: float) -> None:
    position, center, extent = _camera_position(bounds, azim, elev)
    plotter.camera_position = (position.tolist(), center.tolist(), [0, 0, 1])
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = extent * scale
    plotter.camera.clipping_range = (1.0, extent * 20.0)


def _add_meshes(plotter, layers: list[PreparedLayer]) -> None:
    for layer in layers:
        if layer.rgba is not None:
            mesh = layer.mesh.copy(deep=False)
            mesh.cell_data["rgba"] = layer.rgba
            plotter.add_mesh(
                mesh,
                scalars="rgba",
                rgb=True,
                smooth_shading=True,
                show_edges=False,
                lighting=True,
                culling=layer.culling,
                ambient=0.35,
                diffuse=0.70,
                specular=0.05,
            )
        else:
            plotter.add_mesh(
                layer.mesh,
                color=layer.color,
                opacity=layer.opacity,
                smooth_shading=True,
                show_edges=False,
                lighting=True,
                culling=layer.culling,
                ambient=0.35,
                diffuse=0.70,
                specular=0.05,
            )


def _add_points_and_vectors(plotter, targets, scenario, isocenter: np.ndarray, show_vectors: bool) -> None:
    import pyvista as pv

    plotter.add_mesh(pv.Sphere(radius=1.35, theta_resolution=24, phi_resolution=24, center=isocenter), color="white")
    for _number, structure in targets:
        start = scene3d._centroid(structure)  # noqa: SLF001
        plotter.add_mesh(pv.Sphere(radius=0.65, theta_resolution=16, phi_resolution=16, center=start), color="white")
        if not show_vectors:
            continue
        end = scene3d.forward_transform_point(start, scenario, isocenter)
        if float(np.linalg.norm(end - start)) < 0.05:
            continue
        line = pv.Line(start, end).tube(radius=0.22, n_sides=12)
        plotter.add_mesh(line, color=scene3d.VECTOR_COLOR, opacity=0.86, smooth_shading=True)
        plotter.add_mesh(pv.Sphere(radius=0.75, theta_resolution=16, phi_resolution=16, center=end), color=scene3d.VECTOR_COLOR)


def _render_panel(
    *,
    layers: list[PreparedLayer],
    bounds: tuple[float, ...],
    targets,
    scenario,
    isocenter: np.ndarray,
    azim: float,
    elev: float,
    size: tuple[int, int],
    show_vectors: bool,
    camera_scale: float,
    show_bounds: bool = True,
) -> Image.Image:
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background(BG)
    plotter.enable_depth_peeling(number_of_peels=16, occlusion_ratio=0.0)
    plotter.enable_anti_aliasing("ssaa")
    _add_meshes(plotter, layers)
    _add_points_and_vectors(plotter, targets, scenario, isocenter, show_vectors)
    if show_bounds:
        plotter.show_bounds(
            bounds=bounds,
            grid="back",
            location="outer",
            color=GRID,
            xtitle="X",
            ytitle="Y",
            ztitle="Z",
            font_size=8,
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
            all_edges=False,
        )
    _configure_camera(plotter, bounds, azim, elev, camera_scale)
    arr = plotter.screenshot(return_img=True)
    plotter.close()
    return Image.fromarray(arr[:, :, :3])


class PanelRenderer:
    def __init__(
        self,
        *,
        layers: list[PreparedLayer],
        bounds: tuple[float, ...],
        targets,
        scenario,
        isocenter: np.ndarray,
        size: tuple[int, int],
        show_vectors: bool,
        camera_scale: float,
        show_bounds: bool = True,
        elev: float = 22.0,
    ) -> None:
        import pyvista as pv

        self.bounds = bounds
        self.camera_scale = camera_scale
        self.elev = elev
        self.plotter = pv.Plotter(off_screen=True, window_size=size)
        self.plotter.set_background(BG)
        self.plotter.enable_depth_peeling(number_of_peels=16, occlusion_ratio=0.0)
        self.plotter.enable_anti_aliasing("ssaa")
        _add_meshes(self.plotter, layers)
        _add_points_and_vectors(self.plotter, targets, scenario, isocenter, show_vectors)
        if show_bounds:
            self.plotter.show_bounds(
                bounds=bounds,
                grid="back",
                location="outer",
                color=GRID,
                xtitle="X",
                ytitle="Y",
                ztitle="Z",
                font_size=8,
                n_xlabels=5,
                n_ylabels=5,
                n_zlabels=5,
                all_edges=False,
            )

    def image(self, azim: float) -> Image.Image:
        _configure_camera(self.plotter, self.bounds, azim, self.elev, self.camera_scale)
        self.plotter.render()
        arr = self.plotter.screenshot(return_img=True)
        return Image.fromarray(arr[:, :, :3])

    def close(self) -> None:
        self.plotter.close()


def _rounded_box(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=TEXT, outline="#354052", bg="#101722") -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    pad_x, pad_y = 8, 4
    box = [xy[0], xy[1], xy[0] + bbox[2] + 2 * pad_x, xy[1] + bbox[3] + 2 * pad_y]
    draw.rounded_rectangle(box, radius=7, fill=bg, outline=outline, width=2)
    draw.text((xy[0] + pad_x, xy[1] + pad_y - 1), text, font=font, fill=fill)


def _focus_label(focus: FocusScene | None) -> str | None:
    if focus is None:
        return None
    details: list[str] = []
    if focus.volume_cc is not None:
        details.append(f"V={focus.volume_cc:.2f} cc")
    if focus.distance_to_iso_mm is not None:
        details.append(f"Iso={focus.distance_to_iso_mm:.1f} mm")
    if details:
        return "  ".join(details)
    return f"PTV{focus.target[0]}"


def _draw_focus_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], focus: FocusScene | None) -> None:
    text = _focus_label(focus)
    if not text:
        return
    bbox = draw.multiline_textbbox((0, 0), text, font=FONT_LEGEND, spacing=2)
    pad_x, pad_y = 7, 5
    x, y = xy[0] + 8, xy[1] + 8
    box = [x, y, x + bbox[2] + 2 * pad_x, y + bbox[3] + 2 * pad_y]
    draw.rounded_rectangle(box, radius=5, fill="#101722", outline="#334052", width=1)
    draw.multiline_text((x + pad_x, y + pad_y - 1), text, fill=TEXT, font=FONT_LEGEND, spacing=2)


def _draw_legend(draw: ImageDraw.ImageDraw, y: int) -> None:
    items = [
        ("PTVs", scene3d.PTV_COLOR),
        ("PTV8 Fokus", FOCUS_PTV_COLOR),
        ("Nominale 20 Gy Isodose", scene3d.NOMINAL_COLOR),
        ("Szenario Isodose", scene3d.DIFF_SHIFTED_COLOR),
        ("PTV ohne 20 Gy nach Shift", scene3d.UNCOVERED_COLOR),
        ("Brainstem", scene3d.OAR_COLORS["Brainstem"]),
        ("Optic Nerves", scene3d.OAR_COLORS["Optic Nerve Left"]),
        ("Chiasma", scene3d.OAR_COLORS["Chiasm"]),
        ("Eyes", scene3d.OAR_COLORS["Eye Left"]),
    ]
    x = 72
    col_w = 350
    row_h = 30
    legend_rows = math.ceil(len(items) / 4)
    draw.rounded_rectangle([48, y - 12, 1472, y + legend_rows * row_h + 14], radius=5, fill="#101722", outline="#263046", width=2)
    for index, (label, color) in enumerate(items):
        col = index % 4
        row = index // 4
        item_x = x + col * col_w
        item_y = y + row * row_h
        draw.rectangle([item_x, item_y, item_x + 44, item_y + 17], fill=color)
        draw.text((item_x + 54, item_y - 3), label, fill=TEXT, font=FONT_LEGEND)


def _compose_combined_frame(
    prepared: PreparedScene,
    *,
    stage_title: str,
    panel_images: list[Image.Image],
    inset_images: dict[str, Image.Image],
) -> np.ndarray:
    canvas = Image.new("RGB", CANVAS_SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    title_bbox = draw.textbbox((0, 0), stage_title, font=FONT_TITLE)
    draw.text(((CANVAS_SIZE[0] - title_bbox[2]) // 2, 24), stage_title, fill=TEXT, font=FONT_TITLE)
    draw.text(
        (CANVAS_SIZE[0] - 36, 78),
        f"{prepared.rx_gy:g} Gy Isodose\nsynchrone Kamera, gleiche Legende",
        fill=MUTED,
        font=FONT_SMALL,
        anchor="ra",
        align="right",
    )
    for scenario, position, panel in zip(prepared.scenarios, PANEL_POSITIONS, panel_images):
        canvas.paste(panel, position)
        panel_draw = ImageDraw.Draw(canvas)
        _rounded_box(panel_draw, (position[0] + 10, position[1] + 10), scene3d._scenario_short_label(scenario), FONT_PANEL)  # noqa: SLF001
        detail = scene3d._scenario_compact_detail(scenario)  # noqa: SLF001
        detail_bbox = panel_draw.textbbox((0, 0), detail, font=FONT_DETAIL)
        panel_draw.rounded_rectangle(
            [
                position[0] + PANEL_SIZE[0] - detail_bbox[2] - 20,
                position[1] + 16,
                position[0] + PANEL_SIZE[0] - 8,
                position[1] + detail_bbox[3] + 28,
            ],
            radius=5,
            fill="#101722",
        )
        panel_draw.text(
            (position[0] + PANEL_SIZE[0] - 14, position[1] + 20),
            detail,
            fill=MUTED,
            font=FONT_DETAIL,
            anchor="ra",
        )
        inset = inset_images.get(scenario.name)
        if inset is not None:
            inset_pos = (position[0] + 4, position[1] + PANEL_SIZE[1] - INSET_SIZE[1] - 18)
            canvas.paste(inset, inset_pos)
            draw.rectangle(
                [
                    inset_pos[0],
                    inset_pos[1],
                    inset_pos[0] + INSET_SIZE[0] - 1,
                    inset_pos[1] + INSET_SIZE[1] - 1,
                ],
                outline=CYAN,
                width=4,
            )
            _draw_focus_label(draw, inset_pos, prepared.focus.get(scenario.name))
    _draw_legend(draw, 1402)
    footer = "Alle Szenarien laufen synchron mit gleicher Kamera und gemeinsamer Legende."
    footer_bbox = draw.textbbox((0, 0), footer, font=FONT_SMALL)
    draw.text(((CANVAS_SIZE[0] - footer_bbox[2]) // 2, 1568), footer, fill=MUTED, font=FONT_SMALL)
    return np.asarray(canvas)


def _compose_individual_frame(
    prepared: PreparedScene,
    *,
    scenario,
    stage_title: str,
    panel: Image.Image,
    inset: Image.Image | None,
) -> np.ndarray:
    size = (2560, 1440)
    canvas = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(canvas)
    title = f"{scene3d._scenario_short_label(scenario)} | {stage_title}"  # noqa: SLF001
    title_bbox = draw.textbbox((0, 0), title, font=FONT_TITLE)
    draw.text(((size[0] - title_bbox[2]) // 2, 28), title, fill=TEXT, font=FONT_TITLE)
    draw.text(
        (size[0] - 42, 76),
        f"{prepared.rx_gy:g} Gy Isodose\n{scene3d._scenario_compact_detail(scenario)}",  # noqa: SLF001
        fill=MUTED,
        font=FONT_SMALL,
        anchor="ra",
        align="right",
    )
    panel_position = (160, 112)
    canvas.paste(panel, panel_position)
    if inset is not None:
        inset_position = (110, size[1] - 760)
        canvas.paste(inset, inset_position)
        draw.rectangle(
            [
                inset_position[0],
                inset_position[1],
                inset_position[0] + inset.size[0] - 1,
                inset_position[1] + inset.size[1] - 1,
            ],
            outline=CYAN,
            width=5,
        )
        _draw_focus_label(draw, inset_position, prepared.focus.get(scenario.name))
    _draw_legend(draw, 1272)
    return np.asarray(canvas)


def render_combined_frame(
    prepared: PreparedScene,
    *,
    stage_key: str,
    stage_title: str,
    azim: float,
    show_vectors: bool,
) -> np.ndarray:
    panel_images: list[Image.Image] = []
    inset_images: dict[str, Image.Image] = {}
    for scenario in prepared.scenarios:
        panel_images.append(
            _render_panel(
                layers=prepared.stage_layers[scenario.name][stage_key],
                bounds=prepared.axis_limits,
                targets=prepared.targets,
                scenario=scenario,
                isocenter=prepared.isocenter,
                azim=azim,
                elev=22,
                size=PANEL_SIZE,
                show_vectors=show_vectors,
                camera_scale=0.66,
            )
        )
        focus = prepared.focus.get(scenario.name)
        if focus is not None and focus.layers.get(stage_key):
            inset_bounds = scene3d._focus_limits(focus.center, focus.radius)  # noqa: SLF001
            inset_images[scenario.name] = _render_panel(
                layers=focus.layers[stage_key],
                bounds=inset_bounds,
                targets=[focus.target],
                scenario=scenario,
                isocenter=prepared.isocenter,
                azim=azim,
                elev=22,
                size=INSET_SIZE,
                show_vectors=show_vectors,
                camera_scale=0.60,
                show_bounds=True,
            )
    return _compose_combined_frame(
        prepared,
        stage_title=stage_title,
        panel_images=panel_images,
        inset_images=inset_images,
    )


def _midphase_azimuths() -> list[tuple[str, str, float, bool]]:
    return final_export._midphase_azimuths()  # noqa: SLF001


def render_individual_frame(
    prepared: PreparedScene,
    *,
    scenario,
    stage_key: str,
    stage_title: str,
    azim: float,
    show_vectors: bool,
) -> np.ndarray:
    panel_size = (2240, 1120)
    inset_size = (460, 330)
    panel = _render_panel(
        layers=prepared.stage_layers[scenario.name][stage_key],
        bounds=prepared.axis_limits,
        targets=prepared.targets,
        scenario=scenario,
        isocenter=prepared.isocenter,
        azim=azim,
        elev=22,
        size=panel_size,
        show_vectors=show_vectors,
        camera_scale=0.63,
    )
    inset = None
    focus = prepared.focus.get(scenario.name)
    if focus is not None and focus.layers.get(stage_key):
        inset = _render_panel(
            layers=focus.layers[stage_key],
            bounds=scene3d._focus_limits(focus.center, focus.radius),  # noqa: SLF001
            targets=[focus.target],
            scenario=scenario,
            isocenter=prepared.isocenter,
            azim=azim,
            elev=22,
            size=inset_size,
            show_vectors=show_vectors,
            camera_scale=0.60,
            show_bounds=True,
        )
    return _compose_individual_frame(prepared, scenario=scenario, stage_title=stage_title, panel=panel, inset=inset)


def write_midphase_images(prepared: PreparedScene, output_dir: Path) -> list[Path]:
    out = output_dir / "midphase" / "all4"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, (stage_key, title, azim, show_vectors) in enumerate(_midphase_azimuths(), start=1):
        frame = render_combined_frame(
            prepared,
            stage_key=stage_key,
            stage_title=title,
            azim=azim,
            show_vectors=show_vectors,
        )
        path = out / f"{index:02d}_{stage_key}.png"
        Image.fromarray(frame).save(path)
        paths.append(path)
    images = [Image.open(path).convert("RGB") for path in paths]
    sheet = Image.new("RGB", (CANVAS_SIZE[0] * 2, CANVAS_SIZE[1] * 3), BG)
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 2) * CANVAS_SIZE[0], (index // 2) * CANVAS_SIZE[1]))
    contact = out / "contact.png"
    sheet.save(contact)
    paths.append(contact)
    return paths


def write_individual_midphase_images(prepared: PreparedScene, output_dir: Path) -> list[Path]:
    root = output_dir / "midphase" / "individual"
    written: list[Path] = []
    for scenario in prepared.scenarios:
        scenario_dir = root / scene3d._sanitize(scenario.name)  # noqa: SLF001
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_paths: list[Path] = []
        for index, (stage_key, title, azim, show_vectors) in enumerate(_midphase_azimuths(), start=1):
            frame = render_individual_frame(
                prepared,
                scenario=scenario,
                stage_key=stage_key,
                stage_title=title,
                azim=azim,
                show_vectors=show_vectors,
            )
            path = scenario_dir / f"{index:02d}_{stage_key}.png"
            Image.fromarray(frame).save(path)
            scenario_paths.append(path)
            written.append(path)
        images = [Image.open(path).convert("RGB") for path in scenario_paths]
        sheet = Image.new("RGB", (2560 * 2, 1440 * 3), BG)
        for index, image in enumerate(images):
            sheet.paste(image, ((index % 2) * 2560, (index // 2) * 1440))
        contact = scenario_dir / "contact.png"
        sheet.save(contact)
        written.append(contact)
    return written


def _create_stage_renderers(prepared: PreparedScene, stage_key: str, show_vectors: bool):
    renderers = []
    for scenario in prepared.scenarios:
        main_renderer = PanelRenderer(
            layers=prepared.stage_layers[scenario.name][stage_key],
            bounds=prepared.axis_limits,
            targets=prepared.targets,
            scenario=scenario,
            isocenter=prepared.isocenter,
            size=PANEL_SIZE,
            show_vectors=show_vectors,
            camera_scale=0.66,
        )
        inset_renderer = None
        focus = prepared.focus.get(scenario.name)
        if focus is not None and focus.layers.get(stage_key):
            inset_renderer = PanelRenderer(
                layers=focus.layers[stage_key],
                bounds=scene3d._focus_limits(focus.center, focus.radius),  # noqa: SLF001
                targets=[focus.target],
                scenario=scenario,
                isocenter=prepared.isocenter,
                size=INSET_SIZE,
                show_vectors=show_vectors,
                camera_scale=0.60,
                show_bounds=True,
            )
        renderers.append((scenario, main_renderer, inset_renderer))
    return renderers


def _create_individual_stage_renderer(prepared: PreparedScene, scenario, stage_key: str, show_vectors: bool):
    panel_size = (2240, 1120)
    inset_size = (460, 330)
    main_renderer = PanelRenderer(
        layers=prepared.stage_layers[scenario.name][stage_key],
        bounds=prepared.axis_limits,
        targets=prepared.targets,
        scenario=scenario,
        isocenter=prepared.isocenter,
        size=panel_size,
        show_vectors=show_vectors,
        camera_scale=0.63,
    )
    inset_renderer = None
    focus = prepared.focus.get(scenario.name)
    if focus is not None and focus.layers.get(stage_key):
        inset_renderer = PanelRenderer(
            layers=focus.layers[stage_key],
            bounds=scene3d._focus_limits(focus.center, focus.radius),  # noqa: SLF001
            targets=[focus.target],
            scenario=scenario,
            isocenter=prepared.isocenter,
            size=inset_size,
            show_vectors=show_vectors,
            camera_scale=0.60,
            show_bounds=True,
        )
    return main_renderer, inset_renderer


def _close_stage_renderers(renderers) -> None:
    for _scenario, main_renderer, inset_renderer in renderers:
        main_renderer.close()
        if inset_renderer is not None:
            inset_renderer.close()


def _close_individual_stage_renderer(main_renderer, inset_renderer) -> None:
    main_renderer.close()
    if inset_renderer is not None:
        inset_renderer.close()


def _render_frame_from_stage_renderers(
    prepared: PreparedScene,
    *,
    stage_title: str,
    azim: float,
    renderers,
) -> np.ndarray:
    panel_images: list[Image.Image] = []
    inset_images: dict[str, Image.Image] = {}
    for scenario, main_renderer, inset_renderer in renderers:
        panel_images.append(main_renderer.image(azim))
        if inset_renderer is not None:
            inset_images[scenario.name] = inset_renderer.image(azim)
    return _compose_combined_frame(
        prepared,
        stage_title=stage_title,
        panel_images=panel_images,
        inset_images=inset_images,
    )


def _render_individual_frame_from_renderers(
    prepared: PreparedScene,
    *,
    scenario,
    stage_title: str,
    azim: float,
    main_renderer,
    inset_renderer,
) -> np.ndarray:
    panel = main_renderer.image(azim)
    inset = inset_renderer.image(azim) if inset_renderer is not None else None
    return _compose_individual_frame(prepared, scenario=scenario, stage_title=stage_title, panel=panel, inset=inset)


def write_video(prepared: PreparedScene, output_dir: Path, fps: int, build_seconds: float, rotation_seconds: float) -> Path:
    stages = [
        (key, title, build_seconds if key in {"ptv", "nominal"} else rotation_seconds, azim_delta, show_vectors)
        for key, title, azim_delta, show_vectors in STAGES
    ]
    out = output_dir / "videos" / "all4"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "12met_iso_shift_all4_staged_grid.mp4"
    writer = imageio.get_writer(path, fps=fps, quality=8, macro_block_size=16)
    current_azim = -55.0
    try:
        for stage_key, title, seconds, azim_delta, show_vectors in stages:
            print(f"stage {stage_key}: building renderers", flush=True)
            renderers = _create_stage_renderers(prepared, stage_key, show_vectors)
            frame_count = max(1, int(round(fps * seconds)))
            try:
                for frame_index in range(frame_count):
                    progress = frame_index / max(frame_count - 1, 1)
                    frame = _render_frame_from_stage_renderers(
                        prepared,
                        stage_title=title,
                        azim=current_azim + azim_delta * progress,
                        renderers=renderers,
                    )
                    writer.append_data(frame)
                    if frame_index == 0 or frame_index == frame_count - 1 or (frame_index + 1) % max(1, frame_count // 4) == 0:
                        print(f"stage {stage_key}: frame {frame_index + 1}/{frame_count}", flush=True)
                current_azim += azim_delta
            finally:
                _close_stage_renderers(renderers)
    finally:
        writer.close()
    return path


def write_individual_videos(
    prepared: PreparedScene,
    output_dir: Path,
    fps: int,
    build_seconds: float,
    rotation_seconds: float,
) -> list[Path]:
    out = output_dir / "videos" / "individual"
    out.mkdir(parents=True, exist_ok=True)
    stages = [
        (key, title, build_seconds if key in {"ptv", "nominal"} else rotation_seconds, azim_delta, show_vectors)
        for key, title, azim_delta, show_vectors in STAGES
    ]
    written: list[Path] = []
    for scenario in prepared.scenarios:
        path = out / f"12met_iso_shift_{scene3d._sanitize(scenario.name)}_staged.mp4"  # noqa: SLF001
        writer = imageio.get_writer(path, fps=fps, quality=8, macro_block_size=16)
        current_azim = -55.0
        try:
            for stage_key, title, seconds, azim_delta, show_vectors in stages:
                print(f"{scenario.name} stage {stage_key}: building renderer", flush=True)
                main_renderer, inset_renderer = _create_individual_stage_renderer(prepared, scenario, stage_key, show_vectors)
                frame_count = max(1, int(round(fps * seconds)))
                try:
                    for frame_index in range(frame_count):
                        progress = frame_index / max(frame_count - 1, 1)
                        frame = _render_individual_frame_from_renderers(
                            prepared,
                            scenario=scenario,
                            stage_title=title,
                            azim=current_azim + azim_delta * progress,
                            main_renderer=main_renderer,
                            inset_renderer=inset_renderer,
                        )
                        writer.append_data(frame)
                        if frame_index == 0 or frame_index == frame_count - 1 or (frame_index + 1) % max(1, frame_count // 3) == 0:
                            print(f"{scenario.name} stage {stage_key}: frame {frame_index + 1}/{frame_count}", flush=True)
                    current_azim += azim_delta
                finally:
                    _close_individual_stage_renderer(main_renderer, inset_renderer)
        finally:
            writer.close()
        written.append(path)
    return written


def write_gif_from_mp4(mp4_path: Path, output_dir: Path) -> Path:
    import cv2

    out = output_dir / "gifs" / "all4"
    out.mkdir(parents=True, exist_ok=True)
    gif_path = out / "12met_iso_shift_all4_staged_grid.gif"
    cap = cv2.VideoCapture(str(mp4_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    step = max(1, round(source_fps / 6.0))
    writer = imageio.get_writer(gif_path, mode="I", fps=source_fps / step, loop=0)
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                frame = cv2.resize(frame, (600, 644), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame)
            index += 1
    finally:
        writer.close()
        cap.release()
    return gif_path


def write_individual_gifs_from_mp4(mp4_paths: list[Path], output_dir: Path) -> list[Path]:
    import cv2

    out = output_dir / "gifs" / "individual"
    out.mkdir(parents=True, exist_ok=True)
    gif_paths: list[Path] = []
    for mp4_path in mp4_paths:
        gif_path = out / mp4_path.with_suffix(".gif").name
        cap = cv2.VideoCapture(str(mp4_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
        step = max(1, round(source_fps / 6.0))
        writer = imageio.get_writer(gif_path, mode="I", fps=source_fps / step, loop=0)
        try:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    frame = cv2.resize(frame, (720, 405), interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(frame)
                index += 1
        finally:
            writer.close()
            cap.release()
        gif_paths.append(gif_path)
    return gif_paths


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _build_scene(args.output_dir, focus_target_number=args.focus_target_number)
    for path in write_midphase_images(prepared, args.output_dir):
        print(path)
    for path in write_individual_midphase_images(prepared, args.output_dir):
        print(path)
    if args.write_video:
        mp4 = write_video(prepared, args.output_dir, args.fps, args.build_seconds, args.rotation_seconds)
        print(mp4)
        individual_mp4s = write_individual_videos(prepared, args.output_dir, args.fps, args.build_seconds, args.rotation_seconds)
        for path in individual_mp4s:
            print(path)
        if args.write_gif:
            print(write_gif_from_mp4(mp4, args.output_dir))
            for path in write_individual_gifs_from_mp4(individual_mp4s, args.output_dir):
                print(path)


if __name__ == "__main__":
    main()
