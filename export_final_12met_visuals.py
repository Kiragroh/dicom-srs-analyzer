from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import render_12met_scenario_visuals as r


STAGES = [
    ("ptv", "1. PTV + OAR", 55.0, False),
    ("nominal", "2. Nominale 20 Gy", 55.0, False),
    ("shifted", "3. Iso-Interaktion", 360.0, True),
    ("diff", "4. Rest ausserhalb nominal", 360.0, True),
    ("miss", "5. PTV ohne 20 Gy", 360.0, True),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final 12-met staged scenario videos and midphase frames.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Paper" / "12met_3d_scenarios",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-dpi", type=int, default=160)
    parser.add_argument("--midphase-dpi", type=int, default=260)
    parser.add_argument("--build-seconds", type=float, default=2.0)
    parser.add_argument("--rotation-seconds", type=float, default=12.0)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--only-combined", action="store_true")
    return parser.parse_args()


def _safe_clean_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    expected = (Path(__file__).resolve().parents[1] / "Paper" / "12met_3d_scenarios").resolve()
    if resolved != expected:
        raise RuntimeError(f"Refusing to clean unexpected output directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    for child in resolved.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _base_render_args(args: argparse.Namespace) -> argparse.Namespace:
    return r.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(args.output_dir),
            "--skip-static",
            "--skip-animation",
            "--no-write-gif",
            "--iso-faces",
            "50000",
            "--ptv-faces",
            "2600",
            "--oar-faces",
            "5200",
            "--anim-iso-faces",
            "16000",
            "--anim-ptv-faces",
            "1200",
            "--anim-oar-faces",
            "1600",
            "--anim-context-faces",
            "1800",
            "--frame-dpi",
            str(args.frame_dpi),
            "--fps",
            str(args.fps),
            "--staged-build-seconds",
            str(args.build_seconds),
            "--staged-rotation-seconds",
            str(args.rotation_seconds),
            "--combined-zoom-factor",
            "1.00",
            "--combined-camera-distance",
            "5.4",
            "--diff-dose-margin-gy",
            "0.05",
        ]
    )


def _decimated_mesh_layer(layer: r.MeshLayer, max_faces: int) -> r.MeshLayer:
    if layer.faces is None or len(layer.faces) <= max_faces or max_faces <= 0:
        return layer
    indices = np.linspace(0, len(layer.faces) - 1, max_faces, dtype=int)
    facecolors = layer.facecolors[indices] if layer.facecolors is not None else None
    return r.MeshLayer(
        layer.name,
        layer.verts,
        layer.faces[indices],
        layer.color,
        layer.alpha,
        facecolors=facecolors,
    )


def _decimated_stage_layers(stage_layers, render_args):
    decimated = {}
    for scenario_name, scenario_stages in stage_layers.items():
        decimated[scenario_name] = {}
        for stage_key, layers in scenario_stages.items():
            decimated[scenario_name][stage_key] = [
                _decimated_mesh_layer(layer, r._animation_face_limit(layer, render_args))  # noqa: SLF001
                for layer in layers
            ]
    return decimated


def _midphase_azimuths() -> list[tuple[str, str, float, bool]]:
    azim = -55.0
    frames = []
    for key, title, azim_delta, show_vectors in STAGES:
        frames.append((key, title, azim + 0.5 * azim_delta, show_vectors))
        azim += azim_delta
    return frames


def _save_contact_sheet(paths: list[Path], output_path: Path, *, cols: int = 2) -> None:
    images = [plt.imread(path)[..., :3] for path in paths]
    height, width = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    canvas = np.zeros((height * rows, width * cols, 3), dtype=np.float32)
    canvas[:] = np.array([5, 10, 18], dtype=np.float32) / 255.0
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width, :] = image
    plt.imsave(output_path, canvas)


def _write_individual_midphase_images(
    output_dir: Path,
    *,
    scenarios,
    stage_layers,
    isocenter,
    targets,
    axis_limits,
    label_points,
    rx_gy: float,
    midphase_dpi: int,
) -> list[Path]:
    written: list[Path] = []
    midphase_root = output_dir / "midphase" / "individual"
    midphase_root.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        scenario_dir = midphase_root / r._sanitize(scenario.name)  # noqa: SLF001
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_paths: list[Path] = []
        for index, (key, title, azim, show_vectors) in enumerate(_midphase_azimuths(), start=1):
            frame = r.render_frame(
                scenario=scenario,
                layers=stage_layers[scenario.name][key],
                isocenter=isocenter,
                targets=targets,
                axis_limits=axis_limits,
                label_points=label_points,
                rx_gy=rx_gy,
                azim=azim,
                frame_dpi=midphase_dpi,
                stage_title=title,
                show_vectors=show_vectors,
            )
            path = scenario_dir / f"{index:02d}_{key}.png"
            plt.imsave(path, frame)
            scenario_paths.append(path)
        contact_path = scenario_dir / "contact.png"
        _save_contact_sheet(scenario_paths, contact_path, cols=2)
        written.extend([*scenario_paths, contact_path])
    return written


def _write_combined_midphase_images(
    output_dir: Path,
    *,
    scenarios,
    stage_layers,
    isocenter,
    targets,
    axis_limits,
    label_points,
    rx_gy: float,
    midphase_dpi: int,
    zoom_factor: float,
    camera_distance: float,
) -> list[Path]:
    written: list[Path] = []
    midphase_dir = output_dir / "midphase" / "all4"
    midphase_dir.mkdir(parents=True, exist_ok=True)
    zoomed_axis_limits = r._scale_axis_limits(axis_limits, zoom_factor)  # noqa: SLF001
    stage_paths: list[Path] = []
    for index, (key, title, azim, show_vectors) in enumerate(_midphase_azimuths(), start=1):
        frame = r.render_combined_frame(
            scenarios=scenarios,
            stage_layers=stage_layers,
            stage_key=key,
            isocenter=isocenter,
            targets=targets,
            axis_limits=zoomed_axis_limits,
            label_points=label_points,
            rx_gy=rx_gy,
            azim=azim,
            frame_dpi=midphase_dpi,
            stage_title=title,
            show_vectors=show_vectors,
            camera_distance=camera_distance,
        )
        path = midphase_dir / f"{index:02d}_{key}.png"
        plt.imsave(path, frame)
        stage_paths.append(path)
    contact_path = midphase_dir / "contact.png"
    _save_contact_sheet(stage_paths, contact_path, cols=2)
    written.extend([*stage_paths, contact_path])
    return written


def main() -> None:
    args = _parse_args()
    if not args.no_clean and not args.only_combined:
        _safe_clean_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    render_args = _base_render_args(args)
    (
        scenarios,
        _static_layers,
        _anim_layers,
        stage_layers,
        isocenter,
        targets,
        axis_limits,
        label_points,
    ) = r.build_scene(render_args)
    video_stage_layers = _decimated_stage_layers(stage_layers, render_args)

    written: list[Path] = []
    individual_video_dir = args.output_dir / "videos" / "individual"
    combined_video_dir = args.output_dir / "videos" / "all4"
    individual_video_dir.mkdir(parents=True, exist_ok=True)
    combined_video_dir.mkdir(parents=True, exist_ok=True)

    if not args.only_combined:
        for scenario in scenarios:
            _gif_path, mp4_path = r.save_staged_animation(
                individual_video_dir,
                scenario=scenario,
                stage_layers=video_stage_layers[scenario.name],
                isocenter=isocenter,
                targets=targets,
                axis_limits=axis_limits,
                label_points=label_points,
                rx_gy=render_args.rx_gy,
                fps=args.fps,
                build_seconds=args.build_seconds,
                rotation_seconds=args.rotation_seconds,
                frame_dpi=args.frame_dpi,
                write_gif=False,
            )
            if mp4_path is not None:
                written.append(mp4_path)

    _gif_path, mp4_path = r.save_combined_staged_animation(
        combined_video_dir,
        scenarios=scenarios,
        stage_layers=video_stage_layers,
        isocenter=isocenter,
        targets=targets,
        axis_limits=axis_limits,
        label_points=label_points,
        rx_gy=render_args.rx_gy,
        fps=args.fps,
        build_seconds=args.build_seconds,
        rotation_seconds=args.rotation_seconds,
        frame_dpi=args.frame_dpi,
        zoom_factor=render_args.combined_zoom_factor,
        camera_distance=render_args.combined_camera_distance,
        write_gif=False,
    )
    if mp4_path is not None:
        written.append(mp4_path)

    if not args.only_combined:
        written.extend(
            _write_individual_midphase_images(
                args.output_dir,
                scenarios=scenarios,
                stage_layers=stage_layers,
                isocenter=isocenter,
                targets=targets,
                axis_limits=axis_limits,
                label_points=label_points,
                rx_gy=render_args.rx_gy,
                midphase_dpi=args.midphase_dpi,
            )
        )
    written.extend(
        _write_combined_midphase_images(
            args.output_dir,
            scenarios=scenarios,
            stage_layers=stage_layers,
            isocenter=isocenter,
            targets=targets,
            axis_limits=axis_limits,
            label_points=label_points,
            rx_gy=render_args.rx_gy,
            midphase_dpi=args.midphase_dpi,
            zoom_factor=render_args.combined_zoom_factor,
            camera_distance=render_args.combined_camera_distance,
        )
    )

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
