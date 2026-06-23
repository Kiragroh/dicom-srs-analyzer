"""
Rigid setup-error impact analysis for SRS plans.

This script is intentionally narrow: it evaluates two summed rotation
scenarios around the plan isocenter, one pure measured-translation
scenario, and one combined small 6D error, then exports target metrics plus
local V12 metrics.

Default scenarios:
  - nominal
  - total 1 deg rotation split equally over Rx/Ry/Rz
  - total 2 deg rotation split equally over Rx/Ry/Rz
  - measured translation: vertical 0.08 cm, longitudinal 0.02 cm,
    lateral 0.02 cm
  - total 0.5 mm translation plus total 0.5 deg rotation, split equally
    over X/Y/Z

Translation convention:
  The default HFS cranial mapping is clinical lateral -> DICOM X,
  vertical -> DICOM Y, longitudinal -> DICOM Z. Values are magnitudes unless
  signed values are passed on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pydicom

from brain_normal_export import (
    build_target_maps,
    clean_text,
    plan_target_prescription_gy,
    referenced_plan_uid,
    referenced_structure_set_uid,
)
from brain_normal_metrics import (
    active_numbers_from_plan_label,
    local_region_mask,
    mask_volume_cc,
    rounded_d99_prescription_gy,
    select_brain_structure,
    structure_mask_on_dose_grid,
    voxel_volume_cc,
)
from dicom_io import DoseData, load_dose, load_plan_meta, load_structures
from metrics import calculate_all_metrics
from shift_scenarios import ShiftScenario, _rotation_matrix, build_shifted_dose


DEFAULT_ROTATION_TOTALS_DEG = (1.0, 2.0)
DEFAULT_LOCAL_MARGIN_MM = 15.0
DEFAULT_V12_THRESHOLD_GY = 12.0
PRESENTATION_DPI = 300


@dataclass(frozen=True)
class DicomPlanSet:
    case_id: str
    rtstruct_path: Path
    rtplan_path: Path
    rtdose_path: Path


def case_hash(*parts: object) -> str:
    text = "|".join(clean_text(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def discover_plan_sets(root: Path) -> list[DicomPlanSet]:
    """Discover RTSTRUCT/RTPLAN/RTDOSE sets by DICOM tags, not file suffixes."""
    records = []
    root = root.resolve()
    script_dir = Path(__file__).resolve().parent

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            path.relative_to(script_dir)
            continue
        except ValueError:
            pass

        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue

        modality = clean_text(getattr(ds, "Modality", ""))
        if modality not in {"RTSTRUCT", "RTPLAN", "RTDOSE"}:
            continue

        try:
            group = path.relative_to(root).parts[0]
        except ValueError:
            group = path.parent.name

        records.append(
            {
                "group": group,
                "path": path,
                "modality": modality,
                "sop_uid": clean_text(getattr(ds, "SOPInstanceUID", "")),
                "ref_plan_uid": referenced_plan_uid(ds),
                "ref_rs_uid": referenced_structure_set_uid(ds),
            }
        )

    if not records:
        return []

    plan_sets: list[DicomPlanSet] = []
    for _group, group in pd.DataFrame(records).groupby("group", dropna=False):
        rs_rows = group[group["modality"].eq("RTSTRUCT")]
        rp_rows = group[group["modality"].eq("RTPLAN")]
        rd_rows = group[group["modality"].eq("RTDOSE")]
        if rs_rows.empty or rp_rows.empty or rd_rows.empty:
            continue

        for _, rd_row in rd_rows.iterrows():
            matching_rp = rp_rows
            if rd_row["ref_plan_uid"]:
                by_uid = rp_rows[rp_rows["sop_uid"].eq(rd_row["ref_plan_uid"])]
                if not by_uid.empty:
                    matching_rp = by_uid
            rp_row = matching_rp.iloc[0]

            matching_rs = rs_rows
            if rp_row["ref_rs_uid"]:
                by_uid = rs_rows[rs_rows["sop_uid"].eq(rp_row["ref_rs_uid"])]
                if not by_uid.empty:
                    matching_rs = by_uid

            plan_sets.append(
                DicomPlanSet(
                    case_id=f"case_{len(plan_sets) + 1:03d}",
                    rtstruct_path=Path(matching_rs.iloc[0]["path"]),
                    rtplan_path=Path(rp_row["path"]),
                    rtdose_path=Path(rd_row["path"]),
                )
            )

    return plan_sets


def rotation_scenario(total_deg: float) -> ShiftScenario:
    per_axis = float(total_deg) / 3.0
    label = f"rot_total_{str(total_deg).replace('.', 'p')}deg_xyz_equal"
    return ShiftScenario(
        name=label,
        rx_deg=per_axis,
        ry_deg=per_axis,
        rz_deg=per_axis,
    )


def measured_translation_scenario(
    *,
    vertical_cm: float,
    longitudinal_cm: float,
    lateral_cm: float,
) -> ShiftScenario:
    vertical_mm = float(vertical_cm) * 10.0
    longitudinal_mm = float(longitudinal_cm) * 10.0
    lateral_mm = float(lateral_cm) * 10.0
    return ShiftScenario(
        name="measured_translation_only",
        dx_mm=lateral_mm,
        dy_mm=vertical_mm,
        dz_mm=longitudinal_mm,
    )


def sixd_equal_scenario(total_mm: float = 0.5, total_deg: float = 0.5) -> ShiftScenario:
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


def scenario_rows(scenarios: Iterable[ShiftScenario]) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        rows.append(
            {
                "Scenario": sc.name,
                "dx_mm": sc.dx_mm,
                "dy_mm": sc.dy_mm,
                "dz_mm": sc.dz_mm,
                "rx_deg": sc.rx_deg,
                "ry_deg": sc.ry_deg,
                "rz_deg": sc.rz_deg,
                "TotalAbsRotationDeg": abs(sc.rx_deg) + abs(sc.ry_deg) + abs(sc.rz_deg),
                "TranslationMagnitudeMm": math.sqrt(sc.dx_mm**2 + sc.dy_mm**2 + sc.dz_mm**2),
                "Convention": (
                    "Rotations are extrinsic Rx->Ry->Rz around plan isocenter. "
                    "Default translation mapping: lateral->DICOM X, vertical->DICOM Y, "
                    "longitudinal->DICOM Z."
                ),
            }
        )
    return pd.DataFrame(rows)


def centroid(structure) -> np.ndarray:
    if not structure.contours:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.vstack(structure.contours).mean(axis=0)


def forward_transform_point(point: np.ndarray, scenario: ShiftScenario, isocenter: np.ndarray) -> np.ndarray:
    t = np.array([scenario.dx_mm, scenario.dy_mm, scenario.dz_mm], dtype=float)
    rotation = _rotation_matrix(scenario.rx_deg, scenario.ry_deg, scenario.rz_deg)
    return rotation @ (point - isocenter) + isocenter + t


def sample_dose_at_points(points_xyz: np.ndarray, dose) -> np.ndarray:
    """
    Vectorized trilinear dose lookup. Out-of-grid points are returned as NaN.
    Supports ShiftedDose rotation through its transform parameters.
    """
    points = np.asarray(points_xyz, dtype=float)
    if points.size == 0:
        return np.array([], dtype=float)

    if getattr(dose, "_has_rotation", False):
        centered = points - dose._isocenter - dose._t
        points = centered @ dose._R_inv.T + dose._isocenter

    ox, oy, oz = np.asarray(dose.origin, dtype=float)
    dx, dy, dz = np.asarray(dose.spacing, dtype=float)
    nz, ny, nx = dose.dose_grid.shape

    ix = (points[:, 0] - ox) / dx
    iy = (points[:, 1] - oy) / dy
    iz = (points[:, 2] - oz) / dz
    valid = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < nx - 1) & (iy < ny - 1) & (iz < nz - 1)

    out = np.full(points.shape[0], np.nan, dtype=float)
    if not np.any(valid):
        return out

    ixv = ix[valid]
    iyv = iy[valid]
    izv = iz[valid]
    x0 = ixv.astype(int)
    y0 = iyv.astype(int)
    z0 = izv.astype(int)
    xd = ixv - x0
    yd = iyv - y0
    zd = izv - z0

    grid = dose.dose_grid
    c000 = grid[z0, y0, x0]
    c100 = grid[z0, y0, x0 + 1]
    c010 = grid[z0, y0 + 1, x0]
    c110 = grid[z0, y0 + 1, x0 + 1]
    c001 = grid[z0 + 1, y0, x0]
    c101 = grid[z0 + 1, y0, x0 + 1]
    c011 = grid[z0 + 1, y0 + 1, x0]
    c111 = grid[z0 + 1, y0 + 1, x0 + 1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    out[valid] = c0 * (1 - zd) + c1 * zd
    return out


def points_for_mask(mask: np.ndarray, reference_dose: DoseData) -> np.ndarray:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return np.empty((0, 3), dtype=float)
    origin = np.asarray(reference_dose.origin, dtype=float)
    spacing = np.asarray(reference_dose.spacing, dtype=float)
    points = np.empty((indices.shape[0], 3), dtype=float)
    points[:, 0] = origin[0] + indices[:, 2] * spacing[0]
    points[:, 1] = origin[1] + indices[:, 1] * spacing[1]
    points[:, 2] = origin[2] + indices[:, 0] * spacing[2]
    return points


def volume_at_threshold_for_mask(
    mask: np.ndarray,
    *,
    eval_dose,
    reference_dose: DoseData,
    threshold_gy: float,
) -> float:
    values = sample_dose_at_points(points_for_mask(mask, reference_dose), eval_dose)
    return float(np.count_nonzero(values >= threshold_gy) * voxel_volume_cc(reference_dose))


def metrics_to_row(metrics: dict) -> dict:
    return {
        "TV_cc": metrics.get("TV", np.nan),
        "Coverage_pct": 100.0 * metrics.get("coverage", np.nan),
        "PaddickCI": metrics.get("paddickCI", np.nan),
        "RTOG_CI": metrics.get("rtogCI", np.nan),
        "HI": metrics.get("HI", np.nan),
        "GI": metrics.get("GI", np.nan),
        "Dmax_Gy": metrics.get("Dmax", np.nan),
        "PIV_cc": metrics.get("PIV", np.nan),
        "V12Gy_metric_cc": metrics.get("V12Gy", np.nan),
        "D2_Gy": metrics.get("D2", np.nan),
        "D50_Gy": metrics.get("D50", np.nan),
        "D98_Gy": metrics.get("D98", np.nan),
        "EffDiameter_mm": metrics.get("effectiveDiameter", np.nan),
        "FinalGrid_mm": metrics.get("finalGrid", np.nan),
        "CI_uncertain": bool(metrics.get("ci_uncertain", False)),
        "GI_uncertain": bool(metrics.get("gi_uncertain", False)),
        "Bridging_suspected": bool(metrics.get("bridging_suspected", False)),
    }


def delta_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "PTV",
        "GTV",
        "Rx_Gy",
        "RxSource",
        "PTVVolumeCc_mask",
        "GTVVolumeCc_mask",
        "LocalRegionVolumeCc",
        "DistToIso_mm",
    ]
    delta_cols = [
        "Coverage_pct",
        "PaddickCI",
        "RTOG_CI",
        "HI",
        "GI",
        "Dmax_Gy",
        "PIV_cc",
        "V12Gy_metric_cc",
        "D2_Gy",
        "D50_Gy",
        "D98_Gy",
        "LocalV12_BrainMinusGTV_cc",
        "LocalV12_BrainMinusPTV_cc",
        "LocalV12_PTV_cc",
        "CentroidDisplacementMm",
    ]
    rows = []
    key_cols = ["CaseId", "TargetNumber"]
    nominal = metrics_df[metrics_df["Scenario"].eq("nominal")]
    nominal_map = {tuple(row[k] for k in key_cols): row for _, row in nominal.iterrows()}
    for _, row in metrics_df.iterrows():
        key = tuple(row[k] for k in key_cols)
        base = nominal_map.get(key)
        if base is None:
            continue
        out = {k: row[k] for k in key_cols}
        out["Scenario"] = row["Scenario"]
        for col in meta_cols:
            out[col] = row.get(col, np.nan)
        for col in delta_cols:
            out[col] = row.get(col, np.nan)
            out[f"Delta_{col}"] = row.get(col, np.nan) - base.get(col, np.nan)
        rows.append(out)
    return pd.DataFrame(rows)


def summary_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "Delta_Coverage_pct",
        "Delta_D98_Gy",
        "Delta_D50_Gy",
        "Delta_Dmax_Gy",
        "Delta_PaddickCI",
        "Delta_GI",
        "Delta_LocalV12_BrainMinusGTV_cc",
        "Delta_LocalV12_BrainMinusPTV_cc",
        "Delta_LocalV12_PTV_cc",
        "CentroidDisplacementMm",
    ]
    for scenario, group in delta_df[~delta_df["Scenario"].eq("nominal")].groupby("Scenario"):
        row = {"Scenario": scenario, "Targets": int(group["TargetNumber"].nunique())}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_explorer_html(metrics_df: pd.DataFrame, delta_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    import plotly.express as px
    import plotly.io as pio

    plot_df = delta_df[~delta_df["Scenario"].eq("nominal")].copy()
    scenario_color_map = {
        "rot_total_1p0deg_xyz_equal": "#2f80ed",
        "rot_total_2p0deg_xyz_equal": "#d64550",
        "measured_translation_only": "#1f9d55",
        "sixd_total_0p5mm_0p5deg_xyz_equal": "#8e44ad",
    }
    hover_cols = {
        "PTV": True,
        "PTVVolumeCc_mask": ":.3f",
        "DistToIso_mm": ":.1f",
        "Delta_D98_Gy": ":.2f",
        "Delta_PaddickCI": ":.3f",
        "Delta_Coverage_pct": ":.2f",
        "Delta_LocalV12_BrainMinusGTV_cc": ":.3f",
    }
    fig_d98 = px.box(
        plot_df,
        x="Scenario",
        y="Delta_D98_Gy",
        points="all",
        color="Scenario",
        color_discrete_map=scenario_color_map,
        hover_data=hover_cols,
        title="D98 change vs nominal",
        labels={"Delta_D98_Gy": "Delta D98 (Gy)"},
    )
    fig_ci = px.box(
        plot_df,
        x="Scenario",
        y="Delta_PaddickCI",
        points="all",
        color="Scenario",
        color_discrete_map=scenario_color_map,
        hover_data=hover_cols,
        title="Paddick CI change vs nominal",
        labels={"Delta_PaddickCI": "Delta Paddick CI"},
    )
    distance_df = plot_df[
        [
            "CaseId",
            "TargetNumber",
            "PTV",
            "PTVVolumeCc_mask",
            "DistToIso_mm",
            "Scenario",
            "Delta_D98_Gy",
            "Delta_PaddickCI",
        ]
    ].melt(
        id_vars=["CaseId", "TargetNumber", "PTV", "PTVVolumeCc_mask", "DistToIso_mm", "Scenario"],
        value_vars=["Delta_D98_Gy", "Delta_PaddickCI"],
        var_name="Metric",
        value_name="DeltaValue",
    )
    distance_df["MetricLabel"] = distance_df["Metric"].map(
        {"Delta_D98_Gy": "D98 (Gy)", "Delta_PaddickCI": "Paddick CI"}
    )
    fig_distance_impact = px.scatter(
        distance_df,
        x="DistToIso_mm",
        y="DeltaValue",
        color="Scenario",
        color_discrete_map=scenario_color_map,
        facet_col="MetricLabel",
        hover_data={
            "PTV": True,
            "PTVVolumeCc_mask": ":.3f",
            "DistToIso_mm": ":.1f",
            "DeltaValue": ":.3f",
            "Metric": False,
            "MetricLabel": False,
        },
        title="Impact vs distance to isocenter: D98 and Paddick CI with trendlines",
        labels={
            "DistToIso_mm": "PTV centroid distance to iso (mm)",
            "DeltaValue": "Delta vs nominal",
            "MetricLabel": "",
        },
    )
    fig_distance_impact.update_yaxes(matches=None)
    for metric_label, subplot_col in {"D98 (Gy)": 1, "Paddick CI": 2}.items():
        metric_data = distance_df[distance_df["MetricLabel"].eq(metric_label)]
        for scenario, scenario_data in metric_data.groupby("Scenario", sort=False):
            fit = scenario_data[["DistToIso_mm", "DeltaValue"]].dropna().sort_values("DistToIso_mm")
            if len(fit) < 3 or fit["DistToIso_mm"].nunique() < 2:
                continue
            coeffs = np.polyfit(fit["DistToIso_mm"], fit["DeltaValue"], deg=1)
            x_fit = np.linspace(float(fit["DistToIso_mm"].min()), float(fit["DistToIso_mm"].max()), 60)
            y_fit = coeffs[0] * x_fit + coeffs[1]
            fig_distance_impact.add_scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                line={"color": scenario_color_map.get(str(scenario), "#555"), "dash": "dash", "width": 2},
                legendgroup=str(scenario),
                showlegend=False,
                hovertemplate=(
                    f"{scenario}<br>linear fit<br>"
                    "Distance=%{x:.1f} mm<br>Delta=%{y:.3f}<extra></extra>"
                ),
                row=1,
                col=subplot_col,
            )
    volume_df = plot_df[
        [
            "CaseId",
            "TargetNumber",
            "PTV",
            "PTVVolumeCc_mask",
            "DistToIso_mm",
            "Scenario",
            "Delta_D98_Gy",
            "Delta_PaddickCI",
        ]
    ].melt(
        id_vars=["CaseId", "TargetNumber", "PTV", "PTVVolumeCc_mask", "DistToIso_mm", "Scenario"],
        value_vars=["Delta_D98_Gy", "Delta_PaddickCI"],
        var_name="Metric",
        value_name="DeltaValue",
    )
    volume_df["MetricLabel"] = volume_df["Metric"].map(
        {"Delta_D98_Gy": "D98 (Gy)", "Delta_PaddickCI": "Paddick CI"}
    )
    fig_volume_impact = px.scatter(
        volume_df,
        x="PTVVolumeCc_mask",
        y="DeltaValue",
        color="Scenario",
        color_discrete_map=scenario_color_map,
        facet_col="MetricLabel",
        hover_data={
            "PTV": True,
            "PTVVolumeCc_mask": ":.3f",
            "DistToIso_mm": ":.1f",
            "DeltaValue": ":.3f",
            "Metric": False,
            "MetricLabel": False,
        },
        title="Impact vs PTV volume: D98 and Paddick CI",
        labels={
            "PTVVolumeCc_mask": "PTV volume (cc)",
            "DeltaValue": "Delta vs nominal",
            "MetricLabel": "",
        },
    )
    fig_volume_impact.update_yaxes(matches=None)
    fig_v12 = px.box(
        plot_df,
        x="Scenario",
        y="Delta_LocalV12_BrainMinusGTV_cc",
        points="all",
        color="Scenario",
        color_discrete_map=scenario_color_map,
        hover_data=hover_cols,
        title="Local V12 Brain-GTV change vs nominal",
        labels={"Delta_LocalV12_BrainMinusGTV_cc": "Delta local V12 Brain-GTV (cc)"},
    )
    fig_disp = px.scatter(
        plot_df,
        x="DistToIso_mm",
        y="CentroidDisplacementMm",
        color="Scenario",
        color_discrete_map=scenario_color_map,
        hover_data=hover_cols,
        title="Centroid displacement by distance to isocenter",
        labels={"DistToIso_mm": "PTV centroid distance to iso (mm)", "CentroidDisplacementMm": "Centroid displacement (mm)"},
    )

    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Rigid setup-error impact</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#222}"
        "table{border-collapse:collapse;font-size:12px;margin:12px 0;width:100%}"
        "th,td{border:1px solid #ddd;padding:4px 6px;text-align:right}"
        "th{text-align:left;background:#f2f2f2;position:sticky;top:0}"
        ".note{max-width:980px;line-height:1.45}.plot{margin:24px 0}</style>",
        "</head><body>",
        "<h1>Rigid setup-error impact</h1>",
        "<p class='note'>Rotations are applied around the DICOM plan isocenter using the repository ShiftedDose transform. "
        "Local V12 metrics are sampled on the fixed anatomy grid in a 15 mm local region around each PTV. "
        "Hover over points to see PTV name, PTV volume and distance to isocenter.</p>",
        "<h2>Scenario Summary</h2>",
        summary_df.round(4).to_html(index=False),
        "<div class='plot'>",
        pio.to_html(fig_d98, full_html=False, include_plotlyjs="cdn"),
        "</div><div class='plot'>",
        pio.to_html(fig_ci, full_html=False, include_plotlyjs=False),
        "</div><div class='plot'>",
        pio.to_html(fig_distance_impact, full_html=False, include_plotlyjs=False),
        "</div><div class='plot'>",
        pio.to_html(fig_volume_impact, full_html=False, include_plotlyjs=False),
        "</div><div class='plot'>",
        pio.to_html(fig_v12, full_html=False, include_plotlyjs=False),
        "</div><div class='plot'>",
        pio.to_html(fig_disp, full_html=False, include_plotlyjs=False),
        "</div>",
        "<h2>Per Target Deltas</h2>",
        delta_df.round(4).to_html(index=False),
        "<h2>Per Target Metrics</h2>",
        metrics_df.round(4).to_html(index=False),
        "</body></html>",
    ]
    out_path.write_text("\n".join(html), encoding="utf-8")


def _scenario_display_name(scenario: str) -> str:
    names = {
        "rot_total_1p0deg_xyz_equal": "1 deg rot total",
        "rot_total_2p0deg_xyz_equal": "2 deg rot total",
        "measured_translation_only": "Measured translation",
        "sixd_total_0p5mm_0p5deg_xyz_equal": "0.5 mm + 0.5 deg 6D",
        "nominal": "Nominal",
    }
    return names.get(str(scenario), str(scenario).replace("_", " "))


def write_presentation_plots(metrics_df: pd.DataFrame, delta_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = output_dir / "presentation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_df = delta_df[~delta_df["Scenario"].eq("nominal")].copy()
    plot_df["ScenarioLabel"] = plot_df["Scenario"].map(_scenario_display_name)
    plot_df["TargetLabel"] = plot_df["PTV"].fillna("").astype(str)
    scenario_order = ["1 deg rot total", "2 deg rot total", "Measured translation", "0.5 mm + 0.5 deg 6D"]
    colors = {
        "1 deg rot total": "#2f80ed",
        "2 deg rot total": "#d64550",
        "Measured translation": "#1f9d55",
        "0.5 mm + 0.5 deg 6D": "#8957e5",
    }

    paths: list[Path] = []

    def save_current(name: str) -> None:
        png = plot_dir / f"{name}.png"
        pdf = plot_dir / f"{name}.pdf"
        plt.savefig(png, dpi=PRESENTATION_DPI, bbox_inches="tight", facecolor="white")
        plt.savefig(pdf, bbox_inches="tight", facecolor="white")
        paths.extend([png, pdf])
        plt.close()

    def scatter_by_scenario(ax, x_col: str, y_col: str, xlabel: str, ylabel: str, title: str) -> None:
        for label in scenario_order:
            sub = plot_df[plot_df["ScenarioLabel"].eq(label)].sort_values(x_col)
            ax.scatter(
                sub[x_col],
                sub[y_col],
                label=label,
                color=colors[label],
                s=58,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.6,
            )
            fit = sub[[x_col, y_col]].dropna()
            if len(fit) >= 3 and fit[x_col].nunique() >= 2:
                coeffs = np.polyfit(fit[x_col], fit[y_col], deg=1)
                x_fit = np.linspace(float(fit[x_col].min()), float(fit[x_col].max()), 50)
                y_fit = coeffs[0] * x_fit + coeffs[1]
                ax.plot(x_fit, y_fit, color=colors[label], linewidth=1.3, alpha=0.55, linestyle="--")
        ax.axhline(0, color="#222", linewidth=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(color="#d7d7d7", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    # Figure 1: per-target D98 waterfall.
    targets = sorted(plot_df["TargetNumber"].unique())
    x = np.arange(len(targets))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(scenario_order))
    plt.figure(figsize=(12, 6.2))
    ax = plt.gca()
    for offset, label in zip(offsets, scenario_order):
        sub = plot_df[plot_df["ScenarioLabel"].eq(label)].set_index("TargetNumber").reindex(targets)
        ax.bar(
            x + offset,
            sub["Delta_D98_Gy"],
            width=width,
            label=label,
            color=colors[label],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="#222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PTV{int(t):02d}" for t in targets], rotation=0)
    ax.set_ylabel("Delta D98 vs nominal (Gy)")
    ax.set_title("Target D98 sensitivity to rigid setup error (20 Gy Rx)")
    ax.legend(frameon=False, ncol=4, loc="lower left")
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save_current("delta_d98_by_target")

    # Figure 2: per-target Paddick CI waterfall.
    plt.figure(figsize=(12, 6.2))
    ax = plt.gca()
    for offset, label in zip(offsets, scenario_order):
        sub = plot_df[plot_df["ScenarioLabel"].eq(label)].set_index("TargetNumber").reindex(targets)
        ax.bar(
            x + offset,
            sub["Delta_PaddickCI"],
            width=width,
            label=label,
            color=colors[label],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="#222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PTV{int(t):02d}" for t in targets], rotation=0)
    ax.set_ylabel("Delta Paddick CI vs nominal")
    ax.set_title("Target Paddick CI sensitivity to rigid setup error (20 Gy Rx)")
    ax.legend(frameon=False, ncol=4, loc="best")
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save_current("delta_paddick_ci_by_target")

    # Figure 3: local Brain-GTV V12 changes.
    plt.figure(figsize=(12, 6.2))
    ax = plt.gca()
    for offset, label in zip(offsets, scenario_order):
        sub = plot_df[plot_df["ScenarioLabel"].eq(label)].set_index("TargetNumber").reindex(targets)
        ax.bar(
            x + offset,
            sub["Delta_LocalV12_BrainMinusGTV_cc"],
            width=width,
            label=label,
            color=colors[label],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="#222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PTV{int(t):02d}" for t in targets], rotation=0)
    ax.set_ylabel("Delta local V12 Brain-GTV (cc)")
    ax.set_title("Local normal-brain V12 sensitivity around each PTV")
    ax.legend(frameon=False, ncol=4, loc="best")
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save_current("delta_local_v12_brain_minus_gtv")

    # Figure 4: displacement vs distance to isocenter.
    plt.figure(figsize=(9.5, 6.5))
    ax = plt.gca()
    for label in scenario_order:
        sub = plot_df[plot_df["ScenarioLabel"].eq(label)]
        ax.scatter(
            sub["DistToIso_mm"],
            sub["CentroidDisplacementMm"],
            label=label,
            color=colors[label],
            s=54,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xlabel("PTV centroid distance to isocenter (mm)")
    ax.set_ylabel("Centroid displacement from scenario (mm)")
    ax.set_title("Rotation impact scales with distance from isocenter")
    ax.legend(frameon=False, loc="best")
    ax.grid(color="#d7d7d7", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save_current("displacement_vs_distance_to_iso")

    # Figure 5: D98 and Paddick CI impact vs distance to isocenter.
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    scatter_by_scenario(
        axes[0],
        "DistToIso_mm",
        "Delta_D98_Gy",
        "PTV centroid distance to isocenter (mm)",
        "Delta D98 vs nominal (Gy)",
        "D98 impact vs distance from isocenter",
    )
    scatter_by_scenario(
        axes[1],
        "DistToIso_mm",
        "Delta_PaddickCI",
        "PTV centroid distance to isocenter (mm)",
        "Delta Paddick CI vs nominal",
        "Paddick CI impact vs distance from isocenter",
    )
    axes[0].legend(frameon=False, ncol=4, loc="best")
    axes[0].set_xlabel("")
    axes[1].set_xlabel("PTV centroid distance to isocenter (mm)")
    fig.tight_layout()
    save_current("distance_impact_d98_ci")

    # Figure 6: D98 and Paddick CI impact vs PTV volume.
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    scatter_by_scenario(
        axes[0],
        "PTVVolumeCc_mask",
        "Delta_D98_Gy",
        "PTV volume (cc)",
        "Delta D98 vs nominal (Gy)",
        "D98 impact vs PTV volume",
    )
    scatter_by_scenario(
        axes[1],
        "PTVVolumeCc_mask",
        "Delta_PaddickCI",
        "PTV volume (cc)",
        "Delta Paddick CI vs nominal",
        "Paddick CI impact vs PTV volume",
    )
    axes[0].legend(frameon=False, ncol=4, loc="best")
    axes[0].set_xlabel("")
    fig.tight_layout()
    save_current("volume_impact_d98_ci")

    # Figure 7: one-slide view of geometry and volume drivers.
    fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5), sharey="row")
    scatter_by_scenario(
        axes[0, 0],
        "DistToIso_mm",
        "Delta_D98_Gy",
        "Distance to isocenter (mm)",
        "Delta D98 (Gy)",
        "D98 vs distance",
    )
    scatter_by_scenario(
        axes[0, 1],
        "PTVVolumeCc_mask",
        "Delta_D98_Gy",
        "PTV volume (cc)",
        "",
        "D98 vs volume",
    )
    scatter_by_scenario(
        axes[1, 0],
        "DistToIso_mm",
        "Delta_PaddickCI",
        "Distance to isocenter (mm)",
        "Delta Paddick CI",
        "Paddick CI vs distance",
    )
    scatter_by_scenario(
        axes[1, 1],
        "PTVVolumeCc_mask",
        "Delta_PaddickCI",
        "PTV volume (cc)",
        "",
        "Paddick CI vs volume",
    )
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.ravel():
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Rigid setup-error impact: distance and target volume", y=0.995, fontsize=16)
    fig.text(0.5, 0.015, "12 single targets, 20 Gy Rx; dashed lines are per-scenario linear fits.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92))
    save_current("impact_distance_volume_slide")

    # Figure 8: compact summary for slides.
    summary = (
        plot_df.groupby("ScenarioLabel", observed=False)
        .agg(
            mean_d98=("Delta_D98_Gy", "mean"),
            min_d98=("Delta_D98_Gy", "min"),
            max_d98=("Delta_D98_Gy", "max"),
            mean_cov=("Delta_Coverage_pct", "mean"),
            mean_v12=("Delta_LocalV12_BrainMinusGTV_cc", "mean"),
        )
        .reindex(scenario_order)
    )
    plt.figure(figsize=(9.5, 5.8))
    ax = plt.gca()
    y = np.arange(len(summary))
    ax.barh(y, summary["mean_d98"], color=[colors[label] for label in summary.index], height=0.56)
    for yi, label in enumerate(summary.index):
        ax.plot([summary.loc[label, "min_d98"], summary.loc[label, "max_d98"]], [yi, yi], color="#222", linewidth=1.2)
        ax.scatter([summary.loc[label, "min_d98"], summary.loc[label, "max_d98"]], [yi, yi], color="#222", s=22)
    ax.axvline(0, color="#222", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.set_xlabel("Delta D98 vs nominal (Gy), mean bar with min-max whisker")
    ax.set_title("Plan-impact summary across 12 metastases")
    ax.grid(axis="x", color="#d7d7d7", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save_current("summary_delta_d98")

    return paths


def run_analysis(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    plan_sets = discover_plan_sets(args.data_root)
    if not plan_sets:
        raise RuntimeError(f"No RTSTRUCT/RTPLAN/RTDOSE plan set found under {args.data_root}")

    scenarios: list[ShiftScenario] = [ShiftScenario(name="nominal")]
    scenarios.extend(rotation_scenario(total) for total in args.rotation_total_deg)
    scenarios.append(
        measured_translation_scenario(
            vertical_cm=args.translation_vertical_cm,
            longitudinal_cm=args.translation_longitudinal_cm,
            lateral_cm=args.translation_lateral_cm,
        )
    )
    scenarios.append(sixd_equal_scenario())

    rows = []
    qa_rows = []
    for plan_set in plan_sets:
        structures = load_structures(str(plan_set.rtstruct_path))
        base_dose = load_dose(str(plan_set.rtdose_path))
        meta = load_plan_meta(str(plan_set.rtplan_path))
        if base_dose is None or meta is None or meta.isocenter is None:
            raise RuntimeError(f"Could not load dose/meta/isocenter for {plan_set.case_id}")

        brain = select_brain_structure(structures, allow_external_fallback=args.allow_external_fallback)
        if brain is None:
            raise RuntimeError(f"No brain contour found for {plan_set.case_id}")

        plan_label = clean_text(meta.plan_label)
        active_numbers = active_numbers_from_plan_label(plan_label)
        plan_rx = plan_target_prescription_gy(plan_set.rtplan_path)
        ptv_by_number, gtv_by_number, inventory = build_target_maps(structures)
        target_numbers = sorted(set(ptv_by_number).intersection(gtv_by_number))
        if active_numbers:
            target_numbers = [n for n in target_numbers if n in active_numbers]

        mask_cache = {}

        def mask_for(structure):
            key = structure.name
            if key not in mask_cache:
                mask_cache[key] = structure_mask_on_dose_grid(structure, base_dose)
            return mask_cache[key]

        brain_mask = mask_for(brain.structure)
        forced_rx = float(args.force_rx_gy) if args.force_rx_gy is not None else float("nan")

        qa_rows.append(
            {
                "CaseId": plan_set.case_id,
                "PlanHash": case_hash(plan_set.rtplan_path.name, plan_label),
                "TargetsEvaluated": len(target_numbers),
                "ActiveNumbersFromPlanLabel": ",".join(str(v) for v in sorted(active_numbers)),
                "BrainStructure": brain.structure.name,
                "BrainSource": brain.source,
                "BrainExternalFallback": brain.is_external_fallback,
                "DoseGridShape": "x".join(str(v) for v in base_dose.dose_grid.shape),
                "DoseSpacingMm": ",".join(f"{float(v):g}" for v in base_dose.spacing),
                "LocalMarginMm": args.local_margin_mm,
                "V12ThresholdGy": args.v12_threshold_gy,
                "ForcedRxGy": forced_rx,
                "Note": "Patient identifiers and UIDs are intentionally omitted.",
            }
        )

        for number in target_numbers:
            ptv = ptv_by_number[number]
            gtv = gtv_by_number[number]
            ptv_mask = mask_for(ptv)
            gtv_mask = mask_for(gtv)
            ptv_centroid = centroid(ptv)
            dist_iso = float(np.linalg.norm(ptv_centroid - meta.isocenter))

            if args.force_rx_gy is not None:
                rx = float(args.force_rx_gy)
                rx_source = "forced_cli"
            else:
                rx = rounded_d99_prescription_gy(ptv_mask, base_dose)
                rx_source = "ptv_d99_rounded"
                if not np.isfinite(rx) or rx <= 0:
                    rx = plan_rx
                    rx_source = "rtplan_target_prescription"
            if not np.isfinite(rx) or rx <= 0:
                raise RuntimeError(f"No usable prescription dose for target {number} in {plan_set.case_id}")

            local_region = local_region_mask(ptv, base_dose, args.local_margin_mm) & brain_mask
            local_brain_minus_ptv = local_region & ~ptv_mask
            local_brain_minus_gtv = local_region & ~gtv_mask

            for scenario in scenarios:
                print(f"{plan_set.case_id} target {number:02d} scenario {scenario.name}", flush=True)
                shifted_dose = build_shifted_dose(base_dose, scenario, meta.isocenter)
                target_metrics = calculate_all_metrics(ptv, shifted_dose, rx)
                moved_centroid = forward_transform_point(ptv_centroid, scenario, meta.isocenter)
                displacement = float(np.linalg.norm(moved_centroid - ptv_centroid))

                row = {
                    "CaseId": plan_set.case_id,
                    "PlanHash": case_hash(plan_set.rtplan_path.name, plan_label),
                    "TargetNumber": number,
                    "PTV": ptv.name,
                    "GTV": gtv.name,
                    "Scenario": scenario.name,
                    "Rx_Gy": rx,
                    "RxSource": rx_source,
                    "DistToIso_mm": dist_iso,
                    "CentroidDisplacementMm": displacement,
                    "PTVVolumeCc_mask": mask_volume_cc(ptv_mask, base_dose),
                    "GTVVolumeCc_mask": mask_volume_cc(gtv_mask, base_dose),
                    "LocalRegionVolumeCc": mask_volume_cc(local_region, base_dose),
                    "LocalV12_BrainMinusGTV_cc": volume_at_threshold_for_mask(
                        local_brain_minus_gtv,
                        eval_dose=shifted_dose,
                        reference_dose=base_dose,
                        threshold_gy=args.v12_threshold_gy,
                    ),
                    "LocalV12_BrainMinusPTV_cc": volume_at_threshold_for_mask(
                        local_brain_minus_ptv,
                        eval_dose=shifted_dose,
                        reference_dose=base_dose,
                        threshold_gy=args.v12_threshold_gy,
                    ),
                    "LocalV12_PTV_cc": volume_at_threshold_for_mask(
                        ptv_mask,
                        eval_dose=shifted_dose,
                        reference_dose=base_dose,
                        threshold_gy=args.v12_threshold_gy,
                    ),
                }
                row.update(metrics_to_row(target_metrics))
                rows.append(row)

    metrics_df = pd.DataFrame(rows)
    deltas = delta_table(metrics_df)
    scenarios_df = scenario_rows(scenarios)
    qa = pd.DataFrame(qa_rows)
    summary = summary_table(deltas)
    return {
        "metrics": metrics_df,
        "delta_vs_nominal": deltas,
        "scenario_definitions": scenarios_df,
        "summary": summary,
        "qa": qa,
    }


def write_outputs(sheets: dict[str, pd.DataFrame], output_dir: Path, *, write_html: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in sheets.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    workbook_path = output_dir / "rigid_error_impact.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            sheet_name = name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    if write_html:
        write_explorer_html(
            sheets["metrics"],
            sheets["delta_vs_nominal"],
            sheets["summary"],
            output_dir / "rigid_error_explorer.html",
        )
        write_presentation_plots(
            sheets["metrics"],
            sheets["delta_vs_nominal"],
            output_dir,
        )


def parse_rotation_totals(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one rotation total is required.")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("Rotation totals must be non-negative.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate rigid setup-error impact for SRS DICOM plans.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output" / "rigid_error_impact")
    parser.add_argument("--rotation-total-deg", type=parse_rotation_totals, default=list(DEFAULT_ROTATION_TOTALS_DEG))
    parser.add_argument("--translation-vertical-cm", type=float, default=0.08)
    parser.add_argument("--translation-longitudinal-cm", type=float, default=0.02)
    parser.add_argument("--translation-lateral-cm", type=float, default=0.02)
    parser.add_argument("--local-margin-mm", type=float, default=DEFAULT_LOCAL_MARGIN_MM)
    parser.add_argument("--v12-threshold-gy", type=float, default=DEFAULT_V12_THRESHOLD_GY)
    parser.add_argument("--force-rx-gy", type=float, default=None, help="Override Rx dose for every target, e.g. 20 for a known single-case plan.")
    parser.add_argument("--allow-external-fallback", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    sheets = run_analysis(args)
    write_outputs(sheets, args.output_dir, write_html=not args.no_html)
    print(f"Wrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
