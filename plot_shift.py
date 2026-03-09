"""
SRS Analysis – Shift-Scenario Impact Plots
===========================================
Generates boxplot + swarmplot figures showing the impact of shift/rotation
scenarios on D98, Paddick CI and GI.

One PNG per metric is saved to SHIFT_PLOTS_DIR.  Each figure has two
side-by-side subplots (HA plan | MM plan) or a single subplot if only one
plan type is present.

Entry point:
    generate_shift_plots(metrics_csv_path, output_dir) -> List[str]
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Metrics to plot: (csv_column, display_label, y_axis_label)
PLOT_METRICS: List[Tuple[str, str, str]] = [
    ("D98_Gy",    "D98",         "D98 [Gy]"),
    ("PaddickCI", "Paddick CI",  "Paddick CI"),
    ("GI",        "GI",          "Gradient Index (GI)"),
    ("Coverage_pct", "Coverage", "Coverage [%]"),
]

# Colour palette for scenarios
_NOMINAL_COLOR   = "#dddddd"
_SHIFT_COLORS    = [
    "#61a8e8", "#e87a61", "#61e8a0", "#e8c261",
    "#a061e8", "#e861b0", "#61e8d8", "#e8e261",
]

# Figure style
_FIGSIZE_PER_PLAN = (5.5, 4.5)   # width per subplot
_DPI              = 130


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _order_scenarios(names: List[str]) -> List[str]:
    """Nominal first, then alphabetically sorted shifts."""
    nominal = [n for n in names if str(n).lower() == "nominal"]
    rest    = sorted(n for n in names if str(n).lower() != "nominal")
    return nominal + rest


def _short_label(name: str) -> str:
    """Shorten scenario label for x-axis tick."""
    return str(name).replace("nominal", "nom").replace("_", "\n")


def _jitter(n: int, width: float = 0.18, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width / 2, width / 2, n)


# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------

def _plot_metric(
    ax: plt.Axes,
    data_by_scenario: dict,            # {scenario_name: [float values]}
    ordered_scenarios: List[str],
    metric_col: str,
    y_label: str,
    plan_type: str,
    color_map: dict,
) -> None:
    """Draw boxplot + swarmplot for one metric / one plan type."""
    positions = list(range(len(ordered_scenarios)))

    for xi, sc in enumerate(ordered_scenarios):
        vals = [v for v in data_by_scenario.get(sc, []) if v is not None]
        if not vals:
            continue

        color = color_map.get(sc, _SHIFT_COLORS[xi % len(_SHIFT_COLORS)])

        # Boxplot (filled, subtle)
        bp = ax.boxplot(
            vals,
            positions=[xi],
            widths=0.45,
            patch_artist=True,
            notch=False,
            showfliers=False,
            medianprops=dict(color="#ffffff", linewidth=2.0),
            boxprops=dict(facecolor=color + "55", edgecolor=color, linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.0, linestyle="--"),
            capprops=dict(color=color, linewidth=1.2),
        )

        # Swarmplot (jittered strip)
        jx = _jitter(len(vals), width=0.30)
        ax.scatter(
            xi + jx, vals,
            color=color,
            s=22,
            alpha=0.75,
            edgecolors="none",
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([_short_label(s) for s in ordered_scenarios],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(plan_type, fontsize=10, fontweight="bold", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)


# ---------------------------------------------------------------------------
# Per-metric figure
# ---------------------------------------------------------------------------

def _generate_metric_figure(
    metrics_df: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    y_label: str,
    output_dir: str,
) -> Optional[str]:
    """Generate and save one figure for *metric_col*."""
    if metric_col not in metrics_df.columns:
        logger.warning("Shift plots: column '%s' not in metrics – skipped.", metric_col)
        return None

    nom_df = metrics_df[metrics_df["Scenario"].astype(str).str.lower() == "nominal"] \
             if "Scenario" in metrics_df.columns else metrics_df
    plan_types = sorted(metrics_df["PlanType"].dropna().unique().tolist())
    all_scenarios = _order_scenarios(
        metrics_df["Scenario"].dropna().unique().tolist()
        if "Scenario" in metrics_df.columns else ["nominal"]
    )

    if not all_scenarios:
        return None

    # Color assignment: nominal = light gray, others get palette colours
    color_map = {}
    sc_idx = 0
    for sc in all_scenarios:
        if sc.lower() == "nominal":
            color_map[sc] = "#bbbbbb"
        else:
            color_map[sc] = _SHIFT_COLORS[sc_idx % len(_SHIFT_COLORS)]
            sc_idx += 1

    n_plans = len(plan_types)
    fig_w = max(_FIGSIZE_PER_PLAN[0] * n_plans, 5.5)
    fig, axes = plt.subplots(
        1, max(n_plans, 1),
        figsize=(fig_w, _FIGSIZE_PER_PLAN[1]),
        squeeze=False,
    )
    fig.patch.set_facecolor("#f8f9fb")

    for col_i, pt in enumerate(plan_types):
        ax = axes[0][col_i]
        ax.set_facecolor("#f8f9fb")

        pt_df = metrics_df[metrics_df["PlanType"] == pt]

        data_by_sc: dict = {}
        for sc in all_scenarios:
            sc_vals = pt_df[pt_df["Scenario"].astype(str) == sc][metric_col] \
                      if "Scenario" in pt_df.columns else pt_df[metric_col]
            data_by_sc[sc] = [_safe_float(v) for v in sc_vals.tolist()
                               if _safe_float(v) is not None]

        _plot_metric(ax, data_by_sc, all_scenarios, metric_col, y_label, pt, color_map)

    # Shared legend for scenario colours
    handles = [
        mpatches.Patch(facecolor=color_map[sc], edgecolor="#555",
                       label=_short_label(sc).replace("\n", " "))
        for sc in all_scenarios
    ]
    fig.legend(
        handles=handles,
        title="Scenario",
        loc="lower center",
        ncol=min(len(all_scenarios), 6),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle(
        f"Shift-Scenario Impact — {metric_name}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    safe_name = metric_col.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(output_dir, f"shift_impact_{safe_name}.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Shift plot saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_shift_plots(
    metrics_csv_path: str,
    output_dir: str,
    metrics: Optional[List[Tuple[str, str, str]]] = None,
) -> List[str]:
    """
    Read *metrics_csv_path* and generate boxplot+swarmplot figures for
    the specified metrics (default: D98, PaddickCI, GI, Coverage).

    Returns list of generated PNG file paths.
    Skips gracefully if the CSV does not exist or is empty.
    """
    if not os.path.isfile(metrics_csv_path):
        logger.warning("Shift plots: metrics CSV not found at %s – skipped.", metrics_csv_path)
        return []

    try:
        df = pd.read_csv(metrics_csv_path)
    except Exception as exc:
        logger.error("Shift plots: failed to read CSV: %s", exc)
        return []

    if df.empty:
        logger.warning("Shift plots: metrics CSV is empty – skipped.")
        return []

    if "PlanType" not in df.columns or "Scenario" not in df.columns:
        logger.warning("Shift plots: required columns missing in CSV – skipped.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    plot_specs = metrics if metrics is not None else PLOT_METRICS

    paths: List[str] = []
    for metric_col, metric_name, y_label in plot_specs:
        path = _generate_metric_figure(df, metric_col, metric_name, y_label, output_dir)
        if path:
            paths.append(path)

    logger.info("Shift plots: %d figure(s) generated in %s", len(paths), output_dir)
    return paths
