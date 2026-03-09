"""
SRS Analysis – DVH Plot Generation
===================================
Generates cumulative dose-volume histogram (DVH) plots for each plan,
showing all active PTVs on a single figure.

Output: one PNG per patient/plan in DVH_PLOTS_DIR.
"""

import logging
import os
from typing import List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dvh import compute_dvh

logger = logging.getLogger(__name__)

# Plot styling
_FIGSIZE = (10, 6)
_DPI = 150
_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#c0392b", "#16a085",
]


def generate_dvh_plot(
    patient_folder: str,
    plan_type: str,
    structures: list,
    dose,
    active_rows: list,
    output_dir: str,
    scenario: str = "nominal",
) -> Optional[str]:
    """
    Generate a DVH plot for all active structures in *active_rows*.
    
    Args:
        patient_folder: patient identifier
        plan_type: HA or MM
        structures: list of Structure objects
        dose: DoseData object (or ShiftedDose)
        active_rows: list of dicts from mapping table (active PTVs)
        output_dir: base output directory
        scenario: scenario label (e.g. "nominal" or "1/0/0//0/0/0")
    
    Returns:
        Path to saved PNG, or None if no structures to plot.
    """
    if not active_rows:
        return None

    struct_map = {s.name: s for s in structures}
    
    # Collect DVH data for each structure
    dvh_data: List[Dict] = []
    for row in active_rows:
        orig_name = str(row.get("StructureName_Original", "")).strip()
        new_name = str(row.get("NewStructureName", orig_name)).strip()
        struct = struct_map.get(orig_name)
        if struct is None:
            continue
        
        # Compute DVH
        try:
            dose_bins, volume_pct = compute_dvh(struct, dose)
            if dose_bins is None or volume_pct is None:
                continue
            dvh_data.append({
                "name": new_name,
                "dose_bins": dose_bins,
                "volume_pct": volume_pct,
            })
        except Exception as exc:
            logger.warning("DVH computation failed for %s/%s: %s", patient_folder, orig_name, exc)
            continue
    
    if not dvh_data:
        logger.info("No DVH data for %s/%s – skipping plot.", patient_folder, plan_type)
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    fig.patch.set_facecolor("#f8f9fb")
    ax.set_facecolor("#ffffff")

    # Plot each DVH curve
    for i, item in enumerate(dvh_data):
        color = _COLORS[i % len(_COLORS)]
        ax.plot(
            item["dose_bins"],
            item["volume_pct"],
            label=item["name"],
            color=color,
            linewidth=2.0,
            alpha=0.9,
        )

    # Styling
    ax.set_xlabel("Dose [Gy]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Volume [%]", fontsize=11, fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # Title
    safe_sc = scenario.replace("/", "-").replace(" ", "_")
    title = f"DVH – {patient_folder} / {plan_type}"
    if scenario != "nominal":
        title += f" / {safe_sc}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Legend
    ax.legend(
        loc="best",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    plt.tight_layout()

    # Save
    safe_patient = patient_folder.replace(" ", "_").replace("/", "_")
    safe_scenario = scenario.replace("/", "-").replace(" ", "_")
    filename = f"dvh_{safe_patient}_{plan_type}_{safe_scenario}.png"
    out_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    
    logger.info("DVH plot saved: %s", out_path)
    return out_path
