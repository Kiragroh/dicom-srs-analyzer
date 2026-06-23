"""
Generate the interactive 3D HTML viewer for the rigid setup-error scenarios.

This wrapper keeps the existing plot_3d_html renderer but discovers the local
RTSTRUCT/RTPLAN/RTDOSE set by DICOM tags, so it also works for DICOM files
without a .dcm suffix or the repository's original zz_* folder convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brain_normal_export import build_target_maps, clean_text
from brain_normal_metrics import active_numbers_from_plan_label, rounded_d99_prescription_gy, structure_mask_on_dose_grid
from dicom_io import PlanFiles, load_dose, load_plan_meta, load_structures
from excel_io import MAPPING_COLUMNS
from plot_3d_html import generate_3d_html
from rigid_error_impact import discover_plan_sets, measured_translation_scenario, rotation_scenario, sixd_equal_scenario
from structure_mapping import compute_centroid


def build_mapping_for_plan(plan_files: PlanFiles, force_rx_gy: float | None = None) -> pd.DataFrame:
    structures = load_structures(plan_files.rs_path)
    dose = load_dose(plan_files.rd_path)
    meta = load_plan_meta(plan_files.rp_path)
    if dose is None or meta is None:
        raise RuntimeError("Could not load dose or plan metadata for 3D mapping.")

    ptv_by_number, gtv_by_number, _inventory = build_target_maps(structures)
    active_numbers = active_numbers_from_plan_label(clean_text(meta.plan_label))
    target_numbers = sorted(set(ptv_by_number).intersection(gtv_by_number))
    if active_numbers:
        target_numbers = [number for number in target_numbers if number in active_numbers]

    rows = []
    for number in target_numbers:
        structure = ptv_by_number[number]
        mask = structure_mask_on_dose_grid(structure, dose)
        rx = float(force_rx_gy) if force_rx_gy is not None else rounded_d99_prescription_gy(mask, dose)
        centroid = compute_centroid(structure)
        dist_iso = float(np.linalg.norm(centroid - meta.isocenter)) if centroid is not None else float("nan")

        row = {column: "" for column in MAPPING_COLUMNS}
        row.update(
            {
                "PatientFolder": plan_files.patient_folder,
                "PatientID": "case_001",
                "PlanType": plan_files.plan_type,
                "PlanLabel": clean_text(meta.plan_label),
                "StructureName_Original": structure.name,
                "Volume_cc": f"{float(np.count_nonzero(mask) * np.prod(dose.spacing) / 1000.0):.3f}",
                "IsPTV_candidate_auto": "True",
                "ExcludeFromAnalysis": "",
                "ExcludeReason": "",
                "Prescription_Gy_detected": f"{rx:.0f}" if np.isfinite(rx) else "",
                "Prescription_Gy_reference": f"{rx:.0f}" if np.isfinite(rx) else "",
                "NewStructureName": f"PTV{number:02d}_{rx:.0f}Gy" if np.isfinite(rx) else f"PTV{number:02d}",
                "Comment": f"DistToIso_mm={dist_iso:.1f}",
                "CI_bridging": "",
                "GI_bridging": "",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def run(args: argparse.Namespace) -> list[str]:
    plan_sets = discover_plan_sets(args.data_root)
    if not plan_sets:
        raise RuntimeError(f"No RTSTRUCT/RTPLAN/RTDOSE plan set found under {args.data_root}")
    plan_set = plan_sets[0]

    plan_files = PlanFiles(
        patient_folder=plan_set.rtstruct_path.parent.name,
        patient_folder_path=str(plan_set.rtstruct_path.parent),
        plan_type="SRS",
        rp_path=str(plan_set.rtplan_path),
        rs_path=str(plan_set.rtstruct_path),
        rd_path=str(plan_set.rtdose_path),
    )

    mapping_df = build_mapping_for_plan(plan_files, force_rx_gy=args.force_rx_gy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(args.output_dir / "viewer_mapping.csv", index=False)

    scenarios = [
        rotation_scenario(1.0),
        rotation_scenario(2.0),
        measured_translation_scenario(
            vertical_cm=args.translation_vertical_cm,
            longitudinal_cm=args.translation_longitudinal_cm,
            lateral_cm=args.translation_lateral_cm,
        ),
        sixd_equal_scenario(),
    ]

    import plot_3d as p3d

    presets = {
        "low": (1500, 3),
        "medium": (5000, 2),
        "high": (12000, 1),
        "ultra": (40000, 1),
    }
    p3d._MAX_FACES, p3d._ISO_Z_STRIDE = presets[args.resolution]

    return generate_3d_html(
        plan_files_list=[plan_files],
        mapping_df=mapping_df,
        scenarios=scenarios,
        output_dir=str(args.output_dir),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 3D HTML for rigid setup-error scenarios.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output" / "rigid_error_impact" / "3d_view")
    parser.add_argument("--translation-vertical-cm", type=float, default=0.08)
    parser.add_argument("--translation-longitudinal-cm", type=float, default=0.02)
    parser.add_argument("--translation-lateral-cm", type=float, default=0.02)
    parser.add_argument("--force-rx-gy", type=float, default=None)
    parser.add_argument("--resolution", choices=["low", "medium", "high", "ultra"], default="medium")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = run(args)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
