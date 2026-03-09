"""
SRS Analysis Pipeline - Main Entry Point
==========================================
Orchestrates all phases:
  A  – Directory scan and plan detection
  B  – DICOM loading, PTV mapping table build/update
  C  – Nominal metric computation + orthogonal slice views
  D  – (Optional) 6D shift-scenario metrics
  E  – HTML report generation
  F  – (Optional) DICOM export with new UIDs

Usage:
    cd <srs_analysis folder>
    python main.py

Toggle DEBUG_MODE, ENABLE_SHIFT_SCENARIOS etc. in config.py.
"""

import logging
import os
import sys

import pandas as pd

# Ensure the srs_analysis package folder is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

import config
from config import (
    OUTPUT_DIR, MAPPING_EXCEL_PATH, METRICS_CSV_PATH,
    SHIFT_SCENARIOS_PATH, HTML_REPORT_PATH, LOG_FILE_PATH,
    DEBUG_MODE, ENABLE_SHIFT_SCENARIOS,
    ENABLE_DICOM_EXPORT, ENABLE_VIEWS,
    ENABLE_COMPARISON, COMPARE_REPORT_PATH,
    ENABLE_SHIFT_PLOTS, SHIFT_PLOTS_DIR,
    ENABLE_DVH_PLOTS, DVH_PLOTS_DIR,
)
from dicom_io import (
    scan_patient_folders, find_plan_files, apply_debug_filter, load_plan,
)
from excel_io import (
    load_mapping, save_mapping, upsert_mapping,
    build_mapping_rows, assign_new_names_inplace,
    get_excluded_structures, get_active_structures,
)
from metrics import compute_metrics_for_plan, results_to_dataframe
from dicom_export import export_plan_dicom
from slice_views import render_structure_views
from shift_scenarios import (
    create_default_scenarios_table, load_scenarios,
    build_shifted_dose, NOMINAL_SCENARIO, scenario_label,
)
from html_report import generate_report
from comparison import generate_comparison_report
from plot_shift import generate_shift_plots
from dvh_plot import generate_dvh_plot


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO,
                        format=fmt, handlers=handlers)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase A – Scan
# ---------------------------------------------------------------------------

def phase_a() -> list:
    logger.info("=" * 60)
    logger.info("PHASE A – Directory scan")
    logger.info("=" * 60)
    logger.info("Data root: %s", config.DATA_ROOT)
    if DEBUG_MODE:
        logger.info("[DEBUG MODE ACTIVE] patients=%d  plan_types=%s",
                    config.DEBUG_MAX_PATIENTS, config.DEBUG_PLAN_TYPES)

    folder_names = scan_patient_folders(config.DATA_ROOT)
    all_plan_files = []
    for folder in folder_names:
        plans = find_plan_files(folder, config.DATA_ROOT)
        all_plan_files.extend(plans)

    logger.info("Total plans found (before debug filter): %d", len(all_plan_files))
    filtered = apply_debug_filter(all_plan_files, folder_names)
    logger.info("Plans to process: %d", len(filtered))

    for pf in filtered:
        logger.info("  %-30s  %s  RS=%s  RD=%s  RP=%s",
                    pf.patient_folder, pf.plan_type,
                    "ok" if pf.rs_path else "MISSING",
                    "ok" if pf.rd_path else "MISSING",
                    "ok" if pf.rp_path else "MISSING")

    return filtered


# ---------------------------------------------------------------------------
# Phase B – Build / update mapping table
# ---------------------------------------------------------------------------

def phase_b(plan_files_list: list) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("PHASE B – PTV mapping table build/update")
    logger.info("=" * 60)

    existing_df = load_mapping(MAPPING_EXCEL_PATH)
    all_new_rows = []

    for pf in plan_files_list:
        try:
            structures, dose, meta = load_plan(pf)
        except Exception as exc:
            logger.error("Failed to load plan %s/%s: %s", pf.patient_folder, pf.plan_type, exc)
            continue

        if not structures:
            logger.warning("No structures loaded for %s/%s – skipping.", pf.patient_folder, pf.plan_type)
            continue

        rows = build_mapping_rows(pf, structures, dose, meta)
        all_new_rows.extend(rows)

    if not all_new_rows:
        logger.warning("No PTV candidate rows generated – mapping table unchanged.")
        return existing_df

    updated_df = upsert_mapping(existing_df, all_new_rows)

    # Assign NewStructureName where missing
    updated_df = assign_new_names_inplace(updated_df)

    save_mapping(updated_df, MAPPING_EXCEL_PATH)

    # Log exclusion summary
    get_excluded_structures(updated_df)

    return updated_df


# ---------------------------------------------------------------------------
# Phase C – Nominal metrics
# ---------------------------------------------------------------------------

def phase_c(
    plan_files_list: list,
    mapping_df: pd.DataFrame,
) -> tuple:
    """
    Returns (all_results, views_map).
    views_map: {(patient_folder, plan_type, struct_orig_name, scenario): {view: path}}
    """
    logger.info("=" * 60)
    logger.info("PHASE C – Nominal metric computation + views")
    logger.info("=" * 60)

    all_results = []
    views_map: dict = {}
    active_df = get_active_structures(mapping_df)

    for pf in plan_files_list:
        mask = (
            (active_df["PatientFolder"] == pf.patient_folder) &
            (active_df["PlanType"] == pf.plan_type)
        )
        plan_active = active_df[mask]

        if plan_active.empty:
            logger.info("No active PTV structures for %s/%s – skipping.", pf.patient_folder, pf.plan_type)
            continue

        try:
            structures, dose, meta = load_plan(pf)
        except Exception as exc:
            logger.error("Load error for %s/%s: %s", pf.patient_folder, pf.plan_type, exc)
            continue

        isocenter = meta.isocenter if meta else None

        results = compute_metrics_for_plan(
            plan_files=pf,
            structures=structures,
            dose=dose,
            plan_meta=meta,
            active_rows=plan_active,
            scenario="nominal",
            isocenter=isocenter,
        )
        all_results.extend(results)
        logger.info("  %s/%s: %d metric result(s)", pf.patient_folder, pf.plan_type, len(results))

        # Generate orthogonal slice views
        if ENABLE_VIEWS and dose is not None and isocenter is not None:
            struct_map = {s.name: s for s in structures}
            for _, row in plan_active.iterrows():
                orig_name = str(row.get("StructureName_Original", "")).strip()
                struct = struct_map.get(orig_name)
                if struct is None:
                    continue
                rx = _resolve_rx_from_row(row)
                if rx is None:
                    continue
                vpaths = render_structure_views(
                    patient_folder=pf.patient_folder,
                    plan_type=pf.plan_type,
                    structure=struct,
                    dose=dose,
                    isocenter=isocenter,
                    rx_dose=rx,
                    scenario="nominal",
                )
                if vpaths:
                    views_map[(pf.patient_folder, pf.plan_type, orig_name, "nominal")] = vpaths

        # Generate DVH plot for this plan (nominal scenario)
        if ENABLE_DVH_PLOTS and dose is not None:
            generate_dvh_plot(
                patient_folder=pf.patient_folder,
                plan_type=pf.plan_type,
                structures=structures,
                dose=dose,
                active_rows=plan_active.to_dict("records"),
                output_dir=DVH_PLOTS_DIR,
                scenario="nominal",
            )

    return all_results, views_map


def _resolve_rx_from_row(row) -> "Optional[float]":
    for col in ("Prescription_Gy_reference", "Prescription_Gy_detected"):
        val = row.get(col)
        try:
            v = float(val)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Phase D – Shift scenarios
# ---------------------------------------------------------------------------

def phase_d(plan_files_list: list, mapping_df: pd.DataFrame, nominal_results: list, views_map: dict, scenarios: list = None) -> list:
    logger.info("=" * 60)
    logger.info("PHASE D – Shift-scenario metrics")
    logger.info("=" * 60)

    if not ENABLE_SHIFT_SCENARIOS:
        logger.info("ENABLE_SHIFT_SCENARIOS=False – skipping shift analysis.")
        return nominal_results

    if scenarios is None:
        create_default_scenarios_table(SHIFT_SCENARIOS_PATH)
        scenarios = load_scenarios(SHIFT_SCENARIOS_PATH)
    shift_scenarios = [s for s in scenarios if s.name != "nominal"]
    logger.info("Loaded %d shift scenario(s): %s", len(shift_scenarios), [s.name for s in shift_scenarios])

    all_results = list(nominal_results)  # keep nominals
    active_df = get_active_structures(mapping_df)

    for pf in plan_files_list:
        mask = (
            (active_df["PatientFolder"] == pf.patient_folder) &
            (active_df["PlanType"] == pf.plan_type)
        )
        plan_active = active_df[mask]
        if plan_active.empty:
            continue

        try:
            structures, dose, meta = load_plan(pf)
        except Exception as exc:
            logger.error("Load error for %s/%s: %s", pf.patient_folder, pf.plan_type, exc)
            continue

        isocenter = meta.isocenter if meta else None
        if isocenter is None:
            logger.warning("No isocenter for %s/%s – shift scenarios skipped.", pf.patient_folder, pf.plan_type)
            continue

        struct_map = {s.name: s for s in structures}

        for scenario in shift_scenarios:
            sc_lbl = scenario_label(scenario)
            logger.info("  Scenario '%s' (%s) for %s/%s",
                        scenario.name, sc_lbl, pf.patient_folder, pf.plan_type)
            shifted_dose = build_shifted_dose(dose, scenario, isocenter)
            results = compute_metrics_for_plan(
                plan_files=pf,
                structures=structures,
                dose=dose,
                plan_meta=meta,
                active_rows=plan_active,
                scenario=sc_lbl,
                dose_override=shifted_dose,
                isocenter=isocenter,
            )
            all_results.extend(results)

            # Generate views for this shift scenario
            if ENABLE_VIEWS and shifted_dose is not None:
                for _, row in plan_active.iterrows():
                    orig_name = str(row.get("StructureName_Original", "")).strip()
                    struct = struct_map.get(orig_name)
                    if struct is None:
                        continue
                    rx = _resolve_rx_from_row(row)
                    if rx is None:
                        continue
                    vpaths = render_structure_views(
                        patient_folder=pf.patient_folder,
                        plan_type=pf.plan_type,
                        structure=struct,
                        dose=shifted_dose,
                        isocenter=isocenter,
                        rx_dose=rx,
                        scenario=sc_lbl,
                        nominal_dose=dose,  # pass nominal for gray overlay
                    )
                    if vpaths:
                        views_map[(pf.patient_folder, pf.plan_type, orig_name, sc_lbl)] = vpaths

            # Generate DVH plot for this shift scenario
            if ENABLE_DVH_PLOTS and shifted_dose is not None:
                generate_dvh_plot(
                    patient_folder=pf.patient_folder,
                    plan_type=pf.plan_type,
                    structures=structures,
                    dose=shifted_dose,
                    active_rows=plan_active.to_dict("records"),
                    output_dir=DVH_PLOTS_DIR,
                    scenario=sc_lbl,
                )

    return all_results


# ---------------------------------------------------------------------------
# Phase E – HTML report + CSV export
# ---------------------------------------------------------------------------

def phase_e(all_results: list, mapping_df: pd.DataFrame, views_map: dict) -> None:
    logger.info("=" * 60)
    logger.info("PHASE E – Report generation")
    logger.info("=" * 60)

    metrics_df = results_to_dataframe(all_results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        metrics_df.to_csv(METRICS_CSV_PATH, index=False)
        logger.info("Metrics CSV written: %s", METRICS_CSV_PATH)
    except Exception as exc:
        logger.error("Failed to write metrics CSV: %s", exc)

    generate_report(metrics_df, mapping_df, HTML_REPORT_PATH, views_map=views_map)


# ---------------------------------------------------------------------------
# Phase F – DICOM export
# ---------------------------------------------------------------------------

def phase_h() -> None:
    logger.info("=" * 60)
    logger.info("PHASE H – Shift-scenario impact plots")
    logger.info("=" * 60)

    if not ENABLE_SHIFT_PLOTS:
        logger.info("ENABLE_SHIFT_PLOTS=False – skipping shift plots.")
        return

    paths = generate_shift_plots(METRICS_CSV_PATH, SHIFT_PLOTS_DIR)
    if paths:
        logger.info("  Shift plots written to %s", SHIFT_PLOTS_DIR)
    else:
        logger.info("  No shift plots generated (insufficient data or only nominal scenario).")


def phase_g(
    all_results: list,
    mapping_df: pd.DataFrame,
    plan_files_list: list,
    views_map: dict,
) -> None:
    logger.info("=" * 60)
    logger.info("PHASE G – Plan comparison report")
    logger.info("=" * 60)

    if not ENABLE_COMPARISON:
        logger.info("ENABLE_COMPARISON=False – skipping comparison report.")
        return

    from metrics import results_to_dataframe
    metrics_df = results_to_dataframe(all_results)
    generate_comparison_report(
        metrics_df=metrics_df,
        mapping_df=mapping_df,
        plan_files_list=plan_files_list,
        output_path=COMPARE_REPORT_PATH,
        views_map=views_map,
    )


def phase_f(plan_files_list: list) -> None:
    logger.info("=" * 60)
    logger.info("PHASE F – DICOM export (new UIDs)")
    logger.info("=" * 60)

    if not ENABLE_DICOM_EXPORT:
        logger.info("ENABLE_DICOM_EXPORT=False – skipping.")
        return

    create_default_scenarios_table(SHIFT_SCENARIOS_PATH)
    scenarios = load_scenarios(SHIFT_SCENARIOS_PATH)

    total_ok = 0
    for pf in plan_files_list:
        try:
            _, _, meta = load_plan(pf)
            iso = meta.isocenter if meta else None
        except Exception:
            iso = None
        exported = export_plan_dicom(pf, scenarios=scenarios, isocenter=iso)
        total_ok += len(exported)
    logger.info("DICOM export complete: %d folder(s) written", total_ok)


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    logger.info("SRS Analysis Pipeline starting.")
    logger.info(
        "DEBUG_MODE=%s  ENABLE_SHIFT_SCENARIOS=%s  ENABLE_DICOM_EXPORT=%s  ENABLE_VIEWS=%s",
        DEBUG_MODE, ENABLE_SHIFT_SCENARIOS, ENABLE_DICOM_EXPORT, ENABLE_VIEWS,
    )

    # Ensure shift scenarios example table exists (even when shifts are off)
    create_default_scenarios_table(SHIFT_SCENARIOS_PATH)

    # A – scan
    plan_files_list = phase_a()

    if not plan_files_list:
        logger.error("No plans found – aborting.")
        return

    # B – mapping table
    mapping_df = phase_b(plan_files_list)

    # C – nominal metrics + views
    nominal_results, views_map = phase_c(plan_files_list, mapping_df)

    # Load scenarios once (needed by both D and F)
    create_default_scenarios_table(SHIFT_SCENARIOS_PATH)
    scenarios = load_scenarios(SHIFT_SCENARIOS_PATH)

    # D – shift scenarios (conditional)
    all_results = phase_d(plan_files_list, mapping_df, nominal_results, views_map, scenarios)

    # E – reports
    phase_e(all_results, mapping_df, views_map)

    # F – DICOM export
    phase_f(plan_files_list)

    # G – Plan comparison report
    phase_g(all_results, mapping_df, plan_files_list, views_map)

    # H – Shift-scenario impact plots
    phase_h()

    logger.info("Pipeline complete.")
    logger.info("Outputs:")
    logger.info("  Mapping Excel : %s", MAPPING_EXCEL_PATH)
    logger.info("  Metrics CSV   : %s", METRICS_CSV_PATH)
    logger.info("  Shift table   : %s", SHIFT_SCENARIOS_PATH)
    logger.info("  HTML report   : %s", HTML_REPORT_PATH)
    if ENABLE_COMPARISON:
        logger.info("  Compare report: %s", COMPARE_REPORT_PATH)
    if ENABLE_SHIFT_PLOTS:
        logger.info("  Shift plots   : %s", SHIFT_PLOTS_DIR)
    logger.info("  Log file      : %s", LOG_FILE_PATH)


if __name__ == "__main__":
    main()
