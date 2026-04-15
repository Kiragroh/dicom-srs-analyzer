"""
plot_3d_from_dicom.py – Standalone 3D Isodose-vs-Structure Plot Generator
=========================================================================

Generates PNG plots + interactive HTML viewers without running the full pipeline.
Reads from existing DICOM files + shift_scenarios.xlsx + ptv_mapping.xlsx.

Usage
-----
  python plot_3d_from_dicom.py                        # all patients, all scenarios
  python plot_3d_from_dicom.py --patient zz_Paper_MM_UKE_10
  python plot_3d_from_dicom.py --plan-type HA
  python plot_3d_from_dicom.py --out-dir C:/custom/output/3d_plots
  python plot_3d_from_dicom.py --no-png                # HTML only
  python plot_3d_from_dicom.py --no-html               # PNG only

Output
------
  PNG: {out_dir}/{patient}__{plan_type}__{scenario}.png
         Two 3-D views (Front-L / Front-R) with Lambertian shading.
         Gray=PTV (nominal only)  Green=nominal isodose  Magenta=shifted protrusion

  HTML: {out_dir}/html/{patient}__{plan_type}__viewer.html
         Interactive Plotly 3-D viewer – rotate freely, switch scenarios,
         camera presets, anatomical orientation cube.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger("plot_3d_from_dicom")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D isodose-protrusion plots from existing DICOMs."
    )
    parser.add_argument("--patient",   default=None,
                        help="Filter to this patient folder name")
    parser.add_argument("--plan-type", default=None,
                        help="Filter to this plan type (e.g. HA or MM)")
    parser.add_argument("--out-dir",   default=None,
                        help="Output directory (default: config.PLOT_3D_DIR)")
    parser.add_argument("--stride",     type=int, default=None,
                        help="Z-slice stride for isodose marching cubes (overrides --resolution)")
    parser.add_argument("--max-faces",  type=int, default=None,
                        help="Max triangular faces per mesh (overrides --resolution)")
    parser.add_argument("--resolution", default=None,
                        choices=["low", "medium", "high", "ultra"],
                        help="Mesh quality preset  "
                             "low=1500f/stride3  medium=5000f/stride2  "
                             "high=12000f/stride1  ultra=40000f/stride1  "
                             "(default: from config.py)")
    parser.add_argument("--no-png",  action="store_true",
                        help="Skip PNG generation")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip HTML viewer generation")
    args = parser.parse_args()

    from config import (
        DATA_ROOT, MAPPING_EXCEL_PATH, SHIFT_SCENARIOS_PATH,
        PLOT_3D_DIR, DEBUG_MODE,
    )
    from dicom_io import scan_patient_folders, find_plan_files, apply_debug_filter
    from excel_io import load_mapping
    from shift_scenarios import load_scenarios
    import plot_3d as p3d
    from plot_3d_html import generate_3d_html

    _RESOLUTION_PRESETS = {
        "low":    (1500,  3),
        "medium": (5000,  2),
        "high":   (12000, 1),
        "ultra":  (40000, 1),
    }

    if args.resolution:
        preset_faces, preset_stride = _RESOLUTION_PRESETS[args.resolution]
        p3d._MAX_FACES    = preset_faces
        p3d._ISO_Z_STRIDE = preset_stride
        logger.info("Resolution preset '%s': max_faces=%d, iso_stride=%d",
                    args.resolution, preset_faces, preset_stride)

    if args.stride is not None:
        p3d._ISO_Z_STRIDE = args.stride
        logger.info("Z-stride overridden to %d", args.stride)

    if args.max_faces is not None:
        p3d._MAX_FACES = args.max_faces
        logger.info("max_faces overridden to %d", args.max_faces)

    out_dir = args.out_dir or PLOT_3D_DIR
    os.makedirs(out_dir, exist_ok=True)

    folder_names = scan_patient_folders(DATA_ROOT)
    all_plans = []
    for f in folder_names:
        all_plans.extend(find_plan_files(f, DATA_ROOT))

    if DEBUG_MODE:
        plan_files_list = apply_debug_filter(all_plans, folder_names)
    else:
        plan_files_list = all_plans

    if args.patient:
        plan_files_list = [p for p in plan_files_list
                           if p.patient_folder == args.patient]
        logger.info("Filtered to patient '%s': %d plan(s)", args.patient,
                    len(plan_files_list))

    if args.plan_type:
        plan_files_list = [p for p in plan_files_list
                           if p.plan_type == args.plan_type]
        logger.info("Filtered to plan_type '%s': %d plan(s)", args.plan_type,
                    len(plan_files_list))

    if not plan_files_list:
        logger.error("No matching plans found. Check --patient / --plan-type.")
        sys.exit(1)

    mapping_df = load_mapping(MAPPING_EXCEL_PATH)
    scenarios  = load_scenarios(SHIFT_SCENARIOS_PATH)

    logger.info("Generating 3D output for %d plan(s), %d scenario(s) …",
                len(plan_files_list), len(scenarios))

    png_paths  = []
    html_paths = []

    if not args.no_png:
        png_paths = p3d.generate_3d_plots(
            plan_files_list=plan_files_list,
            mapping_df=mapping_df,
            scenarios=scenarios,
            output_dir=out_dir,
        )
        print(f"\nPNG: {len(png_paths)} file(s) written to {out_dir}")
        for p in png_paths:
            print("  ", p)

    if not args.no_html:
        html_dir = os.path.join(out_dir, "html")
        html_paths = generate_3d_html(
            plan_files_list=plan_files_list,
            mapping_df=mapping_df,
            scenarios=scenarios,
            output_dir=html_dir,
        )
        print(f"\nHTML: {len(html_paths)} viewer(s) written to {html_dir}")
        for p in html_paths:
            print("  ", p)


if __name__ == "__main__":
    main()
