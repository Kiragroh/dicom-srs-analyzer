"""
SRS Analysis Pipeline - Configuration
======================================
Central configuration file. Edit constants here to control pipeline behaviour.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root directory that contains all patient subfolders
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Output directory (created automatically if missing)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Mapping Excel file path
MAPPING_EXCEL_PATH = os.path.join(OUTPUT_DIR, "ptv_mapping.xlsx")

# Metrics output CSV
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "metrics.csv")

# Shift scenarios input table
SHIFT_SCENARIOS_PATH = os.path.join(OUTPUT_DIR, "shift_scenarios.xlsx")

# HTML report output
HTML_REPORT_PATH = os.path.join(OUTPUT_DIR, "report.html")

# Log file path
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "pipeline.log")

# ---------------------------------------------------------------------------
# Debug / Filter settings
# ---------------------------------------------------------------------------

# Set True to restrict to the first 2 patients and HA plans only
DEBUG_MODE = True

# In debug mode: number of patients to process
DEBUG_MAX_PATIENTS = 15

# In debug mode: only these plan types are processed
DEBUG_PLAN_TYPES = ["HA"]

# ---------------------------------------------------------------------------
# Anonymization settings (for screenshots / public sharing)
# ---------------------------------------------------------------------------

# Set True to anonymize patient IDs and structure names in all outputs
# Patient IDs become: Patient_001, Patient_002, ...
# Structure names become: PTV_A, PTV_B, ... (alphabetically)
ANONYMIZE_OUTPUT = False

# ---------------------------------------------------------------------------
# PTV detection mode
# ---------------------------------------------------------------------------
# Currently implemented:  "name_startswith_ptv"
# Architecturally reserved: "dicom_type_ptv"  (not yet active)

PTV_DETECTION_MODE = "name_startswith_ptv"

# ---------------------------------------------------------------------------
# Shift scenario settings
# ---------------------------------------------------------------------------

# Set True to compute shifted-dose scenarios in addition to the nominal plan
ENABLE_SHIFT_SCENARIOS = True

# ---------------------------------------------------------------------------
# Calculation grid settings
# ---------------------------------------------------------------------------
# The grid step progression used for metric convergence (mm).
# Mirrors the logic in SRS_Metrics_v1.0.0RC1.html (CalculationEngine).
GRID_STEP_SIZES = [1.0, 0.5, 0.25]

# Structures with effective diameter >= this value (mm) use the coarse 1 mm grid
LARGE_TARGET_DIAMETER_MM = 50.0

# Volume convergence tolerance (fraction) for grid refinement
GRID_CONVERGENCE_TOL = 0.05

# Fixed isodose threshold for V12Gy metric (Gy)
V12GY_THRESHOLD = 12.0

# ---------------------------------------------------------------------------
# Boundary-expansion settings for CI/GI uncertainty resolution
# ---------------------------------------------------------------------------
# The structure bounding box is expanded for volume metrics (PIV, V_half_Rx).
# A coarse 1 mm grid is used; structural metrics keep the fine-grid values.
# Expansion is RADIAL (spherical dilation of bbox) to avoid capturing
# dose from adjacent structures at the corners of a rectangular expansion.
#
# Separate margins for CI and GI because:
#   - CI needs the 100% Rx isodose → typically compact → 15 mm sufficient
#   - GI needs the 50% Rx isodose → spreads further → 20 mm recommended
BOUNDARY_MARGIN_CI_MM     = 15.0   # radial expansion (mm) for PIV (100% Rx)
BOUNDARY_MARGIN_GI_MM     = 20.0   # radial expansion (mm) for V_half_Rx (50% Rx)
BOUNDARY_EXPAND_STEP_MM   = 1.0    # coarse grid step for the boundary scan
BOUNDARY_CLOSURE_DELTA_MM = 2.0    # additional expansion for PIV-stability closure check
PIV_CLOSURE_TOL           = 0.02   # 2 %: PIV (100 % Rx) stable within this → no * on CI
VHALF_CLOSURE_TOL         = 0.05   # 5 %: V_half_Rx (50 % Rx) stable within this → no * on GI

# Backward compat alias used in slice_views.py for the bounding-box visualisation
BOUNDARY_FIXED_MARGIN_MM  = BOUNDARY_MARGIN_CI_MM

# Show the +BOUNDARY_MARGIN_CI_MM computation box outline in slice views
SHOW_COMPUTATION_BOUNDARY = True

# ---------------------------------------------------------------------------
# Dose-bridging detection settings
# ---------------------------------------------------------------------------
# For structures with Paddick CI < BRIDGING_CI_THRESHOLD, the pipeline checks
# whether PIV grows when the sampling radius is enlarged from BRIDGING_SMALL_MM
# to BRIDGING_LARGE_MM.  A growth > BRIDGING_PIV_GROWTH_TOL indicates that
# the 100 % Rx isodose of a neighbouring structure is being captured (bridging).
BRIDGING_CI_THRESHOLD     = 0.80   # check bridging only when CI below this
BRIDGING_SMALL_MARGIN_MM  = 10.0   # inner radius for PIV bridging test (mm)
BRIDGING_LARGE_MARGIN_MM  = 11.0   # outer radius = inner + 1 grid step → sharp-edge test
BRIDGING_PIV_GROWTH_TOL   = 0.15   # 15 % PIV growth at that step triggers bridging flag

# ---------------------------------------------------------------------------
# DICOM export settings
# ---------------------------------------------------------------------------

# Set True to write modified RD + RP with new SOPInstanceUIDs into a separate
# folder so they can be imported into a TPS alongside the original CT / RS.
# FrameOfReferenceUID and ReferencedStructureSetSequence are kept unchanged.
ENABLE_DICOM_EXPORT = True

# Output folder for re-stamped DICOM files (one subfolder per patient/plan)
DICOM_EXPORT_DIR = os.path.join(OUTPUT_DIR, "dicom_export")

# ---------------------------------------------------------------------------
# Orthogonal slice view settings
# ---------------------------------------------------------------------------

# Set True to generate axial / coronal / sagittal dose+contour PNGs per PTV
ENABLE_VIEWS = True

# Root folder for PNG files (subfolders: patient/plan/structure)
VIEWS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "views")

# Half-width of the view window around the PTV centroid (mm)
VIEW_HALF_MM = 35.0

# DPI for output PNG images
VIEW_DPI = 120

# ---------------------------------------------------------------------------
# Plan type identifier strings (searched in filename, case-insensitive)
# ---------------------------------------------------------------------------
PLAN_TYPE_HA = "HA"
PLAN_TYPE_MM = "MM"

# Prefix filters for patient folders (folders not starting with these are skipped)
PATIENT_FOLDER_PREFIXES = ["zz_"]
