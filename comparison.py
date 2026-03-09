"""
SRS Analysis – Plan Comparison Report
======================================
Generates a standalone HTML report comparing two plan types (e.g. HA vs MM)
for all patients that have both plans present in the metrics CSV.

Phases covered:
  1  PTV matching by NewStructureName
  2  Side-by-side metric table with delta / % columns
  3  DVH overlay per PTV (cumulative, canvas-rendered)
  4  MLC viewer with control-point slider
  5  Side-by-side dose views (existing PNGs from Phase C)

Entry point:
    generate_comparison_report(
        metrics_df, mapping_df, plan_files_list,
        output_path, views_map
    )
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, VIEWS_OUTPUT_DIR
from dicom_io import PlanFiles, load_plan
from dvh import get_dvh
from mlc_viewer import load_mlc_data

logger = logging.getLogger(__name__)

DVH_CACHE_DIR = os.path.join(OUTPUT_DIR, "dvh_cache")

# Metrics to show in the comparison table (CSV column → display label)
METRIC_COLS = [
    ("TV_cc",           "TV (cc)"),
    ("Coverage_pct",    "Coverage (%)"),
    ("PaddickCI",       "Paddick CI"),
    ("RTOG_CI",         "RTOG CI"),
    ("HI",              "HI"),
    ("GI",              "GI"),
    ("Dmax_Gy",         "Dmax (Gy)"),
    ("PIV_cc",          "PIV (cc)"),
    ("V12Gy_cc",        "V12Gy (cc)"),
    ("D98_Gy",          "D98 (Gy)"),
    ("D2_Gy",           "D2 (Gy)"),
    ("D50_Gy",          "D50 (Gy)"),
    ("DistToIso_mm",    "Dist Iso (mm)"),
]

# For these metrics, HIGHER is better
HIGHER_IS_BETTER = {"Coverage_pct", "PaddickCI", "D98_Gy", "D50_Gy"}
# For these metrics, LOWER is better
LOWER_IS_BETTER  = {"RTOG_CI", "GI", "Dmax_Gy", "PIV_cc", "V12Gy_cc", "DistToIso_mm", "D2_Gy"}
# Others: no strong preference (neutral)


# ---------------------------------------------------------------------------
# Phase 1 – Find comparable plan pairs
# ---------------------------------------------------------------------------

def find_comparison_pairs(metrics_df: pd.DataFrame) -> List[Dict]:
    """
    Return all pairwise plan-type combinations per patient.
    Each dict: {patient_folder, plan_a, plan_b}  (plan_a < plan_b alphabetically)
    Supports any number of plan types (HA, MM, 6D, …).
    """
    from itertools import combinations
    if metrics_df is None or metrics_df.empty:
        return []
    nominal = metrics_df[metrics_df.get("Scenario", pd.Series()) == "nominal"] \
              if "Scenario" in metrics_df.columns else metrics_df
    pairs = []
    for patient, grp in nominal.groupby("PatientFolder"):
        plan_types = sorted(grp["PlanType"].unique().tolist())
        for plan_a, plan_b in combinations(plan_types, 2):
            pairs.append({
                "patient_folder": patient,
                "plan_a": plan_a,
                "plan_b": plan_b,
            })
    logger.info("Comparison: found %d pair(s)", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Phase 2 – Match structures
# ---------------------------------------------------------------------------

def _match_structures(metrics_df: pd.DataFrame, patient: str,
                      plan_a: str, plan_b: str) -> List[Dict]:
    """
    Match structures between plan_a and plan_b for a patient using NewStructureName.
    Returns list of match dicts.
    """
    nom = metrics_df[
        (metrics_df["PatientFolder"] == patient) &
        (metrics_df["Scenario"].eq("nominal") if "Scenario" in metrics_df.columns else True)
    ]
    rows_a = nom[nom["PlanType"] == plan_a]
    rows_b = nom[nom["PlanType"] == plan_b]

    def row_to_metrics(row) -> Dict:
        out = {}
        for col, _ in METRIC_COLS:
            v = row.get(col)
            try:
                out[col] = float(v) if v is not None and str(v) != "nan" else None
            except (TypeError, ValueError):
                out[col] = None
        out["CI_uncertain"]       = bool(row.get("CI_uncertain", False))
        out["GI_uncertain"]       = bool(row.get("GI_uncertain", False))
        out["Bridging_suspected"] = bool(row.get("Bridging_suspected", False))
        out["Rx_dose"]            = float(row.get("Rx_Gy", 0) or 0)
        out["Error"]              = str(row.get("Error", "") or "")
        return out

    name_to_a = {str(r.get("NewStructureName", "") or r.get("StructureName_Original", "")): r
                 for _, r in rows_a.iterrows()}
    name_to_b = {str(r.get("NewStructureName", "") or r.get("StructureName_Original", "")): r
                 for _, r in rows_b.iterrows()}

    all_names = sorted(set(name_to_a) | set(name_to_b))
    matches = []
    for name in all_names:
        ra = name_to_a.get(name)
        rb = name_to_b.get(name)
        entry = {
            "new_name":  name,
            "orig_a":    str(ra.get("StructureName_Original", name)) if ra is not None else None,
            "orig_b":    str(rb.get("StructureName_Original", name)) if rb is not None else None,
            "metrics_a": row_to_metrics(ra) if ra is not None else None,
            "metrics_b": row_to_metrics(rb) if rb is not None else None,
            "only_in":   ("a" if rb is None else ("b" if ra is None else None)),
        }
        matches.append(entry)
    return matches


# ---------------------------------------------------------------------------
# Helpers – DICOM loading / DVH / MLC / views
# ---------------------------------------------------------------------------

def _find_plan_files(patient: str, plan_type: str,
                     plan_files_list: List[PlanFiles]) -> Optional[PlanFiles]:
    for pf in plan_files_list:
        if pf.patient_folder == patient and pf.plan_type == plan_type:
            return pf
    return None


def _relative_path(abs_path: str, report_dir: str) -> str:
    try:
        return os.path.relpath(abs_path, report_dir).replace("\\", "/")
    except ValueError:
        return abs_path.replace("\\", "/")


def _gather_views(patient: str, plan_type: str, orig_name: str,
                  views_map: dict, report_dir: str) -> Dict[str, str]:
    """Return {axial/coronal/sagittal: relative_path} from existing views."""
    key = (patient, plan_type, orig_name, "nominal")
    vpaths = views_map.get(key, {})
    return {k: _relative_path(v, report_dir) for k, v in vpaths.items() if v}


# ---------------------------------------------------------------------------
# Phase 3-4 – Per-patient data gathering
# ---------------------------------------------------------------------------

def _gather_patient_data(
    patient: str,
    plan_a: str,
    plan_b: str,
    metrics_df: pd.DataFrame,
    plan_files_list: List[PlanFiles],
    views_map: dict,
    report_dir: str,
) -> Dict:
    """Load DICOM for both plans, compute DVH, extract MLC, gather views."""
    matches = _match_structures(metrics_df, patient, plan_a, plan_b)

    # Load DICOM for both plans
    pf_a = _find_plan_files(patient, plan_a, plan_files_list)
    pf_b = _find_plan_files(patient, plan_b, plan_files_list)

    structs_a, dose_a, meta_a = ({}, None, None)
    structs_b, dose_b, meta_b = ({}, None, None)

    if pf_a:
        try:
            s, d, m = load_plan(pf_a)
            structs_a = {st.name: st for st in s}
            dose_a, meta_a = d, m
        except Exception as exc:
            logger.error("Comparison: load error %s/%s: %s", patient, plan_a, exc)

    if pf_b:
        try:
            s, d, m = load_plan(pf_b)
            structs_b = {st.name: st for st in s}
            dose_b, meta_b = d, m
        except Exception as exc:
            logger.error("Comparison: load error %s/%s: %s", patient, plan_b, exc)

    # MLC data
    mlc_a = []
    mlc_b = []
    if pf_a and pf_a.rp_path:
        logger.info("Comparison: loading MLC from %s", pf_a.rp_path)
        mlc_a = load_mlc_data(pf_a.rp_path)
        logger.info("Comparison: MLC plan A → %d beams", len(mlc_a))
    else:
        logger.warning("Comparison: no RP path for %s/%s", patient, plan_a)
    
    if pf_b and pf_b.rp_path:
        logger.info("Comparison: loading MLC from %s", pf_b.rp_path)
        mlc_b = load_mlc_data(pf_b.rp_path)
        logger.info("Comparison: MLC plan B → %d beams", len(mlc_b))
    else:
        logger.warning("Comparison: no RP path for %s/%s", patient, plan_b)

    # Plan labels from meta
    label_a = meta_a.plan_label if meta_a else plan_a
    label_b = meta_b.plan_label if meta_b else plan_b

    # Per-structure: DVH + views
    for match in matches:
        # DVH plan A
        orig_a = match.get("orig_a")
        rx_a   = (match["metrics_a"] or {}).get("Rx_dose") or 0
        if orig_a and orig_a in structs_a and dose_a and rx_a:
            match["dvh_a"] = get_dvh(structs_a[orig_a], dose_a, rx_a,
                                      DVH_CACHE_DIR, patient, plan_a)
        else:
            match["dvh_a"] = None

        # DVH plan B
        orig_b = match.get("orig_b")
        rx_b   = (match["metrics_b"] or {}).get("Rx_dose") or 0
        if orig_b and orig_b in structs_b and dose_b and rx_b:
            match["dvh_b"] = get_dvh(structs_b[orig_b], dose_b, rx_b,
                                      DVH_CACHE_DIR, patient, plan_b)
        else:
            match["dvh_b"] = None

        # Views
        if orig_a:
            match["views_a"] = _gather_views(patient, plan_a, orig_a, views_map, report_dir)
        else:
            match["views_a"] = {}
        if orig_b:
            match["views_b"] = _gather_views(patient, plan_b, orig_b, views_map, report_dir)
        else:
            match["views_b"] = {}

    pair_id = (patient + "__" + plan_a + "_vs_" + plan_b).replace(" ", "_")
    return {
        "patient_folder": patient,
        "plan_a":         plan_a,
        "plan_b":         plan_b,
        "label_a":        label_a,
        "label_b":        label_b,
        "structures":     matches,
        "mlc_a":          mlc_a,
        "mlc_b":          mlc_b,
        "pair_id":        pair_id,
    }


# ---------------------------------------------------------------------------
# Phase 5 – HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>SRS Plan Comparison</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0e1117;color:#ccc;font-family:'Segoe UI',sans-serif;font-size:13px}
.layout{display:flex;height:100vh}
.sidebar{width:210px;min-width:210px;overflow-y:auto;background:#0d1117;border-right:2px solid #21262d;padding:10px 8px}
.sidebar h3{color:#61dafb;font-size:12px;letter-spacing:.08em;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #21262d}
.pat-card{cursor:pointer;padding:7px 10px;margin-bottom:4px;border-radius:5px;background:#161b22;border:1px solid transparent;transition:background .15s}
.pat-card:hover{background:#1e2a3a}
.pat-card.active{background:#0d2137;border-color:#61dafb}
.pat-card .pname{font-size:11px;color:#aaddff;font-weight:bold;word-break:break-all}
.pat-card .ptypes{font-size:10px;color:#888;margin-top:2px}
.main{flex:1;overflow-y:auto;padding:20px 24px}
.patient-section{display:none}
.patient-section.active{display:block}
.plan-header{background:#161b22;border-radius:7px;padding:12px 16px;margin-bottom:14px;border:1px solid #21262d}
.plan-header h2{color:#61dafb;font-size:16px;margin-bottom:6px}
.plan-header .plan-badges{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.badge-a{background:#1a3a1a;color:#66cc88;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:bold}
.badge-b{background:#1a2a3a;color:#66aaff;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:bold}
.tabs{display:flex;gap:4px;margin-bottom:0;border-bottom:2px solid #21262d;padding-bottom:0}
.tab{padding:7px 16px;cursor:pointer;background:transparent;color:#888;border-radius:5px 5px 0 0;border:1px solid transparent;border-bottom:none;transition:all .15s;font-size:12px}
.tab:hover{color:#aaddff;background:#161b22}
.tab.active{color:#61dafb;background:#161b22;border-color:#21262d;border-bottom-color:#161b22;margin-bottom:-2px}
.tab-content{display:none;background:#161b22;border:1px solid #21262d;border-top:none;border-radius:0 5px 5px 5px;padding:16px;margin-bottom:16px}
.tab-content.active{display:block}
table{border-collapse:collapse;width:100%}
th{background:#0d1117;color:#88aad8;padding:6px 8px;text-align:left;font-size:11px;border-bottom:1px solid #21262d;position:sticky;top:0}
td{padding:5px 8px;border-bottom:1px solid #1a1a2a;font-size:12px;vertical-align:middle}
tr:hover td{background:#0e1a28}
.struct-name{color:#aaddff;font-weight:bold}
.val-a{color:#66cc88}
.val-b{color:#66aaff}
.delta-better{color:#44dd88;font-weight:bold}
.delta-worse{color:#ff7766;font-weight:bold}
.delta-neutral{color:#999}
.missing{color:#555;font-style:italic}
.warn-badge{color:#ffaa44;font-size:10px;background:#2a2000;border:1px solid #ffaa44;border-radius:3px;padding:1px 4px;margin-left:4px}
.bridge-badge{color:#ff6688;font-size:10px;background:#3a1a22;border:1px solid #ff6688;border-radius:3px;padding:1px 4px;margin-left:4px}
.dvh-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px}
.dvh-card{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:10px}
.dvh-card h4{color:#aaddff;font-size:12px;margin-bottom:8px}
.dvh-legend{display:flex;gap:12px;font-size:10px;color:#888;margin-top:4px}
.leg-a{color:#66cc88}
.leg-b{color:#66aaff}
.mlc-wrap{display:flex;gap:16px;flex-wrap:wrap}
.mlc-panel-box{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:10px;flex:1;min-width:320px}
.mlc-panel-box h4{color:#aaddff;font-size:12px;margin-bottom:6px}
.mlc-ctrls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;font-size:11px}
.mlc-ctrls select{background:#1e2a3a;color:#ccc;border:1px solid #444;border-radius:3px;padding:2px 6px;font-size:11px}
.mlc-ctrls input[type=range]{accent-color:#61dafb;width:120px}
.mlc-ctrls button{background:#1e2a3a;color:#aaddff;border:1px solid #444;border-radius:3px;padding:2px 8px;cursor:pointer;font-size:11px}
.mlc-ctrls button:hover{background:#21316a}
.mlc-meta{font-size:10px;color:#888;margin-bottom:4px;min-height:28px}
.views-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.view-card{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:10px}
.view-card h4{color:#aaddff;font-size:12px;margin-bottom:8px}
.view-orient{display:flex;gap:6px;margin-bottom:6px}
.view-orient span{font-size:10px;color:#888;width:50%;text-align:center}
.view-row-imgs{display:flex;gap:6px}
.view-row-imgs img{width:50%;border-radius:4px;display:block}
.orient-label{font-size:10px;color:#666;text-align:center;margin-bottom:4px}
.no-data{color:#555;font-style:italic;font-size:11px;padding:8px}
select,input{outline:none}
</style>
</head>
<body>
<div class="layout">
<div class="sidebar">
  <h3>PLAN COMPARISON</h3>
  <div id="sidebar-list"></div>
</div>
<div class="main" id="main-content">
  <p style="color:#555;margin-top:40px;text-align:center">&#8592; Select a patient from the sidebar</p>
</div>
</div>

<script>
// ============================================================
// DATA
// ============================================================
const DATA = PLACEHOLDER_DATA;

// ============================================================
// Utility
// ============================================================
function fmt(v, dec) {
  if (v === null || v === undefined) return '<span class="missing">–</span>';
  var n = parseFloat(v);
  if (isNaN(n)) return '<span class="missing">–</span>';
  return n.toFixed(dec !== undefined ? dec : 2);
}
function deltaClass(col, delta) {
  var hib = ["Coverage_pct","PaddickCI"];
  var lib = ["RTOG_CI","GI","Dmax_Gy","PIV_cc","V12Gy_cc","DistToIso_mm"];
  if (delta === null || delta === undefined || isNaN(delta)) return "delta-neutral";
  if (hib.indexOf(col) >= 0) return delta > 0.001 ? "delta-better" : (delta < -0.001 ? "delta-worse" : "delta-neutral");
  if (lib.indexOf(col) >= 0) return delta < -0.001 ? "delta-better" : (delta > 0.001 ? "delta-worse" : "delta-neutral");
  return "delta-neutral";
}
function metricDec(col) {
  var d3 = ["PaddickCI","RTOG_CI","HI","GI"];
  var d1 = ["Coverage_pct","DistToIso_mm","Dmax_Gy","D98_Gy","D2_Gy","Rx_dose"];
  if (d3.indexOf(col) >= 0) return 3;
  if (d1.indexOf(col) >= 0) return 1;
  return 3;
}

// ============================================================
// Sidebar
// ============================================================
(function buildSidebar(){
  var el = document.getElementById("sidebar-list");
  DATA.patients.forEach(function(p, i){
    var card = document.createElement("div");
    card.className = "pat-card";
    card.setAttribute("data-idx", i);
    card.innerHTML = '<div class="pname">' + p.patient_folder.replace(/^zz_/,"") + '</div>' +
      '<div class="ptypes"><span class="badge-a">&#9312; ' + p.plan_a + '</span> vs <span class="badge-b">&#9313; ' + p.plan_b + '</span></div>';
    card.onclick = function(){ showPatient(i); };
    el.appendChild(card);
  });
})();

var activePatient = -1;
var swapState = {};

function getViewData(p) {
  if (!swapState[p.pair_id]) return p;
  var sp = Object.assign({}, p);
  sp.plan_a = p.plan_b; sp.plan_b = p.plan_a;
  sp.label_a = p.label_b; sp.label_b = p.label_a;
  sp.mlc_a = p.mlc_b; sp.mlc_b = p.mlc_a;
  sp.structures = p.structures.map(function(s){
    return Object.assign({}, s, {
      metrics_a: s.metrics_b, metrics_b: s.metrics_a,
      dvh_a: s.dvh_b, dvh_b: s.dvh_a,
      views_a: s.views_b, views_b: s.views_a,
      orig_a: s.orig_b, orig_b: s.orig_a,
      only_in: s.only_in==='a' ? 'b' : (s.only_in==='b' ? 'a' : null),
    });
  });
  return sp;
}

function swapPlans(pairId) {
  swapState[pairId] = !swapState[pairId];
  var p = getViewData(DATA.patients[activePatient]);
  var mc = document.getElementById("main-content");
  mc.innerHTML = buildPatientHTML(p);
  initDVHPlots(p);
  initMLCViewers(p);
}

function showPatient(i) {
  if (activePatient === i) return;
  activePatient = i;
  document.querySelectorAll(".pat-card").forEach(function(c){ c.classList.toggle("active", +c.getAttribute("data-idx") === i); });
  var p = getViewData(DATA.patients[i]);
  var mc = document.getElementById("main-content");
  mc.innerHTML = buildPatientHTML(p);
  initDVHPlots(p);
  initMLCViewers(p);
}

// ============================================================
// Patient HTML
// ============================================================
function buildPatientHTML(p) {
  var html = '';
  html += '<div class="plan-header">';
  html += '<h2>' + p.patient_folder.replace(/^zz_/,"") + ' <span style="color:#888;font-size:13px;font-weight:normal">' + p.plan_a + ' vs ' + p.plan_b + '</span></h2>';
  html += '<div class="plan-badges" style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">';
  html += '<span class="badge-a">&#9312; ' + p.plan_a + ' – ' + p.label_a + '</span>';
  html += '<span style="color:#888">vs</span>';
  html += '<span class="badge-b">&#9313; ' + p.plan_b + ' – ' + p.label_b + '</span>';
  html += '<span style="color:#888;font-size:11px">' + p.structures.length + ' structures</span>';
  html += '<button onclick="swapPlans(\'' + p.pair_id + '\')" style="margin-left:auto;background:#1e2a3a;color:#aaddff;border:1px solid #444;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:11px;white-space:nowrap">&#8646; Swap A / B</button>';
  html += '</div></div>';

  var tabId = "p" + p.pair_id.replace(/\W/g,"");

  html += '<div class="tabs">';
  html += '<div class="tab active" onclick="switchTab(this,\'' + tabId + '_metrics\')">Metrics</div>';
  html += '<div class="tab" onclick="switchTab(this,\'' + tabId + '_dvh\')">DVH Individual</div>';
  html += '<div class="tab" onclick="switchTab(this,\'' + tabId + '_dvhmean\')">DVH Mean ± SD</div>';
  html += '<div class="tab" onclick="switchTab(this,\'' + tabId + '_mlc\')">MLC Viewer</div>';
  html += '<div class="tab" onclick="switchTab(this,\'' + tabId + '_views\')">Dose Views</div>';
  html += '</div>';

  html += '<div id="' + tabId + '_metrics" class="tab-content active">' + buildMetricsTab(p) + '</div>';
  html += '<div id="' + tabId + '_dvh" class="tab-content">' + buildDVHTab(p) + '</div>';
  html += '<div id="' + tabId + '_dvhmean" class="tab-content">' + buildMeanDVHTab(p) + '</div>';
  html += '<div id="' + tabId + '_mlc" class="tab-content">' + buildMLCTab(p) + '</div>';
  html += '<div id="' + tabId + '_views" class="tab-content">' + buildViewsTab(p) + '</div>';


  return html;
}

function switchTab(el, targetId) {
  var tabs = el.parentElement.querySelectorAll(".tab");
  tabs.forEach(function(t){ t.classList.remove("active"); });
  el.classList.add("active");
  var contents = el.parentElement.parentElement.querySelectorAll(".tab-content");
  contents.forEach(function(c){ c.classList.remove("active"); });
  document.getElementById(targetId).classList.add("active");
}

// ============================================================
// Metrics tab
// ============================================================
var METRIC_COLS = PLACEHOLDER_METRIC_COLS;
var METRIC_DEC = {Coverage_pct:1,PaddickCI:3,RTOG_CI:3,HI:3,GI:3,Dmax_Gy:2,
                  PIV_cc:3,V12Gy_cc:3,D98_Gy:2,D2_Gy:2,TV_cc:3,DistToIso_mm:1};

function buildMetricsTab(p) {
  var html = '<div style="overflow-x:auto"><table>';
  html += '<thead><tr><th>Structure</th><th>Rx (Gy)</th>';
  METRIC_COLS.forEach(function(mc){ html += '<th>' + mc[1] + '</th>'; });
  html += '<th>Note</th></tr></thead><tbody>';

  p.structures.forEach(function(s){
    if (!s.metrics_a && !s.metrics_b) return;
    var ma = s.metrics_a || {}, mb = s.metrics_b || {};
    html += '<tr>';
    var badges = '';
    if (s.only_in === 'a') badges = ' <span class="warn-badge">only ' + p.plan_a + '</span>';
    if (s.only_in === 'b') badges = ' <span class="warn-badge">only ' + p.plan_b + '</span>';
    if ((ma.Bridging_suspected || mb.Bridging_suspected)) badges += ' <span class="bridge-badge">bridge</span>';
    html += '<td class="struct-name">' + s.new_name + badges + '</td>';
    var rx = (ma.Rx_dose || mb.Rx_dose || 0);
    html += '<td>' + (rx ? rx.toFixed(0) : '–') + '</td>';
    METRIC_COLS.forEach(function(mc){
      var col = mc[0];
      var dec = METRIC_DEC[col] !== undefined ? METRIC_DEC[col] : 2;
      var va = (ma[col] !== null && ma[col] !== undefined) ? parseFloat(ma[col]) : null;
      var vb = (mb[col] !== null && mb[col] !== undefined) ? parseFloat(mb[col]) : null;
      var delta = (va !== null && vb !== null) ? (vb - va) : null;
      var dc = delta !== null ? deltaClass(col, delta) : 'delta-neutral';
      var sa = va !== null ? ('<span class="val-a">' + va.toFixed(dec) + '</span>') : '<span class="missing">–</span>';
      var sb = vb !== null ? ('<span class="val-b">' + vb.toFixed(dec) + '</span>') : '<span class="missing">–</span>';
      var sd = delta !== null ? ('<span class="' + dc + '">' + (delta >= 0 ? '+' : '') + delta.toFixed(dec) + '</span>') : '<span class="delta-neutral">–</span>';
      html += '<td>' + sa + ' / ' + sb + '<br><small>' + sd + '</small></td>';
    });
    var noteA = (ma.CI_uncertain ? '*CI ' : '') + (ma.GI_uncertain ? '*GI' : '');
    var noteB = (mb.CI_uncertain ? '*CI ' : '') + (mb.GI_uncertain ? '*GI' : '');
    var note  = (noteA || noteB) ? ('<span class="warn-badge">' + (noteA || noteB) + '</span>') : '';
    html += '<td>' + note + '</td>';
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  html += '<p style="font-size:10px;color:#666;margin-top:8px">Values: <span class="val-a">&#9312; ' + p.plan_a + ' (Ref)</span> / <span class="val-b">&#9313; ' + p.plan_b + ' (Comp)</span> &nbsp; Delta = Comp&minus;Ref</p>';
  return html;
}

// ============================================================
// DVH tab (canvas-rendered)
// ============================================================
function buildDVHTab(p) {
  var html = '<div class="dvh-grid">';
  p.structures.forEach(function(s, si){
    if (!s.dvh_a && !s.dvh_b) return;
    html += '<div class="dvh-card">';
    html += '<h4>' + s.new_name + '</h4>';
    html += '<canvas id="dvh_' + p.pair_id.replace(/\W/g,'') + '_' + si + '" width="290" height="200"></canvas>';
    html += '<div class="dvh-legend">';
    if (s.dvh_a) html += '<span class="leg-a">&#9312; ' + p.plan_a + ' (TV=' + (s.dvh_a.total_vol_cc||'?') + ' cc)</span>';
    if (s.dvh_b) html += '<span class="leg-b">&#9313; ' + p.plan_b + ' (TV=' + (s.dvh_b.total_vol_cc||'?') + ' cc)</span>';
    html += '</div></div>';
  });
  html += '</div>';
  return html;
}

function initDVHPlots(p) {
  var pid = p.pair_id.replace(/\W/g,'');
  p.structures.forEach(function(s, si){
    if (!s.dvh_a && !s.dvh_b) return;
    var canvas = document.getElementById("dvh_" + pid + "_" + si);
    if (!canvas) return;
    drawDVH(canvas, s.dvh_a, s.dvh_b, s.metrics_a ? s.metrics_a.Rx_dose : (s.metrics_b ? s.metrics_b.Rx_dose : null));
  });
  // Mean DVH plot
  var meanCanvas = document.getElementById("dvh_mean_" + pid);
  if (meanCanvas) drawMeanDVH(meanCanvas, p);
}

function drawDVH(canvas, dvhA, dvhB, rxDose) {
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;
  var PAD = {top:10, right:10, bottom:28, left:36};
  var pw = W - PAD.left - PAD.right;
  var ph = H - PAD.top - PAD.bottom;

  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, W, H);

  // Determine dose range - round up to nearest 5 Gy for uniform scaling
  var dmax_raw = 0;
  if (dvhA) dmax_raw = Math.max(dmax_raw, dvhA.dose_bins[dvhA.dose_bins.length-1]);
  if (dvhB) dmax_raw = Math.max(dmax_raw, dvhB.dose_bins[dvhB.dose_bins.length-1]);
  if (dmax_raw < 1) return;
  var dmax = Math.ceil(dmax_raw / 5) * 5;  // Round up to nearest 5 Gy
  if (dmax < 5) dmax = 5;

  // Grid - Y axis (volume %)
  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  for (var yi = 0; yi <= 4; yi++) {
    var yp = PAD.top + ph * yi / 4;
    ctx.beginPath(); ctx.moveTo(PAD.left, yp); ctx.lineTo(PAD.left+pw, yp); ctx.stroke();
    ctx.fillStyle = "#666"; ctx.font = "9px sans-serif"; ctx.textAlign = "right";
    ctx.fillText((100 - yi*25) + "%", PAD.left-3, yp+3);
  }
  // Grid - X axis (dose in 5 Gy steps)
  var nSteps = dmax / 5;
  for (var xi = 0; xi <= nSteps; xi++) {
    var xp = PAD.left + pw * (xi / nSteps);
    ctx.beginPath(); ctx.moveTo(xp, PAD.top); ctx.lineTo(xp, PAD.top+ph); ctx.stroke();
    ctx.fillStyle = "#666"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
    ctx.fillText((xi * 5), xp, PAD.top+ph+12);
  }
  ctx.fillStyle = "#888"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("Dose [Gy]", PAD.left + pw/2, H-2);

  // Rx dose line
  if (rxDose) {
    var rxX = PAD.left + (rxDose / dmax) * pw;
    ctx.strokeStyle = "#335533"; ctx.lineWidth = 1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(rxX, PAD.top); ctx.lineTo(rxX, PAD.top+ph); ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawCurve(dvh, color, dashed) {
    if (!dvh) return;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    if (dashed) ctx.setLineDash([5,3]); else ctx.setLineDash([]);
    ctx.beginPath();
    var bins = dvh.dose_bins, vols = dvh.vol_pct;
    var first = true;
    for (var i = 0; i < bins.length; i++) {
      var xp2 = PAD.left + (bins[i] / dmax) * pw;
      var yp2 = PAD.top + ph * (1 - vols[i] / 100);
      if (first) { ctx.moveTo(xp2, yp2); first = false; } else ctx.lineTo(xp2, yp2);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawCurve(dvhA, "#66cc88", false);
  drawCurve(dvhB, "#66aaff", true);
}

// ============================================================
// Mean DVH tab (averaged ± SD)
// ============================================================
function buildMeanDVHTab(p) {
  var html = '<div style="max-width:700px">';
  html += '<h4 style="color:#aaddff;margin-bottom:10px">Mean DVH ± Standard Deviation</h4>';
  html += '<canvas id="dvh_mean_' + p.pair_id.replace(/\W/g,'') + '" width="650" height="400" style="background:#0d1117;border-radius:6px"></canvas>';
  html += '<div style="margin-top:8px;font-size:10px;color:#888">';
  html += '<span class="leg-a">■ &#9312; ' + p.plan_a + ' (mean ± SD)</span> &nbsp;&nbsp; ';
  html += '<span class="leg-b">- - &#9313; ' + p.plan_b + ' (mean ± SD)</span>';
  html += '</div></div>';
  return html;
}

function drawMeanDVH(canvas, p) {
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;
  var PAD = {top:20, right:20, bottom:40, left:50};
  var pw = W - PAD.left - PAD.right;
  var ph = H - PAD.top - PAD.bottom;

  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, W, H);

  // Collect all DVHs for plan A and B
  var dvhsA = [], dvhsB = [];
  p.structures.forEach(function(s){
    if (s.dvh_a && s.dvh_a.dose_bins && s.dvh_a.vol_pct) dvhsA.push(s.dvh_a);
    if (s.dvh_b && s.dvh_b.dose_bins && s.dvh_b.vol_pct) dvhsB.push(s.dvh_b);
  });

  if (dvhsA.length === 0 && dvhsB.length === 0) {
    ctx.fillStyle = "#666"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("No DVH data available", W/2, H/2);
    return;
  }

  // Determine unified dose range (5 Gy steps)
  var dmax_raw = 0;
  dvhsA.concat(dvhsB).forEach(function(dvh){
    dmax_raw = Math.max(dmax_raw, dvh.dose_bins[dvh.dose_bins.length-1]);
  });
  var dmax = Math.ceil(dmax_raw / 5) * 5;
  if (dmax < 5) dmax = 5;

  // Resample all DVHs to uniform dose grid (0.1 Gy steps)
  var doseStep = 0.1;
  var nPts = Math.floor(dmax / doseStep) + 1;
  var uniformDose = [];
  for (var i = 0; i < nPts; i++) uniformDose.push(i * doseStep);

  function resampleDVH(dvh) {
    var out = [];
    for (var i = 0; i < uniformDose.length; i++) {
      var d = uniformDose[i];
      var vol = 0;
      for (var j = 0; j < dvh.dose_bins.length; j++) {
        if (dvh.dose_bins[j] >= d) { vol = dvh.vol_pct[j]; break; }
      }
      out.push(vol);
    }
    return out;
  }

  var resampledA = dvhsA.map(resampleDVH);
  var resampledB = dvhsB.map(resampleDVH);

  // Compute mean ± SD
  function computeMeanSD(arrays) {
    if (arrays.length === 0) return {mean: [], sd: []};
    var n = arrays[0].length;
    var mean = [], sd = [];
    for (var i = 0; i < n; i++) {
      var vals = arrays.map(function(a){ return a[i]; });
      var m = vals.reduce(function(s,v){return s+v;}, 0) / vals.length;
      var variance = vals.reduce(function(s,v){return s+(v-m)*(v-m);}, 0) / vals.length;
      mean.push(m);
      sd.push(Math.sqrt(variance));
    }
    return {mean: mean, sd: sd};
  }

  var statsA = computeMeanSD(resampledA);
  var statsB = computeMeanSD(resampledB);

  // Grid - Y axis
  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  for (var yi = 0; yi <= 4; yi++) {
    var yp = PAD.top + ph * yi / 4;
    ctx.beginPath(); ctx.moveTo(PAD.left, yp); ctx.lineTo(PAD.left+pw, yp); ctx.stroke();
    ctx.fillStyle = "#666"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
    ctx.fillText((100 - yi*25) + "%", PAD.left-5, yp+3);
  }
  // Grid - X axis (5 Gy steps)
  var nSteps = dmax / 5;
  for (var xi = 0; xi <= nSteps; xi++) {
    var xp = PAD.left + pw * (xi / nSteps);
    ctx.beginPath(); ctx.moveTo(xp, PAD.top); ctx.lineTo(xp, PAD.top+ph); ctx.stroke();
    ctx.fillStyle = "#666"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
    ctx.fillText((xi * 5), xp, PAD.top+ph+15);
  }
  ctx.fillStyle = "#aaa"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("Dose [Gy]", PAD.left + pw/2, H-8);
  ctx.save(); ctx.translate(12, PAD.top + ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText("Volume [%]", 0, 0); ctx.restore();

  // Draw SD bands + mean curves
  function drawBand(stats, color) {
    if (stats.mean.length === 0) return;
    ctx.fillStyle = color + "33"; // semi-transparent
    ctx.beginPath();
    for (var i = 0; i < uniformDose.length; i++) {
      var d = uniformDose[i];
      var xp = PAD.left + (d / dmax) * pw;
      var yUpper = PAD.top + ph * (1 - Math.min(100, stats.mean[i] + stats.sd[i]) / 100);
      if (i === 0) ctx.moveTo(xp, yUpper); else ctx.lineTo(xp, yUpper);
    }
    for (var i = uniformDose.length - 1; i >= 0; i--) {
      var d = uniformDose[i];
      var xp = PAD.left + (d / dmax) * pw;
      var yLower = PAD.top + ph * (1 - Math.max(0, stats.mean[i] - stats.sd[i]) / 100);
      ctx.lineTo(xp, yLower);
    }
    ctx.closePath(); ctx.fill();
  }

  function drawMean(stats, color, dashed) {
    if (stats.mean.length === 0) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    if (dashed) ctx.setLineDash([6,4]); else ctx.setLineDash([]);
    ctx.beginPath();
    for (var i = 0; i < uniformDose.length; i++) {
      var d = uniformDose[i];
      var xp = PAD.left + (d / dmax) * pw;
      var yp = PAD.top + ph * (1 - stats.mean[i] / 100);
      if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
    }
    ctx.stroke(); ctx.setLineDash([]);
  }

  drawBand(statsA, "#66cc88");
  drawMean(statsA, "#66cc88", false);
  drawBand(statsB, "#66aaff");
  drawMean(statsB, "#66aaff", true);

  // Legend with N
  ctx.fillStyle = "#888"; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
  ctx.fillText(p.plan_a + " (N=" + dvhsA.length + ")", PAD.left, 12);
  ctx.fillText(p.plan_b + " (N=" + dvhsB.length + ")", PAD.left + 150, 12);
}

// ============================================================
// MLC Viewer tab
// ============================================================
function buildMLCTab(p) {
  if (!p.mlc_a.length && !p.mlc_b.length) {
    return '<p class="no-data">No beam/MLC data available for this patient.</p>';
  }
  var html = '<div class="mlc-wrap">';
  if (p.mlc_a.length) html += buildMLCPanelHTML(p, 'a', '&#9312; ' + p.plan_a + ' – ' + p.label_a);
  if (p.mlc_b.length) html += buildMLCPanelHTML(p, 'b', '&#9313; ' + p.plan_b + ' – ' + p.label_b);
  html += '</div>';
  return html;
}

function buildMLCPanelHTML(p, which, title) {
  var beams = (which === 'a') ? p.mlc_a : p.mlc_b;
  var pid = p.pair_id.replace(/\W/g,'') + "_" + which;
  var beamOpts = '';
  beams.forEach(function(b, i){ beamOpts += '<option value="' + i + '">' + b.name + '</option>'; });
  var html = '<div class="mlc-panel-box">';
  html += '<h4>' + title + '</h4>';
  html += '<div class="mlc-ctrls">';
  html += 'Beam: <select id="mlcBeam_' + pid + '" onchange="mlcUpdate(\'' + pid + '\')">' + beamOpts + '</select>';
  html += '<button onclick="mlcStep(\'' + pid + '\',-1)">&#8249;</button>';
  html += 'CP: <input type="range" id="mlcSlider_' + pid + '" min="0" value="0">';
  html += '<span id="mlcCPLabel_' + pid + '" style="min-width:60px;color:#aaa">0 / 0</span>';
  html += '<button onclick="mlcStep(\'' + pid + '\',1)">&#8250;</button>';
  html += '</div>';
  html += '<div class="mlc-meta" id="mlcMeta_' + pid + '"></div>';
  html += '<canvas id="mlcCanvas_' + pid + '" width="360" height="300" style="background:#090e15;border-radius:4px"></canvas>';
  html += '</div>';
  return html;
}

var mlcState = {};

function initMLCViewers(p) {
  ["a","b"].forEach(function(which){
    var beams = (which === 'a') ? p.mlc_a : p.mlc_b;
    if (!beams.length) return;
    var pid = p.pair_id.replace(/\W/g,'') + "_" + which;
    mlcState[pid] = { beams: beams, beamIdx: 0, cpIdx: 0, patient: p };
    var slider = document.getElementById("mlcSlider_" + pid);
    if (slider) {
      slider.max = Math.max(0, beams[0].control_points.length - 1);
      slider.value = 0;
      // Real-time update while dragging
      slider.addEventListener('input', function(){ mlcUpdateFromSlider(pid); });
    }
    mlcUpdate(pid);
  });
}

function mlcStep(pid, dir) {
  var st = mlcState[pid]; if (!st) return;
  var beam = st.beams[st.beamIdx];
  st.cpIdx = Math.max(0, Math.min(beam.control_points.length - 1, st.cpIdx + dir));
  var slider = document.getElementById("mlcSlider_" + pid);
  if (slider) slider.value = st.cpIdx;
  mlcDraw(pid);
}

function mlcUpdate(pid) {
  var st = mlcState[pid]; if (!st) return;
  var beamSel = document.getElementById("mlcBeam_" + pid);
  if (beamSel) st.beamIdx = parseInt(beamSel.value);
  var slider = document.getElementById("mlcSlider_" + pid);
  var beam = st.beams[st.beamIdx];
  if (slider) {
    slider.max = Math.max(0, beam.control_points.length - 1);
    st.cpIdx = Math.min(st.cpIdx, beam.control_points.length - 1);
    slider.value = st.cpIdx;
  } else { st.cpIdx = 0; }
  mlcDraw(pid);
}

function mlcUpdateFromSlider(pid) {
  var st = mlcState[pid]; if (!st) return;
  var slider = document.getElementById("mlcSlider_" + pid);
  if (slider) st.cpIdx = parseInt(slider.value);
  mlcDraw(pid);
}

function mlcDraw(pid) {
  var st = mlcState[pid]; if (!st) return;
  var canvas = document.getElementById("mlcCanvas_" + pid);
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var beam = st.beams[st.beamIdx];
  var cp = beam.control_points[st.cpIdx] || {};

  // Meta
  var meta = document.getElementById("mlcMeta_" + pid);
  if (meta) {
    meta.innerHTML = 'Beam: <b>' + beam.name + '</b> &nbsp; Gantry: ' + cp.gantry_angle + '° &nbsp; Coll: ' + beam.collimator_angle + '° &nbsp; Couch: ' + beam.couch_angle + '° &nbsp; MU-weight: ' + (cp.meterset_weight||0).toFixed(4);
  }
  var lbl = document.getElementById("mlcCPLabel_" + pid);
  if (lbl) lbl.textContent = st.cpIdx + " / " + (beam.control_points.length-1);

  var W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#090e15"; ctx.fillRect(0, 0, W, H);

  var leafA = cp.leaf_a || [], leafB = cp.leaf_b || [];
  var bounds = beam.leaf_boundaries;
  var n = Math.max(leafA.length, leafB.length, (bounds.length > 0 ? bounds.length - 1 : 0));
  if (n === 0) {
    ctx.fillStyle = "#555"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("No MLC data for this control point", W/2, H/2);
    return;
  }

  // Coordinate system: X axis = leaf direction, Y axis = leaf pair index
  var PAD = {top:10, right:10, bottom:20, left:10};
  var pw = W - PAD.left - PAD.right;
  var ph = H - PAD.top - PAD.bottom;

  // X range from jaws or leaf positions
  var xmin = cp.jaw_x1 !== null ? cp.jaw_x1 : -120;
  var xmax = cp.jaw_x2 !== null ? cp.jaw_x2 : 120;
  var xspan = xmax - xmin;
  // extend a bit
  xmin -= 10; xmax += 10; xspan = xmax - xmin;

  function toX(mm) { return PAD.left + (mm - xmin) / xspan * pw; }
  function toY(idx) { return PAD.top + (idx / n) * ph; }
  var rowH = ph / n;

  // Draw closed MLC (gray background first)
  ctx.fillStyle = "#2a0000";
  ctx.fillRect(PAD.left, PAD.top, pw, ph);

  // Draw open aperture per leaf pair
  for (var i = 0; i < n; i++) {
    var la = leafA[i] !== undefined ? leafA[i] : xmin;
    var lb = leafB[i] !== undefined ? leafB[i] : xmax;
    var yTop = toY(i);
    // open area
    if (lb > la) {
      var x0 = toX(la), x1 = toX(lb);
      ctx.fillStyle = "rgba(120,200,120,0.25)";
      ctx.fillRect(x0, yTop+0.5, x1-x0, rowH-1);
    }
    // leaf edges
    ctx.strokeStyle = "#4a6a4a"; ctx.lineWidth = 0.5;
    ctx.strokeRect(PAD.left, yTop, pw, rowH);
  }

  // Draw closed leaf blocks
  for (var i = 0; i < n; i++) {
    var la = leafA[i] !== undefined ? leafA[i] : xmin;
    var lb = leafB[i] !== undefined ? leafB[i] : xmax;
    var yTop = toY(i);
    // Bank A (left block, xmin to la)
    ctx.fillStyle = "#5a2020";
    ctx.fillRect(PAD.left, yTop+0.5, toX(la)-PAD.left, rowH-1);
    // Bank B (right block, lb to xmax)
    ctx.fillRect(toX(lb), yTop+0.5, PAD.left+pw-toX(lb), rowH-1);
  }

  // Jaw boundaries
  if (cp.jaw_x1 !== null) {
    ctx.strokeStyle = "#ffdd55"; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
    var jx1 = toX(cp.jaw_x1);
    ctx.beginPath(); ctx.moveTo(jx1, PAD.top); ctx.lineTo(jx1, PAD.top+ph); ctx.stroke();
  }
  if (cp.jaw_x2 !== null) {
    var jx2 = toX(cp.jaw_x2);
    ctx.beginPath(); ctx.moveTo(jx2, PAD.top); ctx.lineTo(jx2, PAD.top+ph); ctx.stroke();
  }
  ctx.setLineDash([]);

  // Axis labels
  ctx.fillStyle = "#666"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("X [mm]", PAD.left + pw/2, H-4);
  ctx.fillStyle = "#444"; ctx.textAlign = "center";
  ctx.fillText(xmin.toFixed(0), PAD.left, H-4);
  ctx.fillText(xmax.toFixed(0), PAD.left+pw, H-4);
  ctx.fillText("0", toX(0), H-4);
}

// ============================================================
// Dose Views tab
// ============================================================
function buildViewsTab(p) {
  var structs = p.structures.filter(function(s){
    return Object.keys(s.views_a||{}).length > 0 || Object.keys(s.views_b||{}).length > 0;
  });
  if (!structs.length) return '<p class="no-data">No dose view images available. Enable ENABLE_VIEWS in config.py and re-run.</p>';
  var html = '<div class="views-grid">';
  structs.forEach(function(s){
    html += '<div class="view-card">';
    html += '<h4>' + s.new_name + '</h4>';
    var orients = ["axial","coronal","sagittal"];
    orients.forEach(function(or){
      var va = (s.views_a||{})[or];
      var vb = (s.views_b||{})[or];
      if (!va && !vb) return;
      html += '<div class="orient-label">' + or.charAt(0).toUpperCase()+or.slice(1) + '</div>';
      html += '<div class="view-orient"><span>&#9312; ' + p.plan_a + '</span><span>&#9313; ' + p.plan_b + '</span></div>';
      html += '<div class="view-row-imgs">';
      html += va ? '<img src="' + va + '" alt="' + or + ' A">' : '<div style="width:50%;background:#1a1a1a;border-radius:4px;display:flex;align-items:center;justify-content:center;min-height:60px"><span style="color:#333;font-size:10px">n/a</span></div>';
      html += vb ? '<img src="' + vb + '" alt="' + or + ' B">' : '<div style="width:50%;background:#1a1a1a;border-radius:4px;display:flex;align-items:center;justify-content:center;min-height:60px"><span style="color:#333;font-size:10px">n/a</span></div>';
      html += '</div>';
    });
    html += '</div>';
  });
  html += '</div>';
  return html;
}
</script>
</body>
</html>"""


def _build_html(patients_data: List[Dict], report_dir: str) -> str:
    metric_cols_js = json.dumps([[c, l] for c, l in METRIC_COLS])

    # Serialize data – replace non-JSON-safe numpy types
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    data_json = json.dumps({"patients": _clean(patients_data)}, ensure_ascii=False)

    html = _HTML_TEMPLATE
    html = html.replace("PLACEHOLDER_DATA", data_json)
    html = html.replace("PLACEHOLDER_METRIC_COLS", metric_cols_js)
    return html


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_comparison_report(
    metrics_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    plan_files_list: List[PlanFiles],
    output_path: str,
    views_map: Optional[dict] = None,
) -> None:
    """
    Generate compare_report.html.  Safe to call even when no pairs exist.
    """
    if views_map is None:
        views_map = {}

    report_dir = os.path.dirname(output_path)
    os.makedirs(report_dir, exist_ok=True)

    pairs = find_comparison_pairs(metrics_df)
    if not pairs:
        logger.info("Comparison: no HA+MM pairs found in metrics – skipping report.")
        return

    patients_data: List[Dict] = []
    for pair in pairs:
        logger.info("Comparison: processing %s (%s vs %s)",
                    pair["patient_folder"], pair["plan_a"], pair["plan_b"])
        try:
            data = _gather_patient_data(
                patient=pair["patient_folder"],
                plan_a=pair["plan_a"],
                plan_b=pair["plan_b"],
                metrics_df=metrics_df,
                plan_files_list=plan_files_list,
                views_map=views_map,
                report_dir=report_dir,
            )
            patients_data.append(data)
        except Exception as exc:
            logger.error("Comparison: error processing %s: %s",
                         pair["patient_folder"], exc)

    if not patients_data:
        logger.info("Comparison: no data gathered – report not written.")
        return

    html = _build_html(patients_data, report_dir)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Comparison report written: %s", output_path)
    except Exception as exc:
        logger.error("Comparison: failed to write report: %s", exc)
