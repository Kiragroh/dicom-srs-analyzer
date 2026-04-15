"""
SRS Analysis Pipeline - HTML Report Generator
===============================================
Produces a self-contained HTML report with:
  - Main overview table (all patients / plans / scenarios)
  - Per-patient sections with per-PTV metric details
  - Scenario comparison (nominal vs. shift scenarios)
  - Excluded-structure documentation section
  - Simple colour-coded bar / cell highlights for key metrics
"""

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import HTML_REPORT_PATH, OUTPUT_DIR, ANONYMIZE_OUTPUT
from anonymization import anonymize_patient_id, anonymize_structure_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour thresholds for metric highlighting
# (values chosen as common SRS quality benchmarks – adjust as needed)
# ---------------------------------------------------------------------------
_THRESHOLDS = {
    "Coverage_pct":  {"green": 95.0,   "yellow": 90.0,  "higher_is_better": True},
    "PaddickCI":     {"green": 0.70,   "yellow": 0.50,  "higher_is_better": True},
    "RTOG_CI":       {"green": 2.0,    "yellow": 2.5,   "higher_is_better": False},
    "HI":            {"green": 0.30,   "yellow": 0.50,  "higher_is_better": False},
    "GI":            {"green": 3.5,    "yellow": 5.0,   "higher_is_better": False},
}


def _cell_colour(col: str, value) -> str:
    if col not in _THRESHOLDS or value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    cfg = _THRESHOLDS[col]
    g, y = cfg["green"], cfg["yellow"]
    hib  = cfg["higher_is_better"]
    if hib:
        if value >= g:  return "background:#2d5a2d; color:#aaffaa;"
        if value >= y:  return "background:#5a4a2d; color:#ffdd88;"
        return "background:#5a2d2d; color:#ffaaaa;"
    else:
        if value <= g:  return "background:#2d5a2d; color:#aaffaa;"
        if value <= y:  return "background:#5a4a2d; color:#ffdd88;"
        return "background:#5a2d2d; color:#ffaaaa;"


def _fmt(value, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmu(value, decimals: int, uncertain: bool) -> str:
    """Format a metric value, prepending * if the result is flagged uncertain."""
    s = _fmt(value, decimals)
    if uncertain and s != "-":
        return f'<span class="uncertain">*</span>{s}'
    return s


def _da(s: str) -> str:
    """Escape a string for use as an HTML data-attribute value."""
    return s.replace('&', '&amp;').replace('"', '&quot;').replace("'", '&#39;')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    metrics_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    path: str = HTML_REPORT_PATH,
    views_map: Optional[Dict[Tuple[str, str, str, str], Dict[str, str]]] = None,
) -> None:
    """
    Build and write the HTML report.

    *metrics_df*  – result of metrics.results_to_dataframe()
    *mapping_df*  – the full PTV mapping table (including excluded rows)
    *views_map*   – optional dict keyed by (patient_folder, plan_type,
                    struct_original_name, scenario) →
                    {'axial': abs_path, 'coronal': abs_path, 'sagittal': abs_path}.
    """
    if views_map is None:
        views_map = {}
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Build JS-safe views dict: key = "pat||plan||struct||scenario", values = relative paths
    report_dir = os.path.dirname(os.path.abspath(path))
    views_js: Dict[str, Dict[str, str]] = {}
    for (pat, plan, struct, sc), vpaths in views_map.items():
        key = f"{pat}||{plan}||{struct}||{sc}"
        views_js[key] = {
            vname: os.path.relpath(vpath, report_dir).replace("\\", "/")
            for vname, vpath in vpaths.items()
            if os.path.isfile(vpath)
        }
    views_json = json.dumps(views_js, ensure_ascii=False)

    excluded_df = mapping_df[
        mapping_df["ExcludeFromAnalysis"].astype(str).str.strip().str.lower()
        .isin({"true", "1", "ja", "yes", "x"})
    ]

    html = _build_html(metrics_df, excluded_df, views_json, path)

    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("HTML report written to %s", path)
    except Exception as exc:
        logger.error("Failed to write HTML report: %s", exc)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _build_html(
    metrics_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    views_json: str,
    report_path: str,
) -> str:
    sections = []

    # --- Overview ---
    sections.append(_section_overview(metrics_df))

    # --- Plan summary (means per patient/plan) ---
    sections.append(_section_plan_summary(metrics_df))

    # --- Per-patient ---
    if not metrics_df.empty:
        for patient in sorted(metrics_df["PatientFolder"].unique()):
            pat_df = metrics_df[metrics_df["PatientFolder"] == patient]
            sections.append(_section_patient(patient, pat_df))

    # --- Excluded structures ---
    if not excluded_df.empty:
        sections.append(_section_excluded(excluded_df))

    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SRS Analysis Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 16px; background:#2d2d2d; color:#e0e0e0; font-size:13px; }}
  h1   {{ color:#61dafb; border-bottom:2px solid #555; padding-bottom:6px; }}
  h2   {{ color:#aaddff; margin-top:2em; border-bottom:1px solid #555; padding-bottom:4px; }}
  h3   {{ color:#cce8ff; }}
  table {{ border-collapse:collapse; width:100%; margin-bottom:1.5em; font-size:12px; }}
  th   {{ background:#3c3c3c; color:#ccc; padding:5px 8px; border:1px solid #555;
         text-align:left; white-space:nowrap; }}
  td   {{ padding:4px 8px; border:1px solid #444; white-space:nowrap; }}
  tr:nth-child(even) {{ background:#333; }}
  tr:hover {{ background:#3a4a5a; }}
  .scenario-nominal {{ background:#1e3d1e; }}
  .scenario-shift   {{ background:#3d1e1e; }}
  .badge-ha  {{ background:#1a4a7a; color:#9dcfff; padding:2px 6px; border-radius:3px; }}
  .badge-mm  {{ background:#4a1a4a; color:#d9a0ff; padding:2px 6px; border-radius:3px; }}
  .section   {{ background:#3c3c3c; border-radius:6px; padding:12px 16px;
                margin-bottom:1.5em; box-shadow:0 2px 6px rgba(0,0,0,0.4); }}
  details    {{ margin-bottom:0.8em; }}
  summary    {{ cursor:pointer; font-weight:bold; color:#aaddff; padding:4px 0; }}
  .excluded-row {{ background:#4a3030; color:#ffaaaa; }}
  .meta-note {{ color:#888; font-size:11px; margin-bottom:0.5em; }}
  tr[data-struct] {{ cursor:pointer; }}
  tr.row-selected {{ outline:2px solid #61dafb !important; }}
  .uncertain {{ color:#ffaa44; font-size:0.72em; vertical-align:super;
               font-weight:bold; font-style:normal; line-height:1; }}
  .mean-row td {{ background:#1e3050 !important; color:#aaccff; font-style:italic; }}
  .mean-row td:first-child {{ color:#88aadd; font-weight:bold; }}
  .uncertain-badge {{ color:#ffaa44; font-size:10px; font-weight:bold; margin-left:6px; }}
  .bridging-badge {{ color:#ff6688; font-size:10px; font-weight:bold; margin-left:6px;
                     background:#3a1a22; border:1px solid #ff6688; border-radius:3px; padding:1px 4px; }}
  .ci_low {{ background-color: #ffcccc; }}
  .gi_high {{ background-color: #ffdddd; }}
  #view-panel {{
    display:none; position:fixed; top:0; right:0;
    width:300px; height:100vh; background:#1a1a2e;
    border-left:2px solid #61dafb; padding:14px 12px;
    overflow-y:auto; z-index:1000;
    box-shadow:-6px 0 24px rgba(0,0,0,0.6);
  }}
  #view-panel h4 {{ color:#61dafb; margin:0 0 8px 0; font-size:13px;
                   word-break:break-word; padding-right:20px; }}
  #view-panel-close {{
    position:absolute; top:10px; right:12px;
    cursor:pointer; color:#aaa; font-size:18px; line-height:1;
  }}
  #view-panel-close:hover {{ color:#61dafb; }}
  .view-label {{ color:#888; font-size:10px; margin:6px 0 2px 0; }}
  #view-panel img {{ width:100%; border-radius:4px; display:block; }}
  .legend-box {{
    background:#1e2a1e; border:1px solid #4a8a4a; border-radius:6px;
    padding:10px 16px; margin-bottom:1.2em; font-size:12px; color:#ccc;
    display:flex; gap:24px; flex-wrap:wrap; align-items:flex-start;
  }}
  .legend-box h4 {{ margin:0 0 6px 0; color:#aaddaa; font-size:12px; letter-spacing:.05em; }}
  .legend-item {{ display:flex; gap:6px; align-items:baseline; }}
  .legend-star {{ color:#ffaa44; font-style:italic; font-weight:bold; font-size:14px; }}
</style>
</head>
<body>
<h1>SRS Analysis Report</h1>
<p class="meta-note">Generated by srs_analysis pipeline &nbsp;|&nbsp;
Click any table row to view orthogonal dose images</p>
<div class="legend-box">
  <div>
    <h4>LEGEND</h4>
    <div class="legend-item">
      <span class="legend-star">*</span>
      <span><strong>Uncertain metric</strong> &ndash; the relevant isodose was not fully enclosed within the sampling region. Reported value may be underestimated.<br>
      <em>Paddick CI / RTOG CI</em>: 100&nbsp;%&nbsp;Rx isodose not fully contained &rarr; PIV may be underestimated.<br>
      <em>GI</em>: 50&nbsp;%&nbsp;Rx isodose not fully contained &rarr; V&frac12;Rx may be underestimated.</span>
    </div>
    <div class="legend-item" style="margin-top:8px">
      <span class="bridging-badge" style="font-size:11px">bridge</span>
      <span><strong>Dose bridging suspected</strong> &ndash; Paddick CI &lt;&nbsp;0.8 and PIV grows by &gt;&nbsp;15&nbsp;% when the sampling radius increases from 5&nbsp;mm to 10&nbsp;mm. A neighbouring high-dose region is likely being captured &rarr; PIV / CI may be artificially inflated.</span>
    </div>
    <div class="legend-item" style="margin-top:8px">
      <span class="ci_low" style="padding:1px 6px;border-radius:3px">low CI</span>
      <span><strong>Low Conformity Index</strong> &ndash; Paddick CI &lt;&nbsp;0.7.</span>
    </div>
    <div class="legend-item" style="margin-top:4px">
      <span class="gi_high" style="padding:1px 6px;border-radius:3px">high GI</span>
      <span><strong>High Gradient Index</strong> &ndash; GI &gt;&nbsp;5.0.</span>
    </div>
  </div>
  <div>
    <h4>COLOUR CODE</h4>
    <div class="legend-item"><span style="background:#2d5a2d;color:#aaffaa;padding:1px 6px;border-radius:3px">green</span>&nbsp;within target range</div>
    <div class="legend-item" style="margin-top:4px"><span style="background:#5a4a2d;color:#ffdd88;padding:1px 6px;border-radius:3px">yellow</span>&nbsp;borderline</div>
    <div class="legend-item" style="margin-top:4px"><span style="background:#5a2d2d;color:#ffaaaa;padding:1px 6px;border-radius:3px">red</span>&nbsp;outside target range</div>
  </div>
</div>
{body}
<div id="view-panel">
  <span id="view-panel-close" title="Close">&times;</span>
  <h4 id="view-panel-title"></h4>
  <div id="view-panel-imgs"></div>
</div>
<script>
(function(){{
  const V={views_json};
  function showPanel(pat,plan,struct,sc){{
    const key=pat+'||'+plan+'||'+struct+'||'+sc;
    const v=V[key];
    const panel=document.getElementById('view-panel');
    if(!v||(!v.axial&&!v.coronal&&!v.sagittal)){{panel.style.display='none';return;}}
    document.getElementById('view-panel-title').textContent=struct+' \u2013 '+sc;
    let h='';
    if(v.axial)   h+='<div class="view-label">Axial</div><img src="'+v.axial+'" loading="lazy">';
    if(v.coronal) h+='<div class="view-label">Coronal</div><img src="'+v.coronal+'" loading="lazy">';
    if(v.sagittal)h+='<div class="view-label">Sagittal</div><img src="'+v.sagittal+'" loading="lazy">';
    document.getElementById('view-panel-imgs').innerHTML=h;
    panel.style.display='block';
  }}
  document.addEventListener('DOMContentLoaded',function(){{
    document.querySelectorAll('tr[data-struct]').forEach(function(tr){{
      tr.addEventListener('click',function(){{
        document.querySelectorAll('tr.row-selected').forEach(function(r){{r.classList.remove('row-selected');}});
        tr.classList.add('row-selected');
        showPanel(tr.dataset.patient,tr.dataset.plan,tr.dataset.struct,tr.dataset.scenario);
      }});
    }});
    document.getElementById('view-panel-close').addEventListener('click',function(){{
      document.getElementById('view-panel').style.display='none';
      document.querySelectorAll('tr.row-selected').forEach(function(r){{r.classList.remove('row-selected');}});
    }});
  }});
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _plan_means_row(sub_df: pd.DataFrame, label: str) -> str:
    """One mean row for a group of nominal rows; CI/GI means exclude uncertain flags."""
    nom = sub_df[sub_df["Scenario"] == "nominal"] if "Scenario" in sub_df.columns else sub_df
    if nom.empty:
        return ""

    def smean(series, mask=None):
        s = pd.to_numeric(series if mask is None else series[mask], errors="coerce").dropna()
        return s.mean() if len(s) else float("nan")

    ci_ok = ~nom.get("CI_uncertain", pd.Series([False]*len(nom), index=nom.index)).astype(bool)
    gi_ok = ~nom.get("GI_uncertain", pd.Series([False]*len(nom), index=nom.index)).astype(bool)
    n  = len(nom)
    nc = ci_ok.sum()
    ng = gi_ok.sum()
    cn = f" (n={nc}/{n})" if nc < n else ""
    gn = f" (n={ng}/{n})" if ng < n else ""

    return (
        f'<tr class="mean-row">'
        f"<td colspan='3' style='font-weight:bold'>{label} &#x2205; (n={n}, excl. uncertain)</td>"
        f"<td>{_fmt(smean(nom['TV_cc']),2)}</td>"
        f"<td>-</td>"
        f"<td>{_fmt(smean(nom['Coverage_pct']),1)}</td>"
        f"<td>{_fmt(smean(nom['PaddickCI'],ci_ok),3)}{cn}</td>"
        f"<td>{_fmt(smean(nom['RTOG_CI'],ci_ok),2)}{cn}</td>"
        f"<td>{_fmt(smean(nom['HI']),3)}</td>"
        f"<td>{_fmt(smean(nom['GI'],gi_ok),2)}{gn}</td>"
        f"<td>{_fmt(smean(nom['Dmax_Gy']),2)}</td>"
        f"<td>{_fmt(smean(nom['PIV_cc']),3)}</td>"
        f"<td>{_fmt(smean(nom['V12Gy_cc']),3)}</td>"
        f"<td></td>"
        f"</tr>"
    )


def _section_plan_summary(df: pd.DataFrame) -> str:
    """Standalone section: one mean row per patient x plan (nominal only, excl. uncertain)."""
    nominal = df[df["Scenario"] == "nominal"] if not df.empty else df
    if nominal.empty:
        return ""

    rows_html = []
    for patient in sorted(nominal["PatientFolder"].unique()):
        for plan_type in sorted(nominal[nominal["PatientFolder"] == patient]["PlanType"].unique()):
            sub = nominal[(nominal["PatientFolder"] == patient) & (nominal["PlanType"] == plan_type)]
            rows_html.append(_plan_means_row(sub, f"{patient} / {plan_type}"))

    if not any(rows_html):
        return ""

    return f"""<div class="section">
<h2>Plan Summary (nominal means, uncertain values excluded)</h2>
<p class="meta-note">One row per patient/plan. CI and GI means exclude structures marked uncertain (*).
 n=used/total shown in parentheses.</p>
<table>
<thead><tr>
  <th colspan="3">Patient / Plan</th>
  <th>TV (cc)</th><th>Dist</th><th>Coverage (%)</th><th>Paddick CI</th><th>RTOG CI</th>
  <th>HI</th><th>GI</th><th>Dmax (Gy)</th><th>PIV (cc)</th><th>V12Gy (cc)</th><th></th>
</tr></thead>
<tbody>{chr(10).join(r for r in rows_html if r)}</tbody>
</table>
</div>"""


def _section_overview(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div class="section"><h2>Overview</h2><p>No metric results available.</p></div>'

    nominal = df[df["Scenario"] == "nominal"]
    rows_html = []
    for _, r in nominal.sort_values(["PatientFolder", "PlanType", "NewStructureName"]).iterrows():
        badge_cls = "badge-ha" if str(r.get("PlanType", "")).upper() == "HA" else "badge-mm"
        pat   = _da(str(r.get('PatientFolder', '')))
        plan  = _da(str(r.get('PlanType', '')))
        orig  = _da(str(r.get('StructureName_Original', '')))
        ci_u  = bool(r.get('CI_uncertain', False))
        gi_u  = bool(r.get('GI_uncertain', False))
        br_u  = bool(r.get('Bridging_suspected', False))
        ci_l  = bool(r.get('CI_low', False))
        gi_h  = bool(r.get('GI_high', False))
        
        # Apply anonymization if enabled
        display_patient = anonymize_patient_id(r['PatientFolder']) if ANONYMIZE_OUTPUT else r['PatientFolder']
        display_struct = anonymize_structure_name(r.get('NewStructureName','') or r.get('StructureName_Original','')) if ANONYMIZE_OUTPUT else (r.get('NewStructureName','') or r.get('StructureName_Original',''))
        
        bridge_span = '<span class="bridging-badge">bridge</span>' if br_u else ''
        ci_span = '<span class="ci_low">low CI</span>' if ci_l else ''
        gi_span = '<span class="gi_high">high GI</span>' if gi_h else ''
        rows_html.append(
            f'<tr data-patient="{pat}" data-plan="{plan}" '
            f'data-struct="{orig}" data-scenario="nominal">'
            f"<td>{display_patient}</td>"
            f"<td><span class='{badge_cls}'>{r['PlanType']}</span></td>"
            f"<td>{display_struct}</td>"
            f"<td>{_fmt(r.get('TV_cc'),3)}</td>"
            f"<td>{_fmt(r.get('DistToIso_mm'),1)}</td>"
            f"<td style='{_cell_colour('Coverage_pct', _safe_float(r.get('Coverage_pct')))}'>{_fmt(r.get('Coverage_pct'),1)}</td>"
            f"<td style='{_cell_colour('PaddickCI', _safe_float(r.get('PaddickCI')))}'>{_fmu(r.get('PaddickCI'),3,ci_u)}{bridge_span}{ci_span}</td>"
            f"<td style='{_cell_colour('RTOG_CI', _safe_float(r.get('RTOG_CI')))}'>{_fmu(r.get('RTOG_CI'),3,ci_u)}{ci_span}</td>"
            f"<td style='{_cell_colour('HI', _safe_float(r.get('HI')))}'>{_fmt(r.get('HI'),3)}</td>"
            f"<td style='{_cell_colour('GI', _safe_float(r.get('GI')))}'>{_fmu(r.get('GI'),2,gi_u)}{gi_span}</td>"
            f"<td>{_fmt(r.get('Dmax_Gy'),2)}</td>"
            f"<td>{_fmt(r.get('PIV_cc'),3)}</td>"
            f"<td>{_fmt(r.get('V12Gy_cc'),3)}</td>"
            f"<td>{r.get('Error','')}</td>"
            f"</tr>"
        )

    # Mean rows per plan type (excluding uncertain)
    mean_rows = []
    for pt in sorted(nominal["PlanType"].unique()):
        mean_rows.append(_plan_means_row(nominal[nominal["PlanType"] == pt], pt))

    return f"""<div class="section">
<h2>Overview – Nominal Plans</h2>
<table>
<thead><tr>
  <th>Patient</th><th>Plan</th><th>Structure</th>
  <th>TV (cc)</th><th>Dist Iso (mm)</th><th>Coverage (%)</th><th>Paddick CI</th><th>RTOG CI</th>
  <th>HI</th><th>GI</th><th>Dmax (Gy)</th><th>PIV (cc)</th><th>V12Gy (cc)</th>
  <th>Note</th>
</tr></thead>
<tbody>{chr(10).join(rows_html)}{chr(10).join(mean_rows)}</tbody>
</table>
</div>"""


def _section_patient(patient: str, pat_df: pd.DataFrame) -> str:
    sub_sections = []
    for plan_type in sorted(pat_df["PlanType"].unique()):
        plan_df = pat_df[pat_df["PlanType"] == plan_type]
        sub_sections.append(_subsection_plan(plan_type, plan_df, patient))

    return f"""<div class="section">
<h2>Patient: {patient}</h2>
{"".join(sub_sections)}
</div>"""


def _subsection_plan(plan_type: str, plan_df: pd.DataFrame, patient: str) -> str:
    structure_sections = []

    for struct in sorted(plan_df["StructureName_Original"].unique()):
        struct_df = plan_df[plan_df["StructureName_Original"] == struct]
        rows_html = []
        pat_da   = _da(patient)
        plan_da  = _da(plan_type)
        orig_da  = _da(struct)
        
        # Get display name for structure (anonymized if enabled)
        display_struct = anonymize_structure_name(struct) if ANONYMIZE_OUTPUT else struct
        
        for _, r in struct_df.iterrows():
            sc = str(r.get("Scenario", "nominal"))
            row_cls = "scenario-nominal" if sc == "nominal" else "scenario-shift"
            sc_da   = _da(sc)
            ci_u = bool(r.get('CI_uncertain', False))
            gi_u = bool(r.get('GI_uncertain', False))
            br_u = bool(r.get('Bridging_suspected', False))
            ci_l = bool(r.get('CI_low', False))
            gi_h = bool(r.get('GI_high', False))
            bridge_span2 = '<span class="bridging-badge">bridge</span>' if br_u else ''
            ci_span2 = '<span class="ci_low">low CI</span>' if ci_l else ''
            gi_span2 = '<span class="gi_high">high GI</span>' if gi_h else ''
            rows_html.append(
                f'<tr class="{row_cls}" data-patient="{pat_da}" data-plan="{plan_da}" '
                f'data-struct="{orig_da}" data-scenario="{sc_da}">'
                f"<td>{sc}</td>"
                f"<td>{_fmt(r.get('TV_cc'),3)}</td>"
                f"<td>{_fmt(r.get('DistToIso_mm'),1)}</td>"
                f"<td style='{_cell_colour('Coverage_pct', _safe_float(r.get('Coverage_pct')))}'>{_fmt(r.get('Coverage_pct'),1)}</td>"
                f"<td style='{_cell_colour('PaddickCI', _safe_float(r.get('PaddickCI')))}'>{_fmu(r.get('PaddickCI'),3,ci_u)}{bridge_span2}{ci_span2}</td>"
                f"<td style='{_cell_colour('RTOG_CI', _safe_float(r.get('RTOG_CI')))}'>{_fmu(r.get('RTOG_CI'),3,ci_u)}{ci_span2}</td>"
                f"<td style='{_cell_colour('HI', _safe_float(r.get('HI')))}'>{_fmt(r.get('HI'),3)}</td>"
                f"<td style='{_cell_colour('GI', _safe_float(r.get('GI')))}'>{_fmu(r.get('GI'),2,gi_u)}{gi_span2}</td>"
                f"<td>{_fmt(r.get('Dmax_Gy'),2)}</td>"
                f"<td>{_fmt(r.get('PIV_cc'),3)}</td>"
                f"<td>{_fmt(r.get('V12Gy_cc'),3)}</td>"
                f"<td>{_fmt(r.get('D98_Gy'),2)}</td>"
                f"<td>{_fmt(r.get('D2_Gy'),2)}</td>"
                f"<td>{r.get('Error','')}</td>"
                f"</tr>"
            )
        new_name = struct_df["NewStructureName"].iloc[0] if not struct_df.empty else ""
        label = f"{new_name} ({display_struct})" if new_name else display_struct
        n_ci_u = struct_df.get("CI_uncertain", pd.Series([False]*len(struct_df), index=struct_df.index)).astype(bool).sum()
        n_gi_u = struct_df.get("GI_uncertain", pd.Series([False]*len(struct_df), index=struct_df.index)).astype(bool).sum()
        n_br   = struct_df.get("Bridging_suspected", pd.Series([False]*len(struct_df), index=struct_df.index)).astype(bool).sum()
        n_ci_l = struct_df.get("CI_low", pd.Series([False]*len(struct_df), index=struct_df.index)).astype(bool).sum()
        n_gi_h = struct_df.get("GI_high", pd.Series([False]*len(struct_df), index=struct_df.index)).astype(bool).sum()
        unc_hint = ""
        if n_ci_u: unc_hint += f' <span class="uncertain-badge">* CI ({n_ci_u}x)</span>'
        if n_gi_u: unc_hint += f' <span class="uncertain-badge">* GI ({n_gi_u}x)</span>'
        if n_br:   unc_hint += f' <span class="bridging-badge">bridge ({n_br}x)</span>'

        structure_sections.append(f"""
<details>
<summary>{label}{unc_hint} <small style='color:#888;font-weight:normal'>(click row to view images)</small></summary>
<table>
<thead><tr>
  <th>Scenario</th><th>TV (cc)</th><th>Dist Iso (mm)</th><th>Coverage (%)</th>
  <th>Paddick CI</th><th>RTOG CI</th><th>HI</th><th>GI</th>
  <th>Dmax (Gy)</th><th>PIV (cc)</th><th>V12Gy (cc)</th>
  <th>D98 (Gy)</th><th>D2 (Gy)</th><th>Note</th>
</tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
</details>""")

    badge_cls = "badge-ha" if plan_type.upper() == "HA" else "badge-mm"
    mean_html = _plan_means_row(plan_df, plan_type)
    if mean_html:
        mean_block = (
            '<table style="font-size:11px;margin-bottom:1em">'
            '<thead><tr><th colspan="3">Mean (nominal, excl. uncertain)</th>'
            '<th>TV (cc)</th><th>Dist</th>'
            '<th>Cov (%)</th><th>Paddick CI</th><th>RTOG CI</th><th>HI</th><th>GI</th>'
            '<th>Dmax</th><th>PIV</th><th>V12Gy</th><th></th></tr></thead>'
            f'<tbody>{mean_html}</tbody></table>'
        )
    else:
        mean_block = ""
    return f"""<h3>Plan: <span class="{badge_cls}">{plan_type}</span></h3>
{mean_block}{chr(10).join(structure_sections)}"""


def _section_excluded(excluded_df: pd.DataFrame) -> str:
    rows_html = []
    for _, r in excluded_df.iterrows():
        rows_html.append(
            f"<tr class='excluded-row'>"
            f"<td>{anonymize_patient_id(r.get('PatientFolder','')) if ANONYMIZE_OUTPUT else r.get('PatientFolder','')}</td>"
            f"<td>{r.get('PlanType','')}</td>"
            f"<td>{anonymize_structure_name(r.get('StructureName_Original','')) if ANONYMIZE_OUTPUT else r.get('StructureName_Original','')}"
            f"<td>{r.get('ExcludeReason','')}</td>"
            f"<td>{r.get('Comment','')}</td>"
            f"</tr>"
        )

    return f"""<div class="section">
<h2>Excluded PTV Candidates (documentation only)</h2>
<p class="meta-note">These structures were marked as excluded in the mapping table
and are NOT included in any metric calculations or aggregations.</p>
<table>
<thead><tr>
  <th>Patient</th><th>Plan</th><th>Structure</th>
  <th>Exclude Reason</th><th>Comment</th>
</tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
</div>"""


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
