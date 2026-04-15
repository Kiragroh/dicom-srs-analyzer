"""
plot_3d_html.py – Interactive 3-D HTML Viewer per Patient / Plan
=================================================================

Generates one self-contained HTML file per patient/plan.
All scenario meshes are embedded as JSON.

Features:
  • Scenario tabs (nominal + all shifted)
  • Plotly Mesh3d with proper per-vertex Lambertian lighting
  • Camera preset buttons (Front-L, Front-R, Top, Side)
  • Anatomical orientation cube synced to main view
  • Consistent axis limits across all scenarios
  • Dark theme, fully offline-capable (only Plotly CDN required)
"""

import json
import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly trace builders
# ─────────────────────────────────────────────────────────────────────────────

def _mesh_trace(verts: Optional[np.ndarray], faces: Optional[np.ndarray],
                color: str, name: str, alpha: float = 1.0,
                hovertext: Optional[str] = None,
                layer: Optional[str] = None) -> Optional[dict]:
    """
    Build a Plotly Mesh3d trace.
    alpha=1.0 → fully opaque (WebGL depth-buffer works correctly, no artifacts).
    alpha<1.0 → semi-transparent (painter's algorithm, may have sorting glitches
                but acceptable for the "background" layer that peeks out).
    hovertext → custom tooltip (if None, uses name only).
    layer → 'struct', 'nominal', or 'shifted' for visibility toggle.
    """
    if verts is None or faces is None or len(faces) == 0:
        return None
    
    trace = {
        "type": "mesh3d",
        "x": [round(float(v), 3) for v in verts[:, 0]],
        "y": [round(float(v), 3) for v in verts[:, 1]],
        "z": [round(float(v), 3) for v in verts[:, 2]],
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "color": color,
        "opacity": alpha,
        "flatshading": False,
        "lighting": {
            "ambient":   0.40,
            "diffuse":   0.70,
            "roughness": 0.45,
            "specular":  0.25,
            "fresnel":   0.10,
        },
        "lightposition": {"x": -300, "y": -200, "z": 500},
        "name": name,
        "showscale": False,
    }
    
    if layer:
        trace["layer"] = layer
    
    if hovertext:
        trace["hovertemplate"] = hovertext + "<extra></extra>"
    else:
        trace["hoverinfo"] = "name"
    
    return trace


def _isocenter_trace(iso: np.ndarray) -> dict:
    return {
        "type": "scatter3d",
        "x": [round(float(iso[0]), 3)],
        "y": [round(float(iso[1]), 3)],
        "z": [round(float(iso[2]), 3)],
        "mode": "markers+text",
        "marker": {"size": 7, "color": "#ffffff", "symbol": "cross",
                   "line": {"color": "#aaaaaa", "width": 1}},
        "text": ["ISO"],
        "textfont": {"size": 10, "color": "#ffffff"},
        "textposition": "top center",
        "name": "Isocenter",
        "hoverinfo": "name+x+y+z",
        "showlegend": False,
    }


def _orientation_traces() -> list:
    """Six arrow traces for the orientation cube: ±X ±Y ±Z with L/R A/P S/I labels."""
    axes = [
        ([0, 1.2], [0, 0],   [0, 0],   "R (+X)", "#ff5555"),
        ([0,-1.2], [0, 0],   [0, 0],   "L (−X)", "#ff9999"),
        ([0, 0],   [0, 1.2], [0, 0],   "A (+Y)", "#55ff55"),
        ([0, 0],   [0,-1.2], [0, 0],   "P (−Y)", "#99ff99"),
        ([0, 0],   [0, 0],   [0, 1.2], "S (+Z)", "#5599ff"),
        ([0, 0],   [0, 0],   [0,-1.2], "I (−Z)", "#99bbff"),
    ]
    traces = []
    for xs, ys, zs, label, color in axes:
        short = label.split()[0]
        traces.append({
            "type": "scatter3d",
            "x": xs, "y": ys, "z": zs,
            "mode": "lines+text",
            "line": {"color": color, "width": 6},
            "text": ["", short],
            "textfont": {"size": 13, "color": color, "family": "Arial Black"},
            "textposition": "top center",
            "hovertext": label,
            "hoverinfo": "text",
            "showlegend": False,
        })
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# HTML template  (uses ###MARKER### placeholders, no Python format() needed)
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>###TITLE###</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body {
  background: #0d1117;
  color: #ccc;
  font-family: 'Segoe UI', system-ui, sans-serif;
  display: flex;
  flex-direction: column;
}
header {
  background: #161b22;
  border-bottom: 1px solid #30363d;
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
  min-height: 40px;
}
header h1 { font-size: 13px; font-weight: 600; color: #e6edf3; }
.badge {
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 12px;
  padding: 2px 9px;
  font-size: 10px;
  color: #8b949e;
  margin-left: auto;
}
.content {
  display: flex;
  flex: 1;
  min-height: 0;
}
/* ── Sidebar ────────────────────────────────────────────────────── */
.sidebar {
  width: 215px;
  min-width: 215px;
  background: #161b22;
  border-right: 1px solid #30363d;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: 10px 8px;
  gap: 14px;
}
.sidebar section { display: flex; flex-direction: column; gap: 5px; }
.sidebar h3 {
  font-size: 10px;
  font-weight: 600;
  color: #6e7681;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  padding-bottom: 2px;
  border-bottom: 1px solid #21262d;
}
.sc-btn {
  display: block;
  width: 100%;
  padding: 6px 9px;
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 5px;
  color: #ccc;
  font-size: 11px;
  text-align: left;
  cursor: pointer;
  transition: all 0.12s;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.sc-btn:hover  { background: #30363d; color: #e6edf3; }
.sc-btn.active { background: #1f6feb; border-color: #388bfd; color: #fff; font-weight: 600; }
.view-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
.vbtn {
  padding: 5px 4px;
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 5px;
  color: #ccc;
  font-size: 11px;
  cursor: pointer;
  text-align: center;
  transition: all 0.12s;
}
.vbtn:hover  { background: #30363d; color: #e6edf3; }
.vbtn.active { background: #1a7f37; border-color: #2ea043; color: #fff; }
#ori-plot { height: 165px; }
.legend-row { display: flex; align-items: center; gap: 8px; font-size: 11px; color: #ccc; }
.swatch { width: 16px; height: 11px; border-radius: 3px; flex-shrink: 0; }
/* ── Main view ──────────────────────────────────────────────────── */
.main-view {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.sc-title {
  background: #161b22;
  border-bottom: 1px solid #30363d;
  padding: 5px 14px;
  font-size: 11px;
  color: #8b949e;
  flex-shrink: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
#main-plot { flex: 1; min-height: 0; }
</style>
</head>
<body>

<header>
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#388bfd" stroke-width="2">
    <circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/>
  </svg>
  <h1>SRS 3-D Viewer — ###PLAN_LABEL###</h1>
  <span class="badge">###N_SCENARIOS### scenarios</span>
</header>

<div class="content">

  <div class="sidebar">

    <section>
      <h3>&#127776; Szenarien</h3>
      ###SCENARIO_BUTTONS###
    </section>

    <section>
      <h3>&#128247; Kamera</h3>
      <div class="view-grid">
        <button class="vbtn active" id="vbtn-fl"   onclick="setCamera('fl')">Front-L</button>
        <button class="vbtn"        id="vbtn-fr"   onclick="setCamera('fr')">Front-R</button>
        <button class="vbtn"        id="vbtn-top"  onclick="setCamera('top')">Top (↓)</button>
        <button class="vbtn"        id="vbtn-side" onclick="setCamera('side')">Side</button>
      </div>
    </section>

    <section>
      <h3>&#128065; Sichtbarkeit</h3>
      <div style="display:flex;flex-direction:column;gap:6px;font-size:11px;">
        <label style="display:flex;align-items:center;gap:7px;cursor:pointer;color:#ccc;">
          <input type="checkbox" id="vis-struct" checked onchange="toggleLayer('struct')">
          <span>Struktur (grau)</span>
        </label>
        <label style="display:flex;align-items:center;gap:7px;cursor:pointer;color:#ccc;">
          <input type="checkbox" id="vis-nominal" checked onchange="toggleLayer('nominal')">
          <span>Nominale Isodose (grün)</span>
        </label>
        <label style="display:flex;align-items:center;gap:7px;cursor:pointer;color:#ccc;">
          <input type="checkbox" id="vis-shifted" checked onchange="toggleLayer('shifted')">
          <span>Verschobene Isodose (magenta)</span>
        </label>
      </div>
    </section>

    <section>
      <h3>&#127759; Anatomische Orientierung</h3>
      <div id="ori-plot"></div>
      <div style="font-size:9px;color:#555;line-height:1.5;padding-top:3px;">
        R=Rechts&nbsp; L=Links<br>
        A=Anterior&nbsp; P=Posterior<br>
        S=Superior&nbsp; I=Inferior
      </div>
    </section>

    <section>
      <h3>&#9679; Legende</h3>
      ###LEGEND_HTML###
    </section>

    <section>
      <h3>&#9881; Auflösung</h3>
      <div style="font-size:10px;color:#6e7681;line-height:1.7;">
        ###MESH_INFO###
        <div style="margin-top:5px;color:#484f58;">
          Höhere Auflösung → <code>--resolution ultra</code>
        </div>
      </div>
    </section>

  </div>

  <div class="main-view">
    <div class="sc-title" id="sc-title">###FIRST_TITLE###</div>
    <div id="main-plot"></div>
  </div>

</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────
const SCENARIOS   = ###SCENARIOS_JSON###;
const AXIS_LIMITS = ###AXIS_LIMITS_JSON###;
const ORI_TRACES  = ###ORI_TRACES_JSON###;

// ── Camera presets ────────────────────────────────────────────────────────
const CAMERAS = {
  fl:   { eye: { x:-1.5, y:-1.5, z: 0.8 }, up: { x:0, y:0, z:1 }, center: { x:0,y:0,z:0 } },
  fr:   { eye: { x: 1.5, y:-1.5, z: 0.8 }, up: { x:0, y:0, z:1 }, center: { x:0,y:0,z:0 } },
  top:  { eye: { x: 0,   y:-0.1, z: 2.8 }, up: { x:0, y:1, z:0 }, center: { x:0,y:0,z:0 } },
  side: { eye: { x:-2.8, y: 0,   z: 0   }, up: { x:0, y:0, z:1 }, center: { x:0,y:0,z:0 } },
};

// ── Scene layout (shared, consistent axis limits) ─────────────────────────
const sceneBase = {
  xaxis: { range:[AXIS_LIMITS[0],AXIS_LIMITS[1]], showgrid:true, gridcolor:'#1c1c2c',
           title:{ text:'X (mm)', font:{color:'#555',size:10} },
           tickfont:{color:'#555',size:9}, backgroundcolor:'#0d1117',
           showbackground:true, zerolinecolor:'#2a2a3c' },
  yaxis: { range:[AXIS_LIMITS[2],AXIS_LIMITS[3]], showgrid:true, gridcolor:'#1c1c2c',
           title:{ text:'Y (mm)', font:{color:'#555',size:10} },
           tickfont:{color:'#555',size:9}, backgroundcolor:'#0d1117',
           showbackground:true, zerolinecolor:'#2a2a3c' },
  zaxis: { range:[AXIS_LIMITS[4],AXIS_LIMITS[5]], showgrid:true, gridcolor:'#1c1c2c',
           title:{ text:'Z (mm)', font:{color:'#555',size:10} },
           tickfont:{color:'#555',size:9}, backgroundcolor:'#0d1117',
           showbackground:true, zerolinecolor:'#2a2a3c' },
  aspectmode: 'cube',
  bgcolor: '#0d1117',
  camera: CAMERAS.fl,
  dragmode: 'orbit',
};

const baseLayout = {
  scene: sceneBase,
  paper_bgcolor: '#0d1117',
  plot_bgcolor:  '#0d1117',
  margin: { l:0, r:0, t:0, b:0 },
  showlegend: false,
  uirevision: 'locked',  // keeps camera when updating data
};

// ── Orientation cube layout ───────────────────────────────────────────────
const oriLayout = {
  scene: {
    xaxis:{ range:[-1.8,1.8], showgrid:false, showticklabels:false,
            backgroundcolor:'#0a0a14', showbackground:true, zerolinecolor:'#1a1a28' },
    yaxis:{ range:[-1.8,1.8], showgrid:false, showticklabels:false,
            backgroundcolor:'#0a0a14', showbackground:true, zerolinecolor:'#1a1a28' },
    zaxis:{ range:[-1.8,1.8], showgrid:false, showticklabels:false,
            backgroundcolor:'#0a0a14', showbackground:true, zerolinecolor:'#1a1a28' },
    aspectmode:'cube', bgcolor:'#0a0a14',
    camera: CAMERAS.fl,
    dragmode: false,
  },
  paper_bgcolor:'#0a0a14',
  margin:{ l:0, r:0, t:0, b:0 },
  showlegend: false,
};

// ── State ─────────────────────────────────────────────────────────────────
let currentScenario = Object.keys(SCENARIOS)[0];

// ── Layer visibility state ────────────────────────────────────────────
let layerVisibility = {
  struct: true,
  nominal: true,
  shifted: true,
};

function _applyVisibilityDefaults(sc) {
  // Reset struct visibility to scenario default (ON for nominal, OFF for shifted)
  const structDefault = sc.defaultStructVisible !== false;
  layerVisibility.struct = structDefault;
  const cb = document.getElementById('vis-struct');
  if (cb) cb.checked = structDefault;
  // shifted checkbox only relevant in shifted scenarios – reset to true always
  layerVisibility.shifted = true;
  const cbsh = document.getElementById('vis-shifted');
  if (cbsh) cbsh.checked = true;
}

function _filteredTraces(sc) {
  return sc.traces.filter(t => {
    if (!t.layer) return true;  // isocenter, always visible
    return layerVisibility[t.layer];
  });
}

// ── Scenario switching ────────────────────────────────────────────────
function showScenario(name) {
  if (name === currentScenario) return;
  currentScenario = name;
  const sc = SCENARIOS[name];

  // Reset visibility defaults for new scenario
  _applyVisibilityDefaults(sc);

  // Preserve current camera
  const mainEl = document.getElementById('main-plot');
  let cam = CAMERAS.fl;
  try { cam = mainEl._fullLayout.scene.camera; } catch(e) {}

  const layout = JSON.parse(JSON.stringify(baseLayout));
  layout.scene.camera = cam;

  Plotly.react('main-plot', _filteredTraces(sc), layout, { responsive:true,
    displayModeBar:true, modeBarButtonsToRemove:['toImage'], displaylogo:false });
  document.getElementById('sc-title').textContent = sc.title;

  document.querySelectorAll('.sc-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('scbtn-' + name);
  if (btn) btn.classList.add('active');
}

// ── Toggle layer visibility ───────────────────────────────────────────
function toggleLayer(layer) {
  layerVisibility[layer] = document.getElementById('vis-' + layer).checked;

  const sc = SCENARIOS[currentScenario];
  const mainEl = document.getElementById('main-plot');
  let cam = CAMERAS.fl;
  try { cam = mainEl._fullLayout.scene.camera; } catch(e) {}

  const layout = JSON.parse(JSON.stringify(baseLayout));
  layout.scene.camera = cam;

  Plotly.react('main-plot', _filteredTraces(sc), layout, { responsive:true,
    displayModeBar:true, modeBarButtonsToRemove:['toImage'], displaylogo:false });
}

// ── Camera presets ────────────────────────────────────────────────────────
function setCamera(preset) {
  Plotly.relayout('main-plot', { 'scene.camera': CAMERAS[preset] });
  document.querySelectorAll('.vbtn').forEach(b => b.classList.remove('active'));
  document.getElementById('vbtn-' + preset).classList.add('active');
}

// ── Sync orientation cube with main plot ──────────────────────────────────
document.getElementById('main-plot').addEventListener('plotly_relayout', function(ev) {
  if (!ev || !ev['scene.camera']) return;
  Plotly.relayout('ori-plot', { 'scene.camera': ev['scene.camera'] });
}, false);

// ── Init orientation cube ─────────────────────────────────────────────────
Plotly.newPlot('ori-plot', ORI_TRACES, oriLayout,
  { staticPlot:false, responsive:true, displayModeBar:false });

// ── Init: apply defaults for the first scenario ───────────────────────────
_applyVisibilityDefaults(SCENARIOS[currentScenario]);

// ── Init main plot ────────────────────────────────────────────────────────
Plotly.newPlot('main-plot',
  _filteredTraces(SCENARIOS[currentScenario]),
  baseLayout,
  { responsive:true, displayModeBar:true,
    modeBarButtonsToRemove:['toImage'], displaylogo:false }
).then(function() {
  // Wire up relayout sync after plot is ready
  document.getElementById('main-plot').on('plotly_relayout', function(ev) {
    if (ev && ev['scene.camera']) {
      Plotly.relayout('ori-plot', { 'scene.camera': ev['scene.camera'] });
      // Deactivate all camera preset buttons (user is freely rotating)
      document.querySelectorAll('.vbtn').forEach(b => b.classList.remove('active'));
    }
  });
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate_3d_html(
    plan_files_list,
    mapping_df,
    scenarios,
    output_dir: str,
) -> List[str]:
    """
    Generate one self-contained interactive HTML per patient/plan.
    Reuses marching-cubes mesh functions from plot_3d.py.
    """
    from dicom_io import load_plan
    from excel_io import get_active_structures
    from plot_3d import (
        _marching_cubes_dose,
        _voxelize_and_mesh_per_structure,
        _transform_verts,
        _resolve_rx_for_plan,
        _ISO_NOM_COLOR,
        _ISO_SHIFT_COLOR,
        _STRUCT_COLOR,
    )

    os.makedirs(output_dir, exist_ok=True)
    active_df = get_active_structures(mapping_df)
    paths: List[str] = []

    all_scenarios = list(scenarios)
    if not any(s.name == "nominal" for s in all_scenarios):
        from shift_scenarios import ShiftScenario
        all_scenarios = [ShiftScenario(name="nominal")] + all_scenarios

    for pf in plan_files_list:
        mask = (
            (active_df["PatientFolder"] == pf.patient_folder) &
            (active_df["PlanType"]      == pf.plan_type)
        )
        if active_df[mask].empty:
            continue
        active_names = set(active_df[mask]["StructureName_Original"].tolist())

        try:
            structures, dose, meta = load_plan(pf)
        except Exception as exc:
            logger.warning("HTML 3D: load failed %s/%s: %s",
                           pf.patient_folder, pf.plan_type, exc)
            continue

        if meta is None or meta.isocenter is None or dose is None:
            continue

        isocenter = meta.isocenter
        rx_dose   = _resolve_rx_for_plan(mapping_df, pf.patient_folder,
                                          pf.plan_type)
        if rx_dose is None or rx_dose <= 0:
            continue

        logger.info("HTML 3D: building meshes %s/%s …",
                    pf.patient_folder, pf.plan_type)
        iso_verts, iso_faces = _marching_cubes_dose(dose, rx_dose)
        if iso_verts is None:
            logger.warning("HTML 3D: isodose mesh empty %s/%s – skipped.",
                           pf.patient_folder, pf.plan_type)
            continue
        logger.info("HTML 3D:   isodose %d faces", len(iso_faces))

        # Per-PTV meshes: one mesh per active structure with individual hover
        per_ptv = _voxelize_and_mesh_per_structure(structures, active_names, dose)
        total_str_faces = sum(len(f) for _, _, f in per_ptv)
        logger.info("HTML 3D:   structure %d PTV(s), %d total faces",
                    len(per_ptv), total_str_faces)

        # ── Axis limits (from nominal meshes) ─────────────────────────────
        ref_verts = [iso_verts] + [v for _, v, _ in per_ptv if v is not None]
        combined  = np.vstack(ref_verts)
        lo, hi    = combined.min(axis=0), combined.max(axis=0)
        center    = (lo + hi) * 0.5
        half      = float(np.max(hi - lo)) * 0.55
        axis_limits = [
            round(center[0] - half, 2), round(center[0] + half, 2),
            round(center[1] - half, 2), round(center[1] + half, 2),
            round(center[2] - half, 2), round(center[2] + half, 2),
        ]

        # ── Pre-build per-PTV structure traces (reused in all scenarios) ──
        struct_traces = []
        for ptv_name, sv, sf in per_ptv:
            centroid = sv.mean(axis=0)
            dist     = float(np.linalg.norm(centroid - isocenter))
            hover    = (f"<b>{ptv_name}</b><br>"
                        f"Abstand zum Isocenter: <b>{dist:.1f} mm</b>")
            t = _mesh_trace(sv, sf, _STRUCT_COLOR,
                            ptv_name, alpha=1.0,
                            hovertext=hover, layer="struct")
            if t:
                struct_traces.append(t)

        # ── Build per-scenario trace sets ──────────────────────────────────
        scenarios_data = {}
        for sc in all_scenarios:
            is_nominal = (
                sc.name == "nominal" or (
                    sc.dx_mm == 0 and sc.dy_mm == 0 and sc.dz_mm == 0 and
                    sc.rx_deg == 0 and sc.ry_deg == 0 and sc.rz_deg == 0
                )
            )
            traces = []

            if is_nominal:
                # Isodose opaque (back), structure opaque (front).
                t = _mesh_trace(iso_verts, iso_faces, _ISO_NOM_COLOR,
                                f"Nominal {rx_dose:.0f} Gy isodose",
                                alpha=1.0, layer="nominal")
                if t: traces.append(t)
                traces.extend(struct_traces)
                title = "Nominal  (Struktur vs. Isodose)"
            else:
                iso_sh = _transform_verts(iso_verts, isocenter, sc)
                # Shifted (alpha=0.72) drawn first (behind).
                # Nominal opaque (alpha=1.0) drawn second → correctly occludes.
                t = _mesh_trace(iso_sh, iso_faces, _ISO_SHIFT_COLOR,
                                "Shifted isodose protrusion", alpha=0.72,
                                layer="shifted")
                if t: traces.append(t)
                t = _mesh_trace(iso_verts, iso_faces, _ISO_NOM_COLOR,
                                f"Nominal {rx_dose:.0f} Gy isodose",
                                alpha=1.0, layer="nominal")
                if t: traces.append(t)
                # Structure included (hidden by default in shifted)
                traces.extend(struct_traces)
                has_rot = bool(sc.rx_deg or sc.ry_deg or sc.rz_deg)
                title = (f"{sc.name}  "
                         f"Δ=({sc.dx_mm:+.1f},{sc.dy_mm:+.1f},"
                         f"{sc.dz_mm:+.1f}) mm")
                if has_rot:
                    title += (f"  rot=({sc.rx_deg:.1f},"
                              f"{sc.ry_deg:.1f},{sc.rz_deg:.1f})°")

            traces.append(_isocenter_trace(isocenter))
            scenarios_data[sc.name] = {
                "traces": traces,
                "title": title,
                "defaultStructVisible": is_nominal,
            }

        # ── Build HTML ────────────────────────────────────────────────────
        folder_clean = (pf.patient_folder[3:]
                        if pf.patient_folder.startswith("zz_")
                        else pf.patient_folder)
        plan_label = f"{folder_clean} / {pf.plan_type}"
        safe_pat   = pf.patient_folder.replace(" ", "_")
        first_name = list(scenarios_data.keys())[0]

        scenario_buttons = ""
        for name, data in scenarios_data.items():
            active_cls = " active" if name == first_name else ""
            label      = name.replace("_", " ")
            scenario_buttons += (
                f'<button class="sc-btn{active_cls}" '
                f'id="scbtn-{name}" '
                f'onclick="showScenario(\'{name}\')">'
                f'{label}</button>\n'
            )

        legend_html = (
            f'<div class="legend-row"><div class="swatch" '
            f'style="background:{_ISO_NOM_COLOR}"></div>Nominale Isodose</div>\n'
            f'<div class="legend-row"><div class="swatch" '
            f'style="background:{_ISO_SHIFT_COLOR}"></div>Isodosen-Protrusion</div>\n'
            f'<div class="legend-row"><div class="swatch" '
            f'style="background:{_STRUCT_COLOR};border:1px solid #666"></div>PTV Struktur</div>\n'
            f'<div class="legend-row"><div class="swatch" '
            f'style="background:#fff;border-radius:50%;width:11px;height:11px;border:1px solid #aaa"></div>Isocenter</div>\n'
        )

        ori_traces = _orientation_traces()

        try:
            from plot_3d import _MAX_FACES, _ISO_Z_STRIDE
            _used_faces  = _MAX_FACES
            _used_stride = _ISO_Z_STRIDE
        except ImportError:
            _used_faces, _used_stride = 12000, 1

        iso_nf  = len(iso_faces) if iso_faces is not None else 0
        str_nf  = sum(len(f) for _, _, f in per_ptv)
        mesh_info = (
            f"Isodose: <b>{iso_nf:,}</b> Dreiecke<br>"
            f"Struktur: <b>{str_nf:,}</b> Dreiecke<br>"
            f"Stride: {_used_stride} &nbsp; Limit: {_used_faces:,}"
        )

        html = _TEMPLATE
        html = html.replace("###TITLE###",          f"SRS 3D – {plan_label}")
        html = html.replace("###PLAN_LABEL###",     plan_label)
        html = html.replace("###N_SCENARIOS###",    str(len(scenarios_data)))
        html = html.replace("###SCENARIO_BUTTONS###", scenario_buttons)
        html = html.replace("###LEGEND_HTML###",    legend_html)
        html = html.replace("###MESH_INFO###",      mesh_info)
        html = html.replace("###FIRST_TITLE###",    scenarios_data[first_name]["title"])
        html = html.replace("###SCENARIOS_JSON###", json.dumps(scenarios_data,
                                                               ensure_ascii=False))
        html = html.replace("###AXIS_LIMITS_JSON###", json.dumps(axis_limits))
        html = html.replace("###ORI_TRACES_JSON###",  json.dumps(ori_traces))

        out_path = os.path.join(output_dir, f"{safe_pat}__{pf.plan_type}__viewer.html")
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(html)
            paths.append(out_path)
            logger.info("HTML 3D saved: %s", out_path)
        except Exception as exc:
            logger.error("HTML 3D save failed %s: %s", out_path, exc)

    logger.info("HTML 3D: %d file(s) written to %s", len(paths), output_dir)
    return paths
