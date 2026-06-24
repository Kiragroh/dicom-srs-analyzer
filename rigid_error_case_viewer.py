"""
Build the versioned interactive Plotly viewer for the 12-met rigid-error case.

The generic DICOM viewer only knows PTV and isodose meshes. This exporter uses
the staged 12-met scene builder so the checked-in example viewer also contains
the OAR context and the cyan undercovered PTV surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ANALYZER_DIR = Path(__file__).resolve().parent
CASE_BUILDER_DIR = ANALYZER_DIR.parent
if str(CASE_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_BUILDER_DIR))

import render_12met_scenario_visuals as scene3d  # noqa: E402


STAGE_MODES = [
    ("ptv", "1. PTV + OAR", ("oar", "ptv")),
    ("nominal", "2. Nominale 20 Gy", ("oar", "nominal", "ptv")),
    ("shifted", "3. Iso-Interaktion", ("oar", "shifted")),
    ("diff", "4. Rest ausserhalb nominal", ("oar", "diff", "ptv")),
    ("miss", "5. PTV ohne 20 Gy", ("oar", "ptv", "miss")),
]

DEFAULT_STAGE = "shifted"
FOCUS_TARGET_NUMBER = 8
FOCUS_TARGET_LABEL = f"PTV{FOCUS_TARGET_NUMBER}"
FOCUS_PTV_COLOR = "#fff1a8"

TRACE_LIMITS = {
    "oar": 3000,
    "ptv": 1400,
    "nominal": 0,
    "shifted": 0,
    "diff": 0,
    "miss": 1400,
}

TRACE_ALPHA = {
    "oar": 0.34,
    "ptv": 0.34,
    "nominal": 0.48,
    "shifted": 0.68,
    "diff": 0.68,
    "miss": 0.72,
}

OAR_DISPLAY = {
    "Eye Left": ("Eyes", "oar"),
    "Eye Right": ("Eyes", "oar"),
    "Optic Nerve Left": ("Optic Nerves", "oar"),
    "Optic Nerve Right": ("Optic Nerves", "oar"),
    "Chiasm": ("Chiasma", "oar"),
    "Brainstem": ("Brainstem", "oar"),
}


def _scene_args(output_dir: Path, resolution: str) -> argparse.Namespace:
    presets = {
        "light": (9000, 900, 2000),
        "standard": (18000, 1600, 5200),
        "dense": (32000, 2600, 9000),
    }
    iso_faces, ptv_faces, oar_faces = presets[resolution]
    return scene3d.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--skip-static",
            "--skip-animation",
            "--no-write-gif",
            "--iso-faces",
            str(iso_faces),
            "--ptv-faces",
            str(ptv_faces),
            "--oar-faces",
            str(oar_faces),
            "--axis-pad",
            "0.03",
            "--diff-dose-margin-gy",
            "0.05",
        ]
    )


def _decimate_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if faces is None or max_faces <= 0 or len(faces) <= max_faces:
        return faces
    indices = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
    return faces[indices]


def _decimate_faces_and_colors(
    faces: np.ndarray,
    facecolors: np.ndarray | None,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if faces is None or max_faces <= 0 or len(faces) <= max_faces:
        return faces, facecolors
    indices = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
    return faces[indices], facecolors[indices] if facecolors is not None else None


def _compact_mesh(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return verts[used], remap[faces]


def _round_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in values]


def _trace_style(layer: scene3d.MeshLayer) -> tuple[str, str, str, str]:
    if layer.name == "Nominale Isodose":
        return "nominal", "Nominale 20 Gy Isodose", scene3d.NOMINAL_COLOR, "Nominale 20 Gy Isodose"
    if layer.name == "Verschobene Isodose":
        return "shifted", "Szenario Isodose", scene3d.DIFF_SHIFTED_COLOR, "Szenario Isodose"
    if layer.name == "Iso-Interaktion":
        return "shifted", "Iso-Interaktion", scene3d.NOMINAL_COLOR, "Szenario Isodose"
    if layer.name == "Abweichung zur Nominalisodose":
        return "diff", "Szenario Isodose ausserhalb nominal", scene3d.DIFF_SHIFTED_COLOR, "Rest ausserhalb nominal"
    if layer.name.startswith("PTV ohne 100% "):
        target = layer.name.removeprefix("PTV ohne 100% ")
        return "miss", f"PTV ohne 20 Gy: {target}", scene3d.UNCOVERED_COLOR, "PTV ohne 20 Gy nach Shift"
    if layer.name.startswith("PTV/Mets "):
        target = layer.name.removeprefix("PTV/Mets ")
        color = FOCUS_PTV_COLOR if target.startswith(f"PTV{FOCUS_TARGET_NUMBER:02d}") else scene3d.PTV_COLOR
        legend = f"{FOCUS_TARGET_LABEL} Fokus" if color == FOCUS_PTV_COLOR else "PTVs"
        return "ptv", target, color, legend
    if layer.name in scene3d.OAR_COLORS:
        display, category = OAR_DISPLAY.get(layer.name, (layer.name, "oar"))
        return category, layer.name, scene3d.OAR_COLORS[layer.name], display
    return "other", layer.name, layer.color, layer.name


def _css_rgba(facecolors: np.ndarray | None) -> list[str] | None:
    if facecolors is None:
        return None
    colors = np.asarray(facecolors, dtype=float)
    if colors.size == 0:
        return None
    if float(np.nanmax(colors)) <= 1.0:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255)
    rgba = []
    for red, green, blue, alpha in colors:
        rgba.append(f"rgba({int(red)},{int(green)},{int(blue)},{float(alpha) / 255.0:.3f})")
    return rgba


def _target_name_from_layer(category: str, name: str) -> str | None:
    if category == "ptv":
        return name
    if category == "miss":
        return name.removeprefix("PTV ohne 20 Gy: ")
    return None


def _hover_text(category: str, name: str, target_info: dict[str, dict[str, float]]) -> str:
    safe_name = name.replace("<", "&lt;").replace(">", "&gt;")
    target_name = _target_name_from_layer(category, name)
    if target_name and target_name in target_info:
        info = target_info[target_name]
        return (
            f"<b>{safe_name}</b><br>"
            f"PTV-Volumen: <b>{info['volume_cc']:.3f} cc</b><br>"
            f"Abstand zum Isozentrum: <b>{info['distance_to_iso_mm']:.1f} mm</b>"
            "<extra></extra>"
        )
    return f"<b>{safe_name}</b><extra></extra>"


def _mesh_trace(layer: scene3d.MeshLayer, target_info: dict[str, dict[str, float]]) -> dict | None:
    if layer.verts is None or layer.faces is None or len(layer.faces) == 0:
        return None
    category, name, color, legend_label = _trace_style(layer)
    facecolors = np.asarray(layer.facecolors, dtype=float) if layer.facecolors is not None else None
    faces, facecolors = _decimate_faces_and_colors(
        np.asarray(layer.faces, dtype=np.int64),
        facecolors,
        TRACE_LIMITS.get(category, 3000),
    )
    verts, faces = _compact_mesh(np.asarray(layer.verts, dtype=float), faces)
    trace = {
        "type": "mesh3d",
        "x": _round_list(verts[:, 0]),
        "y": _round_list(verts[:, 1]),
        "z": _round_list(verts[:, 2]),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "color": color,
        "opacity": TRACE_ALPHA.get(category, float(layer.alpha)),
        "flatshading": False,
        "lighting": {
            "ambient": 0.70,
            "diffuse": 0.58,
            "roughness": 0.86,
            "specular": 0.08,
            "fresnel": 0.03,
        },
        "lightposition": {"x": -250, "y": -260, "z": 420},
        "name": legend_label,
        "legendgroup": legend_label,
        "showlegend": False,
        "layer": category,
        "hovertemplate": _hover_text(category, name, target_info),
        "showscale": False,
    }
    css_facecolors = _css_rgba(facecolors)
    if css_facecolors is not None:
        trace["facecolor"] = css_facecolors
    return trace


def _isocenter_trace(isocenter: np.ndarray) -> dict:
    return {
        "type": "scatter3d",
        "x": [round(float(isocenter[0]), 2)],
        "y": [round(float(isocenter[1]), 2)],
        "z": [round(float(isocenter[2]), 2)],
        "mode": "markers",
        "marker": {"size": 5, "color": "#ffffff", "symbol": "cross"},
        "name": "Isozentrum",
        "hoverinfo": "name+x+y+z",
        "showlegend": False,
        "layer": "iso",
    }


def _orientation_traces() -> list[dict]:
    axes = [
        ([0, 1.2], [0, 0], [0, 0], "R", "#ff5555"),
        ([0, -1.2], [0, 0], [0, 0], "L", "#ff9999"),
        ([0, 0], [0, 1.2], [0, 0], "A", "#55ff55"),
        ([0, 0], [0, -1.2], [0, 0], "P", "#99ff99"),
        ([0, 0], [0, 0], [0, 1.2], "S", "#5599ff"),
        ([0, 0], [0, 0], [0, -1.2], "I", "#99bbff"),
    ]
    traces = []
    for xs, ys, zs, label, color in axes:
        traces.append(
            {
                "type": "scatter3d",
                "x": xs,
                "y": ys,
                "z": zs,
                "mode": "lines+text",
                "line": {"color": color, "width": 6},
                "text": ["", label],
                "textfont": {"size": 13, "color": color, "family": "Arial Black"},
                "textposition": "top center",
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
    return traces


def _collect_case_layers(
    scenario_name: str,
    static_layers: dict[str, list[scene3d.MeshLayer]],
    stage_layers: dict[str, dict[str, list[scene3d.MeshLayer]]],
) -> list[scene3d.MeshLayer]:
    static = static_layers[scenario_name]
    staged = stage_layers[scenario_name]
    oars = [layer for layer in static if layer.name in scene3d.OAR_COLORS]
    nominal = [layer for layer in static if layer.name == "Nominale Isodose"]
    interaction = [layer for layer in staged["shifted"] if layer.name == "Iso-Interaktion"]
    ptvs = [layer for layer in staged["ptv"] if layer.name.startswith("PTV/Mets ")]
    diff = [layer for layer in staged["diff"] if layer.name == "Abweichung zur Nominalisodose"]
    miss = [layer for layer in staged["miss"] if layer.name.startswith("PTV ohne 100% ")]
    return [*oars, *interaction, *nominal, *diff, *miss, *ptvs]


def _scenario_payload(
    scenarios: list,
    static_layers: dict[str, list[scene3d.MeshLayer]],
    stage_layers: dict[str, dict[str, list[scene3d.MeshLayer]]],
    isocenter: np.ndarray,
    target_info: dict[str, dict[str, float]],
) -> dict:
    payload = {}
    for scenario in scenarios:
        traces = []
        for layer in _collect_case_layers(scenario.name, static_layers, stage_layers):
            trace = _mesh_trace(layer, target_info)
            if trace is not None:
                traces.append(trace)
        traces.append(_isocenter_trace(isocenter))
        payload[scenario.name] = {
            "title": f"{scene3d._scenario_short_label(scenario)} | {scene3d._scenario_compact_detail(scenario)}",  # noqa: SLF001
            "buttonLabel": scene3d._scenario_short_label(scenario),  # noqa: SLF001
            "traces": traces,
        }
    return payload


def _target_info(
    targets: list[tuple[int, object]],
    isocenter: np.ndarray,
    data_root: Path,
) -> dict[str, dict[str, float]]:
    plan_sets = scene3d.discover_plan_sets(data_root)
    if not plan_sets:
        return {}
    dose = scene3d.load_dose(str(plan_sets[0].rtdose_path))
    if dose is None:
        return {}

    info: dict[str, dict[str, float]] = {}
    for number, structure in targets:
        mask = scene3d._structure_mask_on_dose_grid(structure, dose)  # noqa: SLF001
        volume_cc = float(np.count_nonzero(mask) * np.prod(dose.spacing) / 1000.0)
        centroid = scene3d._centroid(structure)  # noqa: SLF001
        distance_to_iso_mm = float(np.linalg.norm(centroid - isocenter))
        values = {
            "volume_cc": volume_cc,
            "distance_to_iso_mm": distance_to_iso_mm,
        }
        info[getattr(structure, "name", f"PTV{number:02d}")] = values
        info[f"PTV{number:02d}"] = values
    return info


def _legend_html() -> str:
    entries = [
        ("PTVs", scene3d.PTV_COLOR),
        (f"{FOCUS_TARGET_LABEL} Fokus", FOCUS_PTV_COLOR),
        ("Nominale 20 Gy Isodose", scene3d.NOMINAL_COLOR),
        ("Szenario Isodose", scene3d.DIFF_SHIFTED_COLOR),
        ("PTV ohne 20 Gy nach Shift", scene3d.UNCOVERED_COLOR),
        ("Brainstem", scene3d.OAR_COLORS["Brainstem"]),
        ("Optic Nerves", scene3d.OAR_COLORS["Optic Nerve Left"]),
        ("Chiasma", scene3d.OAR_COLORS["Chiasm"]),
        ("Eyes", scene3d.OAR_COLORS["Eye Left"]),
    ]
    return "\n".join(
        f'<div class="legend-row"><div class="swatch" style="background:{color}"></div>{label}</div>'
        for label, color in entries
    )


def _mode_buttons() -> str:
    html = []
    for key, label, _layers in STAGE_MODES:
        active = " active" if key == DEFAULT_STAGE else ""
        html.append(
            f'<button class="mode-btn{active}" id="mode-{key}" onclick="setViewMode(\'{key}\')">{label}</button>'
        )
    return "\n".join(html)


def _scenario_buttons(scenarios_data: dict) -> str:
    first = next(iter(scenarios_data))
    buttons = []
    for name, data in scenarios_data.items():
        active = " active" if name == first else ""
        buttons.append(
            f'<button class="sc-btn{active}" id="scbtn-{name}" onclick="showScenario(\'{name}\')">'
            f'{data["buttonLabel"]}</button>'
        )
    return "\n".join(buttons)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>12Met Rigid Error 3D Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body { background:#0d1117; color:#ccc; font-family:'Segoe UI',system-ui,sans-serif; display:flex; flex-direction:column; }
header { background:#161b22; border-bottom:1px solid #30363d; padding:8px 16px; display:flex; align-items:center; gap:10px; min-height:40px; flex-shrink:0; }
header h1 { font-size:13px; font-weight:600; color:#e6edf3; }
.badge { margin-left:auto; background:#21262d; border:1px solid #30363d; border-radius:12px; padding:2px 9px; font-size:10px; color:#8b949e; }
.content { display:flex; flex:1; min-height:0; }
.sidebar { width:232px; min-width:232px; background:#161b22; border-right:1px solid #30363d; display:flex; flex-direction:column; overflow-y:auto; padding:10px 8px; gap:14px; }
.sidebar section { display:flex; flex-direction:column; gap:5px; }
.sidebar h3 { font-size:10px; font-weight:600; color:#6e7681; text-transform:uppercase; letter-spacing:.07em; padding-bottom:2px; border-bottom:1px solid #21262d; }
.sc-btn, .mode-btn, .vbtn { background:#21262d; border:1px solid #30363d; border-radius:5px; color:#ccc; cursor:pointer; transition:all .12s; }
.sc-btn { display:block; width:100%; padding:6px 9px; font-size:11px; text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sc-btn:hover, .mode-btn:hover, .vbtn:hover { background:#30363d; color:#e6edf3; }
.sc-btn.active { background:#1f6feb; border-color:#388bfd; color:#fff; font-weight:600; }
.mode-grid { display:grid; grid-template-columns:1fr; gap:4px; }
.mode-btn { padding:6px 7px; font-size:11px; text-align:left; }
.mode-btn.active { background:#8957e5; border-color:#a371f7; color:#fff; }
.view-grid { display:grid; grid-template-columns:1fr 1fr; gap:4px; }
.vbtn { padding:5px 4px; font-size:11px; text-align:center; }
.vbtn.active { background:#1a7f37; border-color:#2ea043; color:#fff; }
.checks { display:flex; flex-direction:column; gap:6px; font-size:11px; color:#ccc; }
.checks label { display:flex; align-items:center; gap:7px; cursor:pointer; }
#ori-plot { height:150px; }
.legend-row { display:flex; align-items:center; gap:8px; font-size:11px; color:#ccc; }
.swatch { width:16px; height:11px; border-radius:3px; flex-shrink:0; }
.main-view { flex:1; display:flex; flex-direction:column; min-width:0; }
.sc-title { background:#161b22; border-bottom:1px solid #30363d; padding:5px 14px; font-size:11px; color:#8b949e; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
#main-plot { flex:1; min-height:0; }
code { color:#8b949e; }
</style>
</head>
<body>
<header>
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#388bfd" stroke-width="2">
    <circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/>
  </svg>
  <h1>12Met Rigid Error - interaktiver 3-D Viewer</h1>
  <span class="badge">4 scenarios, 5 views</span>
</header>
<div class="content">
  <div class="sidebar">
    <section>
      <h3>Szenarien</h3>
      ###SCENARIO_BUTTONS###
    </section>
    <section>
      <h3>Darstellung</h3>
      <div class="mode-grid">###MODE_BUTTONS###</div>
    </section>
    <section>
      <h3>Kamera</h3>
      <div class="view-grid">
        <button class="vbtn active" id="vbtn-fl" onclick="setCamera('fl')">Front-L</button>
        <button class="vbtn" id="vbtn-fr" onclick="setCamera('fr')">Front-R</button>
        <button class="vbtn" id="vbtn-top" onclick="setCamera('top')">Top</button>
        <button class="vbtn" id="vbtn-side" onclick="setCamera('side')">Side</button>
      </div>
    </section>
    <section>
      <h3>Sichtbarkeit</h3>
      <div class="checks">
        <label><input type="checkbox" id="vis-oar" checked onchange="toggleLayer('oar')">Organe</label>
        <label><input type="checkbox" id="vis-ptv" checked onchange="toggleLayer('ptv')">PTVs</label>
        <label><input type="checkbox" id="vis-nominal" checked onchange="toggleLayer('nominal')">Nominale 20 Gy</label>
        <label><input type="checkbox" id="vis-shifted" checked onchange="toggleLayer('shifted')">Szenario-Isodose</label>
        <label><input type="checkbox" id="vis-diff" checked onchange="toggleLayer('diff')">Rest ausserhalb nominal</label>
        <label><input type="checkbox" id="vis-miss" checked onchange="toggleLayer('miss')">PTV ohne 20 Gy</label>
      </div>
    </section>
    <section>
      <h3>Orientierung</h3>
      <div id="ori-plot"></div>
      <div style="font-size:9px;color:#555;line-height:1.5;padding-top:3px;">
        R/L = rechts/links<br>A/P = anterior/posterior<br>S/I = superior/inferior
      </div>
    </section>
    <section>
      <h3>Legende</h3>
      ###LEGEND_HTML###
    </section>
  </div>
  <div class="main-view">
    <div class="sc-title" id="sc-title">###FIRST_TITLE###</div>
    <div id="main-plot"></div>
  </div>
</div>
<script>
const SCENARIOS = ###SCENARIOS_JSON###;
const AXIS_LIMITS = ###AXIS_LIMITS_JSON###;
const VIEW_MODES = ###VIEW_MODES_JSON###;
const ORI_TRACES = ###ORI_TRACES_JSON###;

const CAMERAS = {
  fl:   { eye:{x:-1.5,y:-1.5,z:0.8}, up:{x:0,y:0,z:1}, center:{x:0,y:0,z:0} },
  fr:   { eye:{x: 1.5,y:-1.5,z:0.8}, up:{x:0,y:0,z:1}, center:{x:0,y:0,z:0} },
  top:  { eye:{x: 0.0,y:-0.1,z:2.8}, up:{x:0,y:1,z:0}, center:{x:0,y:0,z:0} },
  side: { eye:{x:-2.8,y: 0.0,z:0.0}, up:{x:0,y:0,z:1}, center:{x:0,y:0,z:0} },
};

const sceneBase = {
  xaxis:{ range:[AXIS_LIMITS[0],AXIS_LIMITS[1]], showgrid:true, gridcolor:'#1c1c2c',
    title:{text:'X (mm)',font:{color:'#555',size:10}}, tickfont:{color:'#555',size:9},
    backgroundcolor:'#0d1117', showbackground:true, zerolinecolor:'#2a2a3c' },
  yaxis:{ range:[AXIS_LIMITS[2],AXIS_LIMITS[3]], showgrid:true, gridcolor:'#1c1c2c',
    title:{text:'Y (mm)',font:{color:'#555',size:10}}, tickfont:{color:'#555',size:9},
    backgroundcolor:'#0d1117', showbackground:true, zerolinecolor:'#2a2a3c' },
  zaxis:{ range:[AXIS_LIMITS[4],AXIS_LIMITS[5]], showgrid:true, gridcolor:'#1c1c2c',
    title:{text:'Z (mm)',font:{color:'#555',size:10}}, tickfont:{color:'#555',size:9},
    backgroundcolor:'#0d1117', showbackground:true, zerolinecolor:'#2a2a3c' },
  aspectmode:'cube',
  bgcolor:'#0d1117',
  camera:CAMERAS.fl,
  dragmode:'orbit',
};

const baseLayout = {
  scene:sceneBase,
  paper_bgcolor:'#0d1117',
  plot_bgcolor:'#0d1117',
  margin:{l:0,r:0,t:0,b:0},
  showlegend:false,
  uirevision:'rigid-error-case',
};

const oriLayout = {
  scene:{
    xaxis:{range:[-1.8,1.8],showgrid:false,showticklabels:false,backgroundcolor:'#0a0a14',showbackground:true,zerolinecolor:'#1a1a28'},
    yaxis:{range:[-1.8,1.8],showgrid:false,showticklabels:false,backgroundcolor:'#0a0a14',showbackground:true,zerolinecolor:'#1a1a28'},
    zaxis:{range:[-1.8,1.8],showgrid:false,showticklabels:false,backgroundcolor:'#0a0a14',showbackground:true,zerolinecolor:'#1a1a28'},
    aspectmode:'cube', bgcolor:'#0a0a14', camera:CAMERAS.fl, dragmode:false
  },
  paper_bgcolor:'#0a0a14',
  margin:{l:0,r:0,t:0,b:0},
  showlegend:false,
};

let currentScenario = Object.keys(SCENARIOS)[0];
let viewMode = '###DEFAULT_STAGE###';
let layerVisibility = { oar:true, ptv:true, nominal:true, shifted:true, diff:true, miss:true, iso:true };

function _activeTraces(sc) {
  const modeLayers = new Set(VIEW_MODES[viewMode].layers);
  return sc.traces.filter(t => {
    if (t.layer === 'iso') return true;
    return modeLayers.has(t.layer) && layerVisibility[t.layer] !== false;
  });
}

function _title(sc) {
  return sc.title + ' | ' + VIEW_MODES[viewMode].label;
}

function _redrawCurrent() {
  const sc = SCENARIOS[currentScenario];
  const mainEl = document.getElementById('main-plot');
  let cam = CAMERAS.fl;
  try { cam = mainEl._fullLayout.scene.camera; } catch(e) {}
  const layout = JSON.parse(JSON.stringify(baseLayout));
  layout.scene.camera = cam;
  Plotly.react('main-plot', _activeTraces(sc), layout, {
    responsive:true, displayModeBar:true, modeBarButtonsToRemove:['toImage'], displaylogo:false
  });
  document.getElementById('sc-title').textContent = _title(sc);
}

function showScenario(name) {
  currentScenario = name;
  _redrawCurrent();
  document.querySelectorAll('.sc-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('scbtn-' + name);
  if (btn) btn.classList.add('active');
}

function setViewMode(mode) {
  viewMode = mode;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('mode-' + mode);
  if (btn) btn.classList.add('active');
  _redrawCurrent();
}

function toggleLayer(layer) {
  layerVisibility[layer] = document.getElementById('vis-' + layer).checked;
  _redrawCurrent();
}

function setCamera(preset) {
  Plotly.relayout('main-plot', { 'scene.camera': CAMERAS[preset] });
  document.querySelectorAll('.vbtn').forEach(b => b.classList.remove('active'));
  document.getElementById('vbtn-' + preset).classList.add('active');
}

Plotly.newPlot('ori-plot', ORI_TRACES, oriLayout, { staticPlot:false, responsive:true, displayModeBar:false });
Plotly.newPlot('main-plot', _activeTraces(SCENARIOS[currentScenario]), baseLayout, {
  responsive:true, displayModeBar:true, modeBarButtonsToRemove:['toImage'], displaylogo:false
}).then(function() {
  document.getElementById('sc-title').textContent = _title(SCENARIOS[currentScenario]);
  document.getElementById('main-plot').on('plotly_relayout', function(ev) {
    if (ev && ev['scene.camera']) {
      Plotly.relayout('ori-plot', { 'scene.camera': ev['scene.camera'] });
      document.querySelectorAll('.vbtn').forEach(b => b.classList.remove('active'));
    }
  });
});
</script>
</body>
</html>
"""


def build_html(output_dir: Path, resolution: str) -> str:
    args = _scene_args(output_dir, resolution)
    (
        scenarios,
        static_layers,
        _anim_layers,
        stage_layers,
        isocenter,
        targets,
        axis_limits,
        _label_points,
    ) = scene3d.build_scene(args)

    scenario_data = _scenario_payload(
        scenarios,
        static_layers,
        stage_layers,
        isocenter,
        _target_info(targets, isocenter, args.data_root),
    )
    view_modes = {key: {"label": label, "layers": list(layers)} for key, label, layers in STAGE_MODES}
    first_title = next(iter(scenario_data.values()))["title"] + " | " + view_modes[DEFAULT_STAGE]["label"]

    html = HTML_TEMPLATE
    html = html.replace("###SCENARIO_BUTTONS###", _scenario_buttons(scenario_data))
    html = html.replace("###MODE_BUTTONS###", _mode_buttons())
    html = html.replace("###LEGEND_HTML###", _legend_html())
    html = html.replace("###FIRST_TITLE###", first_title)
    html = html.replace("###SCENARIOS_JSON###", json.dumps(scenario_data, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("###AXIS_LIMITS_JSON###", json.dumps([round(float(v), 2) for v in axis_limits]))
    html = html.replace("###VIEW_MODES_JSON###", json.dumps(view_modes, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("###ORI_TRACES_JSON###", json.dumps(_orientation_traces(), separators=(",", ":")))
    html = html.replace("###DEFAULT_STAGE###", DEFAULT_STAGE)
    return html


def write_viewers(html: str, paths: list[Path]) -> list[Path]:
    written = []
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        written.append(path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the interactive 12-met rigid-error case viewer.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=ANALYZER_DIR / "docs" / "rigid_error_case",
        help="Versioned docs output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYZER_DIR / "output" / "rigid_error_case",
        help="Local generated output directory.",
    )
    parser.add_argument("--resolution", choices=["light", "standard", "dense"], default="standard")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    html = build_html(args.output_dir, args.resolution)
    paths = write_viewers(html, [args.docs_dir / "viewer.html", args.output_dir / "viewer.html"])
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
