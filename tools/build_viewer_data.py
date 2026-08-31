# tools/build_viewer_data.py
"""Fold a run's traces, its scenarios and the topology into one self-contained
HTML reader (run-viewer design, "The fold script" and "The viewer").

Reading a rollout today means holding an episode's geometry, the forecast
revision, the exposure arithmetic and up to eleven LLM calls per acting hour in
your head at once, across seven episodes and three runs each. Every number in
docs/superpowers/2026-08-28-control-arm-findings.md was hand-derived; its
sharpest finding is a claim about a point moving between two hours of one run
of one episode, and it took a manual reconstruction to see. It should have
taken a scrubber.

Post-hoc and read-only. It never re-runs, re-scores or edits anything -- a UI
that can mutate a run is a second harness.

Runs in THIS repo's env: pure stdlib plus pyyaml, and it imports nothing from
multilayer_optical_network. It is NOT the offline-build exception.

Run: python tools/build_viewer_data.py --out eval/viewer/index.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
TRACES_DIR = REPO_ROOT / "eval" / "traces"
SCENARIOS_DIR = (REPO_ROOT / "src" / "storm_reoptimizer" / "eval"
                 / "scenarios")
TOPOLOGY_PATH = (REPO_ROOT / "src" / "storm_reoptimizer" / "data"
                 / "toy_india_topology.json")
DEFAULT_OUT = REPO_ROOT / "eval" / "viewer" / "index.html"


def load_topology(path: Path) -> dict:
    """143 nodes and 180 edges, for geographic orientation and for the aerial
    overlay. utf-8-sig: this file has a BOM."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        "nodes": {n["id"]: [n["lat"], n["lon"]]
                  for n in raw["graph"]["nodes"]},
        "edges": [{"src": e["src"], "dst": e["dst"],
                   "mount_type": e["mount_type"],
                   "length_km": e["length_km"]}
                  for e in raw["graph"]["edges"]],
    }


def load_episode(path: Path) -> dict:
    """One scenario YAML's geometry and gold block.

    The cone rings are GeoJSON, hence lon-first. Flip to [lat, lon] HERE, once,
    so nothing downstream has to remember which convention it is holding."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    forecast: dict = {}
    for issued_at, horizons in (raw.get("forecast") or {}).items():
        forecast[issued_at] = {
            horizon: {
                "polygon": [[lat, lon] for lon, lat
                            in cone["cone"]["coordinates"][0]],
                "center": cone["center"],
                "width_km": cone["width_km"],
            }
            for horizon, cone in horizons.items()}
    gold = dict(raw.get("gold") or {})
    gold["gold_spare_action"] = (
        raw.get("metadata") or {}).get("gold_spare_action")
    return {
        "id": raw["id"], "pair": raw.get("pair"),
        "hours": raw["hours"], "decision_hour": raw["decision_hour"],
        "actionable_service": raw["service_under_test"],
        "damage_radius_km": raw["damage_radius_km"],
        "lead_time_hours": raw["lead_time_hours"],
        "spares_on_hand": raw["spares_on_hand"],
        "gold": gold, "forecast": forecast, "runs": [],
    }


def _exposure_rows(hour: dict) -> list[dict]:
    """One row per service the hour's ground truth exposed, per horizon,
    marked shown or omitted by comparing `observation` against `projected`.

    The gap between what was true and what the agent was shown is the entire
    subject of the eval-fairness design; this is where it becomes visible."""
    observation = hour.get("observation") or {}
    shown = set((hour.get("projected") or {}).get("exposure") or {})
    rows = []
    for svc, per_horizon in (observation.get("exposure") or {}).items():
        for horizon, entry in per_horizon.items():
            rows.append({"service_id": svc, "horizon": horizon,
                         "shown": svc in shown, **entry})
    rows.sort(key=lambda r: (-r.get("expected_capacity_at_risk_gbps", 0.0),
                            r["service_id"], r["horizon"]))
    return rows


def _competing_services(hour: dict) -> list[str]:
    """The actionable service plus every member of every restorable group at
    this hour -- the only services that actually contend for the depot's one
    spare transponder pair (observation._restorable_groups' own
    depot-eligible, co-terminating definition). `hour.services` (the full
    roster `get_services` returns, ~500+ on the real eval state) is
    deliberately NOT the map's drawing set: most of it is background traffic
    the storm never comes near, and drawing all of it buries the one rival
    claim that actually matters in noise."""
    observation = hour.get("observation") or {}
    members = {
        member
        for groups in (observation.get("restorable_groups") or {}).values()
        for group in groups
        for member in group.get("members", ())
    }
    actionable = (observation.get("actionable_service")
                 or observation.get("service_under_test"))
    if actionable:
        members.add(actionable)
    return sorted(members)


def _mark_committed(hour: dict) -> None:
    """Flag the candidate that actually committed, so each hour opens on what
    happened and divergence is a deliberate click."""
    for step in hour.get("iterations") or ():
        chosen = (step.get("objective") or {}).get("choice")
        for candidate in ((step.get("menu") or {}).get("candidates") or ()):
            candidate["committed"] = (
                step.get("outcome") == "committed"
                and candidate.get("candidate_label") == chosen)


def load_run(path: Path) -> dict:
    """One trace, enriched. Tolerates a pre-Phase-A trace: every key added by
    the recording change is optional and defaults to empty, so the 21 archived
    control rollouts load beside a fresh run (run-viewer design, §6.2)."""
    trace = json.loads(path.read_text(encoding="utf-8"))
    hours = []
    for hour in trace.get("hours") or ():
        enriched = dict(hour)
        observation = enriched.get("observation") or {}
        enriched["actionable_service"] = (
            observation.get("actionable_service")
            or observation.get("service_under_test"))
        enriched["exposure_rows"] = _exposure_rows(enriched)
        enriched["competing_services"] = _competing_services(enriched)
        enriched.setdefault("service_points", {})
        enriched.setdefault("service_paths", {})
        enriched.setdefault("unmapped_nodes", {})
        _mark_committed(enriched)
        hours.append(enriched)
    return {"decider_name": trace["decider_name"],
            "run_index": trace["run_index"],
            "terminal_status": trace["terminal_status"],
            "spares_remaining": trace["spares_remaining"],
            "ledger_debits": trace.get("ledger_debits") or [],
            "actions": trace.get("actions") or [],
            "oms_nodes": trace.get("oms_nodes") or {},
            "tool_calls": trace.get("tool_calls"),
            "hours": hours}


def fold(traces_dir: Path, scenarios_dir: Path, topology_path: Path) -> dict:
    episodes = {}
    for path in sorted(scenarios_dir.glob("*.yaml")):
        episode = load_episode(path)
        episodes[episode["id"]] = episode
    for path in sorted(traces_dir.glob("*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        episode = episodes.get(trace.get("scenario_id"))
        if episode is None:
            print(f"build_viewer_data: WARNING skipping {path.name}: no "
                  f"scenario {trace.get('scenario_id')!r}", flush=True)
            continue
        episode["runs"].append(load_run(path))
    for episode in episodes.values():
        episode["runs"].sort(
            key=lambda r: (r["decider_name"], r["run_index"]))
    return {"topology": load_topology(topology_path), "episodes": episodes}


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>storm-reoptimizer run viewer</title>
<style>__CSS__</style></head>
<body>
<header id="controls">
  <select id="episode"></select>
  <select id="run"></select>
  <label><input type="checkbox" id="plant"> aerial plant</label>
  <button id="reset-view" type="button">reset view</button>
  <div id="legend">
    <span><span class="swatch" style="background:#1a1a1a"></span>working</span>
    <span><span class="swatch" style="background:#888"></span>protection</span>
    <span><span class="swatch" style="background:#1a4fcc"></span>working (spotlit)</span>
    <span><span class="swatch" style="background:#a626cc"></span>protection (spotlit)</span>
    <span><span class="swatch" style="background:#cc6633"></span>committed route</span>
  </div>
  <span id="gold"></span>
</header>
<main>
  <svg id="map" viewBox="0 0 800 800" preserveAspectRatio="xMidYMid meet">
    <g id="layer-plant"></g><g id="layer-cones"></g>
    <g id="layer-paths"></g><g id="layer-candidate"></g>
    <g id="layer-points"></g>
  </svg>
  <aside id="panels">
    <section id="saw"></section>
    <section id="said"></section>
  </aside>
</main>
<footer><div id="scrubber"></div></footer>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>__JS__</script>
</body></html>
"""


_CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: ui-monospace, "Cascadia Code", Consolas,
       monospace; font-size: 13px; color: #1a1a1a; background: #fff; }
#controls { display: flex; align-items: center; gap: 10px; padding: 8px 12px;
            border-bottom: 1px solid #ccc; }
#controls select, #controls label { font: inherit; }
#gold { margin-left: auto; font-weight: bold; }
#reset-view { font: inherit; cursor: pointer; }
#legend { display: flex; gap: 10px; font-size: 11px; color: #555;
          align-items: center; white-space: nowrap; }
.swatch { display: inline-block; width: 14px; height: 3px;
          margin-right: 3px; vertical-align: middle; }
main { display: grid; grid-template-columns: 1fr 380px; gap: 0;
       height: calc(100vh - 90px); }
#map { width: 100%; height: 100%; background: #f7f7f5; cursor: grab; }
#map.dragging { cursor: grabbing; }
#panels { overflow-y: auto; border-left: 1px solid #ccc; padding: 8px; }
#panels section { margin-bottom: 16px; }
#panels h3 { margin: 0 0 4px 0; font-size: 12px; text-transform: uppercase;
             letter-spacing: 0.04em; color: #555; }
table { border-collapse: collapse; width: 100%; font: inherit; }
table td, table th { border: 1px solid #ddd; padding: 2px 5px;
                      text-align: left; white-space: nowrap; }
th { background: #f0f0ee; }
.shown, .omitted { display: inline-block; padding: 0 4px; border-radius: 3px;
                    font-size: 11px; font-weight: bold; }
.shown { background: #d6f0d6; color: #146214; }
.omitted { background: #eee; color: #777; }
.candidate-row { cursor: default; }
.candidate-row.committed { cursor: pointer; font-weight: bold; }
.candidate-row.committed:hover { background: #f0f4ff; }
.candidate-row.selected { outline: 2px solid #cc6633; }
.exposure-row { cursor: pointer; }
.exposure-row:hover { background: #f0f4ff; }
.exposure-row.selected { outline: 2px solid #1a4fcc; background: #eaf0ff; }
footer { border-top: 1px solid #ccc; padding: 6px 12px; }
#scrubber { display: flex; gap: 1px; }
.hcell { flex: 1; padding: 3px 2px; text-align: center; cursor: pointer;
         border: 1px solid #ddd; background: #f5f5f3; }
.hcell.current { background: #3366cc; color: #fff; }
.hcell.decision { border-color: #cc6633; border-width: 2px; }
.hcell .strip { height: 4px; margin-top: 2px; }
.hcell .strip.match { background: #2e8b2e; }
.hcell .strip.mismatch { background: #cc3333; }
.hcell .strip.unknown { background: #ccc; }
.aerial { stroke-dasharray: 4 3; }
.gap-marker { stroke: #cc3333; stroke-width: 2; }
pre.reasoning { white-space: pre-wrap; background: #f7f7f5; padding: 6px;
                border: 1px solid #eee; margin: 4px 0; }
.rejection { color: #a33; margin: 2px 0; }
.contested { background: #fff4e0; border: 1px solid #e0c080; padding: 4px;
             margin: 4px 0; }
.step-label { margin: 10px 0 2px; font-weight: bold; }
details.raw-json { margin: 4px 0 10px; }
details.raw-json summary { cursor: pointer; font-size: 11px; color: #555; }
details.raw-json pre { max-height: 320px; overflow: auto; background: #f7f7f5;
                        border: 1px solid #eee; padding: 6px; margin: 4px 0;
                        white-space: pre-wrap; word-break: break-word; }
"""


_JS = r"""
const P = JSON.parse(document.getElementById('payload').textContent);
let state = {episode: null, run: null, hourIndex: 0, spotlight: null,
             candidateVisible: false};

// ---- geometry -------------------------------------------------------

const BBOX = {latMin: 8, latMax: 35, lonMin: 68, lonMax: 97};

function project(lat, lon) {
    const x = (lon - BBOX.lonMin) / (BBOX.lonMax - BBOX.lonMin) * 800;
    const y = (BBOX.latMax - lat) / (BBOX.latMax - BBOX.latMin) * 800;
    return [x, y];
}

let EDGE_INDEX = null;
function buildEdgeIndex() {
    EDGE_INDEX = new Map();
    for (const e of P.topology.edges) {
        EDGE_INDEX.set(e.src + '|' + e.dst, e);
        EDGE_INDEX.set(e.dst + '|' + e.src, e);
    }
}

function isAerial(nodeA, nodeB) {
    if (!EDGE_INDEX) buildEdgeIndex();
    const e = EDGE_INDEX.get(nodeA + '|' + nodeB);
    return !!e && e.mount_type === 'aerial';
}

// Namespace read off the existing <svg> node rather than spelled out as a
// URL literal, so this file carries no bare protocol-scheme substring.
const SVG_NS = document.getElementById('map').namespaceURI;

function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
}

function layer(name) { return document.getElementById('layer-' + name); }

function clearLayer(name) {
    const g = layer(name);
    while (g.firstChild) g.removeChild(g.firstChild);
}

// ---- zoom / pan ---------------------------------------------------------
// viewBox manipulation, not a CSS transform, so strokes and text stay crisp
// at any zoom level.

const FULL_VIEW = {x: 0, y: 0, w: 800, h: 800};
let view = {x: 0, y: 0, w: 800, h: 800};

function applyView() {
    document.getElementById('map').setAttribute(
        'viewBox', `${view.x} ${view.y} ${view.w} ${view.h}`);
}

function resetView() {
    view = {...FULL_VIEW};
    applyView();
}

function svgPoint(evt) {
    const svg = document.getElementById('map');
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
}

function zoomAt(p, factor) {
    const newW = Math.min(FULL_VIEW.w, Math.max(20, view.w * factor));
    const newH = Math.min(FULL_VIEW.h, Math.max(20, view.h * factor));
    view.x = p.x - (p.x - view.x) * (newW / view.w);
    view.y = p.y - (p.y - view.y) * (newH / view.h);
    view.w = newW;
    view.h = newH;
    applyView();
}

document.getElementById('map').addEventListener('wheel', (ev) => {
    ev.preventDefault();
    zoomAt(svgPoint(ev), ev.deltaY < 0 ? 0.9 : 1.1);
}, {passive: false});

let panState = null;
document.getElementById('map').addEventListener('mousedown', (ev) => {
    panState = {startClientX: ev.clientX, startClientY: ev.clientY,
                startView: {...view}};
    document.getElementById('map').classList.add('dragging');
});
window.addEventListener('mousemove', (ev) => {
    if (!panState) return;
    const rect = document.getElementById('map').getBoundingClientRect();
    const scaleX = panState.startView.w / rect.width;
    const scaleY = panState.startView.h / rect.height;
    view.x = panState.startView.x - (ev.clientX - panState.startClientX) * scaleX;
    view.y = panState.startView.y - (ev.clientY - panState.startClientY) * scaleY;
    applyView();
});
window.addEventListener('mouseup', () => {
    panState = null;
    document.getElementById('map').classList.remove('dragging');
});
document.getElementById('reset-view').addEventListener('click', resetView);

// ---- plant ------------------------------------------------------------

function drawPlant(showAerial) {
    clearLayer('plant');
    const g = layer('plant');
    const nodes = P.topology.nodes;
    for (const e of P.topology.edges) {
        const a = nodes[e.src];
        const b = nodes[e.dst];
        if (!a || !b) continue;
        const [x1, y1] = project(a[0], a[1]);
        const [x2, y2] = project(b[0], b[1]);
        const aerial = showAerial && e.mount_type === 'aerial';
        const path = svgEl('path', {
            d: `M${x1},${y1}L${x2},${y2}`,
            stroke: '#ccc', 'stroke-width': 1, fill: 'none',
            class: aerial ? 'aerial' : '',
        });
        g.appendChild(path);
    }
}

// ---- polylines ----------------------------------------------------------

function drawPolyline(nodes, opts) {
    const weight = opts.weight || 2;
    const colour = opts.colour || '#333';
    const layerName = opts.layer || 'paths';
    const g = layer(layerName);
    const topoNodes = P.topology.nodes;
    const hour = currentHour();
    const unmapped = (hour && hour.unmapped_nodes) || {};
    for (let i = 0; i < nodes.length - 1; i++) {
        const a = nodes[i];
        const b = nodes[i + 1];
        const ca = topoNodes[a];
        const cb = topoNodes[b];
        if (!ca || !cb || unmapped[a] || unmapped[b]) {
            // A node the topology lacks breaks the line with a visible gap
            // marker rather than joining across it.
            const known = ca || cb;
            if (known) {
                const [gx, gy] = project(known[0], known[1]);
                g.appendChild(svgEl('circle', {
                    cx: gx, cy: gy, r: 5, class: 'gap-marker', fill: 'none',
                }));
            }
            continue;
        }
        const [x1, y1] = project(ca[0], ca[1]);
        const [x2, y2] = project(cb[0], cb[1]);
        const aerial = isAerial(a, b);
        g.appendChild(svgEl('path', {
            d: `M${x1},${y1}L${x2},${y2}`,
            stroke: colour, 'stroke-width': weight, fill: 'none',
            class: aerial ? 'aerial' : '',
        }));
    }
}

// ---- cones --------------------------------------------------------------

function drawCones(episode, hour) {
    clearLayer('cones');
    const g = layer('cones');
    const forecast = episode.forecast || {};
    const issuances = Object.keys(forecast).sort();
    // The issuance "in force" at this hour is the latest one issued at or
    // before this hour.
    let inForce = null;
    for (const issued of issuances) {
        if (issued <= hour.hour) inForce = issued;
    }
    if (inForce === null && issuances.length) inForce = issuances[0];
    for (const issued of issuances) {
        const horizons = forecast[issued];
        const graded = issued === inForce;
        for (const horizonKey in horizons) {
            const cone = horizons[horizonKey];
            const pts = (cone.polygon || []).map(([lat, lon]) => project(lat, lon));
            if (!pts.length) continue;
            const d = 'M' + pts.map(p => p.join(',')).join('L') + 'Z';
            g.appendChild(svgEl('path', {
                d,
                fill: graded ? 'rgba(51,102,204,0.12)' : 'rgba(150,150,150,0.06)',
                stroke: graded ? '#3366cc' : '#999',
                'stroke-width': graded ? 1.5 : 1,
                'stroke-dasharray': graded ? '' : '2 2',
                'data-issued': issued, 'data-horizon': horizonKey,
            }));
        }
    }
}

// ---- services -------------------------------------------------------

function drawService(id, hour, opts) {
    opts = opts || {};
    const dim = !!opts.dim;
    const spot = !!opts.spotlight;
    const paths = (hour.service_paths || {})[id];
    if (paths) {
        if (paths.working) {
            drawPolyline(paths.working, {
                weight: spot ? 4 : (dim ? 1 : (opts.emphasis ? 3.5 : 2.5)),
                colour: spot ? '#1a4fcc' : (dim ? '#ddd' : '#1a1a1a'),
                layer: 'paths'});
        }
        if (paths.protection) {
            drawPolyline(paths.protection, {
                weight: spot ? 3 : (dim ? 1 : (opts.emphasis ? 2 : 1.2)),
                colour: spot ? '#a626cc' : (dim ? '#ddd' : '#888'),
                layer: 'paths'});
        }
    }
    const point = (hour.service_points || {})[id];
    if (point) {
        const [x, y] = project(point[0], point[1]);
        layer('points').appendChild(svgEl('circle', {
            cx: x, cy: y, r: spot ? 7 : (opts.emphasis ? 6 : 4),
            fill: spot ? '#ffd23f' : (dim ? '#ddd' : (opts.colour || '#cc3333')),
            stroke: spot ? '#1a1a1a' : '#fff',
            'stroke-width': spot ? 2 : 1,
            'data-service': id,
        }));
    }
}

// ---- candidates -------------------------------------------------------

function drawCandidate(candidate, omsNodes) {
    clearLayer('candidate');
    omsNodes = omsNodes || {};

    const newLightpaths = candidate.new_lightpaths || [];

    // An ip_reroute (and any candidate with no new_lightpaths at all) lights
    // nothing new -- by construction it only re-homes IP traffic onto a
    // lightpath that already exists, and that lightpath is already on the
    // map as this service's working or protection line (drawService, drawn
    // every hour regardless of which candidate is spotlit). There is no
    // second, different route to overlay, so say that plainly instead of
    // reporting it as a gap.
    if (newLightpaths.length === 0 &&
        (candidate.reused_lightpaths || []).length > 0) {
        const note = svgEl('text', {
            x: 10, y: 20, fill: '#555', 'font-size': 12,
        });
        note.textContent = 'no new route: this candidate reuses an ' +
            "existing lightpath, already drawn as the service's " +
            'working/protection line';
        layer('candidate').appendChild(note);
        return;
    }

    // reused_lightpaths are bare lightpath-id strings in every trace on
    // disk today, never {oms_sequence} objects, and this payload carries
    // no lightpath -> OMS map to resolve them -- a genuine payload-shape
    // gap, not something fixable here. Track whether any reused leg fails
    // to resolve, so a candidate that mixes an unresolvable reused leg
    // with a resolvable new leg does not silently draw the new leg alone
    // as if it were the WHOLE route.
    const omsSeq = [];
    let unresolvedReused = false;
    for (const lp of candidate.reused_lightpaths || []) {
        const seq = (lp && lp.oms_sequence) ? lp.oms_sequence
            : (typeof lp === 'string' ? [lp] : null);
        if (!seq) { unresolvedReused = true; continue; }
        for (const oms of seq) {
            if (omsNodes[oms]) omsSeq.push(oms);
            else unresolvedReused = true;
        }
    }
    for (const lp of newLightpaths) {
        omsSeq.push(...(lp.oms_sequence || []));
    }

    if (unresolvedReused) {
        // Honestly-empty rather than plausibly-partial: say why, draw
        // nothing that could be mistaken for the full route.
        const note = svgEl('text', {
            x: 10, y: 20, fill: '#a33', 'font-size': 12,
        });
        note.textContent = 'candidate route incomplete: a reused ' +
            'lightpath has no OMS geometry in this payload';
        layer('candidate').appendChild(note);
        return;
    }

    const nodeSeq = [];
    for (const oms of omsSeq) {
        const pair = omsNodes[oms];
        if (!pair) continue;
        if (nodeSeq.length === 0) nodeSeq.push(pair[0]);
        else if (nodeSeq[nodeSeq.length - 1] !== pair[0]) nodeSeq.push(pair[0]);
        nodeSeq.push(pair[1]);
    }
    if (nodeSeq.length >= 2) {
        drawPolyline(nodeSeq, {weight: 3, colour: '#cc6633', layer: 'candidate'});
    }
}

// ---- panels -------------------------------------------------------------

function currentEpisode() { return P.episodes[state.episode]; }
function currentRun() {
    const ep = currentEpisode();
    if (!ep) return null;
    return ep.runs[state.run] || null;
}
function currentHour() {
    const run = currentRun();
    if (!run) return null;
    return run.hours[state.hourIndex] || null;
}

function selectedIteration(hour) {
    if (!hour || !hour.iterations || !hour.iterations.length) return null;
    return hour.iterations[hour.iterations.length - 1];
}

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function committedCandidate(hour) {
    const iter = selectedIteration(hour);
    const candidates = (iter && iter.menu && iter.menu.candidates) || [];
    return candidates.find(c => c.committed) || null;
}

// A priced candidate menu, rendered wherever it was ACTUALLY shown to the
// model (inline in "What it said", at the objective step of its own
// iteration) rather than pinned in a panel of its own at the top -- there is
// no "menu" independent of an hour and an iteration, so there is no honest
// place to show one before the reasoning it fed.
function candidateTable(candidates) {
    const table = document.createElement('table');
    table.innerHTML = '<tr><th>label</th><th>lever</th><th>spares</th>' +
        '<th>restored</th><th>shortfall</th><th>cost</th></tr>';
    (candidates || []).forEach((c) => {
        const tr = document.createElement('tr');
        const committed = !!c.committed;
        tr.className = 'candidate-row' + (committed ? ' committed' : '') +
            (committed && state.candidateVisible ? ' selected' : '');
        // spares_needed is site -> transponder count (ledger.py's per-site
        // ledger, exposure-and-depot design §4.1) -- render each site's
        // charge rather than a single count, since a hybrid candidate can
        // charge two DIFFERENT sites unevenly.
        const sparesText = Object.entries(c.spares_needed || {})
            .map(([site, n]) => `${site}:${n}`).join(', ') || '-';
        tr.innerHTML =
            `<td>${esc(c.candidate_label)}</td><td>${esc(c.lever)}</td>` +
            `<td>${esc(sparesText)}</td>` +
            `<td>${esc(c.restored_gbps)}</td>` +
            `<td>${esc(c.shortfall_gbps)}</td>` +
            `<td>${esc(JSON.stringify(c.cost_vector || {}))}</td>`;
        if (committed) {
            tr.title = state.candidateVisible
                ? 'click to hide its route on the map'
                : 'click to show its route on the map';
            tr.addEventListener('click', () => {
                state.candidateVisible = !state.candidateVisible;
                renderAll();
            });
        }
        table.appendChild(tr);
    });
    return table;
}

// The cheaper menu the CONSTRAINTS decision sees, before any avoid set narrows
// it -- no cost vector, no committed flag, it is never what gets picked.
function unconstrainedMenuTable(menu) {
    const table = document.createElement('table');
    table.innerHTML = '<tr><th>label</th><th>lever</th><th>spares</th></tr>';
    ((menu && menu.candidates) || []).forEach((c) => {
        const tr = document.createElement('tr');
        const sparesText = Object.entries(c.spares_needed || {})
            .map(([site, n]) => `${site}:${n}`).join(', ') || '-';
        tr.innerHTML = `<td>${esc(c.candidate_label)}</td>` +
            `<td>${esc(c.lever)}</td><td>${esc(sparesText)}</td>`;
        table.appendChild(tr);
    });
    return table;
}

// The literal JSON body one API call received (agent.py's _user_content
// wraps this exact shape and json.dumps's it) -- collapsed by default since
// it duplicates the curated tables above it, but never trimmed or
// re-summarized: this is "everything", verbatim.
function rawJson(label, obj) {
    const details = document.createElement('details');
    details.className = 'raw-json';
    const summary = document.createElement('summary');
    summary.textContent = 'raw payload sent to ' + label;
    details.appendChild(summary);
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(obj, null, 2);
    details.appendChild(pre);
    return details;
}

function stepLabel(text) {
    const div = document.createElement('div');
    div.className = 'step-label';
    div.textContent = text;
    return div;
}

function renderSaw(hour) {
    const el = document.getElementById('saw');
    el.innerHTML = '<h3>What the agent saw (click a row to spotlight it)</h3>';
    if (!hour) return;
    // Only what the agent was actually shown -- the omitted rows this table
    // used to include (project_observation's own trim) are noise here; the
    // trim itself is still checked in tests/eval/test_build_viewer_data.py.
    const rows = (hour.exposure_rows || []).filter(r => r.shown);
    const table = document.createElement('table');
    table.innerHTML = '<tr><th>service</th><th>horizon</th><th>offset_km</th>' +
        '<th>p_cut</th><th>demand_gbps</th><th>ECAR</th></tr>';
    for (const r of rows) {
        const tr = document.createElement('tr');
        tr.className = 'exposure-row' +
            (r.service_id === state.spotlight ? ' selected' : '');
        tr.innerHTML =
            `<td>${esc(r.service_id)}</td><td>${esc(r.horizon)}</td>` +
            `<td>${esc(r.offset_km)}</td><td>${esc(r.p_cut)}</td>` +
            `<td>${esc(r.demand_gbps)}</td>` +
            `<td>${esc(r.expected_capacity_at_risk_gbps)}</td>`;
        tr.addEventListener('click', () => {
            state.spotlight = state.spotlight === r.service_id
                ? null : r.service_id;
            renderSaw(hour);
            renderMap();
        });
        table.appendChild(tr);
    }
    el.appendChild(table);

    const obs = hour.observation || {};
    const totals = obs.horizon_totals || {};
    const totalsPre = document.createElement('pre');
    totalsPre.className = 'reasoning';
    totalsPre.textContent = 'horizon_totals: ' + JSON.stringify(totals, null, 1);
    el.appendChild(totalsPre);

    // Every field below is Observation.to_dict()'s own per-hour ledger
    // state -- what the agent actually knew going into THIS hour, not a
    // run-final scalar (spares_remaining) or a flat episode constant
    // (the YAML's initial spares_on_hand / lead_time_hours). Nothing here
    // is recomputed from run.actions.
    const misc = document.createElement('pre');
    misc.className = 'reasoning';
    misc.textContent =
        `spares_on_hand: ${JSON.stringify(obs.spares_on_hand)}\n` +
        `spares_spent: ${JSON.stringify(obs.spares_spent)}\n` +
        `lead_time_hours: ${JSON.stringify(obs.lead_time_hours || {})}\n` +
        `risk_group_ids: ${JSON.stringify(obs.risk_group_ids || {})}\n` +
        `actions_taken: ${JSON.stringify(obs.actions_taken || [])}`;
    el.appendChild(misc);

    el.appendChild(rawJson('timing', {observation: hour.projected}));
}

function renderSaid(hour) {
    const el = document.getElementById('said');
    el.innerHTML = '<h3>What it said</h3>';
    if (!hour) return;
    if (!hour.timing) {
        el.appendChild(document.createTextNode('(no reasoning recorded)'));
        return;
    }
    el.appendChild(stepLabel('1. Timing'));
    const timing = document.createElement('div');
    timing.innerHTML = `<b>action:</b> ${esc(hour.timing.action)}`;
    const timingReasoning = document.createElement('pre');
    timingReasoning.className = 'reasoning';
    timingReasoning.textContent = hour.timing.reasoning || '';
    el.appendChild(timing);
    el.appendChild(timingReasoning);
    if (hour.timing.contested_claim) {
        const cc = document.createElement('div');
        cc.className = 'contested';
        cc.textContent = 'contested_claim (timing): ' +
            JSON.stringify(hour.timing.contested_claim);
        el.appendChild(cc);
    }
    el.appendChild(rawJson('timing', {observation: hour.projected}));

    for (const rej of hour.rejections || []) {
        const d = document.createElement('div');
        d.className = 'rejection';
        d.textContent = 'rejected: ' + JSON.stringify(rej);
        el.appendChild(d);
    }

    // Each iteration replays BOTH remaining decisions in the order the model
    // actually received them: the unconstrained menu and the constraints
    // call's raw input, then what it decided; then the PRICED menu that
    // decision produced and the objective call's raw input, then what it
    // chose. The candidate table therefore sits right where it was seen --
    // immediately before the decision it fed -- not in a panel of its own.
    (hour.iterations || []).forEach((it, idx) => {
        const h = document.createElement('div');
        h.innerHTML = `<b>iteration ${esc(it.iteration)}</b> -- ` +
            `menu ${esc(it.menu_status)} (${esc(it.menu_size)}) -- ` +
            `outcome ${esc(it.outcome)}`;
        el.appendChild(h);

        if (idx === 0 && hour.unconstrained_menu) {
            el.appendChild(stepLabel(
                '2. Constraints -- menu before constraining'));
            el.appendChild(unconstrainedMenuTable(hour.unconstrained_menu));
        }
        el.appendChild(rawJson(`constraints, iteration ${it.iteration}`,
            {observation: it.projected,
             unconstrained_menu: hour.unconstrained_menu}));

        const constraints = it.constraints || {};
        el.appendChild(stepLabel('2. Constraints decision'));
        const cPre = document.createElement('pre');
        cPre.className = 'reasoning';
        cPre.textContent = `avoid: ${JSON.stringify(constraints.avoid || {})}\n` +
            (constraints.reasoning || '');
        el.appendChild(cPre);
        if (constraints.contested_claim) {
            const cc = document.createElement('div');
            cc.className = 'contested';
            cc.textContent = 'contested_claim (constraints): ' +
                JSON.stringify(constraints.contested_claim);
            el.appendChild(cc);
        }

        el.appendChild(stepLabel('3. Objective -- priced candidate menu'));
        el.appendChild(candidateTable((it.menu && it.menu.candidates) || []));
        el.appendChild(rawJson(`objective, iteration ${it.iteration}`,
            {observation: it.projected, menu: it.menu}));

        const objective = it.objective || {};
        el.appendChild(stepLabel('3. Objective decision'));
        const oPre = document.createElement('pre');
        oPre.className = 'reasoning';
        oPre.textContent = `choice: ${objective.choice}\n` +
            (objective.reasoning || '');
        el.appendChild(oPre);
        // Both decisions can carry a contested_claim, and T3's graded
        // spend/conserve judgement lives in the objective decision's
        // `choice` -- render both, independently labelled, rather than
        // silently preferring one when both are populated (final-review
        // fix, 2026-08-29).
        if (objective.contested_claim) {
            const cc = document.createElement('div');
            cc.className = 'contested';
            cc.textContent = 'contested_claim (objective): ' +
                JSON.stringify(objective.contested_claim);
            el.appendChild(cc);
        }
    });
}

// ---- scrubber -------------------------------------------------------

function renderScrubber(episode, run) {
    const el = document.getElementById('scrubber');
    el.innerHTML = '';
    if (!episode || !run) return;
    const gold = episode.gold || {};
    // Actions live on the run, each tagged with the hour it landed in --
    // hour records carry no `actions` key of their own.
    const actionsByHour = {};
    for (const a of run.actions || []) {
        (actionsByHour[a.hour] = actionsByHour[a.hour] || []).push(a);
    }
    run.hours.forEach((h, i) => {
        const cell = document.createElement('div');
        cell.className = 'hcell' +
            (i === state.hourIndex ? ' current' : '') +
            (h.hour === episode.decision_hour ? ' decision' : '');
        const label = document.createElement('div');
        label.textContent = h.hour;
        cell.appendChild(label);

        const strip = document.createElement('div');
        const acted = actionsByHour[h.hour] || [];
        let cls = 'unknown';
        if (gold.gold_spare_action && acted.length) {
            // Any site charged (a non-empty, non-zero `spares` dict) spends
            // a physical spare (optical_reroute); an empty dict does not
            // (ip_reroute / rate-reduce) -- that split is what
            // gold_spare_action ("conserve" vs "spend") is judging.
            // Action.spares (site -> count) replaced the old scalar
            // `pairs: int` (exposure-and-depot design, §4.1); reading
            // `a.pairs` here always evaluated to `(undefined || 0) > 0` ->
            // false, silently misreporting every action as "conserve"
            // (whole-branch final review, finding 2).
            const spent = acted.some(
                a => Object.values(a.spares || {}).some(n => n > 0));
            const agentAction = spent ? 'spend' : 'conserve';
            cls = agentAction === gold.gold_spare_action ? 'match' : 'mismatch';
        }
        strip.className = 'strip ' + cls;
        strip.title = acted.length
            ? `agent action(s): ${JSON.stringify(acted)}`
            : 'no action this hour';
        cell.appendChild(strip);

        cell.addEventListener('click', () => {
            state.hourIndex = i;
            state.candidateVisible = false;
            renderAll();
        });
        el.appendChild(cell);
    });
}

// ---- top-level render -------------------------------------------------

function renderMap() {
    clearLayer('paths');
    clearLayer('points');
    clearLayer('candidate');
    drawPlant(document.getElementById('plant').checked);
    const episode = currentEpisode();
    const hour = currentHour();
    if (!episode || !hour) { clearLayer('cones'); return; }
    drawCones(episode, hour);
    // Only the services that actually contend for this hour's spare (the SUT
    // plus every restorable-group member) -- not hour.services, the full
    // ~500+-service network roster, most of which the storm never touches.
    const services = hour.competing_services || [];
    const spotlightId = services.includes(state.spotlight) ? state.spotlight : null;
    for (const svc of services) {
        if (svc === spotlightId) continue;
        const emphasis = svc === (hour.actionable_service || episode.actionable_service);
        drawService(svc, hour, {
            emphasis,
            dim: !!spotlightId,
            colour: emphasis ? '#cc3333' : '#3366cc',
        });
    }
    // Drawn last, in its own pass, so it sits on top of every dimmed path.
    if (spotlightId) {
        drawService(spotlightId, hour, {spotlight: true});
    }
    if (state.candidateVisible) {
        const committed = committedCandidate(hour);
        if (committed) {
            const run = currentRun();
            drawCandidate(committed, run ? run.oms_nodes : {});
        }
    }
}

function renderGold() {
    const episode = currentEpisode();
    const el = document.getElementById('gold');
    if (!episode) { el.textContent = ''; return; }
    const gold = episode.gold || {};
    el.textContent = `gold: ${gold.label || ''} (${gold.gold_spare_action || ''})`;
}

function renderAll() {
    const episode = currentEpisode();
    const run = currentRun();
    const hour = currentHour();
    renderMap();
    renderSaw(hour);
    renderSaid(hour);
    renderScrubber(episode, run);
    renderGold();
}

function populateDropdowns() {
    const epSel = document.getElementById('episode');
    epSel.innerHTML = '';
    Object.keys(P.episodes).sort().forEach((id) => {
        const opt = svgOptionLike(id, id);
        epSel.appendChild(opt);
    });
    epSel.addEventListener('change', () => {
        state.episode = epSel.value;
        state.run = 0;
        state.hourIndex = 0;
        state.spotlight = null;
        state.candidateVisible = false;
        resetView();
        populateRunDropdown();
        renderAll();
    });
    state.episode = epSel.value;
    populateRunDropdown();
}

function svgOptionLike(value, text) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    return opt;
}

function populateRunDropdown() {
    const runSel = document.getElementById('run');
    runSel.innerHTML = '';
    const episode = currentEpisode();
    (episode ? episode.runs : []).forEach((r, i) => {
        runSel.appendChild(svgOptionLike(i, `${r.decider_name} #${r.run_index}`));
    });
    runSel.onchange = () => {
        state.run = Number(runSel.value);
        state.hourIndex = 0;
        state.spotlight = null;
        state.candidateVisible = false;
        resetView();
        renderAll();
    };
    state.run = 0;
}

document.getElementById('plant').addEventListener('change', renderMap);

document.addEventListener('keydown', (ev) => {
    const run = currentRun();
    if (!run) return;
    if (ev.key === 'ArrowLeft' && state.hourIndex > 0) {
        state.hourIndex -= 1;
        state.candidateVisible = false;
        renderAll();
    } else if (ev.key === 'ArrowRight' && state.hourIndex < run.hours.length - 1) {
        state.hourIndex += 1;
        state.candidateVisible = false;
        renderAll();
    }
});

populateDropdowns();
renderAll();
"""


def render_html(payload: dict) -> str:
    """One file, double-click, no server and no dependencies.

    The payload is INLINED rather than fetched: fetch() from file:// fails
    CORS in Chrome, which would mean running a local HTTP server on every
    inspection -- a tax paid on exactly the workflow this design exists to
    remove (run-viewer design, §5.3)."""
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (_HTML_TEMPLATE
            .replace("__CSS__", _CSS)
            .replace("__JS__", _JS)
            .replace("__PAYLOAD__", blob))


def main() -> None:
    p = argparse.ArgumentParser(
        prog="build_viewer_data",
        description="Fold eval traces into a single self-contained HTML "
                    "reader.")
    p.add_argument("--traces", default=str(TRACES_DIR),
                   help=f"Directory of trace JSON (default: {TRACES_DIR}).")
    p.add_argument("--scenarios", default=str(SCENARIOS_DIR),
                   help="Directory of scenario YAML.")
    p.add_argument("--topology", default=str(TOPOLOGY_PATH),
                   help="The topology JSON the runs were driven against.")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"Output HTML file (default: {DEFAULT_OUT}).")
    args = p.parse_args()
    payload = fold(Path(args.traces), Path(args.scenarios),
                   Path(args.topology))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    runs = sum(len(e["runs"]) for e in payload["episodes"].values())
    print(f"wrote {out}: {len(payload['episodes'])} episodes, {runs} runs, "
          f"{out.stat().st_size // 1024} KB", flush=True)


if __name__ == "__main__":
    main()
