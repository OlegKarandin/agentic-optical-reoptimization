# storm-reoptimizer — Agentic Disaster Reoptimization for Optical Networks

An agentic application that reoptimizes and restores a multi-layer (IP-over-optical)
network in response to environmental disasters — storms, floods, heatwaves — that
create **correlated failures the design-time risk model never anticipated**.

The canonical case: a storm sweeps a region. Working and protection paths of a
service are SRLG-disjoint — certified disjoint at design time against buried-conduit
and shared-amplifier groups — yet *both* contain aerial fiber inside the storm's
footprint. No static SRLG check flags this. The agent detects the exposure by
synthesizing a risk group that did not exist when the paths were certified, replans
for disjointness against it, and verifies the reroute survives physically.

This repo is the **application and the agent**. It does **not** reimplement QoT,
routing, spectrum assignment, or the network model — those live in the
`multilayer-optical-mcp` server, which this project depends on as an MCP server. This repo owns
exactly what the server deliberately excludes: event interpretation, geo/asset
mapping, the agent's reasoning loop, and the demo GUI.

---

## What the agent is actually for

This is the hard-won lesson the design rests on, and it bounds the agent's job
tightly. Geometry, QoT, routing, disjoint-path computation, and scoring a fixed
objective are **all deterministic** and all live in the server or in plain GIS
code. An agent that "discovers" reroutes or "computes" disjointness adds nothing
and should not be built — a solver does it better.

The agent earns its place in exactly two kinds of step, and nowhere else:

1. **Converting open-ended input into formal objects the deterministic machinery
   consumes.** A free-text operator report ("barge hit the bridge at the north
   crossing"), a novel event type with no pre-tabulated asset filter, a fuzzy
   constraint mid-incident ("don't touch the eastern ring, maintenance opens in
   20 minutes"). These become risk groups, asset filters, and solver constraints.
2. **Setting objectives/weights from event and service *state* — not from service
   class alone.** Restoration mode (optical-restore vs. IP-reroute vs.
   rate-reduce) is only an agentic decision when the right choice depends on event
   state (spare-transponder scarcity under a forecast of more failures, extent of
   the hazard region). If the weighting is a function of service class alone, it is
   a lookup table and the agent is dead weight.

Everything between those two steps — find the assets, build the group, route, check
QoT, score — is a deterministic tool call. If a demo only ever varies service
class, a reviewer correctly calls it a decision tree. Build the cases where the
weighting and constraints come from event state and operator intent.

---

## What lives here vs. in the server

| Concern | Here (app) | Server (`multilayer-optical-mcp`) |
|---|---|---|
| QoT / GSNR | — | ✔ (GNPy adapter) |
| Routing, RSA, disjoint-path | — | ✔ (solvers) |
| Network model, snapshots, validate/commit | — | ✔ |
| Risk groups as abstract asset lists | — | ✔ (`define_risk_group`) |
| **Event ingestion** (weather/fire/flood feeds) | ✔ | — |
| **Geo mapping** (`map_geo_event_to_assets`) | ✔ | — |
| **Asset filter selection** (storm→aerial, flood→buried) | ✔ | — |
| **Agent reasoning loop** | ✔ | — |
| **Weight/constraint setting from context** | ✔ | — |
| **Demo GUI** | ✔ | — |

The seam is a hard rule: the server stays disaster-agnostic so it is reusable; this
repo holds everything storm-shaped. Geometry intersection is **plain GIS code
(Shapely/PostGIS), not the agent** — the agent only handles the inputs GIS can't
parse and the choices a table can't pre-enumerate.

---

## Components

```
   Event feeds (weather / fire / flood APIs, operator free-text)
                          |
                  Event interpreter  ──────────────┐
            (LLM only for unstructured/novel input) │
                          |                          │
                   Geo / asset mapper                │  agent reasoning
            (Shapely/PostGIS: polygon ∩ spans,       │  loop (LLM):
             filter by mount type)                    │  interpret →
                          |                          │  set weights/
                   risk-group definition  ───────────┤  constraints →
                          |                          │  call tools →
              multilayer-optical-mcp server (MCP) ───┘  observe → replan
            (QoT, routing, disjointness, validate, commit)
                          |
                     Demo GUI (map of India; storm sweeps;
                     exposed aerial spans light up; agent reroutes)
```

- **Event interpreter.** Normalizes heterogeneous feeds (a fire perimeter and a
  flood polygon arrive in different shapes) into a single geometry + event-type. The
  LLM is invoked **only** for unstructured input (operator free-text) or event types
  with no pre-tabulated filter. Structured feeds with known types go through a plain
  `(event_type → filter)` table — no agent.
- **Geo / asset mapper.** `map_geo_event_to_assets(geometry, filter)` — intersect
  event geometry with span geometries, filter by mount type. Pure GIS. Spatial data
  in PostGIS if persisted; Shapely in-memory for the demo.
- **Risk-group definition.** Hands the resulting asset list to the server's
  `define_risk_group`. The app produces the list; the server stores the partition.
- **Agent reasoning loop.** Interpret intent → set weights/constraints from event
  state → call server tools → observe structured results → replan. Never solves;
  orchestrates.
- **Demo GUI.** The showcase. A map of India, a storm track advancing frame by
  frame, aerial spans inside the cone lighting up as the dynamic risk group, exposed
  services flashing, the agent rerouting and lines redrawing.

---

## Event input contract & upstream adapters

**Internal format (the contract everything normalizes to):**
```
{ geometry: GeoJSON,        # polygon / track cone / perimeter, not a point
  event_type: enum,         # storm | flood | fire | heat | structure | ...
  valid_at: timestamp,      # one object per forecast hour for moving events
  attributes: { ... } }     # event-specific scalars (wind speed, water level, peak temp)
```
`map_geo_event_to_assets(geometry, filter)` consumes `geometry`; the
`(event_type → filter)` table consumes `event_type`. Moving events (a storm cone
advancing) are a *sequence* of these objects, one per hour, yielding a sequence of
time-stamped risk groups.

**The mismatch to design around:** existing weather MCP servers do **not** output
hazard geometry. The common contract across all of them (Open-Meteo, OpenWeatherMap,
NWS-based) is *point query (lat/lon) → scalar variables over a time series* —
temperature, wind, precip, humidity per coordinate. None emit a storm cone or flood
polygon. The geometry exists upstream (NWS alerts carry GeoJSON polygons; NHC
publishes forecast cones) but most MCP wrappers flatten it to a text area
description. So: assume the *connection* pattern (standard stdio/HTTP MCP, lat/lon
in, JSON out — trivial to wire), but **do not assume their output is your input.**

**Two pluggable upstream adapters, both normalizing to the internal format:**

1. **Point-sampling adapter.** Query a generic weather MCP server over a grid
   covering the region, threshold on wind/precip/temperature, construct the hazard
   polygon from points that exceed threshold. Works with *any* lat/lon weather
   server, vendor-agnostic; crude (reconstructs geometry the source discarded).
   Acceptable for live demo, not for production.
2. **Native-geometry adapter.** Ingest published hazard polygons directly —
   NWS/NHC storm cones, USGS/flood-service extents, fire-perimeter GeoJSON. This is
   what you actually want for storms: a hurricane forecast *cone* is a first-class
   published product, not something to reconstruct from samples.

**For the demo, use neither live source.** Script a known historical track (Cyclone
Amphan over the Bay of Bengal / eastern India) as a sequence of GeoJSON cone polygons
and drive the showcase off that — deterministic, repeatable, no API keys, and it
sidesteps geometry reconstruction entirely. Wire a live adapter later only to
demonstrate pluggability. The internal `{geometry, event_type, valid_at}` contract is
identical whether the source is a scripted track, a sampled grid, or a native feed —
that is the point of normalizing.

---

## The three scenarios

### 1. Storm (moving polygon, aerial filter, cut risk → replan)
Ingest the forecast cone (hourly positions, next N hours). Per hour, intersect with
span geometry, filter to aerial/lashed, emit a time-stamped risk group. For each
service, audit whether working and protection both intersect any hour's group — the
SRLG-disjoint-but-both-aerial exposure. Reroute clear of the **full forecast cone**
(so the reroute isn't re-exposed at t+3h) and verify QoT under current loading. If
spares are scarce because earlier cuts in the same front consumed them, escalate to
a restoration-mode choice with weights set from forecast severity. The moving cone
makes this scenario-1-plus: the safe region shrinks over time, forcing replan
against a *moving* partition.

### 2. Flood (same machinery, inverted filter)
Same pipeline; the filter is buried handholes, manholes, splice closures, and
low-lying/river-crossing spans — aerial spans over the flood polygon are *safe*. A
storm-tuned filter would clear a flooded route and kill the service, so filter
selection is load-bearing. **Honest scope:** if event types are enumerable, the
`(event_type → filter)` map is a table and the agent adds nothing here. Flood earns
the agent **only** as the demo of *runtime filter selection from a natural-language
report* ("the river's over the levee at the north crossing") — otherwise it's a
lookup. Build it as that, or don't claim agentic value for it.

### 3. Heatwave (degradation, not cut → pre-emption + cross-layer remedy)
No assets are cut. A region-wide ambient-temperature event raises EDFA noise figure
and span loss across thermally-exposed equipment (aerial spans, above-ground huts)
in *correlation*, so many margins erode together — the binary up/down monitor sees
nothing. Map heat region → exposed assets, apply forecast `ΔNF/Δloss` via the
server's `inject_degradation`, recompute QoT under loading, calibrate against live
telemetry. Identify the *cluster* of lightpaths whose margin crosses zero before the
peak, then evaluate three remedies on a branch — reroute, modulation downshift, IP
grooming onto survivors — and choose by heat extent (a wide heat dome makes reroute
weak and pushes toward downshift/groom). Act pre-emptively *before* the peak; the
timing is the agentic call, the allocation is the solver's. **Caveat:** heat barely
affects fiber; it affects the *active/lossy elements* (EDFA NF). This is defensible
physics but thinner in the field-outage literature than storm/flood — treat as a
modeled, forward-looking case, lead the project with storm.

---

## Server tools this app calls

By scenario, all on the `multilayer-optical-mcp` MCP server (this app calls, never reimplements):

- **All three:** `define_risk_group`, `get_services`, `check_disjointness` /
  `get_exposure`, `compute_disjoint_paths`, `compute_qot`,
  `recompute_qot_under_loading`, `validate_plan`, `commit_plan` (approval-gated),
  `reconcile` (read back actual state after commit; disaster commits are exactly the
  partial-failure case), snapshot `branch`/`restore`/`diff`.
- **Storm scarcity / cross-layer:** `evaluate_objective` (weights set by the agent),
  `solve_allocation`, `simulate_ip_routing`, `get_affected_services`,
  `reroute_service`.
- **Heatwave:** `inject_degradation`, `get_telemetry` (if exposed),
  `get_transceiver_modes`, `set_modulation_format`.

The app owns one function the server does not expose, by design:
`map_geo_event_to_assets(geometry, filter)` — pure GIS, lives here.

### The sibling side split into two repos (2026-08-18)

What this doc calls "the server" is now two repos:

- **`multilayer-optical-network`** — the deterministic simulator (IP-over-optical
  model, GNPy adapter, solvers, validator). Published as a standalone,
  pip-installable library, not MCP-specific.
- **`multilayer-optical-mcp-server`** — the MCP tool surface (`server.py`) this
  app talks to at runtime. Depends on `multilayer-optical-network`.

This app's runtime pipeline (`scenario_storm.py` and friends) is unaffected by
the split — it still only ever calls `mcp_client.call_tool_json` against the
MCP server subprocess, per the hard seam above. Nothing in `src/` or `tests/`
imports either sibling repo directly.

**One narrow, deliberate exception:** an *offline* build-time script (outside
`src/` and `tests/` — see `tools/`) that constructs this app's seeded demo
service (`storm-svc-1`, `satna`↔`allahabad`) and serializes it to a state
file, per the server's own `--state` flag (`multilayer-optical-mcp --topology
… --state …`). This script imports `multilayer_optical_network` directly,
because there is no MCP tool that persists a new service on a live server —
`solve_allocation` computes on a discarded clone by design (confirmed against
source; the server's own design spec for this file format explicitly rejects
adding a `create_service` plan op, since "the storm scenario never" adds a
demand to an already-running server). The offline script runs once (or on
demand, ~2-3s for one demand against the 143-node toy topology, not the
"minutes to hours" a full operating-network build takes) via the sibling
`multilayer-optical-mcp` conda env's python, producing a checked-in-or-
regeneratable state JSON — never imported into this app's own runtime
process. The pipeline still loads that state exclusively through the real
server's `--state` flag over stdio, exactly like every other input.

---

## Build order

1. **Server dependency + toy topology.** Stand up `multilayer-optical-mcp` against a small,
   seeded India-shaped topology with mount-type attributes and span geometries.
   Confirm `define_risk_group` + `check_disjointness` round-trip before anything
   else.
2. **Scripted event track.** Encode Cyclone Amphan as a sequence of GeoJSON cone
   polygons with timestamps, normalized to the internal `{geometry, event_type,
   valid_at}` contract. This is the demo's data source — no live API. Get it right
   here so every later step replays deterministically.
3. **Geo mapper.** `map_geo_event_to_assets` in Shapely against the toy geometry.
   Storm cone ∩ aerial spans → asset list → server risk group. No agent yet.
4. **Storm scenario, deterministic.** Table-driven filter, scripted call sequence
   (map → define group → audit exposure → disjoint replan → QoT verify → validate)
   over the scripted track's hourly cones. Prove the pipeline end to end with **no
   LLM**. This is the spine; everything agentic hangs off it.
5. **Demo GUI.** Animate the deterministic storm run — the showcase. Star-driver;
   build it before adding agent sophistication.
6. **Agent loop.** Wrap steps 3–4 in the reasoning loop. Introduce the LLM only at
   the two justified points: unstructured-input interpretation and context→weights.
7. **Flood + heatwave.** Flood as the natural-language-filter-selection demo;
   heatwave reusing `inject_degradation`.
8. **Storm-scarcity cross-layer.** The strongest single story: the storm creates the
   spare-transponder scarcity that makes the restoration-mode weighting non-trivial.
   Combines scenarios 1 and 3.
9. **Live upstream adapter (optional).** Add a point-sampling or native-geometry
   adapter to prove the internal contract is source-agnostic. Not needed for the
   showcase; demonstrates pluggability.

Lead every milestone with the deterministic version, then add the agent only where
the two justified steps require it. A working deterministic storm + GUI is the
star-worthy artifact; the agent makes it handle the inputs and tradeoffs a script
can't.

---

## Evaluation

The agent's value is **not** measured by "did it restore the network" — a solver
does that. Measure the two things only the agent can do:

- **Interpretation accuracy.** Given unstructured event reports / novel event types,
  does it produce the correct geometry, filter, and risk group? Compare against a
  human-labeled set.
- **Weighting/constraint quality.** On scenarios engineered so the right restoration
  mode flips with event state (Example A: same service, opposite correct choice
  under spare scarcity), does it choose correctly? A fixed-policy baseline must fail
  these by construction — if a decision tree matches the agent, the scenario doesn't
  justify the agent.

Baseline to beat: a deterministic pipeline with a `(service_class → policy)` table.
The agent only wins on inputs the table can't key on. Report where it *doesn't* win,
too.

---

## Explicitly out of scope

- Anything the server owns: QoT, routing, RSA, disjointness computation, the network
  model, validate/commit. This app calls those; it does not reimplement them.
- Sub-50 ms protection switching. Restoration here is the slow loop (reoptimization,
  what-if, novel-event response). Fast protection stays in the deterministic control
  plane and is never the agent's job.
- Claiming agentic value where a lookup table suffices. Stated per-scenario; flood
  in particular is honest about this.
- Control-plane signalling — handed off via the server's `commit_plan` to whatever
  control plane the operator runs.
