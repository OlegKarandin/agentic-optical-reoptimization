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

### The toy topology's mount types are scenario design (2026-08-30)

`mount_type` is read only by this repo — `geo_mapper.py`, `events/filters.py`
and the viewer. It appears zero times in `multilayer_optical_network`: it
affects no routing, no QoT, no spectrum and no allocation, so changing one
edge's mount type produces a bit-identical rebuilt state at the same seed and
moves only the topology fingerprint.

`satna <-> jabalpur` was changed from `buried` to `aerial` on 2026-08-30
(exposure-and-depot design, §3.2, Option B) so that satna has a third aerial
direction that shares no aerial span with `storm-svc-1`'s working
(`satna <-> rewa`) or protection (`satna <-> jhansi`) legs. Without it, every
satna-homed claimant's storm exposure is perfectly correlated with the service
under test's, and the spare contention the eval scores is not a contest.

This is legitimate — the toy topology is ours, built for this project — on the
condition that it is recorded rather than quietly changed. It is recorded here
and asserted in `tests/eval/test_build_eval_state.py`.

### The T1 pair moved to a jalgaon-homed SUT (2026-09-05) — no topology change

The redesigned T1 pair (spec `docs/superpowers/specs/2026-09-05-t1-spend-or-hold-redesign.md`)
needed CLAUDE.md's canonical case for real: a protected service whose working
AND protection legs both leave their home site on aerial spans inside the cone,
so 1:1 switchover does not save it, plus one buried escape and an independent
aerial claimant corridor out of the same site. `storm-svc-1` never was that —
its protection leg (`satna <-> jhansi`) sits outside every T1 cone.

Three new stage-2 pins were added to `tools/build_eval_state.py` (`T1_PINS`),
so `eval/states/loaded-s17.json` was rebuilt:

- `t1-svc-jalgaon-indore` — `jalgaon <-> indore`, 300 G, protected. Its real
  legs (solver-chosen, read back live) are working `jalgaon -> khandwa -> dhar
  -> indore` and protection `jalgaon -> buldhana -> amravati -> nagpur ->
  bhandara -> raipur -> jabalpur -> indore`: two AERIAL first hops out of the
  depot.
- `t1-claimant-jalgaon-dhulia-fwd` / `-rev` — `jalgaon <-> dhulia`, 100 G each
  direction, unprotected.

Depot is `jalgaon`; the escape is a new lightpath over one of its BURIED spurs
(`surat` / `aurangabad`).

**No mount type was changed.** `jalgaon` already carries four aerial neighbours
(`akola`, `buldhana`, `dhulia`, `khandwa`) and two buried ones (`aurangabad`,
`surat`), so the SUT's two legs, the claimant corridor and the escape are four
independent directions out of one site with nothing to edit. The 2026-08-30
`satna <-> jabalpur` change above therefore remains this topology's ONLY
scenario-driven mount-type edit.

Why `dhulia` and not `khandwa` as the claimant far end (this was found the hard
way, live): the harness's post-cut restoration replay has to REACH the far end
after the corridor is cut and the whole storm risk group is avoided, or holding
the spare buys nothing. `khandwa`'s only other link is aerial (`dhar <->
khandwa`) and falls inside the same cone, so the avoid set disconnects it and
both gold rollouts tied at identical loss. `dhulia`'s other link is BURIED
(`dhulia <-> nasik`), which a storm filter can never admit. Both facts are
asserted in `tests/eval/test_build_eval_state.py`.

D1 is unaffected: it still names `storm-svc-1` and the satna claimant family,
stage 1 (the gravity load) runs before any pin and is not re-routed by one,
and the pin list is solved in order. **T2 and T3 no longer name `storm-svc-1`
or the satna claimants at all** -- see the next section.

### The T2/T3 pairs live on their own state files (2026-09-06) -- no topology change

T2 and T3 were rebuilt on the same jalgaon machinery as T1, on the T2/T3
probe redesign
(spec `docs/superpowers/specs/2026-09-06-t2-t3-probe-redesign-design.md`),
and moved off `storm-svc-1`/the satna claimants entirely -- they no longer
share T1's canonical protected-both-legs-aerial story at all. `T1` stays
decidable from the observation alone (does the observation say the claimant
is exposed enough to outweigh the SUT). `T2` and `T3` are decidable only by
an agent that chooses to call the new read-only `probe_restorability` tool:
`T2` asks whether the claimant CAN be restored at all after its cut (the
observation alone points the wrong way in `T2`'s spend half -- it shows a
*bigger* claim where holding the spare is worthless); `T3` asks whether that
restoration NEEDS the spare. `T3`'s two NAMED claims are the same size both
halves (114.8 G), but its two claimants' own exposure is not tied at all --
whichever one is more exposed this half is always the one whose
restorability status decides the pair, so the observation does say WHICH
claimant to look at. What it cannot say is WHY that claimant's loss is
real: the probe is what confirms the more-exposed claimant is genuinely
isolated there, not just larger, so a policy keyed on exposure magnitude
alone (in either orientation) still fails the same way it fails `T1`/`T2`.

Each pair gets its OWN state file, built from `eval/states/loaded-s17.json`
with `tools/build_eval_state.py --pin-set {t2,t3}` --
`eval/states/t2-jalgaon-s17.json` and `t3-jalgaon-s17.json` -- rather than
adding more stage-2 pins to T1's own file. Two reasons, both live-found, not
speculative: T1's own state, menus, gold and frozen scalars must stay
untouched by T2/T3 authoring (a shared file would re-solve T1's pins every
time T2/T3's pins changed), and an early T3 design (see below) added
"survivor" lightpaths that must never appear as free grooms in T2's or T1's
menus.

**Current, shipped pins (`tools/build_eval_state.py`), as they exist right
now -- read the file directly for the exact `src`/`dst`/`demand_gbps`, not
this summary:**

- `T2_PINS`: the SUT, `t2-svc-jalgaon-nagpur` (`jalgaon <-> nagpur`, 300 G,
  protected), plus ONE claimant, `t2-claimant-jalgaon-khandwa`
  (`jalgaon <-> khandwa`, 200 G, unprotected). Both SUT legs are aerial out
  of jalgaon (working via `dhulia`, protection via `buldhana` -- the solver
  assigns these the OPPOSITE way round from the design spec's own GIS
  pre-check, confirmed live and recorded in the authoring note below); the
  claimant's flip is whether `khandwa`'s only other link
  (`khandwa <-> dhar`, aerial) sits inside the storm footprint too, which
  determines whether `route_service` can restore it at all once its own
  corridor is cut.
- `T3_PINS`: the SUT, `t3-svc-jalgaon-nagpur` (`jalgaon <-> nagpur`, 300 G,
  **unprotected** -- deliberately a different posture from T1/T2's
  protected canonical case, since jalgaon has only four aerial neighbours
  and a protected SUT already consumes two of them), plus TWO claimants,
  `t3-claimant-jalgaon-khandwa` (`jalgaon <-> khandwa`, 200 G) and
  `t3-claimant-jalgaon-buldhana` (`jalgaon <-> buldhana`, 200 G), both
  unprotected. **`T3_PINS` carries NO survivor pins.** The original design
  (spec §4.2) called for a THIRD kind of pin -- single-hop unprotected
  "survivor" services along a claimant's
  buried alternative path, sized to leave headroom, so that claimant would
  get a free zero-spare `ip_reroute` groom. Checked live 2026-09-06 (plan
  Task 8, `tools/probe_restorability.py`) at both the originally-committed
  100 G survivor demand and a smaller 50 G: the groom was NOT offered
  either time -- `solve_allocation_model` never lit a dedicated lightpath on
  the survivor path at all, grooming each survivor demand onto unrelated
  pre-existing IP links instead, so no lightpath with spare capacity ever
  sat where the claimant's `route_service` call could reuse it. Per the
  design spec's own documented fallback, the survivor pins were dropped
  entirely and T3 ships as the two-corridor RESTORABILITY variant instead
  (same shape as T2's "restorable or not" flip, on a second corridor) --
  `T3_SURVIVOR_PINS` was removed from `tools/build_eval_state.py` rather
  than left dead, since nothing references it once dropped from `T3_PINS`.
- **The claimant corridor was substituted for a second, independent
  live-topology reason (plan Task 11).** The design spec's §4.2 assumed
  T3's unprotected SUT would leave jalgaon via `buldhana`, so it assigned
  `dhulia` and `khandwa` as the two claimant corridors. The live solver
  assigns the SUT's working leg the long `jalgaon -> dhulia -> ...` route
  instead, whose ONLY near-depot aerial span is `jalgaon <-> dhulia` --
  exactly the `dhulia` claimant's own corridor. Measured live: a `dhulia`
  claimant's `p_cut` is then IDENTICAL to the SUT's own, to the last Sobol
  point, at every cone near jalgaon -- the same "spare contention is not a
  contest" confound this file already records for the pre-2026-08-30 satna
  claimants above -- and cutting that corridor cuts the SUT itself, so both
  halves would grade `spend` and the pair could never flip. `buldhana` was
  substituted as the second corridor: structurally `khandwa`'s twin (exactly
  two aerial links, `jalgaon <-> buldhana` and `buldhana <-> amravati`, with
  the far end's onward link BURIED), so it carries the same
  restorable/not-restorable flip `khandwa` does.

No mount-type change was needed for either pin set: the SUTs' real
near-depot legs (T2's `dhulia`/`buldhana` first hops, T3's `dhulia` working
leg) and the claimant corridors (`khandwa`/`buldhana`) are already aerial
out of jalgaon, and the escape node (`aurangabad`, not `surat` -- `surat`
has no `optical_reroute` candidate for either SUT, found live) is already
buried, from T1's own mount-type inventory.
`tests/eval/test_build_eval_state.py::test_the_t2_and_t3_pin_sets_are_the_specs`
asserts `T2_PINS`/`T3_PINS` verbatim; `test_the_pair_state_files_carry_their_pins_and_the_base`
asserts each state file is a superset of `loaded-s17.json`'s own services.
Full derivation, including the live geometry search each pair needed
because the solver's leg assignment inverted the design spec's own GIS
pre-check twice (once for T2, once for T3): `docs/superpowers/plans/notes/2026-09-06-t2-t3-authoring.md`.

### `ConstraintDecision.basis`/`.level` became derived, downgrading `ForecastBlindBaseline`'s own commit-time check (2026-09-11)

The decider-allocation-redesign plan (Task 1, spec §6.4) made
`ConstraintDecision.basis`/`.level` DERIVED, read-only properties from the
`avoid` set (`basis = "risk_group" if avoid.get("risk_groups") else
"physical"`, `level` mirrors it with `"link"`), rather than independently
settable fields on the wire. This is legitimate and deliberate: every real
call site already passed the same posture, the toy topology carries no
static SRLGs, and the four independent wire fields bought a model four ways
to break its own reroute and no way to improve it. It also **unifies two
call sites that disagreed before it**: `oracle.py`'s `spend_decider` already
avoided its decision-hour risk group under `basis="risk_group"`/
`level="risk_group"` by hand, so the reroute stays clear of the full
forecast cone; `baseline.py`'s `ForecastBlindBaseline.constraints` hardcoded
`basis="physical", level="link"` UNCONDITIONALLY for its own commit-time
disjointness check, regardless of whether its `avoid` named a risk group --
its own deliberate prior choice, since this toy topology has no static
SRLGs to check physically against otherwise.

**Concrete consequence, confirmed live for both the `SMOKE` and
`EXPOSURE_SMOKE` test fixtures:** `ForecastBlindBaseline`'s `avoid` ALWAYS
names the currently-exposed risk group while the service under test is
exposed, so its commit-time check now ALWAYS runs at
`basis="risk_group"`/`level="risk_group"` instead of the old unconditional
`physical`/`link`. `storm-svc-1`'s cheapest reroute into `allahabad` funnels
through its own static protection span `oms_jhansi_allahabad` (BURIED, so
no storm-derived risk group on this topology can ever cover it); before
Task 1 the unconditional physical/link check caught the resulting
working/protection collision as a `disjointness_collapse` and forced a
widen-and-retry onto a genuinely disjoint route. Now `oms_jhansi_allahabad`
sits outside `avoid["risk_groups"]`, `validate_plan` no longer flags it, and
the first candidate commits AS-IS with
`collides_with_protection: {"collides": True, "oms_shared_with_protection":
["oms_jhansi_allahabad"]}` still on it -- `ForecastBlindBaseline`'s own
commit is a real physical collision, uncaught by its own commit-time check.

**What still guards correctness:** this is a downgraded hard gate, not a
silent blind spot. `route_service` still computes `collides_with_protection`
independently of `validate_plan`'s basis, and it still ships in the menu the
objective step sees -- a visible-but-optional signal a smarter decider can
act on, where before Task 1 it was an enforced, unconditional gate every
decider got for free. Guarded by
`tests/eval/test_runner.py::_WidensOnDisjointnessRejection`'s docstring
(the HISTORICAL/CURRENT derivation) and
`test_recording_costs_no_extra_server_calls`, which assert the collision
fact is still present on the committed SMOKE/EXPOSURE_SMOKE candidate.

### The vacuous-avoid guard and probe-answer carrying have confirmed, narrower scopes than their names suggest (2026-09-15)

The 2026-09-15 harness/agent-fixes branch shipped nine items together
(`docs/superpowers/specs/2026-09-15-harness-agent-fixes-design.md`) and the
first paid run under them
(`docs/superpowers/plans/notes/2026-09-15-rerun.md`) confirmed two of those
items work exactly as scoped, and no further -- both are standing facts
about the harness's own mechanics, not about this run's particular scores.

**The vacuous-avoid guard (Item 1) closes the literal empty case, not the
under-inclusive one.** Across all 21 agent rollouts in the 2026-09-15 run, a
fully vacuous `avoid` (`{}`, `{"risk_groups": []}`, `{"risk_groups": [""]}`
-- the exact payloads that produced both of 2026-09-11's `inert_commits`,
2026-09-12 failure-analysis note, Finding 2) never once recurs -- checked
programmatically, zero instances. But the guard has no opinion on an `avoid`
that is non-empty yet under-inclusive: `D1-agent_claude-sonnet-5-1.json`,
hour t0, iteration 1 sends `{"assets": ["fiber_rewa_satna_0"],
"risk_groups": []}` -- one real asset, not the gold eight (missing the
reverse-direction working span and the whole protection corridor) -- and
still gets a menu whose top candidate reuses the current path unchanged.
`route_service` offers it as a zero-spare, zero-path-change candidate, the
objective step takes it as free, and `step["inert"]` fires for the same
reason the OLD vacuous-avoid bug did (a menu still carrying a do-nothing
option), through a payload the guard never rejects because it does legitimately
bind something. **The class of bug is "avoid leaves the currently-used,
exposed corridor reachable," not "avoid is empty" -- Item 1 catches one
instance of that class, not the class itself.**

**Probe-answer carrying (Item 2 / Task 6) carries the answer and its
risk-group id, but not a staleness flag against the CURRENT hour's own
group.** Confirmed working end-to-end for the first time in a paid run: a
probe asked at hour N appears in hour N+1..final's own
`observation.probe_answers_this_episode`, in all three D1 seeds and all
three T3b seeds of the 2026-09-15 run. But the carried entry names only the
risk-group id it was answered UNDER, and nothing on the wire marks that id as
stale when the current hour issues a revised group. `T2b-agent_claude-sonnet-
5-0.json` and `-2.json`, hour t1: the only probe of
`t2-claimant-jalgaon-khandwa` on record is from t0, under `rg_T2b_t0_t3`
(`status: solution`) -- correct at t0, but t1 revises the risk group to
`rg_T2b_t1_t3`, under which khandwa's answer is actually `no_solution`
(confirmed directly: `T2b-agent_claude-sonnet-5-1.json` DOES re-probe under
`rg_T2b_t1_t3` at t1 and gets `no_solution`). Both stale-answer runs'
reasoning explicitly names the correct, stale group id (*"probed full-restore
solutions under rg_T2b_t0_t3"*) rather than treating the answer as unscoped
-- Item 3's scope sentence is being read correctly -- and still treats a
same-id-named-but-superseded answer as current. **The carrying mechanism
moves the answer forward; it does not move forward whether the question is
still the one the current hour is asking.**

Neither of these is a regression from this branch -- both are the first
direct trace evidence of where each item's own designed scope actually ends,
which is exactly what a rerun is for. Full derivation:
`docs/superpowers/plans/notes/2026-09-15-rerun.md`, §3.1 and §3.2.

---

### The vacuous/unknown-id `avoid` guard does not recognize ids named only in a rejection's `shared_assets` (2026-09-15)

Task 2 of this same branch added `_check_named_services`'s CONSTRAINT_TOOL
guard (`src/storm_reoptimizer/eval/agent.py`): it rejects an `avoid` naming
an asset or risk-group id the model was not shown, to catch the same class of
hallucinated-id failure as a bogus `claim_priority` entry. `known_assets`/
`known_groups` are built only from this observation's own storm-derived
`payload["risk_groups"]`. They do NOT include any id named only in a prior
`validation_violations`/`disjointness_collapse` rejection's `shared_assets`
-- a different set, since a rejection can name a buried span (like
`oms_jhansi_allahabad`, see the 2026-09-11 section above) that can never
appear in any storm-derived risk group on this topology in the first place.
A model that reads such a rejection and tries the obvious, correct-seeming
fix -- `avoid.assets` the colliding span -- gets rejected by this guard
instead, burning one of `MAX_ATTEMPTS = 3`; three failed attempts raise, and
`run_episode` never catches a decider exception, so this can kill a rollout
mid-run.

Concrete instance, from the 2026-09-15 paid run:
`D1-agent_claude-sonnet-5-0.json`, hour t0, a `disjointness_collapse`
rejection names `oms_jhansi_allahabad` and related buried spans as
`shared_assets`; the guard would reject an `avoid` naming any of them.
Empirically harmless that run -- 14 decisions across the run reached
`attempts: 2`, none reached 3, 0/21 rollouts were lost to this -- so it
shipped as a recorded limitation rather than a fix this fix-wave had time to
verify live. It remains a real, structurally reachable trap: a future run
with more such collisions could exhaust the retry budget. See
`_check_named_services`'s own docstring for the same note at the code site.

---

### Transponder mate-pairing, and three metrics that now measure what they are named (2026-09-21)

Four fixes from reading the 2026-09-15 T1a/T1b traces line by line
(`docs/superpowers/specs/2026-09-21-transponder-pairing-and-metric-fixes-design.md`).

**The transponder ledger now pairs a counter-propagating run against
whatever this SAME rollout already lit over the same unordered endpoint
pair** (`ledger.spares_needed`'s own docstring has the exact rule). A real
pluggable is TX+RX -- one per end serves both directions -- so a
bidirectional claim authored as two unidirectional pins (T1a's
`t1-claimant-jalgaon-dhulia-fwd`/`-rev`) used to bill as two transponders
per site even though its GSNR was already computed as the
worse-of-both-directions figure a bidirectional pair pays. Scoped to runs
lit BY THIS ROLLOUT (`SpareLedger.lit_runs`, grown by `debit()`) -- never
the lightpaths already present in the seeded state, which the offline
builder already charged; pairing those too would double-count in the
opposite direction. `probe.py`, `replay.py`, `runner.py`'s two menu
projections, and `ClaudeDecider`'s own wire render all thread the same
`lit_runs`, so the trace's recorded menu, the model's own view of it, and
what the replay actually spends can never silently disagree.

**T1a's `gold` moved.** Re-enumerated (`gold.enumerate_outcomes`, never
hand-edited): `outcome_gbps_h.hold` fell from 12700.0 to 12400.0 (the
claimant pair's reverse leg now restores for free alongside the forward
one at t3, losing only t3+t4 instead of staying down the whole episode),
widening the margin over `spend` from 300.0 to 600.0 and lowering
`min_margin_gbps_h` from 175.0 to 100.0. `gold.label` stays `hold`. **A
cross-run comparison of T1's `outcome_gbps_h`/`min_margin_gbps_h` against
the 2026-09-15 run's own recorded numbers is invalid** -- those traces
predate this fix and carry the OLD arithmetic.

**`observation._restorable_groups` now caps a co-terminating group at one
lightpath's nominal capacity** (`MAX_LIGHTPATH_CAPACITY_GBPS`, moved from
`assertions.py` to `observation.py` since it is now a runtime default, not
only a static invariant), split by direction and admitted highest-ECAR-
first. No shipped episode's group exceeds it, so no episode number moved;
this is a latent-correctness fix for the next authored group.

**`scoring.reexposed` and `scoring.cites_flip_variable` changed meaning
between the 2026-09-15 run and this fix -- a cross-run comparison of either
column against that run's recorded values is invalid.** `reexposed` used
to read `affected_by_hour` (realized-cut membership against the
PRE-action path), which fires on every successful pre-emptive reroute --
measured `true` on all six T1b rows in the 2026-09-15 run despite zero
residual exposure on the committed candidate. It now reads the SUT's own
recomputed per-hour exposure after the action's effective hour, and its
own docstring states its reach limit: an episode that publishes no
issuance after the action hour (T1 publishes only t0/t1) can never report
a re-exposure this metric would catch. `cites_flip_variable` was a boolean
AND over policy vocabulary that read 0/6 on the same run's own six
rollouts; it is now `cites_flip_variable_frac`, a fraction, and `escape`/
`uncontested` were dropped from the affected episodes' own `flip_variable`
lists as tokens the observation never surfaces (§6.2's own audit,
applied to all six T episodes and D1 alike, not just T1a/T1b).

**One more gap this leaves, stated rather than left implicit.** Dropping
`escape`/`uncontested` closed the false-negative half of item 4's fix, but
the six T episodes' `flip_variable` is now `[spend, hold, claimant]` --
100% policy vocabulary the system prompt is forbidden from ever using, 0%
grounded entities (service ids, corridor names, compared scalars) the
observation actually surfaces. `cites_flip_variable_frac` therefore still
measures spontaneous policy-word choice for these six episodes, with
partial credit instead of an all-or-nothing gate, not genuine citation of
the fact that distinguishes the twin. Closing that needs a follow-up that
authors real grounded tokens per episode -- deliberately not attempted
here, since inventing one without the original authoring discipline risks
repeating the exact "escape" mistake this fix corrects elsewhere.

**The known gap this does NOT close.** All six T1 rollouts' decision rule
is a comparison of raw ECAR (`sut_ecar_gbps` against the largest competing
claim), while the objective they are graded on is realized Gbps-hours,
which converts at a DIFFERENT rate on each side of the comparison. Raw-ECAR
and expected-Gbps-h reasoning agree on both halves of T1 today (T1a's gap
is ~4.25x on raw ECAR, ~6.4x on expected value -- wider, not narrower,
after this fix), so nothing in this suite currently tests the case where
the two rules disagree. Closing that needs a new authored half (out of
scope here); until then, a policy keyed on raw ECAR alone cannot be told
apart from one that correctly converts to expected value by anything T1
currently measures.

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
