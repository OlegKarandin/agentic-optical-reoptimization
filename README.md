# storm-reoptimizer — agentic disaster reoptimization for optical networks

An agent that reoptimizes a multi-layer (IP-over-optical) network ahead of an
environmental disaster, such as a cyclone, a flood or a heatwave, whose
**correlated failures the design-time risk model never anticipated**. It
ships with a benchmark that must make resource-allocation calls under 
forecast uncertainty that a fixed policy table cannot make.

**Headline result** On the benchmark's three twin-pair tests, 
Claude Opus 5.5 got all 18 decisions right (3 pairs × 2 halves × 3 seeds),
solving every pair on every seed with zero regret. The forecast-blind fixed
policies it is compared against solve none of the pairs, and cannot by
construction.

**▶ [Step through every rollout in the interactive run viewer](https://olegkarandin.github.io/agentic-optical-reoptimization/)**
It shows the storm cones, the exposed spans, what the model was shown at each
hour, what it probed, what it said, and what the replay did with the spare.

---

## The problem

A storm sweeps a region. A protected service's working and protection paths
are SRLG-disjoint: they were certified disjoint at design time against
buried-conduit and shared-amplifier groups. Yet *both* run over aerial fiber
inside the storm's footprint. No static SRLG check flags this. Protection
switching doesn't save the service, because both legs fail together.

The app detects this exposure by synthesizing, from the forecast cone, a
**risk group that did not exist when the paths were certified**. It then asks
the network's solvers for reroutes that avoid it. When restoration resources
are scarce, it decides *who gets them*, before the cut or after.

## What the agent is for (and what it is not)

Geometry, QoT, routing, disjoint-path computation and scoring a fixed
objective are all **deterministic**. They live in plain GIS code and in the
[`multilayer-optical-mcp`](https://github.com/OlegKarandin/multilayer-optical-mcp)
server, built on the
[`multilayer-optical-network`](https://github.com/OlegKarandin/multilayer-optical-network)
simulator (GNPy-based QoT, RSA and disjoint-path solvers, validate/commit). An
LLM that "discovers" reroutes adds nothing a solver doesn't do better.

The agent earns its place in only two kinds of step:

1. **Turning open-ended input into formal objects** the deterministic
   machinery consumes, such as risk groups, asset filters and solver
   constraints.
2. **Setting objectives from event *state*, not from service class.** An
   example is whether to spend the depot's last spare transponder now or hold
   it for a service the storm may cut later. That choice depends on the
   forecast, on who else is exposed, and on whether they could even be
   restored.

The benchmark below is built to test point 2. The bar it sets is a
deterministic pipeline with a `(service_class → policy)` table. If a decision
tree matches the agent, the scenario doesn't justify the agent.

## Architecture

```
   Event feeds / scripted cyclone track (IMD best track, Cyclone Hudhud 2014)
                          |
                  Event interpreter  ─────────────┐
            (normalizes to {geometry, event_type, │
             valid_at, attributes})               │  agent loop (LLM):
                          |                       │  observe → probe →
                   Geo / asset mapper             │  decide timing +
       (Shapely: forecast cone ∩ span geometry,   │  claim priority →
        filtered by mount type: storm → aerial)   │  set constraints →
                          |                       │  choose candidate
                risk-group definition ────────────┤
                          |                       │
          multilayer-optical-mcp server (MCP, stdio)
     (QoT, routing, disjointness, validate_plan, commit_plan)
```

The server is disaster-agnostic and reusable. This repo owns everything storm-shaped: 
event ingestion, geo mapping, the agent loop, the evaluation harness and the viewer. 
Nothing here reimplements routing or QoT. Every network fact comes from a tool call 
to the real server.

The network is a 143-node, 180-edge India backbone derived from Internet
Topology Zoo's `TataNld` graph. Span lengths are great-circle and mount types
(aerial/buried) are randomly selected. A seeded, gravity-loaded
operating state (about 580 services, seed 17) sits on top.

---

## The evaluation

### Episode model

Each episode is an hour-by-hour rollout (`t0` … `t7`) of a scripted cyclone
cone passing over one depot site, `jalgaon`, which holds **one spare
transponder**. The forecast is issued at `t0` and revised at `t1`. Each
issuance predicts the cone at the exposure horizon `t3`. `t1` is also the
last hour an optical reroute (2 h lead time) can land before the storm, so
the decision cannot be put off.

At every decidable hour the model sees a projected **observation** and makes
up to three decisions through tool calls:

| Step | Tool | What it decides |
|---|---|---|
| Timing | `submit_timing_decision` | `act` or `wait`, plus a `claim_priority` ranking of who gets the spare if it is held |
| (optional) | `probe_restorability` | Read-only: *if every asset in risk group X were down, what could the routing tools offer service S, and at what spare cost?* (≤ 4 per decision) |
| Constraints | `submit_constraint_decision` | The `avoid` set (risk groups / assets) that the server's `route_service` must route around |
| Objective | `submit_objective_decision` | Which candidate from the solver's priced menu to commit. The harness then runs `validate_plan` → `commit_plan` on the server |

The observation carries the facts the right answer depends on and nothing
more:

- **Per-service exposure:** `p_cut` at each horizon, computed from the cone
  under Gaussian track uncertainty over 65,535 Sobol points; expected
  capacity at risk (ECAR); and a `p_cut_if_track_revised` band while another
  issuance is still due.
- **A joint `cut_outcomes` table:** which services fail *together*, with
  what probability, and `hours_down_if_cut` with and without restoration.
- **`restorable_groups`**, the spare inventory, lever lead times, deadlines,
  and any probe answers carried forward, each tagged with the risk group it
  was answered under.

The model never sees the realized cuts or a future forecast issuance.
Waiting is what buys the next issuance.

After the storm hits, a deterministic replay restores cut services in the
model's `claim_priority` order using whatever spare it held. A held spare
therefore has a real, simulated value, and holding it is a genuine
alternative to spending it. Every episode is scored on **realized Gbps·h
lost** across all shown services.

### Twin pairs: a fixed policy fails by construction

Each test is a **pair of twin episodes** whose observations a
forecast-blind policy cannot tell apart, but whose correct answers are
opposite: `hold` in the `a` half, `spend` in the `b` half. A pair counts as
solved only if a decider gets **both** halves right on the same seed
(`pair_solved`). Anything that emits the same answer twice scores exactly one
half per pair.

The harness enforces this before a single token is spent (`eval/assertions.py`):

- **Twin-pair discipline.** Shared scalars (spares, lead times, radii,
  revision rate) must be identical across halves, or the pair is cut.
- **No single-variable rule solves the suite.** Every candidate flip
  variable is swept over all six halves; each is blocked either by a **tie**
  (two halves read the same value) or a **reversal** (one pair needs the
  opposite threshold orientation from another), so no single threshold
  answers all six halves.
- **Gold is measured, not asserted.** `gold.enumerate_outcomes` runs both
  scripted oracles (spend now, hold for the replay) through the real server
  and replay and records the Gbps·h each loses. The label is whichever loses
  less, and every half must clear a minimum margin.
- **Realized cuts are forecast outcomes.** The storm the harness injects
  must be one the published forecast assigns real probability to, and the
  forecast's expected value must agree with the gold label.

**Baseline:** `ForecastBlindBaseline`, in two variants (act immediately / act
at the deadline). It uses the same harness, the same tools and the same
solver menus, so any difference in outcome comes from the decisions alone.

### The three tests

All three ask the same question: **spend the depot's last spare on the
service under test now, or hold it for a claimant the storm may cut?** What
varies is the fact the model must establish to answer it.

| Pair | The fact that flips the answer | How it can be learned |
|---|---|---|
| **T1** | How **exposed** the competing claimants are, relative to the service under test | From the observation, but only if the model reasons in expected Gbps·h across joint outcomes rather than comparing raw exposure |
| **T2** | Whether the biggest claimant can be **restored at all** after its cut | Only by probing. In `T2b` the observation shows a *larger* claim (khandwa, p_cut 0.985) in exactly the half where holding the spare is worthless |
| **T3** | Which of two equal-sized claimants is **genuinely isolated** once the storm's risk group is avoided | Only by probing. The claims tie at 114.8 G, and exposure magnitude points the wrong way in one half |

`T2` and `T3` also test **staleness**. The risk group is revised at `t1`,
so a probe answered under `t0`'s group can be wrong an hour later. The
correct move is to re-probe under the *current* group.

### Results

One seed of the operating state (17), three rollouts per decider per
episode, on the real MCP server. The baselines are deterministic and
collapse to one rollout each.

| Decider | Halves correct | T1 solved | T2 solved | T3 solved | Mean `pair_solved` | Regret (Gbps·h) |
|---|---|---|---|---|---|---|
| **agent: claude-opus-5-5** | **18/18** | 3/3 | 3/3 | 3/3 | **1.00** | **0** |
| baseline: immediate | 3/6 | 0/1 | 0/1 | 0/1 | 0.00 | 1800 |
| baseline: at_deadline | 3/6 | 0/1 | 0/1 | 0/1 | 0.00 | 1800 |

Gold outcomes (Gbps·h lost, both policies measured through the real harness):

| Half | Gold | Spend | Hold | Opus 5.5 | Baselines |
|---|---|---|---|---|---|
| T1a | hold | 13 000 | **12 400** | 3/3 ✔ | spend ✘ |
| T1b | spend | **32 500** | 33 100 | 3/3 ✔ | spend ✔ |
| T2a | hold | 33 500 | **32 900** | 3/3 ✔ | spend ✘ |
| T2b | spend | **37 000** | 37 600 | 3/3 ✔ | spend ✔ |
| T3a | hold | 33 500 | **32 900** | 3/3 ✔ | spend ✘ |
| T3b | spend | **14 000** | 14 600 | 3/3 ✔ | spend ✔ |

### What the winning reasoning looks like

Excerpts from Opus 5.5's timing decision at `t1` in three episodes,
verbatim from the audit log:

**T2b (spend).** It probed the largest claimant under the current group,
found it unrestorable, and demoted it:

> Khandwa has by far the largest exposure (197 Gbps at risk, p=0.985), but
> under the current group rg_T2b_t1_t3 the probe returns no_solution, so it
> can't use the spare after its cut. I dropped it to the bottom of the
> ranking. […] If I reroute Nagpur now, the new path is up before the storm,
> which avoids an expected 0.145 × 300 Gbps × 5 h ≈ 218 Gbps-h of downtime. If
> I hold the spare instead, the best use after a cut is Indore:
> 0.165 × 300 × (5−2) h ≈ 149 Gbps-h. […] Acting now on Nagpur is worth more.

**T2a (hold).** Same service, same menu. Here both claimants probe as
restorable, and the model uses the joint outcomes:

> Nagpur only goes down in the 0.145 outcome where every other shown service
> goes down too. Holding the spare and restoring after the cut saves 3 h per
> cut service […] That totals ≈ 434 Gbps·h, about twice what acting on nagpur
> gets.

**T1a (hold).** Decided from the observation alone, in expected value, not
raw exposure:

> Every cut outcome that takes down indore also takes down both claimant
> groups (the 0.117 row). That means holding the spare never leaves it
> unused.

---

## Running it

Requirements: Python ≥ 3.11 and the
[`multilayer-optical-mcp`](https://github.com/OlegKarandin/multilayer-optical-mcp)
server installed (it pulls in `multilayer-optical-network`).

```bash
pip install -e ".[dev,eval,agent]"

# Baselines only: no API key, no cost
python -m storm_reoptimizer.eval.suite

# Include the LLM decider (paid; needs ANTHROPIC_API_KEY)
python -m storm_reoptimizer.eval.suite --include-agent --agent-model claude-opus-5-5
python -m storm_reoptimizer.eval.suite --include-agent --only T2a,T2b

# Fold the traces in eval/traces/ into the single-file HTML viewer
python tools/build_viewer_data.py --out eval/viewer/index.html
```

If the server's console script isn't on `PATH`, set
`STORM_REOPTIMIZER_MCP_SERVER_CMD` to a JSON argv list that launches it. The
server-backed tests look for it via `MULTILAYER_OPTICAL_MCP_PYTHON` (a python
that can import `multilayer_optical_mcp`) and skip without it.

The three seeded operating states the episodes run on are checked in under
`eval/states/`. They were built offline by `tools/build_eval_state.py`, the
one script that imports the simulator library directly, because no MCP tool
persists a new service on a live server.

Tests: `pytest` (≈ 780 tests; the server-backed ones need the server and take most of the ~50 min run).

## Repository layout

```
src/storm_reoptimizer/
  events/            event contract, scripted cyclone tracks, (event_type → filter) table
  geo_mapper.py      forecast cone ∩ span geometry → exposed assets (Shapely)
  scenario_storm.py  deterministic end-to-end storm pipeline (no LLM)
  mcp_client.py      stdio MCP client for the server
  eval/
    scenarios/       the episode YAMLs (T1a/b, T2a/b, T3a/b, D1)
    runner.py        hour-by-hour episode rollout
    observation.py   what the decider sees (and must not see)
    agent.py         the Claude decider: prompt, tools, retry loop
    baseline.py      forecast-blind fixed policies
    probe.py         probe_restorability
    replay.py        post-cut restoration replay
    gold.py          measured gold via both oracles
    assertions.py    pre-flight twin-pair and no-single-rule checks
    scoring.py       per-episode and cross-twin metrics
    suite.py         the runnable suite and results table
tools/               offline builders: states, golds, viewer
eval/states/         seeded operating states the episodes load
tests/               pytest suite
```

## License

MIT. See [LICENSE](LICENSE).
