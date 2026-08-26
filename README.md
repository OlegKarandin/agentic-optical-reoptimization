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

## The evaluation harness

CLAUDE.md's Evaluation section sets the bar this repo has to clear: the agent is
not measured by "did it restore the network" — a solver does that — but by
whether it earns its place over a `(service_class → policy)` table on decisions
the table cannot key on. `src/storm_reoptimizer/eval/` is the benchmark built to
answer that question, and `src/storm_reoptimizer/eval/suite.py` is what wires its
seven episodes — three twin pairs (`T1` timing, `T2` constraints, `T3`
objective) plus one diagnostic (`D1`) — into a single runnable suite: it drives
every episode against one or more deciders over the real `multilayer-optical-mcp`
server (never a mock — CLAUDE.md's hard seam), scores `pair_solved` and the
per-episode metrics in `scoring.py`, and renders the results table below.

Run it with:

```
python -m storm_reoptimizer.eval.suite
```

### Results

The table `render_results_table()` produces, from a real run against
`eval/states/loaded-s17.json` (seed 17) over all seven episodes:

| decider | pair_solved | episodes correct | notes |
|---|---|---|---|
| baseline:at_deadline | 0.00 | 1/7 | fixed policy: same input in both halves, so exactly one half per pair |
| baseline:immediate | 0.00 | 1/7 | fixed policy: same input in both halves, so exactly one half per pair |

Budget: seed(s) [17] x 7 episodes x N=3 = 42 rollouts (both baseline variants
are collapsed to one real rollout per episode by `collapse_deterministic`; the
budget line reports the nominal N=3 the same code path would use for a
non-deterministic decider).

**Claim 1 (provable).** The agent beats every fixed policy that does not read
the forecast: the twins' menus and observables are identical by construction,
so such a policy emits the same answer twice and scores exactly 50%.
**Claim 2 (asserted at build time).** No rule keyed on any single forecast
variable -- and no parameter-free greedy policy -- solves the suite; checked
over the gold labels before any rollout runs. Two distinct static checks now
back this claim, and they ask opposite questions:

- `assertions.assert_no_single_variable_rule_solves` -- per pair, over the
  variables both halves are supposed to SHARE (`rules.OBSERVABLE_VARS` and
  `rules.DERIVED_VARS`): does any of them differ between the two halves in a
  way a single threshold could key on?
- `assertions.assert_no_global_policy_solves_the_suite` -- over the whole
  suite, over the claimant-side aggregate that IS the flip
  (`derived.FLIP_VARS`): does one FIXED threshold, applied UNIFORMLY with one
  fixed orientation, answer all six halves at once -- i.e. could an operator
  deploy a bare number and skip the comparison the agent is meant to make?

Until 2026-08-26 the claimant side of every gold comparison was never
enumerated by either check -- nothing had ever tested whether a bare
threshold on it could answer the suite. Building the second check and running
it against the live server found, honestly: no single uniformly-applied
claimant scalar (checked at the exposure horizon, before it, and peaked over
all horizons) answers all six halves under one threshold and one orientation.
That is not because the scalar is well-chosen -- it is because each of the
three variants has exactly one pair whose two halves are TIED to high
precision by construction (two of the three variants tie because a pair's far
horizon is byte-identical across its halves by design; the third tied
because T1's near-horizon nowcast was byte-identical across its halves before
it was later removed for an unrelated reason). A tied pair predicts the same
label for both halves under any threshold, which caps that variable below
6/6 regardless of geometry -- so this result says the suite's per-pair
design already forecloses a global bare-scalar policy, not that a geometry
retune produced the result. (Separately, and out of scope for this claim: a
related check found a different, structural free-lever escape common to all
three conserve-gold episodes, currently `xfail` pending its own follow-up
workstream -- Claim 2 is about the two checks above, not a claim that every
shortcut in the suite is closed.)

`pair_solved` over three pairs takes values in {0, 1/3, 2/3, 1}: enough to tell
a working harness from a broken one, not enough to separate luck from skill.
Held-out seeds are the path to power.

The flip-variable citation metric is a NECESSARY, NOT SUFFICIENT filter for
"right answer, absent reason". It is entity matching, not reasoning
verification.

### Reading `episodes correct` honestly

The `episodes correct` column above is 1/7 for both baseline variants, not
the "roughly half" a naive reading of "both baselines tie every pair at
exactly one half" might predict. This is not a bug in the harness or a
confounded pair — both `T1a`/`T1b` (`test_each_baseline_variant_scores_
exactly_one_half`) and `T2`/`T3`'s equivalent checks already pass in
`tests/eval/test_episodes.py`, which is the pre-flight signal that would
have caught a genuinely confounded twin. What actually happens, documented
in `docs/superpowers/rehearsals/T2.md` and `T3.md` (their own "Q3" section):

- **T1** is scored by `timing_at_decision_hour`, which reads the raw
  act/wait choice off the trace regardless of whether anything committed.
  `ForecastBlindBaseline` genuinely discriminates here: it answers T1's two
  halves identically (by construction — same observable input) and gets
  exactly one of the two labels right, exactly as Claim 1 requires.
- **T2** and **T3** are scored by `avoid_horizon_at_decision_hour` and
  `chosen_lever_at_decision_hour`, both of which require a **committed**
  candidate and return `None` otherwise. `ForecastBlindBaseline` pins its
  constraints to `basis="physical"`, and on this loaded network every
  reachable candidate for `storm-svc-1` collides with its own static
  protection leg under a physical-basis check — so the baseline never
  actually commits anything in T2 or T3, in either half, for either
  variant. `decision_label` reads `None` on all four of those halves, which
  never equals a real gold label, so those four halves score as *not*
  correct rather than the "exactly one of two" pattern T1 shows.

So the true count is: 2 halves discriminating (both baselines correctly
land 1/2 on T1, as Claim 1 requires) and 4 halves where the baseline never
commits at all (T2, T3) — 2/12 raw label-correct halves across both
variants, not 6/12. **Claim 1 still holds exactly as stated**: it is a claim
about `pair_solved` (a fixed policy that reads only current exposure can
never get *both* halves of a well-built pair right, whether by scoring 1/2
or 0/2), not about hitting 50% raw label accuracy — and `pair_solved` is
`0.00` for both variants over all three pairs, confirmed by the run above.
A baseline that never commits also never solves a pair; it just fails
differently than one that commits and picks wrong.

`D1` (the seventh episode in the "1/7" denominator, and not part of any
pair) is wrong for both baseline variants too, on this real run: its own
`ForecastBlindBaseline._nearest_exposed_horizon` check is a crude "is the
offset inside the cone's own half-width" geometric test, and D1's cone
centre is deliberately placed so `storm-svc-1`'s offset (58.4km) exceeds
that half-width (7.5km) even though the probabilistic cut probability at
that offset is 0.976 (see `D1.yaml`'s own commentary on why the centre
moved off `storm-svc-1`'s point). The baseline reads "wait"; gold is "act".
That is the intended lesson of a diagnostic built to show "hold the spare"
has no excuse here — not a discriminating twin, and not counted in
`pair_solved`, but it is why the single correct episode out of seven is
`T1b` alone rather than `T1b` plus `D1`.

### The one tension the three pairs share

After rebuilding all three pairs on the same underlying comparison rule, they
share one shape: a scarce resource, two claims on it, and an answer that
depends on comparing the claims. That convergence is not accidental — it is
CLAUDE.md's storm-scarcity story, "the strongest single story" — but it
narrows what the suite demonstrates. What genuinely varies is the *decision
the contention lands on*: when to act (T1), how much to constrain (T2), which
candidate to take (T3). What does not vary is the *kind* of judgement.
Breadth comes from the interpretation axis (free-text operator reports, novel
event types), not from more pairs of this shape — that axis is a separate
spec.

## Demo: the LLM decider on `D1`

`D1` is the cheapest real exercise of the agent loop (build order step 6):
one service, one decision hour, three tool calls. This section is a real,
unedited run — `ClaudeDecider(model="claude-sonnet-5")` against
`eval/states/loaded-s17.json`, scoped to `D1` alone (run 0 of 3) — captured
to show exactly what the model is shown and exactly what it says back, not a
paraphrase of either.

**The setup** (`src/storm_reoptimizer/eval/scenarios/D1.yaml`). `storm-svc-1`
(300 Gbps, `satna`↔`allahabad`) routes working via `satna↔rewa` and
protection via `satna↔jhansi` — SRLG-disjoint at design time. A 15 km-wide
storm cone centred on `satna` itself (not on the service's own point) catches
*both* corridors, because they share an origin node. At `storm-svc-1`'s
58.4 km offset, `p_cut = 0.9761`; the next-most-exposed service, `d0361`, sits
at `p_cut = 0.012` — effectively uncontested. `hours_remaining` (1) exactly
equals the optical-reroute lead time (1), so this is also the last hour in
which acting is still possible. Gold: `act`.

**What the model is shown.** `project_observation` trims the loaded state's
573 services down to the service under test plus every service clearing
`p_cut >= 0.005` (`eval/agent.py`'s `P_CUT_ENUMERATION_THRESHOLD`) — here,
just `storm-svc-1` and `d0361`. Reconstructed for readability (the real
payload's `n_services_total` was 573, `omitted_services.count` 571):

```json
{
  "scenario_id": "D1", "service_under_test": "storm-svc-1",
  "hour": "t0", "hours_remaining": 1,
  "cones": {"t1": {"width_km": 15.0,
                   "center": {"lat": 24.58333, "lon": 80.83333}}},
  "exposure": {
    "storm-svc-1": {"t1": {"hours_ahead": 1, "offset_km": 58.4457,
                           "p_cut": 0.9761, "demand_gbps": 300.0}},
    "d0361":       {"t1": {"hours_ahead": 1, "offset_km": 91.7,
                           "p_cut": 0.012,  "demand_gbps": 100.0}}
  },
  "spares_on_hand": 2,
  "lead_time_hours": {"ip_reroute": 0, "hybrid": 1, "optical_reroute": 1},
  "risk_group_ids": {"t1": "rg_D1_t0_t1"},
  "n_services_total": 573,
  "omitted_services": {"count": 571, "p_cut_threshold": 0.005,
                       "max_p_cut": 0.0,
                       "summed_expected_capacity_at_risk_gbps": 0.0}
}
```

**1. Timing** (`submit_timing_decision`) — `action: "act"`:

> storm-svc-1 has p_cut=0.9761 at only 1 hour ahead, with lead times of 0-1h
> for the reroute levers. There is no next issuance that will meaningfully
> change this: the horizon is already t1, hours_remaining=1, so waiting for a
> sharper forecast means the action would land at or after the cut (act
> later than exposure hour - lead_time is already at the edge). Expected
> capacity at risk here is ~293 Gbps, dwarfing d0361's ~1.2 Gbps expected
> risk (p_cut=0.012). Waiting to see if the cone narrows risks missing the
> lead-time window entirely on a service that's already near-certain to be
> cut, and it also risks losing one of the 2 spares to d0361's negligible
> claim if we delay prioritization. Acting now lets us commit inventory and
> routing to the dominant, near-certain risk while lead time still permits
> an effective change.

**2. Constraints** (`submit_constraint_decision`) —
`avoid.risk_groups: ["rg_D1_t0_t1"]`, `basis: "risk_group"`,
`level: "risk_group"`, `protected: false`, `best_effort: false`:

> Only one horizon exists (t1, 1 hour ahead) and it is also the only lead
> time we have (optical_reroute lead=1h, hours_remaining=1) — we must act
> against this exact cone, not a wider speculative one. storm-svc-1 has
> p_cut=0.9761 at t1, essentially certain to be cut, and its
> working+protection paths both sit inside the same forecast-derived risk
> group rg_D1_t0_t1 (the correlation the static disjointness check never
> saw). So the reroute must avoid that risk group specifically. Using
> protected=False/basis=physical/level=link would only dodge the individual
> link seen today, not the correlated cone risk that's the actual hazard;
> using basis=risk_group/level=risk_group directly names the forecast
> hazard we're defending against. protected=True is unusable by the harness
> per the framing, so leave it False and best_effort False since we want a
> real avoiding path, not a degraded one.

This is the load-bearing call of the whole scenario: `basis="risk_group"`
means the reroute is constrained against the *synthesized* forecast group,
not the single link currently down. Get this wrong and the reroute survives
today's cut but lands back inside the same storm cone.

**3. Objective** (`submit_objective_decision`) — 8 candidates came back from
the real solver, all restoring the full 300 Gbps at 1 spare pair. Choice:
`candidate_0`:

> All eight candidates restore the full 300 Gbps with zero shortfall and use
> exactly 1 spare pair, so the choice comes down to secondary costs.
> candidate_0 has the shortest OMS path (7 hops), lowest added_latency
> (2304.85ms, the minimum across the menu), lowest spectrum_used (848,
> second-lowest but tied near-best), and the best scalar (3440.27, lowest of
> all candidates). Its total_margin (397.58) is essentially the best among
> the higher-bitrate 600G options and very close to the overall max. Given
> storm-svc-1 has p_cut=0.9761 at hour 1 (near-certain cut) versus d0361's
> negligible 0.012, this spare is clearly best spent here, and among
> equivalent restorations candidate_0 minimizes latency and spectrum
> footprint while preserving strong margin.

**Outcome.** `menu_status: "solution"`, `outcome: "committed"`,
`lever: "optical_reroute"` — the harness ran the choice through
`validate_plan`/`commit_plan` on the real (cloned) network state and it
held. The loop repeats at `t1` (`hours_remaining=0` now): the model reaffirms
`act`, the same risk group, and the same `candidate_0`.

**Honest gap.** `D1.yaml`'s gold `flip_variable` is
`[uncontested, spare, lead]`. The model's reasoning covers all three
*concepts* (d0361 "doesn't compete meaningfully", the spare pairs, the lead
time) but never writes the literal word "uncontested" — so
`scoring.cites_flip_variable`, which is documented as entity matching over
exact substrings, would likely score this call low despite the decision and
reasoning both being correct. That is the metric working as designed
(NECESSARY, NOT SUFFICIENT), not a decider defect.

Full traces for all three of `D1`'s rollouts live in `eval/traces/D1-agent_
claude-sonnet-5-{0,1,2}.json`; the audit sidecar recording exactly what was
shown on every one of the 18 calls that produced them is
`eval/traces/agent-calls.jsonl`.

## Repository layout

- `src/storm_reoptimizer/` — the app: event interpretation, geo/asset mapping
  (`geo_mapper.py`), the storm scenario pipeline, and the evaluation harness
  (`eval/`).
- `src/storm_reoptimizer/eval/scenarios/*.yaml` — the seven episode files
  (`T1a`/`T1b`, `T2a`/`T2b`, `T3a`/`T3b`, `D1`).
- `docs/superpowers/rehearsals/*.md` — hour-by-hour walkthroughs of each pair
  against the real server, including the "does the baseline discriminate"
  question each pair's own Q3 answers.
- `docs/superpowers/specs/2026-08-19-agent-eval-design.md` — the harness's
  design spec.
- `tests/` — the test suite, including `tests/eval/` for the harness itself.

See `CLAUDE.md` for the full project scope, build order, and what is
explicitly out of scope (QoT, routing, RSA, disjointness — all owned by the
`multilayer-optical-mcp` server this app calls but never reimplements).
