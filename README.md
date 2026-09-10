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
seven episodes — three twin pairs, all grading one spare at one depot (`T1`
how exposed the claimant is, `T2` whether it can be restored at all, `T3`
whether its restoration needs the spare) — plus one diagnostic (`D1`) —
into a single runnable suite: it drives every episode against one or more
deciders over the real `multilayer-optical-mcp`
server (never a mock — CLAUDE.md's hard seam), scores `pair_solved` and the
per-episode metrics in `scoring.py`, and renders the results table below.

Run it with:

```
python -m storm_reoptimizer.eval.suite
```

### Results

The table `render_results_table()` produces, from a real run (2026-09-07,
T2/T3 probe redesign plan, Task 14) against the three pairs' own state files
(`eval/states/loaded-s17.json`, `t2-jalgaon-s17.json`, `t3-jalgaon-s17.json`,
all seed 17) over all seven episodes:

| decider | pair_solved | episodes correct | regret_gbps_h | inert_commits | notes |
|---|---|---|---|---|---|
| baseline:at_deadline | 0.00 | 4/7 | 250.0 | 0 | fixed policy: same input in both halves, so exactly one half per pair |
| baseline:immediate | 0.00 | 4/7 | 250.0 | 0 | fixed policy: same input in both halves, so exactly one half per pair |

Budget: seed(s) [17] x 7 episodes x N=3 = 42 rollouts (both baseline variants
are collapsed to one real rollout per episode by `collapse_deterministic`; the
budget line reports the nominal N=3 the same code path would use for a
non-deterministic decider).

---
**Boundary (added 2026-08-26, Finding #9).** Everything ABOVE this line --
the table and the budget line -- is what `render_results_table()` literally
prints for a real run; re-pasting a fresh table here should only ever touch
that part. Everything BELOW is hand-maintained analysis this plan worked to
preserve (Claims 1/2, the `episodes correct` reading, the shared-shape
discussion) -- it is not runtime output, and a mechanical re-paste of a fresh
results table must not overwrite it.
---

**Re-run 2026-08-31 (Task 15, exposure-and-depot plan), against the corrected
exposure model.** `episodes correct` moved from 1/7 to 2/7 for BOTH baseline
variants — `pair_solved` did not move (still `0.00` for both, over all three
pairs). The extra correct episode is `D1`, not a change within any pair; see
"Reading `episodes correct` honestly" below for why, and
`docs/superpowers/rehearsals/D1.md`'s rewritten Q3 for the live-verified
mechanism. **Superseded 2026-09-06 by the T2/T3 probe redesign plan**, which
moved this further to 4/7 (T2 and T3 rebuilt on T1's own label rule) — see
"Reading `episodes correct` honestly" below for the current mechanism; this
entry is kept only as the historical record of the 1/7 -> 2/7 step.
**Superseded again 2026-09-09 (Task 5, fair-scoring plan): the results
table above is now STALE.** It still prints `4/7`, pasted from the
2026-09-07 run and not re-run since — `timing_at_decision_hour` was
changed to read `record["timing_effective"]` instead of the raw declared
`record["timing"]["action"]` (an act that commits nothing, or commits
inertly, is a wait), which flips `D1`'s label from correct to incorrect for
both baseline variants. The current, correct figure is **3/7** for both
variants; see "Reading `episodes correct` honestly" below for the full
mechanism and evidence. The table above must not be hand-edited per the
Boundary note (Finding #9) — it stays `4/7` until the next real
`render_results_table()` run repastes it; until then, treat this note and
the section below as authoritative over the stale table.

**Claim 1 (provable).** The agent beats every fixed policy that does not read
the forecast: the twins' menus and observables are identical by construction,
so such a policy emits the same answer twice and scores exactly 50%.
**Claim 2 (asserted at build time).** No rule keyed on any single forecast
variable -- and no parameter-free greedy policy -- solves the suite; checked
over the gold labels before any rollout runs. All three pairs are now real,
authored, reviewed twins on ONE spare at ONE depot (`jalgaon`), rebuilt on the
T2/T3 probe redesign (`docs/superpowers/specs/2026-09-06-t2-t3-probe-redesign-design.md`):
`T1` grades how EXPOSED its claimant is (decidable from the observation
alone), `T2` grades whether the claimant can be RESTORED AT ALL after its
cut (decidable only by a `probe_restorability` call, since the observation
alone points the wrong way in `T2`'s spend half), and `T3` grades whether
that restoration NEEDS the spare. `T3`'s two halves show the decider the
same NAMED claim size, 114.8 G, both times, but its two claimants' own
exposure is not tied -- whichever claimant is more exposed this half is
always the one whose (constant, identity-determined) restorability status
decides the pair, so the observation does say WHICH claimant matters. What
it cannot say is WHY: the probe is what confirms the more-exposed claimant
is genuinely isolated there, not merely bigger, so a policy keyed on
exposure magnitude alone (in either orientation) still fails the same way
it fails on `T1`/`T2` -- a narrower, claimant-identity-keyed rule would
solve `T3` from the observation alone, but that is a different, more
specific kind of "lookup table" than the ones `T1`/`T2` defeat, not proof
`T3` adds nothing. Two distinct static checks back this claim, and they ask
opposite questions:

- `assertions.assert_no_single_variable_rule_solves` -- per pair, over the
  variables both halves are supposed to SHARE (`rules.OBSERVABLE_VARS` and
  `rules.DERIVED_VARS`): does any of them differ between the two halves in a
  way a single threshold could key on?
- `assertions.assert_no_global_policy_solves_the_suite` -- over the whole
  suite, over the claimant-side aggregate that IS the flip
  (`derived.FLIP_VARS`): does one FIXED threshold, applied UNIFORMLY with one
  fixed orientation, answer all six halves at once -- i.e. could an operator
  deploy a bare number and skip the comparison the agent is meant to make?
  Extracted (Task 13, T2/T3 probe redesign plan) into a reusable
  `assertions.global_policy_report` that names WHY each variable falls
  short, not just whether it does.

**The whole-suite sweep (Task 13, `tools/sweep_flip_vars.py`, printed
live against the three shipped pairs) confirms all five `FLIP_VARS`
members are genuinely blocked**, by one of two structural reasons:

| Variable | Blocked by | Best achievable (of 6 halves) |
|---|---|---|
| `claimant_ecar_at_exposure_horizon` | REVERSAL | 5/6 |
| `claimant_ecar_before_exposure_horizon` | TIE | 0/6 |
| `claimant_ecar_peak_over_horizons` | REVERSAL | 5/6 |
| `claimant_ecar_min_over_horizons` | REVERSAL | 5/6 |
| `largest_restorable_group_ecar_gbps` | TIE | 4/6 |

Every pair now publishes exactly ONE horizon (`t3`) per issuance, so
`..._at_exposure_horizon`, `..._peak_over_horizons` and `..._min_over_
horizons` are the identical column: `T1b=831.070` (spend), `T3b=1090.169`
(spend), `T1a=2456.858` (conserve), `T2a=3991.015` (conserve),
`T3a=4094.957` (conserve), `T2b=6737.691` (spend). A TIE means at least one
pair's two halves read the identical value on that variable, so any single
threshold necessarily assigns them the same label -- `before_exposure_
horizon` ties at 0.0 across all six halves (no pair publishes a horizon
before its own single exposure horizon), so there is no split point to
sweep at all and "best achievable" reads 0/6, not the 3/6 a naive
default-to-one-label policy would score outside this sweep's own threshold
search. A REVERSAL means no tie exists, but the best orientation still
misses because one pair's halves are ordered the OPPOSITE way from
another's: `T1` and `T3` both read "more claimant exposure -> conserve"
(spend below conserve in each), but `T2` reads the other way round
(conserve `3991.015` below spend `6737.691`) -- exactly the design spec's
"T2 reverses the orientation any function of claimant exposure would need."

**Reading the same table by PAIR, not by variable, matches the design
spec's own framing (§3).** On `claimant_ecar_at_exposure_horizon` (=peak=
min): `T1`'s own two halves read "more claimant exposure -> conserve"
(spend `831.070` < conserve `2456.858`), and `T3`'s agree (spend `1090.169`
< conserve `4094.957`) -- but `T2` is the pair that reverses it (conserve
`3991.015` < spend `6737.691`), by exposing MORE claimant capacity in the
half where holding the spare is *worthless* (`no_solution` on the probe),
which is exactly the design spec's "forecast-blind / naive-exposure reflex
defeated" story for `T1` and the "REVERSED orientation" story for `T2`. On
`largest_restorable_group_ecar_gbps`: `T3` contributes a TIE
(`T3a`/`T3b` both read `114.763` G, enforced bit-identically by
`tools/derive_t1.py --require-flip-tie largest_restorable_group_ecar_gbps`)
-- but `T1` and `T2` alone already interleave that same variable (sorted:
`T1b=24.643` spend, `T2a=114.763` conserve, `T1a=148.931` conserve,
`T2b=197.076` spend), so `T3`'s tie is a real, ADDITIONAL blocker of the
gate on this one variable, not the sole one an earlier draft of the design
spec claimed (corrected there, 2026-09-07 followup).

**One known, honestly-carried limitation.** `T3`'s originally intended flip
("does the claimant's restoration need the spare at all", via a zero-spare
`ip_reroute` groomed onto survivor lightpaths) was checked live
(`tools/probe_restorability.py` against the built survivor pins, both at
100 G and 50 G demand) and found NOT offered -- `solve_allocation_model`
never lit a dedicated lightpath on the survivor path at all, so no free
groom ever existed for `route_service` to reuse. Per the design spec's own
documented fallback, `T3` ships as the two-corridor RESTORABILITY variant
instead (same shape as `T2`'s "restorable or not" flip, on a second
claimant corridor) rather than its originally intended "needs the spare or
not" contrast -- a real, acknowledged loss of variety between `T2` and
`T3`, not a hidden one. Recorded in full in
`docs/superpowers/plans/notes/2026-09-06-t2-t3-authoring.md` (Task 8) and
`docs/superpowers/specs/2026-09-06-t2-t3-probe-redesign-design.md` §4.2.

`pair_solved` over three pairs takes values in {0, 1/3, 2/3, 1}: enough to tell
a working harness from a broken one, not enough to separate luck from skill.
Held-out seeds are the path to power.

The flip-variable citation metric is a NECESSARY, NOT SUFFICIENT filter for
"right answer, absent reason". It is entity matching, not reasoning
verification.

### The W3.2 measurement

Earlier revisions of this harness gated `horizon_totals` — per horizon, the
service under test's own expected capacity at risk against the summed
expected capacity at risk of every other service — behind a second agent
arm (`--agent-rival-totals`), to isolate whether handing the model that
comparison changed its answers. That two-arm design is gone: rival totals
are now always computed and always shown, in the observation and the system
prompt, for every agent run. There is exactly one agent arm, named
`agent:{model}` (e.g. `agent:claude-sonnet-5`), and one audit sidecar
(`eval/traces/agent-calls.jsonl`). The rationale for shipping totals
unconditionally, and what the (now-historical) two-arm comparison found, is
recorded in `docs/superpowers/2026-08-28-control-arm-findings.md` and
`docs/superpowers/2026-08-29-shared-depot-arm-predictions.md`.

Run it with:

```
python -m storm_reoptimizer.eval.suite --include-agent
```

**Before running `--include-agent` for a real, paid arm**, run
`python tools/probe_contested_claim_schema.py` (both branches: a null claim
and a non-null claim) and confirm ACCEPTED. `contested_claim` is a nullable
OBJECT field in the tool schema; whether the Anthropic API's strict tool use
actually accepts that shape has only been reasoned about, not verified
against a live call, in the environment that built this harness (no
`ANTHROPIC_API_KEY` was available). If the probe comes back REJECTED,
`decisions.py`'s `CONTESTED_CLAIM_SCHEMA` needs to switch to the flat
two-scalar fallback the probe script's own docstring describes
(`contested_claim_service_id` with a `"none"` sentinel plus
`contested_claim_ecar_gbps`) before trusting any `contested_claim` field a
real run produces.

**One further check to make during a run, for W3.3.** Open the constraints
records for `T3b` at hour `t1` in the audit sidecar and read the
`reasoning`. The design's acceptance for the unconstrained-menu probe is
that the constraints decision *references the 0-pair candidate* — the entry
that reuses `storm-svc-1`'s current working lightpath, which the agent
previously deleted upstream without ever seeing it. Explicitly **not** an
acceptance criterion: that `T3b`'s label flips. That is the measurement, not
the test.

### Reading `episodes correct` honestly

The `episodes correct` column above is 3/7 for both baseline variants
(re-measured 2026-09-07, T2/T3 probe redesign plan, Task 14, against the
rebuilt T2/T3 pairs, then revised again by Task 5 of the fair-scoring plan
below once `timing_at_decision_hour` was changed to read `timing_effective`
instead of the raw declared `timing.action` — see `D1`'s own paragraph
below) — exactly the "roughly half" a reading of "both baselines tie every
pair at exactly one half" predicts, not the 2/7 an earlier revision of this
section reported. This is not a bug in the
harness or a confounded pair — `T1a`/`T1b` (`test_each_baseline_variant_
scores_exactly_one_half`) and `T2`/`T3`'s equivalent checks already pass in
`tests/eval/test_episodes.py`, which is the pre-flight signal that would
have caught a genuinely confounded twin. What actually happens:

- **T1** is scored by `spare_action_by_deadline` (`scoring.decision_label`),
  which reads whether a decider-origin spend debit lands at or before the
  relevant deadline, off the trace, regardless of which specific candidate
  got committed. `ForecastBlindBaseline` genuinely discriminates here: it
  answers T1's two halves identically (by construction — same observable
  input) and gets exactly one of the two labels right, exactly as Claim 1
  requires.
- **T2** and **T3** are now scored by that SAME `spare_action_by_deadline`
  rule — not by `avoid_horizon_at_decision_hour` or
  `chosen_lever_at_decision_hour`, the two rules an earlier revision of this
  section described. Both of those were RETIRED by the T2/T3 probe redesign
  (`scoring.LABEL_RULES` no longer lists either name; `decision_label` now
  raises `ValueError` if a scenario's `metadata.label_rule` names one) along
  with the `storm-svc-1`-based pair they were built for. `T2` and `T3` are
  also no longer built on `storm-svc-1`: both SUTs are now jalgaon-homed
  services whose escape route is a genuinely disjoint `optical_reroute` over
  the buried `jalgaon <-> aurangabad` spur, so `ForecastBlindBaseline`'s
  `basis="physical"` constraint no longer collides with a static protection
  leg the way it did against the old `storm-svc-1`-based pair. The baseline
  now commits normally in both halves of both pairs, exactly as it does on
  `T1`, and lands exactly one correct half per pair for the identical
  reason Claim 1 gives for `T1`.

So the true count is: 6 halves discriminating (all three pairs correctly
land 1/2, as Claim 1 requires) plus `D1` — 6/12 raw label-correct halves
across the three pairs' six halves, plus `D1` INCORRECT in both variants
(see below), for 3/7 raw episodes correct per variant (`T1b`, one of
`T2a`/`T2b`, one of `T3a`/`T3b` — a "one correct half per pair, nothing
else" count, with `D1` contributing nothing to either side). **Claim 1
still holds exactly as stated**: it is a claim about `pair_solved` (a fixed
policy that reads only current exposure can never get *both* halves of a
well-built pair right), not about hitting any particular raw
label-accuracy fraction — and `pair_solved` is `0.00` for both variants
over all three pairs, confirmed by the run above. See
`docs/superpowers/rehearsals/T2.md` and `T3.md` §8 for the live, per-pair
numbers this section summarizes. `D1` cannot move this claim either way:
`pair_solved` is computed over `T1`/`T2`/`T3` only (`D1` declares no
`pair` key, so `rules._by_pair` groups it alone and `cross_twin_metrics`
never runs on it), and even if it did, `D1`'s own `label_rule`
(`timing_at_decision_hour`) is not the rule any pair is scored on — `T1`,
`T2` and `T3` are all `spare_action_by_deadline` (grep the scenario
files: `timing_at_decision_hour` is `D1`'s alone). Task 5's change to
`timing_at_decision_hour` therefore cannot touch Claim 1 by construction,
independent of what it does to `D1`'s own raw score.

`D1` (the seventh episode, and not part of any pair) is now INCORRECT for
both baseline variants, which is a change from the 4/7-with-`D1`-correct
count an earlier revision of this section reported. `ForecastBlindBaseline`
still answers `timing.action = "act"` at `t0` in both variants — that has
not changed — but Task 5 of the fair-scoring plan (2026-09-09) moved
`decision_label`'s `timing_at_decision_hour` branch off the raw declared
`timing.action` and onto `record["timing_effective"]`
(`scoring.timing_at_decision_hour`, same fallback expression
`first_shot_correct` already used), on the reasoning that an act that
commits nothing (or commits inertly) is a wait.
`docs/superpowers/rehearsals/D1.md`'s own Q3 already documents,
live-verified against the real server (2026-08-31, predating this plan),
that neither baseline
variant ever actually commits anything in `D1`: `ForecastBlindBaseline`'s
hardcoded `basis="physical"` constraint collides with `storm-svc-1`'s own
static protection leg on every one of its 5 retries at `t0`
(`outcome: None, committed_lever: None` for both variants — see Q3's
verbatim probe output), so `timing_effective` at `t0` is `"wait"` for both,
not `"act"`. `decision_label(D1)` under the new rule is therefore `"wait"`
for both variants, against `gold.label = "act"` — INCORRECT. This is not a
discriminating twin and is still not counted in `pair_solved`, but it is
why the total lands on 3/7 rather than the 4/7 an earlier revision of this
section reported.

### The one tension the three pairs share

After rebuilding all three pairs on the same underlying comparison rule, they
share one shape: a scarce resource, two claims on it, and an answer that
depends on comparing the claims. That convergence is not accidental — it is
CLAUDE.md's storm-scarcity story, "the strongest single story" — but it
narrows what the suite demonstrates. What genuinely varies is the *fact the
decider must establish to get it right*: `T1` how EXPOSED the claimant is
(readable from the observation alone), `T2` whether the claimant can be
RESTORED AT ALL after its cut (readable only via `probe_restorability`,
since the observation alone points the wrong way in `T2`'s spend half), and
`T3` whether that restoration NEEDS the spare (the observation says which
claimant is more exposed, but only the probe says whether that claimant's
loss is real). What does not vary is the underlying *kind* of judgement --
comparing two claims on one scarce resource. Breadth comes from the
interpretation axis (free-text operator reports, novel event types), not
from more pairs of this shape — that axis is a separate spec.

## Demo: the LLM decider on `D1`

`D1` is the cheapest real exercise of the agent loop (build order step 6):
one service, one decision hour, three tool calls. This section is a real,
unedited run — `ClaudeDecider(model="claude-sonnet-5")` against
`eval/states/loaded-s17.json`, scoped to `D1` alone (run 0 of 3) — captured
to show exactly what the model is shown and exactly what it says back, not a
paraphrase of either.

**This captured trace predates the 2026-08-30 exposure-model correction
(Task 14, exposure-and-depot plan) and is kept here as an unedited historical
record, not re-run for this document.** Regenerating it would need a real,
paid `--include-agent` call, which this rewrite did not make. Under the
corrected model, `storm-svc-1`'s own `offset_km`/`p_cut` at this cone are now
`0.000000`/`1.000000` (not `58.4457`/`0.9761` below), and the real
`claimant-satna-jabalpur-fwd`/`-rev` pair now reads `p_cut = 1.000000` at
this exact cone too — a genuine, depot-eligible competing claim the observation
below never shows the model, because the pin that creates that claimant
postdates this trace. See `docs/superpowers/rehearsals/D1.md`'s corrected
banner and Q3 for the current, live-verified numbers and for what changed in
the baseline's own behaviour as a result. The reasoning quoted below is still
a genuine, faithful account of what this model said against the numbers it
was actually shown; only the numbers themselves are now historical.

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
just `storm-svc-1` and `d0361`. Reconstructed for readability, captured
pre-2026-08-30 per the caveat above (the real payload's `n_services_total`
was 573, `omitted_services.count` 571):

```json
{
  "scenario_id": "D1", "actionable_service": "storm-svc-1",
  "hour": "t0", "hours_remaining": 1,
  "cones": {"t1": {"width_km": 15.0,
                   "center": {"lat": 24.58333, "lon": 80.83333}}},
  "exposure": {
    "storm-svc-1": {"t1": {"hours_ahead": 1, "offset_km": 58.4457,
                           "p_cut": 0.9761, "demand_gbps": 300.0}},
    "d0361":       {"t1": {"hours_ahead": 1, "offset_km": 91.7,
                           "p_cut": 0.012,  "demand_gbps": 100.0}}
  },
  "horizon_totals": {
    "t1": {"sut_ecar_gbps": 292.83, "non_sut_total_ecar_gbps": 1.2}
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

**2026-09-09 (Task 7, fair-scoring plan): the JSON above is historical and
must not be read as today's wire shape.** Tasks 1-2 of that plan trimmed
`offset_km`/`width_km`/`damage_radius_km` and the `cones` key itself (now
`horizons`, a plain list of horizon-hour strings with no geometry) out of
what `project_observation` actually sends the model — they were found to be
distractors the agent double-counted against `p_cut`, which already prices
distance in. This captured D1 trace predates that trim (and, per the caveat
above, the exposure-model correction too), so it still shows `cones` and
`offset_km` — left as-is because the whole block is an unedited historical
record, not a current-format spec. See "Current wire format" below for a
same-length, live example against today's trimmed shape.

`horizon_totals` (`eval/agent.py`'s `P_CUT_ENUMERATION_THRESHOLD` neighbor,
`_horizon_totals` in `observation.py`) is now historical, as of the
2026-09-10 decider-allocation-redesign plan: the `Observation` dataclass
still computes and records it (every trace keeps it), but
`project_observation` no longer sends it to the model — `restorable_groups`
is what the prompt describes instead, and it is the field a live capture
below actually carries. The `contested_claim` field is now on the TIMING
decision only, not on all three: `{"service_id": ..., "expected_
capacity_at_risk_gbps": ...}` naming the strongest rival claim it weighed,
or `null` for "there is none". It is elicited, not scored — see "What to
read afterwards" in `docs/superpowers/2026-08-29-shared-depot-arm-
predictions.md`.

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

### Current wire format (2026-09-10, decider-allocation-redesign plan)

The `D1` trace above is frozen and, per the note under its JSON, now shows a
stale field set. For a live comparison, this is `tools/probe_episode.py
--dump-prompt t0` run today against `T2a`'s own state
(`eval/states/t2-jalgaon-s17.json`) — the same `project_observation` payload
a real `ClaudeDecider` would receive at `t0`, this episode's first hour
(chosen over `t1` because `next_issuance` is still populated here — `t1`
itself is `T2a`'s last issuance, so it comes back `null`), trimmed to the
SUT plus its two largest claimants (`t1-svc-jalgaon-indore` at 208.5 Gbps
expected capacity at risk and `t2-claimant-jalgaon-khandwa` at 191.8; the
full payload also lists the tied `d0346`/`d0422` pair and the tied
`t1-claimant-jalgaon-dhulia-{fwd,rev}` pair, each member at 60.0, elided
here):

```json
{
  "scenario_id": "T2a", "actionable_service": "t2-svc-jalgaon-nagpur",
  "hour": "t0", "hours_remaining": 7, "issued_at": "t0",
  "exposure": {
    "t2-svc-jalgaon-nagpur": {"t3": {"hours_ahead": 3, "p_cut": 0.6,
      "demand_gbps": 300.0, "expected_capacity_at_risk_gbps": 180.0,
      "p_cut_if_track_revised": {"revision_radius_km": 90.0, "min": 0.0,
                                 "max": 0.42, "mean": 0.093}}},
    "t1-svc-jalgaon-indore": {"t3": {"hours_ahead": 3, "p_cut": 0.695,
      "demand_gbps": 300.0, "expected_capacity_at_risk_gbps": 208.5,
      "p_cut_if_track_revised": {"revision_radius_km": 90.0, "min": 0.0,
                                 "max": 0.414, "mean": 0.091}}},
    "t2-claimant-jalgaon-khandwa": {"t3": {"hours_ahead": 3, "p_cut": 0.959,
      "demand_gbps": 200.0, "expected_capacity_at_risk_gbps": 191.8,
      "p_cut_if_track_revised": {"revision_radius_km": 90.0, "min": 0.005,
                                 "max": 0.954, "mean": 0.289}}}
  },
  "spares_on_hand": 1,
  "lead_time_hours": {"ip_reroute": 0, "hybrid": 2, "optical_reroute": 2},
  "risk_group_ids": {"t3": "rg_T2a_t0_t3"},
  "iteration": 0, "last_rejection": null,
  "actions_taken": [], "spares_spent": 0,
  "restorable_groups": {"t3": [
    {"endpoints": ["indore", "jalgaon"], "members": ["t1-svc-jalgaon-indore"],
     "ecar_gbps": 208.5},
    {"endpoints": ["jalgaon", "khandwa"],
     "members": ["t2-claimant-jalgaon-khandwa"], "ecar_gbps": 191.8}
  ]},
  "issuance_schedule": ["t0", "t1"],
  "deadline_hour": {"ip_reroute": "t3", "hybrid": "t1", "optical_reroute": "t1"},
  "next_issuance": {"hour": "t1"},
  "standing_claim_priority": [],
  "horizons": ["t3"],
  "n_services_total": 580,
  "omitted_services": {
    "p_cut_threshold": 0.005,
    "below_threshold": {"count": 296, "max_p_cut": 0.0,
                        "summed_expected_capacity_at_risk_gbps": 0},
    "ineligible_for_depot": {"count": 277}
  }
}
```

Note what is gone relative to the `D1` example above: no `cones` key, and no
`offset_km`/`width_km`/`damage_radius_km` anywhere in `exposure` — `p_cut`
is the only distance-derived number the model sees now. `horizons` replaces
`cones` as a plain list of horizon-hour strings. `omitted_services` also now
splits `below_threshold` (quiet services) from `ineligible_for_depot`
(exposed but not depot-eligible) rather than D1's single flat bucket, and
`ineligible_for_depot` is reduced further still, to a bare `count` — it no
longer carries `max_p_cut`/`summed_expected_capacity_at_risk_gbps` the way
`below_threshold` still does, because nothing in the suite ever keyed a
decision off those two figures for a bucket of services that cannot draw on
this depot at all. Relative to the previous (2026-09-09) capture this one
replaces: no `services` key (the roster left the wire in an earlier task)
and no `horizon_totals` key (see the note above); every `exposure` row now
carries a `p_cut_if_track_revised` band while another issuance is still
scheduled; `next_issuance` and `standing_claim_priority` are new top-level
fields; and `restorable_groups` is on the wire in place of `horizon_totals`.

Two more keys exist on the wire but not in the JSON above, because they
appear only on the constraints and objective requests, never on timing:
`decided_this_hour` (the timing decision already made this hour — its
`action`, `reasoning`, `contested_claim`, `claim_priority`, and any probe
answers obtained — so the later steps EXECUTE that decision rather than
re-deciding it) and `attempts_this_hour` (every avoid set already tried
this hour, the menu status and size it produced, and what was answered,
so a repeated attempt with no new information is visible as such). A
third, `risk_groups` (the named groups' own asset-level contents, needed to
name `risk_groups[<id>].assets` in `avoid`), appears on the constraints
request only — the timing and objective steps have no use for it and it is
large. Neither JSON block on this page claims to be the full payload.

**The decidable-hours rule.** The decider is not called at every hour in an
episode's `hours`, only at one where a decision could have content: a spare
is still on hand at `depot_site`, the actionable service carries nonzero
`p_cut` at some horizon, and this hour carries an issuance of its own in
`issuance_schedule` (`runner.is_decidable`'s own docstring gives the full
reasoning). On the shipped suite this is exactly the issuance hours, and it
applies identically to every arm — agent and baseline alike — so
`EpisodeTrace.tool_calls` stays comparable across them: the T episodes drop
from 8 timing calls to 2, D1 from 2 to 1. A skipped hour still appears in
the trace — the harness records a `wait` with `"skipped": true` and never
projects an observation or calls the decider at all — and the viewer marks
it accordingly (see "Viewing a run" below).

**The probe-nudge note.** The probe's "what the answer means for the
spare" text (`_SYSTEM_PROMPT_TAIL` in `agent.py`) carries three sentences.
The first two state semantics: what a `full_restore_candidates`/
`min_spares_needed_by_site` answer implies for whether a service can use
the spare at all. The third — "The answer is as relevant to the services
you would keep the spare for as to the one you can act on." — is a
deliberate nudge, kept as an experiment for the first measured run made
after this change and reported as such either way; see spec 2 of
`docs/superpowers/specs/2026-09-10-decider-allocation-redesign-design.md`.

## Viewing a run

`python tools/build_viewer_data.py` folds the scenario YAMLs, the recorded
traces in `eval/traces/`, and the toy topology into `eval/viewer/index.html`
— a single self-contained, double-clickable file with no server and no
network calls needed. Open it in a browser to step hour-by-hour through an
episode: the exposure map, the candidate menu, and "What it said" (each
decision's reasoning, alongside its `contested_claim` when the model
recorded one). The per-hour gold-vs-agent strip reads each hour's
`gold_spare_action` (`spend` or `conserve`) against whether the committed
action actually spent a physical spare pair. An hour the decidable-hours
rule skipped (see "Current wire format" above) renders with its own
`skipped` marker in "What it said" instead of a reasoning block — the
harness recorded a `wait` there without ever calling the decider, so there
is no reasoning to show.

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
