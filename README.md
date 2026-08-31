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
| baseline:at_deadline | 0.00 | 2/7 | fixed policy: same input in both halves, so exactly one half per pair |
| baseline:immediate | 0.00 | 2/7 | fixed policy: same input in both halves, so exactly one half per pair |

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
mechanism.

**Claim 1 (provable).** The agent beats every fixed policy that does not read
the forecast: the twins' menus and observables are identical by construction,
so such a policy emits the same answer twice and scores exactly 50%.
**Claim 2 (asserted at build time).** No rule keyed on any single forecast
variable -- and no parameter-free greedy policy -- solves the suite; checked
over the gold labels before any rollout runs. This claim is now backed by a
WIDER enumeration than any earlier version of this document reported: **all
seven** claimant-side summary variables the harness can derive (the five
members of `derived.FLIP_VARS` -- `claimant_ecar_at_exposure_horizon`,
`claimant_ecar_before_exposure_horizon`, `claimant_ecar_peak_over_horizons`,
`claimant_ecar_min_over_horizons`, `largest_restorable_group_ecar_gbps` --
plus two further summaries of the same per-horizon map that are NOT in
`FLIP_VARS`, `claimant_ecar_at_earliest_horizon` and `claimant_ecar_
median_over_horizons`), swept jointly against the REAL satna-homed claimant
(`claimant-satna-jabalpur-fwd`/`-rev`, Task 14, 2026-08-30) rather than the
narrative placeholder claimants earlier drafts of this document used. Two
distinct static checks back this claim, and they ask opposite questions:

- `assertions.assert_no_single_variable_rule_solves` -- per pair, over the
  variables both halves are supposed to SHARE (`rules.OBSERVABLE_VARS` and
  `rules.DERIVED_VARS`): does any of them differ between the two halves in a
  way a single threshold could key on?
- `assertions.assert_no_global_policy_solves_the_suite` -- over the whole
  suite, over the claimant-side aggregate that IS the flip
  (`derived.FLIP_VARS`): does one FIXED threshold, applied UNIFORMLY with one
  fixed orientation, answer all six halves at once -- i.e. could an operator
  deploy a bare number and skip the comparison the agent is meant to make?

**The full 7-variable sweep (Task 13, `tools/derive_episodes.py`, recorded in
`docs/superpowers/plans/notes/2026-08-30-joint-tuning.md`) confirms all seven
are genuinely blocked**, by one of two structural reasons:

| Variable | Blocked by | Best achievable (of 6 halves) |
|---|---|---|
| `claimant_ecar_at_exposure_horizon` | TIE (T2, T3 each share a byte-identical far horizon) | 4/6 |
| `claimant_ecar_before_exposure_horizon` | TIE (T1 publishes no horizon before its own exposure horizon) | 4/6 |
| `claimant_ecar_peak_over_horizons` | INTERLEAVE (no tie) | 5/6 |
| `claimant_ecar_min_over_horizons` | TIE (T2's own min reads its shared far horizon in both halves) | 5/6 |
| `largest_restorable_group_ecar_gbps` | TIE (T2, T3, same far-horizon reason) | 4/6 |
| `claimant_ecar_at_earliest_horizon` (not in `FLIP_VARS`) | INTERLEAVE (identical column to `peak_over_horizons` in this suite) | 5/6 |
| `claimant_ecar_median_over_horizons` (not in `FLIP_VARS`) | INTERLEAVE (no tie) | 5/6 |

A TIE means at least one pair's two halves read the identical value on that
variable, so any single threshold necessarily assigns them the same label --
an upper bound on what a bare-scalar policy can score, confirmed tight in
every row above except `before_exposure_horizon` (bound 5/6, not tight: T2a's
own value sits out of order relative to T1a's independently of the tie). An
INTERLEAVE means no two halves tie, but sorting all six values still puts a
`conserve`-labelled half below a `spend`-labelled one, so no single threshold
in either orientation separates the column; confirmed by live-bisecting each
interleaved variable's own binding edge and cross-checking the bisected point
against a closed-form `max(spend values) - min(conserve values)` computation
with no bisection at all (the two agree to `1e-6`).

**The tightest real margin in the whole construction is 31.798 G**
(`claimant_ecar_peak_over_horizons` and `claimant_ecar_at_earliest_horizon`,
both blocked by the SAME pair of values: `T2a`'s network-wide total, `687.276
G`, against `T1a`'s, `655.477 G` -- shrinking `T2a`'s total by more than
31.798 G would re-solve both checks 6/6). The second-tightest is
`claimant_ecar_median_over_horizons`'s **49.973 G** (`T2a` vs `T3b`). Both
margins are two to three orders of magnitude above `cone.py`'s own documented
~2.56e-4 Sobol sampling noise floor, so neither is a coin-flip against
numerical noise. A prior draft of the underlying joint-tuning note reported a
SMALLER, INCORRECT margin here (14.586 G, read off the smallest sorted-
adjacent gap rather than the actual value the interleave has to clear) before
a live review caught and corrected it -- see the note's own "What was wrong
in the first pass" section for the full account; the 31.798 G figure above is
the corrected, live-bisected one.

**This 7-variable sweep replaces every narrower or earlier version of this
claim.** A pre-2026-08-30 version of this document swept only four
claimant-side variants against OLD, narrative placeholder claimants
(`d0029`/`d0348` for T1, eight kanpur/agra services for T2, `d0462`/`d0212`/
`d0363` for T3) -- none of which were real, depot-eligible competitors for
`storm-svc-1`'s own spare pool under the corrected per-site ledger (Task 9).
That version's own account of a near-miss (a threshold on
`claimant_ecar_min_over_horizons` solving the OLD suite 6/6 at a single
global 89.35 G, forcing a retune of T2's near cone) is now moot: the claimant
identity it was tuned against no longer exists in the shipped episodes, and
the CURRENT geometry was derived fresh against the real claimant by Task 13's
own `tools/derive_episodes.py` run, not by patching the old numbers. See
`docs/superpowers/plans/notes/2026-08-30-joint-tuning.md` for the complete,
reproducible derivation (re-run the tool against the live server to
regenerate every figure above digit for digit).

**Separately, and out of scope for this claim, two known findings this
rewrite carries forward honestly rather than silently drops:**

1. A related check (`assert_wait_gold_has_no_free_escape`, W1.6) found that
   storm-svc-1's own static protection lightpath is a free, zero-pair
   `ip_reroute` candidate under every conserve-gold half's near-neutral
   `reference_avoid={}` -- genuinely exploitable on exactly one of the three
   pairs, `T1a` (taking it flips T1a's graded label from gold's `wait` to
   `act`), and correctly a no-op on `T2b`/`T3b` (the same free candidate
   reads the SAME label gold does there).
   `test_a_conserve_gold_with_an_unexploitable_free_escape_passes` confirms
   this live; only `T1a`'s case remains `xfail`, pending its own follow-up
   workstream. Claim 2 is about the two checks above, not a claim that every
   shortcut in the suite is closed.
2. T2's and T3's OLDER `avoid_horizon_at_decision_hour` scoring dimension
   (separate from the spend/conserve flip this claim is about) turned out to
   be structurally broken by this same plan's own `satna<->jabalpur`
   topology change, for the `T2a`/`T3a` halves specifically -- documented in
   full, investigated to a definitive conclusion, and deliberately deferred
   rather than fixed:
   `docs/superpowers/2026-08-31-t2-t3-wide-avoid-finding.md`. This does not
   touch the claim above (which is about the spend/conserve flip variable,
   verified independently three times in Task 13), but a reader auditing the
   suite's overall honesty should know about it.

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

The `episodes correct` column above is 2/7 for both baseline variants
(re-measured 2026-08-31, Task 15, against the corrected exposure model --
it was 1/7 before Task 14's rebuild, and the change is `D1` becoming correct,
not a change within any pair; see below), not the "roughly half" a naive
reading of "both baselines tie every pair at exactly one half" might predict.
This is not a bug in the harness or a
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

`D1` (the seventh episode, and not part of any pair) is now CORRECT for both
baseline variants, and this is the entire reason `episodes correct` moved
from 1/7 to 2/7 — verified live for this rewrite, not inferred from the model
change alone (`ForecastBlindBaseline` run for real against the current server
state: both variants' `timing.action` at `t0` is `"act"`, matching
`gold.label`). This reverses the pre-Task-14 finding this section used to
report. `ForecastBlindBaseline._nearest_exposed_horizon` is a crude "is the
offset inside the cone's own half-width" geometric test, and under the OLD
midpoint-based exposure model, D1's cone centre made `storm-svc-1`'s offset
(58.4 km) exceed that half-width (7.5 km) even though the probabilistic cut
probability at that offset was 0.976 — the baseline read "wait" against a
gold "act". Task 3's corrected model redefines `offset_km` as the distance to
the NEAREST POINT OF THE REAL CUTTABLE SPAN, and D1's cone is centred exactly
on `satna`, which is a literal endpoint of `storm-svc-1`'s own real span — so
the corrected offset is `0.000000 km`, which DOES satisfy the containment
test. `decision_label` for D1 (`timing_at_decision_hour`) reads the raw
timing action regardless of whether anything later commits, and it is now
`"act"` in both variants, matching gold. (A second, independent finding,
recorded in `docs/superpowers/rehearsals/D1.md`'s rewritten Q3: the baseline
still never physically COMMITS anything here, for the identical
buried-protection-leg/`basis="physical"` reason T2 and T3 document — but that
does not change `decision_label`, which is scored on the raw timing action.)
This is not a discriminating twin and is not counted in `pair_solved`, but it
is why the two correct episodes out of seven are `T1b` and `D1` together, not
`T1b` alone.

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
just `storm-svc-1` and `d0361`. Reconstructed for readability (the real
payload's `n_services_total` was 573, `omitted_services.count` 571):

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

`horizon_totals` (`eval/agent.py`'s `P_CUT_ENUMERATION_THRESHOLD` neighbor,
`_horizon_totals` in `observation.py`) is always present now, summed over
every service with a representative point regardless of what
`project_observation` trims for display — the two operands (`sut_ecar_gbps`
against `non_sut_total_ecar_gbps`) every gold rationale in the suite
compares. Every decision the model submits (timing, constraints, objective)
also carries a `contested_claim` field: `{"service_id": ..., "expected_
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

## Viewing a run

`python tools/build_viewer_data.py` folds the scenario YAMLs, the recorded
traces in `eval/traces/`, and the toy topology into `eval/viewer/index.html`
— a single self-contained, double-clickable file with no server and no
network calls needed. Open it in a browser to step hour-by-hour through an
episode: the exposure map, the candidate menu, and "What it said" (each
decision's reasoning, alongside its `contested_claim` when the model
recorded one). The per-hour gold-vs-agent strip reads each hour's
`gold_spare_action` (`spend` or `conserve`) against whether the committed
action actually spent a physical spare pair.

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
