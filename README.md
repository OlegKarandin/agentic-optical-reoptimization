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
over the gold labels before any rollout runs.

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
