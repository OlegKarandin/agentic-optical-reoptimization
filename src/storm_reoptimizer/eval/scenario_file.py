# src/storm_reoptimizer/eval/scenario_file.py
"""The scenario file contract (eval design spec, "Scenario file contract").
One YAML per episode; six twin halves plus one diagnostic.

Parsing is STRICT -- an unknown or missing key raises rather than defaulting.
A misspelled key here is a silently different episode: `spares_on_hand_` typed
for `spares_on_hand` would run green and quietly remove the spare contention
the pair is built on.

Six keys the spec's illustrative YAML omits but its assertions require:
`hours` (the ordered hour labels the rollout walks), `decision_hour` (the `d`
the prefix-share assertion is defined against), `reference_avoid`
(REFERENCE_AVOID -- "any avoid set fixed per pair and applied to both halves,
declared in the scenario file, not derived"), `flip_variable` (the tokens the
flip-variable-citation check matches against the agent's reasoning),
`metadata` (the observable scalars the one-variable check enumerates), and
`gold.label` (the single categorical a candidate rule must predict, since the
three pairs' gold answers are drawn from three different vocabularies)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

_TOP_LEVEL_KEYS = {
    "id", "pair", "seed", "state_file", "service_under_test", "track",
    "hours", "decision_hour", "lead_time_hours", "spares_on_hand",
    "depot_site", "spare_inventory",
    "damage_radius_km", "reference_avoid", "forecast", "realized", "gold",
    "flip_variable", "metadata",
}
_OPTIONAL_TOP_LEVEL_KEYS = {"pair"}   # omitted for singleton episodes (D1)
_GOLD_KEYS = {"survived", "max_spares_wasted", "decision_at_t0", "label",
              "rationale", "outcome_gbps_h", "min_margin_gbps_h"}
# T1 spend-or-hold redesign (Task 8): the OUTCOME the oracle enumerator
# (a later task) scores each of the two candidate decisions against, so
# `scoring.episode_metrics` can grade the episode on Gbps-hours lost against
# the best of those two outcomes (`regret_gbps_h`) rather than on a bare
# act/wait label match alone. Optional -- only episodes using the
# `spare_action_by_deadline` label rule populate them; every other episode's
# gold stays exactly as strict as it always was.
_OPTIONAL_GOLD_KEYS = {"outcome_gbps_h", "min_margin_gbps_h"}
_HORIZON_KEYS = {"cone", "width_km", "center"}


class ScenarioFileError(ValueError):
    """A scenario file that is malformed, incomplete, or self-inconsistent."""


@dataclass(frozen=True)
class ConeAtHorizon:
    cone: dict            # GeoJSON Polygon
    width_km: float       # cross-track diameter at the service's chord
    center: dict          # {"lat": ..., "lon": ...} -- the cone axis


@dataclass(frozen=True)
class Issuance:
    issued_at: str                        # "t0", "t1", ...
    horizons: dict[str, ConeAtHorizon]    # horizon hour -> cone


@dataclass(frozen=True)
class Gold:
    survived: tuple[str, ...]
    max_spares_wasted: int
    decision_at_t0: str      # "act" | "wait"
    label: str               # the single categorical rules.py predicts
    rationale: str           # the expected-cost arithmetic behind the label
    # T1 spend-or-hold redesign: the two candidate decisions' own outcome, in
    # Gbps-hours lost -- {"spend": ..., "hold": ...} -- so `scoring.
    # episode_metrics` can compute `regret_gbps_h` against the BEST of the
    # two, not against the label alone. Populated by a later task's oracle
    # enumerator; None for every episode not using the
    # `spare_action_by_deadline` label rule.
    outcome_gbps_h: dict[str, float] | None = None
    # The REQUIRED margin floor, in Gbps-hours: the best outcome in
    # outcome_gbps_h must beat every other by at least this much, or the
    # episode is invalid. Enforced -- not merely recorded -- by
    # `assertions.assert_gold_matches_outcomes`, which raises PairInvalid
    # when the measured margin falls below it, so that a gold label is not
    # merely correct but correct by a margin an author could not have hit by
    # accident. `gold.gold_from_outcomes` computes it as a fraction of the
    # smaller SCOPE-ONLY total (the SUT plus its declared claimants),
    # floored at 1.0. None (every episode not using the
    # `spare_action_by_deadline` label rule) declares NO floor and skips the
    # check.
    min_margin_gbps_h: float | None = None


@dataclass(frozen=True)
class ScenarioFile:
    id: str
    pair: str | None
    seed: int
    state_file: str
    service_under_test: str
    track: str
    hours: tuple[str, ...]
    decision_hour: str
    lead_time_hours: int          # the optical_reroute value; see lead_time()
    spares_on_hand: int           # transponder PAIRS
    depot_site: str               # the site whose depot is scarce this episode
    spare_inventory: dict[str, int]  # per-site spare transponder counts
    damage_radius_km: float
    reference_avoid: dict
    forecast: dict[str, Issuance]        # issue hour -> Issuance
    realized: dict[str, tuple[str, ...]]  # hour -> asset ids cut
    gold: Gold
    flip_variable: tuple[str, ...]
    metadata: dict[str, Any]


def _require_keys(where: str, got: dict, allowed: set[str],
                  optional: set[str] = frozenset()) -> None:
    unknown = set(got) - allowed
    if unknown:
        raise ScenarioFileError(f"{where}: unknown key(s) {sorted(unknown)}")
    missing = allowed - set(got) - set(optional)
    if missing:
        raise ScenarioFileError(f"{where}: missing key(s) {sorted(missing)}")


def load_scenario(path: str | Path) -> ScenarioFile:
    """Parse one episode YAML. Raises ScenarioFileError on anything the
    contract does not allow."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScenarioFileError(f"{path}: top level must be a mapping")
    _require_keys(str(path), raw, _TOP_LEVEL_KEYS, _OPTIONAL_TOP_LEVEL_KEYS)

    hours = tuple(raw["hours"])
    if raw["decision_hour"] not in hours:
        raise ScenarioFileError(
            f"{path}: decision_hour {raw['decision_hour']!r} not in hours {hours}")

    forecast: dict[str, Issuance] = {}
    for issued_at, horizons in raw["forecast"].items():
        if issued_at not in hours:
            raise ScenarioFileError(
                f"{path}: forecast issue hour {issued_at!r} not in hours {hours}")
        parsed: dict[str, ConeAtHorizon] = {}
        # Horizon keys must appear in ASCENDING `hours` order within one
        # issuance. Not merely cosmetic: `oracle.latest_horizon` and every
        # caller of it (escape_objective, spend_decider, assertions.
        # assert_both_legs_exposed/assert_spend_is_real) pick "the latest
        # horizon" as `next(reversed(issuance.horizons))` -- the last dict
        # key -- rather than re-deriving it from `scenario.hours` on every
        # call (the way `observation.latest_issuance` and
        # `baseline._nearest_exposed_horizon` do for their own, different,
        # "which issuance/horizon" questions). Guaranteeing the order HERE,
        # once, at load time, is what makes that last-key idiom a structural
        # fact about a loaded `ScenarioFile` rather than a silent convention
        # an out-of-order YAML could violate undetected.
        last_horizon_index = -1
        for horizon, cone in horizons.items():
            if horizon not in hours:
                raise ScenarioFileError(
                    f"{path}: forecast horizon {horizon!r} not in hours {hours}")
            horizon_index = hours.index(horizon)
            if horizon_index <= hours.index(issued_at):
                raise ScenarioFileError(
                    f"{path}: issuance {issued_at!r} has horizon {horizon!r} at "
                    f"or before its own issue hour -- an issuance can only "
                    f"forecast the future")
            if horizon_index <= last_horizon_index:
                raise ScenarioFileError(
                    f"{path}: issuance {issued_at!r} lists horizon {horizon!r} "
                    f"out of chronological order -- horizons within one "
                    f"issuance must appear in ascending `hours` order, since "
                    f"callers pick 'the latest horizon' as the last one "
                    f"listed")
            last_horizon_index = horizon_index
            _require_keys(f"{path}:{issued_at}:{horizon}", cone, _HORIZON_KEYS)
            parsed[horizon] = ConeAtHorizon(
                cone=cone["cone"], width_km=float(cone["width_km"]),
                center=cone["center"])
        forecast[issued_at] = Issuance(issued_at=issued_at, horizons=parsed)

    realized: dict[str, tuple[str, ...]] = {}
    for hour, assets in (raw["realized"] or {}).items():
        if hour not in hours:
            raise ScenarioFileError(
                f"{path}: realized hour {hour!r} not in hours {hours}")
        realized[hour] = tuple(assets)

    _require_keys(f"{path}:gold", raw["gold"], _GOLD_KEYS,
                  optional=_OPTIONAL_GOLD_KEYS)
    raw_outcome = raw["gold"].get("outcome_gbps_h")
    raw_margin = raw["gold"].get("min_margin_gbps_h")
    gold = Gold(
        survived=tuple(raw["gold"]["survived"]),
        max_spares_wasted=int(raw["gold"]["max_spares_wasted"]),
        decision_at_t0=raw["gold"]["decision_at_t0"],
        label=raw["gold"]["label"],
        rationale=raw["gold"]["rationale"],
        outcome_gbps_h=({str(k): float(v) for k, v in raw_outcome.items()}
                       if raw_outcome is not None else None),
        min_margin_gbps_h=(float(raw_margin) if raw_margin is not None
                          else None))
    if gold.decision_at_t0 not in {"act", "wait"}:
        raise ScenarioFileError(
            f"{path}: gold.decision_at_t0 must be 'act' or 'wait', "
            f"got {gold.decision_at_t0!r}")

    depot_site = raw["depot_site"]
    spare_inventory = {str(k): int(v) for k, v in raw["spare_inventory"].items()}
    if depot_site not in spare_inventory:
        raise ScenarioFileError(
            f"{path}: depot_site {depot_site!r} has no entry in "
            f"spare_inventory {sorted(spare_inventory)}; the site whose depot "
            f"is scarce must declare how scarce it is")
    declared = int(raw["metadata"]["spares_on_hand"])
    if spare_inventory[depot_site] != declared:
        raise ScenarioFileError(
            f"{path}: metadata.spares_on_hand={declared} but "
            f"spare_inventory[{depot_site!r}]={spare_inventory[depot_site]}. "
            f"The declared scalar IS the depot site's inventory -- "
            f"rules.OBSERVABLE_VARS and assertions.SHARED_SCALARS enumerate "
            f"it, and a disagreement means they are enumerating a number the "
            f"harness does not use")

    # Strict, same treatment as depot_site/spare_inventory above (Task 12,
    # exposure-and-depot plan): the services a gold rationale's claimant
    # arithmetic names, so `assertions.claimant_service_ids` can turn "did
    # the author's arithmetic name a real, eligible, exposed service" into a
    # checkable question without regexing English prose. Required (may be an
    # empty list []) rather than defaulted -- a misspelled or omitted key
    # here would silently exempt an episode from every claimant-side
    # dimensional-coherence invariant, exactly the class of defect this
    # plan's Task 12 exists to catch.
    claimant_services = raw["metadata"].get("claimant_services")
    if not isinstance(claimant_services, list) or not all(
            isinstance(s, str) for s in claimant_services):
        raise ScenarioFileError(
            f"{path}: metadata.claimant_services is required and must be a "
            f"list of service id strings (may be empty, []) -- got "
            f"{claimant_services!r}")

    # CONDITIONALLY strict (2026-08-30 review fix): required together
    # whenever claimant_services is non-empty, exempt when it is empty (D1).
    # `assertions.assert_claim_is_one_lightpath` is invariant 4 -- the check
    # written specifically to catch D2 (a claimant aggregate billed across
    # THREE co-terminating groups where the honest figure was the largest
    # single one) -- and making its input optional unconditionally would
    # leave that guard permanently dead: every episode with a real claimant
    # list already has a real claimed figure sitting in its OWN
    # gold.rationale prose today, the same prose claimant_services was just
    # read from, so there is no reason to wait for Task 14 to populate this.
    # `claimed_competing_ecar_at` is required alongside the value because
    # the claim's horizon is NOT always the SUT's own exposure horizon (T2/
    # T3 bill their claim at the NEAR horizon while the SUT's own exposure
    # horizon is the FAR one) -- see assert_claim_is_one_lightpath's own
    # docstring for the full statement of why that can't be inferred.
    if claimant_services:
        claimed = raw["metadata"].get("claimed_competing_ecar_gbps")
        claimed_at = raw["metadata"].get("claimed_competing_ecar_at")
        if not isinstance(claimed, (int, float)) or isinstance(claimed, bool):
            raise ScenarioFileError(
                f"{path}: metadata.claimant_services is non-empty "
                f"({claimant_services!r}), so metadata."
                f"claimed_competing_ecar_gbps is required and must be a "
                f"number -- got {claimed!r}")
        if not isinstance(claimed_at, str) or claimed_at not in hours:
            raise ScenarioFileError(
                f"{path}: metadata.claimant_services is non-empty, so "
                f"metadata.claimed_competing_ecar_at is required and must "
                f"name one of this episode's hours {hours} -- got "
                f"{claimed_at!r}")

    return ScenarioFile(
        id=raw["id"], pair=raw.get("pair"), seed=int(raw["seed"]),
        state_file=raw["state_file"],
        service_under_test=raw["service_under_test"], track=raw["track"],
        hours=hours, decision_hour=raw["decision_hour"],
        lead_time_hours=int(raw["lead_time_hours"]),
        spares_on_hand=int(raw["spares_on_hand"]),
        depot_site=depot_site, spare_inventory=spare_inventory,
        damage_radius_km=float(raw["damage_radius_km"]),
        reference_avoid=raw["reference_avoid"], forecast=forecast,
        realized=realized, gold=gold,
        flip_variable=tuple(raw["flip_variable"]), metadata=raw["metadata"])


class _ScenarioDumper(yaml.SafeDumper):
    """A `SafeDumper` that renders any MULTI-LINE string in block-literal
    (`|`) style -- the style every hand-authored scenario file already uses
    for `gold.rationale` -- instead of `yaml.safe_dump`'s own default for a
    string containing `\\n` (an escaped, blank-line-separated plain scalar,
    confirmed directly: `yaml.safe_dump({"a": "line one\\nline two\\n"})`).
    Applied to every string, not special-cased to the `rationale` key alone,
    since that is both simpler and robust to any other multi-line field a
    future scenario might carry."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_ScenarioDumper.add_representer(str, _represent_str)


def dump_scenario(scenario: ScenarioFile) -> str:
    """The reverse of `load_scenario`: serialize `scenario` back to the YAML
    text the file format expects, such that `load_scenario(write(
    dump_scenario(scenario))) == scenario`. Needed so `tools/compute_gold.py`
    can rewrite a scenario file's `gold:` block with freshly computed
    numbers without a human hand-editing YAML.

    Built on `dataclasses.asdict(scenario)` -- `Gold`'s own dataclass fields
    already match the file's `gold:` block key-for-key, and every tuple
    field (`hours`, `flip_variable`, `gold.survived`, ...) naturally dumps as
    a YAML list (PyYAML's safe representer maps `tuple` to `list`) -- with
    `forecast` reshaped by hand: `Issuance`/`ConeAtHorizon` carry THEIR OWN
    field names (`issued_at`, `horizons`), which do NOT match the file's
    compact `{issued_at: {horizon: {cone, width_km, center}}}` nesting the
    way `Gold`'s fields happen to coincide with `gold:`'s. `realized` is
    rebuilt the same explicit way for symmetry and to guarantee plain
    `list`s (not tuples-that-happen-to-render-as-lists) regardless of how
    `dataclasses.asdict` treats it.

    `pair` is OMITTED entirely (not written as `pair: null`) when `None` --
    matching how a real singleton episode (D1) is authored today, since
    `_OPTIONAL_TOP_LEVEL_KEYS` only ever tolerates a MISSING key, not a
    key present with a null value written by a human.

    5-decimal cone-coordinate floats are left exactly as loaded: no
    rounding or reformatting is applied anywhere in this function, only
    PyYAML's own default float rendering (which reproduces the shortest
    string that round-trips to the same float -- the original text,
    whenever that text was already minimal)."""
    raw = dataclasses.asdict(scenario)
    if raw.get("pair") is None:
        raw.pop("pair", None)
    raw["forecast"] = {
        issued_at: {
            horizon: {"cone": cone.cone, "width_km": cone.width_km,
                      "center": cone.center}
            for horizon, cone in issuance.horizons.items()
        }
        for issued_at, issuance in scenario.forecast.items()
    }
    raw["realized"] = {hour: list(assets)
                       for hour, assets in scenario.realized.items()}
    return yaml.dump(raw, Dumper=_ScenarioDumper, sort_keys=False)


def load_all_scenarios(directory: str | Path | None = None
                       ) -> dict[str, ScenarioFile]:
    """Every episode in `directory` (default: the packaged scenarios/), keyed
    by scenario id."""
    root = Path(directory) if directory is not None else SCENARIOS_DIR
    out: dict[str, ScenarioFile] = {}
    for path in sorted(root.glob("*.yaml")):
        scenario = load_scenario(path)
        out[scenario.id] = scenario
    return out
