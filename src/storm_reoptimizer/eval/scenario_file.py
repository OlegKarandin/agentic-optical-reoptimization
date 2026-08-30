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
              "rationale"}
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
        for horizon, cone in horizons.items():
            if horizon not in hours:
                raise ScenarioFileError(
                    f"{path}: forecast horizon {horizon!r} not in hours {hours}")
            if hours.index(horizon) <= hours.index(issued_at):
                raise ScenarioFileError(
                    f"{path}: issuance {issued_at!r} has horizon {horizon!r} at "
                    f"or before its own issue hour -- an issuance can only "
                    f"forecast the future")
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

    _require_keys(f"{path}:gold", raw["gold"], _GOLD_KEYS)
    gold = Gold(
        survived=tuple(raw["gold"]["survived"]),
        max_spares_wasted=int(raw["gold"]["max_spares_wasted"]),
        decision_at_t0=raw["gold"]["decision_at_t0"],
        label=raw["gold"]["label"],
        rationale=raw["gold"]["rationale"])
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
