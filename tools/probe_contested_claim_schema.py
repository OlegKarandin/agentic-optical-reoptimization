"""Does strict tool use accept a nullable OBJECT with its own `required` list?

`strict_tool_schema` (eval/agent.py) rewrites a list-valued `type` into an
`anyOf`. `OBJECTIVE_JSON_SCHEMA`'s `priority` already exercises that for
`["array", "null"]`, but an object branch carries `required`,
`additionalProperties` and `properties` with it, and the strict-tool-use
validator may reject that shape. The eval-fairness design flags this as an
implementation risk and pre-specifies a flat fallback; this settles which one
we build (eval-fairness design, §5.2 "Implementation risk").

Runs in THIS repo's env and makes ONE real, tiny API call. It is NOT the
offline-build exception: it imports nothing from multilayer_optical_network.

Run: python tools/probe_contested_claim_schema.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from storm_reoptimizer.eval.agent import DEFAULT_MODEL, strict_tool_schema

NULLABLE_OBJECT = {
    "type": "object", "additionalProperties": False,
    "required": ["action", "contested_claim"],
    "properties": {
        "action": {"type": "string", "enum": ["act", "wait"]},
        "contested_claim": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["service_id", "expected_capacity_at_risk_gbps"],
            "properties": {
                "service_id": {"type": "string"},
                "expected_capacity_at_risk_gbps": {"type": "number"},
            },
        },
    },
}


def main() -> None:
    import anthropic

    adapted = strict_tool_schema(NULLABLE_OBJECT)
    print("adapted schema:")
    print(json.dumps(adapted, indent=2))

    tool = {"name": "probe", "description": "Probe.", "strict": True,
            "input_schema": adapted}
    try:
        response = anthropic.Anthropic().messages.create(
            model=DEFAULT_MODEL, max_tokens=512, tools=[tool],
            tool_choice={"type": "tool", "name": "probe",
                         "disable_parallel_tool_use": True},
            messages=[{"role": "user", "content": (
                "Call `probe` with action 'wait' and contested_claim null.")}])
    except Exception as exc:                      # noqa: BLE001 -- report it
        print(f"REJECTED: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(
            "probe_contested_claim_schema: nullable object rejected; build "
            "the FLAT fallback (contested_claim_service_id with a 'none' "
            "sentinel + contested_claim_ecar_gbps).")
    block = next(b for b in response.content
                 if getattr(b, "type", None) == "tool_use")
    print(f"ACCEPTED: {json.dumps(block.input)}", flush=True)
    print("Now re-run mentally with a NON-null claim to confirm the object "
          "branch validates too -- the null branch alone proves little.",
          flush=True)


if __name__ == "__main__":
    main()
