"""Domain tools exposed through the Flower-hosted model's function calls."""

from __future__ import annotations

import json
import math


def schema(name, description, properties):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TEXT = {"type": "string"}
SCHEMAS = [
    schema("list_sites", "Discover configured sites and their supported analyses.", {}),
    schema(
        "list_studies",
        "List current cold-start studies at a repository site.",
        {"site": TEXT},
    ),
    schema(
        "inspect_study",
        "Read aggregate results, missing coverage, paired comparisons, "
        "selection sensitivity and provenance; never exports raw shots.",
        {"site": TEXT, "study": TEXT},
    ),
    schema(
        "list_cases",
        "List heating cases at a site; demos are explicitly synthetic.",
        {"site": TEXT},
    ),
    schema(
        "diagnose_heating",
        "Rank heating explanations using local held-out samples.",
        {"site": TEXT, "case_id": TEXT},
    ),
    schema(
        "validate_heating",
        "Run six bounded toy-model rollouts; no TORAX jobs or "
        "physical actuation. Test a heating scale under model uncertainty.",
        {
            "site": TEXT,
            "case_id": TEXT,
            "power_scale": {"type": "number", "minimum": 0.5, "maximum": 1.5},
            "steps": {"type": "integer", "minimum": 10, "maximum": 80},
        },
    ),
]


class Toolbox:
    def __init__(self, sites, max_calls=16):
        self.sites, self.max_calls = sites, max_calls
        self.used = 0
        self.audit = []

    def execute(self, name: str, arguments: dict) -> dict:
        if self.used >= self.max_calls:
            return {"error": "Tool budget exhausted"}
        self.used += 1
        try:
            spec = next((s for s in SCHEMAS if s["name"] == name), None)
            if spec is None or not isinstance(arguments, dict):
                raise ValueError("Unknown tool or malformed arguments")
            if set(arguments) != set(spec["parameters"]["required"]):
                raise ValueError("Tool argument names do not match its schema")
            for key, value in arguments.items():
                kind = spec["parameters"]["properties"][key]["type"]
                if kind == "string" and (
                    not isinstance(value, str) or len(value) > 120
                ):
                    raise ValueError("Invalid string argument")
                if kind in ("number", "integer"):
                    constraints = spec["parameters"]["properties"][key]
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or (kind == "integer" and type(value) is not int)
                        or not constraints["minimum"] <= value <= constraints["maximum"]
                    ):
                        raise ValueError(
                            "Numeric argument is outside its allowed range"
                        )
            if name == "list_sites":
                output = {"sites": [s.describe() for s in self.sites.values()]}
            else:
                args = dict(arguments)
                site = self.sites.get(args.pop("site"))
                if site is None:
                    raise ValueError("Site is not configured")
                output = site.call(name, args)
            # Fail closed on NaN and unexpected non-JSON adapter output.
            json.dumps(output, allow_nan=False)
        except (ValueError, TypeError, KeyError, OSError):
            output = {
                "error": "Analysis failed: check tool arguments, site access, "
                "and artifact schema. No result is available."
            }
        try:
            json.dumps(arguments, allow_nan=False)
            audit_arguments = arguments
        except (ValueError, TypeError):
            audit_arguments = {"invalid_arguments": True}
        self.audit.append(
            {"tool": name, "arguments": audit_arguments, "result": output}
        )
        return output
