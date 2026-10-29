"""Four-condition execution state, including the prediction mixture with no separate fit."""

import json
from pathlib import Path

from .config import CONDITIONS
from .protocol import utc_now, write_json


def start_suite(run, protocol_sha256):
    """Create one exclusive status record for the complete four-condition comparison."""

    state = {
        "status": "running", "phase": "preparation", "utc": utc_now(),
        "protocol_sha256": protocol_sha256, "condition_order": list(CONDITIONS),
        "conditions": {name: {"status": "pending", "requires_training": name != "prediction_mixture"}
                       for name in CONDITIONS},
        "events": [],
    }
    write_json(Path(run) / "suite_status.json", state, exclusive=True)
    return state


def condition_event(run, condition, phase, status, **details):
    """Print and persist each condition transition without treating preparation as final completion."""

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown comparison condition: {condition}")
    event = {"utc": utc_now(), "condition": condition, "phase": phase, "status": status, **details}
    path = Path(run) / "suite_status.json"
    if path.exists():
        state = json.loads(path.read_text())
        state["phase"], state["utc"] = phase, event["utc"]
        state["conditions"][condition].update({"status": status, "last_phase": phase, **details})
        state["events"].append(event)
        if status == "failed":
            state["status"] = "failed"
        write_json(path, state)
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def finish_suite(run, *, status, verification=None, error=None):
    """Mark the whole suite complete only after four test results and CPU verification."""

    path = Path(run) / "suite_status.json"
    state = json.loads(path.read_text())
    if status == "completed" and (
        any(state["conditions"][name]["status"] != "completed" for name in CONDITIONS)
        or not verification or verification.get("status") != "passed"
    ):
        raise ValueError("Four completed conditions and successful verification are required.")
    state.update(status=status, utc=utc_now(), phase="completed" if status == "completed" else "failed")
    if verification is not None:
        state["verification"] = verification
    if error is not None:
        state["error"] = error
    write_json(path, state)
    return state
