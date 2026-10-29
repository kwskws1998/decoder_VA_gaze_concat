"""One-command execution of exactly four conditions, including failure propagation."""

import json

import pytest

import environment_runtime
from holdout_experiments.config import CONDITIONS, REMAINING_CONDITIONS, Settings
from holdout_experiments import execution
from holdout_experiments.suite import condition_event


def mocked_suite(tmp_path, monkeypatch, *, fail_on=None, conditions=CONDITIONS):
    """Exercise real orchestration and status files with no GPU fits or test predictions."""

    from holdout_experiments import engine, reporting
    output = tmp_path / "run"
    calls = []

    def prepare(run, data_dir, settings, **kwargs):
        """Create only the temporary run directory for the orchestrator fixture."""

        run.mkdir(exist_ok=False)
        assert tuple(kwargs["conditions"]) == conditions
        return {"protocol_sha256": "fixture_protocol", "conditions": list(conditions)}

    def train(run, condition):
        """Record three required fits and raise deterministically when requested."""

        calls.append(("train", condition))
        if condition == fail_on:
            raise RuntimeError("fixture failure")

    def mixture(run):
        """Confirm both dependencies are selected before the zero-fit mixture condition."""

        assert calls == [("train", "baseline"), ("train", "companion")]
        calls.append(("prepare", "prediction_mixture"))

    def evaluate(run):
        """Mark four ordered test results without loading test rows in this fixture."""

        calls.append(("evaluate", conditions))
        for condition in conditions:
            condition_event(run, condition, "test_evaluation", "running")
            condition_event(run, condition, "test_evaluation", "completed")
        return {"conditions": list(conditions), "condition_count": len(conditions)}

    def verify(run):
        """Complete verification only after the four test events."""

        calls.append(("verify", len(conditions)))
        return {"status": "passed", "arms": len(conditions)}

    monkeypatch.setattr(environment_runtime, "require_active_conda", lambda: {})
    monkeypatch.setattr(execution, "prepare", prepare)
    monkeypatch.setattr(execution, "estimate_runtime", lambda lock: {"fixture": True})
    monkeypatch.setattr(engine, "train", train)
    monkeypatch.setattr(reporting, "prepare_prediction_mixture", mixture)
    monkeypatch.setattr(reporting, "evaluate", evaluate)
    monkeypatch.setattr(reporting, "verify_results", verify)
    return output, calls


def test_one_command_runs_all_four_conditions_and_finishes_after_verification(tmp_path, monkeypatch):
    output, calls = mocked_suite(tmp_path, monkeypatch)
    result = execution.execute(output=output, stage="run", settings=Settings())
    assert result["conditions"] == list(CONDITIONS)
    assert calls == [
        ("train", "baseline"), ("train", "companion"), ("prepare", "prediction_mixture"),
        ("train", "redistribution"), ("evaluate", CONDITIONS), ("verify", 4),
    ]
    state = json.loads((output / "suite_status.json").read_text())
    assert state["status"] == "completed" and state["condition_order"] == list(CONDITIONS)
    assert all(state["conditions"][name]["status"] == "completed" for name in CONDITIONS)
    assert state["conditions"]["prediction_mixture"]["requires_training"] is False
    preparation_order = [event["condition"] for event in state["events"]
                         if event["phase"] in ("training", "mixture_preparation") and event["status"] == "running"]
    assert preparation_order == list(CONDITIONS)
    assert state["verification"]["status"] == "passed"


def test_failed_condition_stops_the_suite_without_test_or_restart(tmp_path, monkeypatch):
    output, calls = mocked_suite(tmp_path, monkeypatch, fail_on="companion")
    with pytest.raises(RuntimeError, match="fixture failure"):
        execution.execute(output=output, stage="run", settings=Settings())
    assert calls == [("train", "baseline"), ("train", "companion")]
    state = json.loads((output / "suite_status.json").read_text())
    assert state["status"] == "failed"
    assert state["conditions"]["companion"]["status"] == "failed"
    assert state["conditions"]["prediction_mixture"]["status"] == "pending"
    assert state["conditions"]["redistribution"]["status"] == "pending"
    assert not (output / "evaluation_started.json").exists()


def test_full_run_cannot_silently_omit_a_condition(tmp_path):
    with pytest.raises(ValueError, match="four conditions"):
        execution.execute(output=tmp_path / "run", conditions=("baseline", "companion", "redistribution"))
    assert not (tmp_path / "run").exists()


def test_remaining_three_runs_two_fits_then_all_three_test_outputs(tmp_path, monkeypatch):
    output, calls = mocked_suite(tmp_path, monkeypatch, conditions=REMAINING_CONDITIONS)
    result = execution.execute(output=output, stage="run", settings=Settings(), conditions=REMAINING_CONDITIONS)
    assert calls == [("train", "baseline"), ("train", "companion"), ("prepare", "prediction_mixture"),
                     ("evaluate", REMAINING_CONDITIONS), ("verify", 3)]
    assert result["condition_count"] == 3
    state = json.loads((output / "suite_status.json").read_text())
    assert state["status"] == "completed"
    assert state["condition_order"] == list(REMAINING_CONDITIONS)
    assert "redistribution" not in state["conditions"]


def test_remaining_three_failure_does_not_reach_test(tmp_path, monkeypatch):
    output, calls = mocked_suite(tmp_path, monkeypatch, conditions=REMAINING_CONDITIONS, fail_on="companion")
    with pytest.raises(RuntimeError, match="fixture failure"):
        execution.execute(output=output, stage="run", settings=Settings(), conditions=REMAINING_CONDITIONS)
    assert calls == [("train", "baseline"), ("train", "companion")]
    assert not (output / "evaluation_started.json").exists()
