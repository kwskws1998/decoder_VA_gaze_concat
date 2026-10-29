"""Python API shared by direct-script and module entry points."""

import json
from pathlib import Path

from trt_pipeline.config import DATA_DIR

from .config import ARMS, CONDITIONS, Settings
from .estimate import estimate_runtime
from .protocol import load_lock, prepare, write_json


def execute(*, output, data_dir=DATA_DIR, settings=None, stage="run", arm="baseline", conditions=CONDITIONS):
    """Dispatch explicit stages without changing a previously locked configuration."""

    if stage == "check-runtime":
        from .engine import runtime_check
        return runtime_check()
    if stage not in ("prepare", "run", "train", "estimate", "seal", "evaluate", "verify"):
        raise ValueError(f"Unknown stage: {stage}")
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if tuple(conditions) != CONDITIONS:
        raise ValueError(f"The full suite requires these four conditions in order: {CONDITIONS}")
    output = Path(output).resolve()
    if stage in ("prepare", "run"):
        settings = Settings() if settings is None else settings
        if stage == "run":
            from environment_runtime import require_active_conda
            require_active_conda()
        lock = prepare(output, data_dir, settings)
        estimate = estimate_runtime(lock)
        write_json(output / "runtime_estimate.json", estimate)
        print(json.dumps({"settings": settings.as_dict(), "runtime_estimate": estimate}, indent=2), flush=True)
        if stage == "prepare":
            return estimate
        from .engine import train
        from .reporting import evaluate, prepare_prediction_mixture, verify_results
        from .suite import condition_event, finish_suite, start_suite
        start_suite(output, lock["protocol_sha256"])
        current = None
        phase = "preparation"
        try:
            for condition in CONDITIONS:
                current = condition
                phase = "mixture_preparation" if condition == "prediction_mixture" else "training"
                condition_event(output, condition, phase, "running")
                if condition == "prediction_mixture":
                    prepare_prediction_mixture(output)
                else:
                    train(output, condition)
                condition_event(output, condition, phase, "waiting_for_test")
            current = None
            phase = "test_evaluation"
            result = evaluate(output)
            phase = "verification"
            verification = verify_results(output)
            finish_suite(output, status="completed", verification=verification)
            return result
        except BaseException as error:
            if current is not None:
                condition_event(output, current, phase, "failed", error=str(error))
            finish_suite(output, status="failed", error={"phase": phase,
                         "type": type(error).__name__, "message": str(error)})
            raise
    lock, locked = load_lock(output)
    if settings is not None and settings != locked:
        raise ValueError("Settings differ from the locked run. Declare changes in a new run before training.")
    if stage == "estimate":
        return estimate_runtime(lock)
    if stage == "train":
        from .engine import train
        return train(output, arm)
    from .reporting import evaluate, seal_selection, verify_results
    return {"seal": seal_selection, "evaluate": evaluate, "verify": verify_results}[stage](output)
