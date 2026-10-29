"""Conda-only setup ownership and detailed direct-Python launch contracts, without installation."""

import importlib
from types import SimpleNamespace

import pytest

import environment_runtime as environment
import run_experiments as entry
import setup_environment as setup
from holdout_experiments.config import Settings
from holdout_experiments import execution


def test_active_conda_requires_matching_prefix_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(environment.sys, "prefix", str(tmp_path))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    with pytest.raises(RuntimeError, match="conda"):
        environment.require_active_conda()
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    with pytest.raises(RuntimeError, match="conda"):
        environment.require_active_conda()
    (tmp_path / "conda-meta").mkdir()
    assert environment.require_active_conda()["prefix"] == str(tmp_path)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "different"))
    with pytest.raises(RuntimeError, match="sys.prefix"):
        environment.require_active_conda()
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    (tmp_path / "pyvenv.cfg").write_text("home = another python")
    with pytest.raises(RuntimeError, match="conda"):
        environment.require_active_conda()


def test_optional_setup_uses_only_requirements_and_never_creates_an_environment(tmp_path, monkeypatch, capsys):
    """The normal dependency path is a single pip requirements command inside active conda."""

    ownership = {"prefix": "/conda/envs/trt_va", "python_version": [3, 11, 12]}
    monkeypatch.setattr(setup, "require_active_conda", lambda: ownership)
    monkeypatch.setattr(setup, "PROJECT_ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        """Stop any actual installation or GPU check during plan inspection."""

        raise AssertionError("Dry run attempted execution")

    monkeypatch.setattr(setup, "run", forbidden)
    monkeypatch.setattr(setup, "check_gpu", forbidden)
    result = setup.main(["--dry-run"])
    output = capsys.readouterr().out
    assert result["dry_run"] and "pip install -r" in output
    assert "conda create" not in output and ".venv" not in output
    assert "--index-url" not in output and "torch==" not in output
    assert not list(tmp_path.iterdir())
    assert setup.install_commands("python") == [
        ["python", "-m", "pip", "install", "-r", str(tmp_path / "requirements.txt")],
        ["python", "-m", "pip", "check"],
    ]


def test_optional_check_only_does_not_install(tmp_path, monkeypatch):
    """Checking an already installed conda environment cannot trigger package installation."""

    monkeypatch.setattr(setup, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(setup, "require_active_conda", lambda: {"python_version": [3, 11, 12]})
    monkeypatch.setattr(setup, "check_gpu", lambda *args, **kwargs: {"fixture": "no GPU execution"})
    commands = []
    monkeypatch.setattr(setup, "run", lambda command, **kwargs: commands.append(command))
    setup.main(["--check-only"])
    assert len(commands) == 2
    assert not any("pip" in command for command in commands)


def test_gpu_probe_preserves_conda_command_prefix(monkeypatch):
    seen = []

    def fake_run(command, **kwargs):
        """Verify a list prefix cannot accidentally turn into a system-Python call."""

        seen.append((command, kwargs))
        return SimpleNamespace(stdout='{"bf16_supported": true}')

    monkeypatch.setattr(environment, "run", fake_run)
    prefix = ["conda", "run", "--name", "trt_va", "python"]
    assert environment.check_gpu(prefix, gpu=2)["bf16_supported"]
    assert seen[0][0] == [*prefix, "-c", environment.GPU_PROBE_CODE]
    assert seen[0][1]["environment"]["CUDA_VISIBLE_DEVICES"] == "2"


def test_default_python_settings_preserve_the_declared_protocol():
    assert entry.EXPERIMENT_SETTINGS == Settings()
    assert entry.STAGE == "run" and entry.EXPERIMENT_SETTINGS.epochs == 10


def test_detailed_python_arguments_are_actual_settings():
    args = entry.build_parser().parse_args([
        "--stage", "prepare", "--epochs", "3", "--seed", "7", "--split-seed", "9",
        "--train-batch-size", "4", "--eval-batch-size", "2", "--max-length", "128",
        "--fractions", ".7", ".2", ".1", "--learning-rate", "9e-6", "--weight-decay", ".02",
        "--warmup-ratio", ".2", "--max-grad-norm", ".5", "--redistribution-learning-rate", ".002",
        "--sigma-left", ".7", "--sigma-right", "1.8", "--min-sigma", "1e-5",
        "--prediction-alpha", "0.307649721979222", "--noise-std-fraction", ".01",
        "--gradient-accumulation-steps", "1", "--validation-every-epochs", "1",
    ])
    settings = entry.resolve_settings(args)
    assert (settings.epochs, settings.seed, settings.split_seed, settings.batch_size,
            settings.eval_batch_size, settings.max_length) == (3, 7, 9, 4, 2, 128)
    assert settings.fractions == (.7, .2, .1)
    assert (settings.learning_rate, settings.weight_decay, settings.warmup_ratio,
            settings.max_grad_norm) == (9e-6, .02, .2, .5)
    assert (settings.redistribution_learning_rate, settings.sigma_left,
            settings.sigma_right, settings.min_sigma) == (.002, .7, 1.8, 1e-5)


@pytest.mark.parametrize("options", [["--validation-every-epochs", "2"],
                                   ["--gradient-accumulation-steps", "2"],
                                   ["--prediction-alpha", ".5"], ["--noise-std-fraction", ".02"]])
def test_fixed_recipe_cannot_be_changed_through_detailed_arguments(options):
    with pytest.raises(SystemExit):
        entry.build_parser().parse_args(options)


def test_import_is_safe_and_python_main_dispatches_exact_settings(tmp_path, monkeypatch):
    calls = []

    def fake_execute(**kwargs):
        """Observe dispatch without preparing data, downloading weights or fitting a model."""

        calls.append(kwargs)
        return {"status": "fixture_only"}

    monkeypatch.setattr(execution, "execute", fake_execute)
    importlib.reload(entry)
    assert calls == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "old")
    result = entry.main(["--stage", "prepare", "--output", str(tmp_path / "run"),
                         "--gpu", "0", "--epochs", "4", "--learning-rate", "1e-5"])
    assert result["status"] == "fixture_only" and len(calls) == 1
    assert calls[0]["settings"].epochs == 4 and calls[0]["settings"].learning_rate == 1e-5
    assert calls[0]["output"] == tmp_path / "run" and calls[0]["stage"] == "prepare"
    assert calls[0]["conditions"] == ("baseline", "companion", "prediction_mixture", "redistribution")
    assert not list(tmp_path.iterdir())


def test_existing_run_uses_its_locked_settings_and_rejects_changes(monkeypatch, tmp_path):
    from holdout_experiments import protocol
    locked = Settings(epochs=3, learning_rate=1e-5)
    monkeypatch.setattr(protocol, "load_lock", lambda path: ({}, locked))
    args = entry.build_parser().parse_args(["--stage", "train", "--output", str(tmp_path)])
    assert entry.resolve_settings(args) == locked
    monkeypatch.setattr(execution, "load_lock", lambda path: ({}, locked))
    with pytest.raises(ValueError, match="locked run"):
        execution.execute(output=tmp_path, stage="train", settings=Settings())


def test_runtime_rejects_non_conda_before_loading_models(monkeypatch):
    from holdout_experiments.engine import runtime_check

    def invalid_environment():
        """Stop before package or CUDA checks for a wrong environment owner."""

        raise RuntimeError("Activate conda first")

    monkeypatch.setattr(environment, "require_active_conda", invalid_environment)
    with pytest.raises(RuntimeError, match="Activate conda"):
        runtime_check()


def test_changed_compute_settings_do_not_claim_a_calibrated_eta():
    from holdout_experiments.estimate import estimate_runtime
    lock = {"partitions": {name: {"rows": rows} for name, rows in
                           (("train", 43034), ("val", 5377), ("test", 5380))},
            "total_updates_per_fit": 26900, "settings": Settings(batch_size=8).as_dict()}
    estimate = estimate_runtime(lock)
    assert not estimate["calibration_matches_compute_settings"]
    assert estimate["estimated_hours_without_extra_buffer"] is None
    assert estimate["planning_hours_with_25pct_buffer"] is None


def test_three_condition_estimate_counts_only_two_fits():
    from holdout_experiments.config import REMAINING_CONDITIONS
    from holdout_experiments.estimate import estimate_runtime
    lock = {"conditions": list(REMAINING_CONDITIONS), "training_arms": ["baseline", "companion"],
            "partitions": {name: {"rows": rows} for name, rows in
                           (("train", 43034), ("val", 5377), ("test", 5380))},
            "total_updates_per_fit": 26900, "settings": Settings().as_dict()}
    estimate = estimate_runtime(lock)
    assert estimate["training_runs"] == 2
    assert set(estimate["per_arm_nominal_hours"]) == {"baseline", "companion"}
