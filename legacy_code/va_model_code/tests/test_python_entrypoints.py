"""Check Python launchers, machine-dependent CUDA selection and matched arguments."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

import run_sigma_experiments as experiments
import setup_environment as setup
import sigma_runtime as runtime


@pytest.mark.parametrize("driver,index", [((12, 0), "cu126"), ((12, 4), "cu126"), ((12, 8), "cu126"), ((13, 0), "cu130"), ((13, 2), "cu130")])
def test_wheel_selection_uses_driver_cuda_family(driver, index):
    assert runtime.choose_torch_index(driver) == index


def test_detects_driver_independently_of_gpu_name(monkeypatch):
    for gpu in ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090", "NVIDIA A100"):
        xml = f"<nvidia_smi_log><driver_version>580.1</driver_version><cuda_version>13.0</cuda_version><gpu><product_name>{gpu}</product_name></gpu></nvidia_smi_log>"
        monkeypatch.setattr(runtime, "run", lambda *args, **kwargs: SimpleNamespace(stdout=xml))
        assert runtime.detect_nvidia_driver() == {"driver_version": "580.1", "driver_cuda": [13, 0], "gpus": [gpu]}


def test_setup_dry_run_does_not_install_or_create_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(setup, "detect_nvidia_driver", lambda: {"driver_cuda": [12, 4], "gpus": ["RTX 4090"]})
    monkeypatch.setattr(setup, "run", lambda *args, **kwargs: pytest.fail("dry-run must not install"))
    destination = tmp_path / "environment with spaces"
    setup.main(["--dry-run", "--venv-dir", str(destination)])
    assert not destination.exists()
    output = capsys.readouterr().out
    assert "torch==2.12.1+cu126" in output
    assert "flash-linear-attention==0.5.2" in output
    assert "--require-fla --smoke-test" in output


def test_kernel_only_setup_preserves_installed_torch_and_validates_before_ready(tmp_path, monkeypatch, capsys):
    """Repair the active environment without reinstalling Torch or declaring untested success."""

    monkeypatch.setattr(setup, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(setup, "detect_nvidia_driver", lambda: pytest.fail("kernel repair must not select another Torch wheel"))
    gpu_checks = []
    commands = []

    def gpu_check(python, **kwargs):
        """Observe the existing runtime checks without initializing CUDA."""

        gpu_checks.append((str(python), kwargs))
        return {"torch": "2.12.1+cu130", "cuda_forward_backward": "passed"}

    def command_run(command, **kwargs):
        """Capture installation and final required GPU smoke-test ordering."""

        commands.append(([str(part) for part in command], kwargs))

    monkeypatch.setattr(setup, "check_gpu", gpu_check)
    monkeypatch.setattr(setup, "run", command_run)
    setup.main(["--current-env", "--kernels-only", "--gpu", "2"])
    assert len(commands) == 3
    install = commands[0][0]
    assert "--only-binary=:all:" in install
    assert "flash-linear-attention==0.5.2" in install
    assert "--constraint" in install
    assert all("torch==" not in part for command, _ in commands for part in command)
    assert "check_training_kernels.py" in commands[-1][0][1]
    assert "--require-fla" in commands[-1][0] and "--smoke-test" in commands[-1][0]
    assert commands[-1][1]["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert len(gpu_checks) == 2
    assert (tmp_path / "environment_setup.json").exists()
    assert "Environment ready" in capsys.readouterr().out


def test_failed_training_kernel_check_does_not_record_environment_ready(tmp_path, monkeypatch, capsys):
    """An import, backward, or CUDA failure must stop setup before a success record."""

    monkeypatch.setattr(setup, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(setup, "check_gpu", lambda *args, **kwargs: {})

    def fail_probe(command, **kwargs):
        """Fail the required final check after a successful simulated pip install."""

        if "check_training_kernels.py" in str(command[1]):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup, "run", fail_probe)
    with pytest.raises(subprocess.CalledProcessError):
        setup.main(["--current-env", "--kernels-only"])
    assert not (tmp_path / "environment_setup.json").exists()
    assert "Environment ready" not in capsys.readouterr().out


def test_kernel_only_dry_run_has_no_cuda_or_install_side_effects(monkeypatch, capsys):
    """Allow inspecting the repair command on machines without CUDA."""

    monkeypatch.setattr(setup, "detect_nvidia_driver", lambda: pytest.fail("unexpected driver query"))
    monkeypatch.setattr(setup, "check_gpu", lambda *args, **kwargs: pytest.fail("unexpected GPU operation"))
    monkeypatch.setattr(setup, "run", lambda *args, **kwargs: pytest.fail("unexpected install"))
    setup.main(["--current-env", "--kernels-only", "--dry-run"])
    output = capsys.readouterr().out
    assert "flash-linear-attention==0.5.2" in output
    assert "--index-url" not in output


def test_runner_selects_virtual_environment_without_activation(tmp_path):
    executable = runtime.environment_python(tmp_path)
    executable.parent.mkdir(parents=True)
    executable.touch()
    assert runtime.select_python(environment_dir=tmp_path) == executable


def test_explicit_venv_interpreter_keeps_its_symlink_path(tmp_path):
    import sys

    executable = tmp_path / 'venv_python'
    executable.symlink_to(sys.executable)
    assert runtime.select_python(explicit=executable) == executable


@pytest.mark.parametrize("mode,folds", [("full", [1, 2]), ("smoke", [1]), ("dry-run", [1, 2])])
def test_all_conditions_have_matched_training_configuration(tmp_path, monkeypatch, mode, folds):
    from va_model_code.tests.test_train_model_contract import train_model_module

    monkeypatch.setattr(experiments, "PROJECT_ROOT", tmp_path)
    args = experiments.build_parser().parse_args(["--mode", mode, "--suite-name", "trial", "--batch-size", "8", "--gradient-accumulation-steps", "2"])
    plans = experiments.build_commands(args, Path("/python with spaces/bin/python"))
    assert [plan['condition'] for plan in plans] == ['raw', 'fixed', 'learned']
    for plan in plans:
        command = plan['command']
        assert command[0] == '/python with spaces/bin/python'
        parsed = train_model_module._build_parser().parse_args(command[3:])
        train_model_module._validate_args(parsed)
        assert list(parsed.held_out_folds) == folds
        assert parsed.train_batch_size * parsed.gradient_accumulation_steps == 16
        assert parsed.seed == 42 and parsed.epochs == 10
        assert parsed.sentence_only and parsed.precision == 'bf16'
        assert parsed.learning_rate == 6e-6
        assert parsed.redistribution_learning_rate == (1e-3 if plan['condition'] == 'learned' else None)


def test_runner_aborts_on_failed_training_before_next_condition(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(experiments, "check_gpu", lambda *args, **kwargs: {"bf16_supported": True})
    monkeypatch.setattr(experiments, "prepare_data", lambda *args, **kwargs: None)
    calls = []

    def fail(command, **kwargs):
        """Fail the first actual training command."""
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(experiments, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        experiments.main(["--suite-name", "failure_test"])
    assert len(calls) == 1


def test_partial_fold_pair_is_not_overwritten(tmp_path, monkeypatch):
    (tmp_path / 'full_dataset_fold1.csv').write_text('existing data')
    monkeypatch.setattr(experiments, 'run', lambda *args, **kwargs: pytest.fail('must not prepare over partial data'))
    with pytest.raises(FileNotFoundError, match='Only one fold'):
        experiments.prepare_data(tmp_path, '/python', {})


def test_python_subprocesses_do_not_invoke_a_shell(monkeypatch):
    captured = {}

    def observe(command, **kwargs):
        """Capture subprocess options without executing a command."""
        captured.update(kwargs)
        captured['command'] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, 'run', observe)
    runtime.run(['/path with spaces/python', '-m', 'pip', 'check'])
    assert captured['command'][0] == '/path with spaces/python'
    assert captured.get('shell', False) is False


@pytest.mark.parametrize(
    "stdout,stderr",
    [("probe started", "CUDA error: out of memory"), ("stdout-only failure", ""), ("", "")],
)
def test_gpu_probe_failure_preserves_child_diagnostics(monkeypatch, stdout, stderr):
    """Exercise a real failed subprocess without requiring CUDA or model packages."""

    probe = (
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        "sys.exit(7)\n"
    )
    monkeypatch.setattr(runtime, "GPU_PROBE_CODE", probe)
    with pytest.raises(RuntimeError, match="CUDA forward/backward check failed") as caught:
        runtime.check_gpu(sys.executable)
    message = str(caught.value)
    assert "exit status 7" in message
    assert "nvidia-smi" in message
    if stdout:
        assert "GPU probe stdout:\n" + stdout in message
    if stderr:
        assert "GPU probe stderr:\n" + stderr in message
    assert isinstance(caught.value.__cause__, subprocess.CalledProcessError)


@pytest.mark.parametrize("selected,expected", [(None, "GPU-existing"), (2, "2")])
def test_gpu_probe_success_preserves_visibility_and_final_json(monkeypatch, selected, expected):
    """Preserve device selection and parse the last JSON line on a successful probe."""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-existing")
    monkeypatch.setattr(
        runtime, "GPU_PROBE_CODE",
        "import json, os\nprint('probe started')\n"
        "print(json.dumps({'visible': os.environ['CUDA_VISIBLE_DEVICES'], 'cuda_forward_backward': 'passed'}))\n",
    )
    result = runtime.check_gpu(sys.executable, gpu=selected)
    assert result == {"visible": expected, "cuda_forward_backward": "passed"}
