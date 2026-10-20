"""Check refactor parity, archive preservation and fail-closed baseline boundaries."""

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from va_model_code.decoder_va.baseline_contracts import reject_experimental_weights
from va_model_code.decoder_va.model import DecoderVARegressor
from va_model_code.tests.test_alignment_training_integration import tiny_native_qwen_backbone
from va_model_code.tests.test_model_loss_evaluation import FakeCausalBackbone, FakeGazeProvider
from va_model_code.tests.test_train_model_contract import train_model_module


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("fusion", ["none", "prefix-concat"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_raw_model_initialization_forward_gradients_and_update_match_snapshot(fusion, bf16, native):
    """Compare complete training steps against the frozen implementation, including RNG."""

    from legacy_code.va_model_code.decoder_va.model import DecoderVARegressor as ArchivedModel

    torch.manual_seed(187)
    backbone = tiny_native_qwen_backbone(torch.float32) if native else FakeCausalBackbone()
    kwargs = {
        "gaze_fusion": fusion,
        "gaze_provider": FakeGazeProvider() if fusion == "prefix-concat" else None,
        "gaze_projection_dim": 8,
    }
    torch.manual_seed(193)
    before = ArchivedModel(copy.deepcopy(backbone), **kwargs)
    original_rng = torch.get_rng_state()
    torch.manual_seed(193)
    after = DecoderVARegressor(copy.deepcopy(backbone), **kwargs)
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert before.state_dict().keys() == after.state_dict().keys()
    for key, value in before.state_dict().items():
        torch.testing.assert_close(after.state_dict()[key], value, rtol=0, atol=0)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [2, 7, 0, 0], [3, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]),
    }
    labels = torch.tensor([[0.2, 0.7], [0.8, 0.3], [0.4, 0.6]])
    predictions = []
    for model in (before, after):
        if native:
            model.gradient_checkpointing_enable({"use_reentrant": False})
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-6)
        torch.manual_seed(197)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            logits = model(**inputs).logits
            loss = (logits - labels).square().mean()
        assert logits.shape == (3, 2) and logits.dtype == torch.float32
        loss.backward()
        predictions.append(logits.detach())
        optimizer.step()
    torch.testing.assert_close(*predictions, rtol=0, atol=0)
    old_parameters = dict(before.named_parameters())
    for name, parameter in after.named_parameters():
        old = old_parameters[name]
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)
        if old.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, old.grad, rtol=0, atol=0)
    assert not any(
        token in name for name in after.state_dict()
        for token in ("gaze_redistributor", "gaze_relative_rotation", "readout_token", "variance_head")
    )


@pytest.mark.parametrize("kwargs", [
    {"gaze_redistribution": {"method": "fixed-gaussian"}},
    {"gaze_relation": {"version": 1, "method": "trt-relative-rotation"}},
    {"gaze_relation": {"version": 3, "method": "trt-readout-rotation"}},
    {"readout": {"version": 1, "mode": "learned-token"}},
    {"prediction_distribution": {"version": 1, "family": "diagonal_gaussian"}},
])
def test_experimental_metadata_is_rejected_instead_of_silently_becoming_raw(kwargs):
    """Make enabled experimental contracts fail before backbone execution."""

    with pytest.raises(ValueError, match="legacy_code"):
        DecoderVARegressor(FakeCausalBackbone(), gaze_fusion="none", **kwargs)


@pytest.mark.parametrize("option", [
    ["--gaze-redistribution", "fixed-gaussian"],
    ["--redistribution-sigma-left", "2"],
    ["--gaze-relation", "trt-readout-rotation"],
    ["--readout", "learned-token"],
    ["--sigma-diagnostics-steps", "50"],
])
def test_experimental_cli_options_are_absent(option):
    """Keep archived options out of the active training interface."""

    with pytest.raises(SystemExit):
        train_model_module._build_parser().parse_args(["qwen", "mse", *option])
    with pytest.raises(SystemExit):
        train_model_module._build_parser().parse_args(["qwen", "heteroscedastic"])


@pytest.mark.parametrize("key", [
    "gaze_redistributor.kernel.log_sigma_left",
    "backbone.layers.3.self_attn.gaze_relative_rotation.alpha",
    "readout_token",
    "variance_head.weight",
])
def test_experimental_state_cannot_hide_behind_raw_metadata(tmp_path, key):
    """Reject experiment weights even when their manifest claims baseline semantics."""

    path = tmp_path / "model.safetensors"
    save_file({key: torch.zeros(2)}, path)
    with pytest.raises(ValueError, match="legacy_code"):
        reject_experimental_weights(path)


def test_active_source_has_no_experimental_imports():
    """Prevent the baseline from accidentally depending on the archived stack."""

    forbidden = {
        "legacy", "legacy_code", "redistribution", "gaze_relation", "gaze_attention",
        "gaze_relation_inputs", "gaze_relation_diagnostics", "gaze_relation_reporting",
        "sigma_diagnostics", "readout", "uncertainty", "sigma_runtime", "run_sigma_experiments",
    }
    paths = list((ROOT / "va_model_code").glob("*.py"))
    paths += list((ROOT / "va_model_code/decoder_va").glob("*.py"))
    paths += [ROOT / "setup_environment.py", ROOT / "environment_runtime.py"]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [entry.name for entry in node.names]
            else:
                continue
            assert not any(forbidden.intersection(module.split(".")) for module in modules), path


def test_archive_snapshot_hashes_and_cli_are_available_from_another_directory(tmp_path):
    """Verify byte preservation and the isolated experiment launcher without training."""

    launcher = ROOT / "legacy_code/run.py"
    result = subprocess.run(
        [sys.executable, "-B", str(launcher), "verify"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "Verified" in result.stdout and "files from" in result.stdout
    result = subprocess.run(
        [sys.executable, "-B", str(launcher), "train", "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "trt-readout-rotation" in result.stdout
    assert "heteroscedastic" in result.stdout


@pytest.mark.parametrize("fusion", ["none", "prefix-concat"])
@pytest.mark.parametrize("schema", [8, 9, 10, 11])
def test_baseline_two_fold_results_keep_packaging_compatibility(tmp_path, fusion, schema):
    """Keep completed historical Raw and current baseline result schemas packageable."""

    import zipfile
    from va_model_code.package_results import package_results
    from va_model_code.tests.test_result_layout import _write_completed_run

    run = tmp_path / "baseline"
    _write_completed_run(run, gaze_fusion=fusion, schema_version=schema)
    archive = package_results(run, results_root=tmp_path)
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        assert not any(name.endswith((".safetensors", ".bin", ".pt")) for name in saved.namelist())


def test_experimental_package_uses_archive_and_original_results_location(tmp_path):
    """Reject a fixed-kernel experiment actively, then package it with archived validators."""

    from legacy_code.va_model_code.decoder_va.redistribution import redistribution_contract
    from va_model_code.package_results import package_results
    from va_model_code.tests.test_result_layout import _write_completed_run

    run = tmp_path / "results/archived_fixed"
    _write_completed_run(run, gaze_fusion="prefix-concat", schema_version=11)
    contract = redistribution_contract("fixed-gaussian")
    paths = [run / "training_parameters.json", *run.glob("heldout_fold*/run_manifest.json")]
    paths += list(run.glob("heldout_fold*/final_model/decoder_va_architecture.json"))
    for path in paths:
        payload = json.loads(path.read_text())
        payload["gaze_redistribution"] = contract
        if "reconstruction" in payload:
            payload["reconstruction"]["gaze_redistribution"] = contract
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="legacy_code"):
        package_results(run, results_root=run.parent)
    code = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "import legacy_code.run as archive; "
        "archive.PROJECT_ROOT = Path(sys.argv[1]); "
        "archive.main(['package', '--run-name', 'archived_fixed'])"
    )
    subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert len(list(run.glob("*_results_only.zip"))) == 1
    assert not (tmp_path / "legacy_code/results").exists()
