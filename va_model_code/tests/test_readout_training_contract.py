"""Protect learned-readout CLI, provenance, packaging, and resume boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import zipfile

import pytest

from va_model_code.decoder_va.gaze_relation import gaze_relation_contract
from va_model_code.decoder_va.paths import condition_slug
from va_model_code.decoder_va.readout import (
    readout_contract,
    readout_concat_order,
    readout_pooling_position,
)
from va_model_code.package_results import package_results
from va_model_code.tests.test_result_layout import (
    _write_completed_run,
    _write_relation_parameter_reports,
)
from va_model_code.tests.test_train_model_contract import (
    _check_resume_pair,
    _expected_resume_manifest,
    train_model_module,
)
from va_model_code.tests.test_overlap_training import offline_overlap_run, _write_fold


def _args(*options):
    """Parse and validate the real command line without model downloads."""

    args = train_model_module._build_parser().parse_args(["qwen", "mse", *options])
    train_model_module._validate_args(args)
    return args


@pytest.mark.parametrize("method", ["none", "trt-readout-rotation"])
def test_learned_readout_cli_accepts_matched_raw_and_rotation(method):
    args = _args("--readout", "learned-token", "--gaze-relation", method)
    assert args.readout == "learned-token"
    assert args.gaze_relation == method
    assert args.gaze_redistribution == "none"
    if method != "none":
        assert args.attn_implementation == "eager"
        assert args.gaze_relation_learning_rate == 1e-3
        assert args.gaze_relation_weight_decay == 0.0


@pytest.mark.parametrize("options,message", [
    (("--gaze-relation", "trt-readout-rotation"), "learned-token"),
    (("--readout", "learned-token", "--gaze-relation", "trt-relative-rotation"), "last-token"),
    (("--readout", "learned-token", "--gaze-relation", "trt-readout-rotation", "--gaze-fusion", "none"), "prefix-concat"),
    (("--readout", "learned-token", "--gaze-relation", "trt-readout-rotation", "--attn-implementation", "sdpa"), "eager"),
])
def test_readout_cli_rejects_operator_mismatch_before_download(options, message):
    with pytest.raises(ValueError, match=message):
        _args(*options)


def test_learned_readout_run_name_cannot_collide_with_last_token_raw():
    shared = dict(model="qwen", finetuning_mode="full", gaze_fusion="prefix-concat", gaze_features=("TRT",), seed=42)
    legacy = condition_slug(**shared)
    raw = condition_slug(**shared, readout=readout_contract("learned-token"))
    rotation = condition_slug(**shared, readout=readout_contract("learned-token"), gaze_relation=gaze_relation_contract("trt-readout-rotation"))
    assert len({legacy, raw, rotation}) == 3
    assert "readout_learned-token" in raw
    assert "trt-readout-rotation" in rotation


def test_learned_readout_dry_run_records_identity_without_model_download(offline_overlap_run, capsys):
    run = offline_overlap_run
    for fold in (1, 2):
        _write_fold(run.data_root, fold, ["same", "other"])
    args = run.arguments("--readout", "learned-token", "--training-text-overlap", "keep", "--dry-run")
    assert train_model_module.run(args) is None
    assert "learned-token" in capsys.readouterr().out
    assert not run.observed["builds"]
    assert not run.observed["trainers"]


@pytest.mark.parametrize("gaze_fusion", ["none", "prefix-concat"])
def test_readout_run_propagates_to_model_and_both_fold_manifests(offline_overlap_run, gaze_fusion):
    run = offline_overlap_run
    for fold in (1, 2):
        _write_fold(run.data_root, fold, ["same", "other"])
    args = run.arguments("--readout", "learned-token", "--gaze-fusion", gaze_fusion, "--training-text-overlap", "keep")
    assert train_model_module.run(args) == run.run_root
    expected = readout_contract("learned-token")
    for construction in run.observed["builds"]:
        assert construction["readout"] == expected
    for relative in ("training_parameters.json", "heldout_fold1/run_manifest.json", "heldout_fold2/run_manifest.json"):
        manifest = json.loads((run.run_root / relative).read_text())
        assert manifest["architecture_manifest_version"] == 10
        assert manifest["readout"] == expected
        assert manifest["pooling_position"] == readout_pooling_position(gaze_fusion, expected)
        assert manifest["gaze_concat_order"] == readout_concat_order(gaze_fusion, expected)
        assert manifest["training_text_overlap_policy"] == "keep"


@pytest.mark.parametrize("method", ["none", "trt-readout-rotation"])
def test_package_preserves_schema10_readout_and_rotation_reports(tmp_path, method):
    run = tmp_path / "learned_readout_run"
    relation = gaze_relation_contract(method)
    _write_completed_run(run, schema_version=10, gaze_fusion="prefix-concat", readout=readout_contract("learned-token"), gaze_relation=relation)
    if method != "none":
        _write_relation_parameter_reports(run, relation)
    archive = package_results(run, results_root=tmp_path)
    assert "readout_learned-token" in archive.name
    with zipfile.ZipFile(archive) as zipped:
        metadata_path = next(name for name in zipped.namelist() if name.endswith("results_manifest.json"))
        assert json.loads(zipped.read(metadata_path))["condition"]["readout"] == readout_contract("learned-token")
        assert not any(name.endswith(".safetensors") for name in zipped.namelist())


@pytest.mark.parametrize("version", [6, 7, 8, 9])
def test_historical_packages_keep_legacy_readout_without_new_required_fields(tmp_path, version):
    run = tmp_path / f"historical_schema{version}"
    _write_completed_run(run, schema_version=version, gaze_fusion="prefix-concat")
    assert package_results(run, results_root=tmp_path).is_file()


@pytest.mark.parametrize("target", ["training_parameters.json", "heldout_fold1/run_manifest.json", "architecture", "reconstruction"])
def test_schema10_packaging_requires_explicit_readout_everywhere(tmp_path, target):
    run = tmp_path / "missing_readout"
    _write_completed_run(run, schema_version=10, gaze_fusion="prefix-concat")
    path = run / ("heldout_fold1/final_model/decoder_va_architecture.json" if target in ("architecture", "reconstruction") else target)
    payload = json.loads(path.read_text())
    source = payload["reconstruction"] if target == "reconstruction" else payload
    source.pop("readout")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="readout"):
        package_results(run, results_root=tmp_path)


@pytest.mark.parametrize("target", ["heldout_fold1/run_manifest.json", "architecture", "reconstruction"])
def test_packaging_rejects_mixed_raw_readout_conditions(tmp_path, target):
    run = tmp_path / "mixed_readout"
    _write_completed_run(run, schema_version=10, gaze_fusion="prefix-concat", readout=readout_contract("learned-token"))
    path = run / ("heldout_fold1/final_model/decoder_va_architecture.json" if target in ("architecture", "reconstruction") else target)
    payload = json.loads(path.read_text())
    source = payload["reconstruction"] if target == "reconstruction" else payload
    source["readout"] = readout_contract()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="readout"):
        package_results(run, results_root=tmp_path)


def test_package_rejects_readout_rotation_report_in_raw_run(tmp_path):
    run = tmp_path / "raw_with_stale_rotation_report"
    _write_completed_run(run, schema_version=10, gaze_fusion="prefix-concat", readout=readout_contract("learned-token"))
    _write_relation_parameter_reports(run, gaze_relation_contract("trt-readout-rotation"))
    with pytest.raises(ValueError, match="gaze_relation"):
        package_results(run, results_root=tmp_path)


def test_resume_accepts_matching_readout_and_rejects_changed_pooling(tmp_path):
    expected = _expected_resume_manifest()
    expected["readout"] = readout_contract("learned-token")
    expected["pooling_position"] = readout_pooling_position(expected["gaze_fusion"], expected["readout"])
    expected["gaze_concat_order"] = readout_concat_order(expected["gaze_fusion"], expected["readout"])
    _check_resume_pair(tmp_path / "valid", copy.deepcopy(expected), expected)
    recorded = copy.deepcopy(expected)
    recorded["readout"] = readout_contract()
    recorded["pooling_position"] = readout_pooling_position(recorded["gaze_fusion"], recorded["readout"])
    recorded["gaze_concat_order"] = readout_concat_order(recorded["gaze_fusion"], recorded["readout"])
    with pytest.raises(ValueError, match="readout"):
        _check_resume_pair(tmp_path / "invalid", recorded, expected)


def test_resume_current_schema_rejects_missing_readout(tmp_path):
    expected = _expected_resume_manifest()
    recorded = copy.deepcopy(expected)
    recorded.pop("readout")
    with pytest.raises(ValueError, match="readout"):
        _check_resume_pair(tmp_path, recorded, expected)


@pytest.mark.parametrize("method,readout_mode", [
    ("trt-relative-rotation", "last-token"),
    ("trt-readout-rotation", "learned-token"),
])
def test_schema10_requires_learned_rotation_coefficient_report(tmp_path, method, readout_mode):
    run = tmp_path / "missing_rotation_report"
    _write_completed_run(
        run, schema_version=10, gaze_fusion="prefix-concat",
        readout=readout_contract(readout_mode), gaze_relation=gaze_relation_contract(method),
    )
    with pytest.raises(FileNotFoundError, match="gaze_relation_parameters"):
        package_results(run, results_root=tmp_path)


def test_schema9_preserves_optional_rotation_report_compatibility(tmp_path):
    run = tmp_path / "historical_rotation_without_report"
    _write_completed_run(
        run, schema_version=9, gaze_fusion="prefix-concat",
        gaze_relation=gaze_relation_contract("trt-relative-rotation"),
    )
    assert package_results(run, results_root=tmp_path).is_file()


@pytest.mark.parametrize("method", ["none", "trt-relative-rotation"])
def test_schema9_last_token_resume_preserves_contract_and_source_metadata(tmp_path, method):
    """A schema-only upgrade leaves legacy raw/rotation weights and metadata untouched."""

    expected = _expected_resume_manifest()
    expected["gaze_relation"] = gaze_relation_contract(method)
    if method != "none":
        expected.update(attn_implementation="eager", gaze_relation_learning_rate=1e-3, gaze_relation_weight_decay=0.0)
    recorded = copy.deepcopy(expected)
    recorded["architecture_manifest_version"] = 9
    recorded.pop("readout")
    _check_resume_pair(tmp_path, recorded, expected)
    saved = json.loads((tmp_path / "run/heldout_fold1/run_manifest.json").read_text())
    assert saved == recorded
    assert expected["architecture_manifest_version"] == 10
    assert expected["readout"] == readout_contract()


def test_schema9_resume_cannot_add_learned_readout(tmp_path):
    expected = _expected_resume_manifest()
    recorded = copy.deepcopy(expected)
    recorded["architecture_manifest_version"] = 9
    recorded.pop("readout")
    expected["readout"] = readout_contract("learned-token")
    expected["pooling_position"] = readout_pooling_position(expected["gaze_fusion"], expected["readout"])
    expected["gaze_concat_order"] = readout_concat_order(expected["gaze_fusion"], expected["readout"])
    with pytest.raises(ValueError, match="unchanged last-token"):
        _check_resume_pair(tmp_path, recorded, expected)


@pytest.mark.parametrize("field,value", [
    ("learning_rate", 9e-4),
    ("gaze_relation_learning_rate", 0.2),
    ("train_batch_size", 32),
    ("pooling_position", "learned_readout_token_after_text"),
    ("gaze_relation", gaze_relation_contract()),
])
def test_schema9_resume_keeps_optimizer_and_operator_checks_strict(tmp_path, field, value):
    expected = _expected_resume_manifest()
    expected.update(
        gaze_relation=gaze_relation_contract("trt-relative-rotation"),
        attn_implementation="eager", gaze_relation_learning_rate=1e-3, gaze_relation_weight_decay=0.0,
    )
    recorded = copy.deepcopy(expected)
    recorded["architecture_manifest_version"] = 9
    recorded.pop("readout")
    recorded[field] = value
    with pytest.raises(ValueError, match=field):
        _check_resume_pair(tmp_path, recorded, expected)


def test_resume_records_source_schema_and_checks_weights_before_training(offline_overlap_run, monkeypatch):
    """The training pipeline preserves migration provenance and invokes all weight guards."""

    run = offline_overlap_run
    for fold in (1, 2):
        _write_fold(run.data_root, fold, ["same", "other"])
    args = run.arguments("--held-out-folds", "1", "--training-text-overlap", "keep")
    train_model_module.run(args)
    manifest_path = run.run_root / "heldout_fold1/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["architecture_manifest_version"] = 9
    manifest.pop("readout")
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = run.run_root / "heldout_fold1/checkpoints/checkpoint-10"
    checkpoint.mkdir(parents=True)
    calls = []

    def record_precision(model, path):
        calls.append(("precision", path))

    def record_relation(model, path):
        calls.append(("relation", path))

    def record_readout(model, path):
        calls.append(("readout", path))

    monkeypatch.setattr(train_model_module, "validate_saved_trainable_precision", record_precision)
    monkeypatch.setattr(train_model_module, "validate_saved_gaze_relation", record_relation)
    monkeypatch.setattr(train_model_module, "validate_saved_readout", record_readout)
    args.resume_from_checkpoint = str(checkpoint)
    train_model_module.run(args)
    saved = json.loads(manifest_path.read_text())
    assert saved["architecture_manifest_version"] == 10
    assert saved["resume_source_architecture_manifest_version"] == 9
    assert saved["readout"] == readout_contract()
    assert calls == [(name, checkpoint / "model.safetensors") for name in ("precision", "relation", "readout")]
