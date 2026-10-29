"""CPU-only protocol, numerical-operator and end-to-end control-flow regression tests."""

import ast
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from holdout_experiments.config import ARMS, Settings
from holdout_experiments.engine import (
    ValidationSelection, _fit, epoch_order, learning_rate_factor,
    loss_for_arm, optimizer_for, predict,
)
from holdout_experiments.model import ExperimentModel, restore_weights, save_model
from holdout_experiments.protocol import (
    digest, grouped_split, load_lock, prepare, read_partition, source_manifest,
    validate_partitions, write_json,
)
from holdout_experiments.redistribution import AsymGaussianRedistributor
from holdout_experiments.reporting import completion_receipts, seal_selection
from trt_pipeline.config import ROOT
from trt_pipeline.core.dataset import VABatchCollator
from trt_pipeline.data import contract_sha256, text_groups
from trt_pipeline.precision import BF16DecoderVARegressor
from trt_pipeline.two_view import TwoViewVAEncoder


def synthetic_frame(rows=90):
    """Include Unicode IDs, cross-source duplicate texts, blanks and escaped characters."""

    texts = [f"item {i // 2}" for i in range(rows)]
    texts[:6] = ["  CAFÉ  ", "cafe\u0301", "x\ty", "x\\y\nnext", "", "  "]
    return pd.DataFrame({
        "index": [f"id漢字{i}" for i in range(rows)], "text": texts,
        "dataset_of_origin": ["sourceA" if i % 3 else "sourceB" for i in range(rows)],
        "valence": np.linspace(.1, .9, rows), "arousal": np.linspace(.9, .1, rows),
    })


def synthetic_run(tmp_path, *, epochs=2):
    """Build the real lock schema around small tables without requiring private data."""

    settings = Settings(epochs=epochs, batch_size=8, eval_batch_size=8)
    frames = grouped_split(synthetic_frame(), settings)
    run = tmp_path / "run"
    (run / "splits").mkdir(parents=True)
    partitions = {}
    for name, frame in frames.items():
        path = run / "splits" / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False, quoting=3, escapechar="\\")
        partitions[name] = {"rows": len(frame), "sha256": digest(path)}
    lock = {"settings": settings.as_dict(), "partitions": partitions,
            "source_sha256": source_manifest(), "updates_per_epoch": (len(frames["train"]) + 7) // 8,
            "total_updates_per_fit": ((len(frames["train"]) + 7) // 8) * epochs}
    lock["protocol_sha256"] = contract_sha256(lock)
    write_json(run / "protocol_lock.json", lock)
    return run, lock, settings, frames


class TinyTokenizer:
    """Provide deterministic tokens without Hugging Face downloads."""

    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def __call__(self, text, **kwargs):
        tokens = [2 + (ord(char) % 15) for char in str(text)[:4]] or [1]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    def save_pretrained(self, directory):
        Path(directory, "tiny_tokenizer.json").write_text("{}")


class TinyBackbone(nn.Module):
    """Exercise true mixed BF16/FP32 storage and causal dependence on gaze tokens."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6, model_type="tiny", use_cache=False)
        self.embedding = nn.Embedding(18, 6, dtype=torch.bfloat16)
        self.projection = nn.Linear(6, 6, dtype=torch.bfloat16)

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        masked = inputs_embeds * attention_mask[:, :, None]
        return SimpleNamespace(last_hidden_state=self.projection(masked.cumsum(dim=1)))


def tiny_model(settings, arm, **kwargs):
    """Use the real base fusion/head and experiment wrapper around a tiny backbone."""

    backbone = TinyBackbone()
    provider = SimpleNamespace(
        feature_indices=(3,),
        compute=lambda ids, mask: ((ids.float().remainder(5) / 5).unsqueeze(-1), mask.bool()),
    )
    base = BF16DecoderVARegressor(
        backbone, native_backbone_dtypes={name: str(p.dtype).removeprefix("torch.")
                                         for name, p in backbone.named_parameters()},
        gaze_provider=provider, gaze_projection_dim=8,
        gaze_projection_dropout=(0.0, 0.0), classifier_dropout=0.0,
    )
    base._reconstruction_config = {"tiny_cpu_test_fixture": True}
    return ExperimentModel(base, arm, settings), TinyTokenizer()


def test_group_split_is_deterministic_source_aware_and_label_blind():
    frame = synthetic_frame()
    first = grouped_split(frame, Settings())
    second = grouped_split(frame, Settings())
    changed = frame.copy()
    changed[["valence", "arousal"]] = 1 - changed[["valence", "arousal"]]
    third = grouped_split(changed, Settings())
    validate_partitions(first)
    assert sum(len(x) for x in first.values()) == len(frame)
    for name in first:
        assert first[name]["index"].tolist() == second[name]["index"].tolist() == third[name]["index"].tolist()
        assert set(first[name]["dataset_of_origin"]) == {"sourceA", "sourceB"}
    owner = {group: name for name, rows in first.items() for group in text_groups(rows)}
    assert owner["café"] in first and "" in owner


def test_split_leakage_and_duplicate_ids_rejected():
    frame = synthetic_frame()
    parts = grouped_split(frame, Settings())
    parts["test"] = pd.concat((parts["test"], parts["train"].iloc[:1]))
    with pytest.raises(ValueError, match="leakage"):
        validate_partitions(parts)
    frame.loc[1, "index"] = frame.loc[0, "index"]
    with pytest.raises(ValueError, match="unique"):
        grouped_split(frame, Settings())


@pytest.mark.parametrize("overrides", [{"epochs": 0}, {"seed": True}, {"split_seed": -1},
                                     {"fractions": (.9, .1, .1)}, {"learning_rate": float("nan")},
                                     {"noise_std_fraction": .02}, {"prediction_alpha": .5}])
def test_invalid_or_changed_fixed_settings_rejected(overrides):
    with pytest.raises(ValueError):
        Settings(**overrides)


def test_lock_and_unicode_tsv_round_trip_and_tamper(tmp_path):
    run, lock, settings, frames = synthetic_run(tmp_path)
    assert load_lock(run)[1] == settings
    for name, frame in frames.items():
        loaded = read_partition(run / "splits" / f"{name}.tsv")
        assert loaded["index"].tolist() == frame["index"].tolist()
        assert loaded["text"].tolist() == frame["text"].tolist()
        np.testing.assert_array_equal(loaded[["valence", "arousal"]], frame[["valence", "arousal"]])
    with (run / "splits/test.tsv").open("a") as stream:
        stream.write("edited")
    with pytest.raises(ValueError, match="test data changed"):
        load_lock(run)


def test_protocol_lock_and_code_change_rejected(tmp_path, monkeypatch):
    run, lock, _, _ = synthetic_run(tmp_path)
    from holdout_experiments import protocol
    monkeypatch.setattr(protocol, "source_manifest", lambda: {"changed": "sha"})
    with pytest.raises(ValueError, match="Source changed"):
        load_lock(run)
    lock["settings"]["epochs"] = 8
    write_json(run / "protocol_lock.json", lock)
    with pytest.raises(ValueError, match="lock has changed"):
        load_lock(run)


def test_real_data_split_counts_and_budget(tmp_path):
    if not (ROOT / "data/paper7_seed42/full_dataset_fold1.csv").exists():
        pytest.skip("Private canonical data not present in this clone")
    run = tmp_path / "real"
    lock = prepare(run)
    assert [lock["partitions"][name]["rows"] for name in ("train", "val", "test")] == [43034, 5377, 5380]
    assert lock["total_updates_per_fit"] == 26900
    assert lock["labels_used_for_splitting"] is False
    parts = {name: read_partition(run / "splits" / f"{name}.tsv") for name in ("train", "val", "test")}
    validate_partitions(parts)
    assert sum(len(p) for p in parts.values()) == 53791
    with pytest.raises(FileExistsError):
        prepare(run)


def test_schedule_and_common_epoch_order():
    assert epoch_order(27, 42, 1) == epoch_order(27, 42, 1)
    assert epoch_order(27, 42, 1) != epoch_order(27, 42, 2)
    assert sorted(epoch_order(27, 42, 1)) == list(range(27))
    assert [learning_rate_factor(s, warmup=2, total=10) for s in (0, 1, 2, 6, 10)] == [0, .5, 1, .5, 0]


def test_validation_selection_cannot_accept_test_or_tune_on_ties():
    selector = ValidationSelection()
    assert selector.observe(partition="val", metrics={"mse_mean": .2}, epoch=1, step=9)
    assert not selector.observe(partition="val", metrics={"mse_mean": .2}, epoch=2, step=18)
    assert not selector.observe(partition="val", metrics={"mse_mean": .3}, epoch=3, step=27)
    with pytest.raises(ValueError, match="Only validation"):
        selector.observe(partition="test", metrics={"mse_mean": .01}, epoch=4, step=36)
    assert selector.epoch == 1 and selector.step == 9


def test_companion_exact_two_view_objective_and_clean_eval():
    settings = Settings()
    torch.manual_seed(12)
    model, tokenizer = tiny_model(settings, "companion")
    reference = TwoViewVAEncoder(copy.deepcopy(model.base_model))
    batch = VABatchCollator(tokenizer)([
        {"input_ids": torch.tensor([2, 4, 6]), "attention_mask": torch.ones(3, dtype=torch.long), "labels": torch.tensor([.2, .8])},
        {"input_ids": torch.tensor([3, 5]), "attention_mask": torch.ones(2, dtype=torch.long), "labels": torch.tensor([.7, .4])},
    ])
    gold = batch.pop("labels")
    model.train()
    reference.train()
    torch.manual_seed(9)
    observed = model(**batch)
    loss = loss_for_arm(observed, gold, "companion")
    torch.manual_seed(9)
    expected = reference(**batch)
    expected_loss = torch.nn.functional.mse_loss(expected["view_predictions"].float(), torch.cat((gold, gold)))
    assert torch.equal(loss, expected_loss)
    loss.backward()
    expected_loss.backward()
    for left, right in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    model.eval()
    assert "view_predictions" not in model(**batch)
    torch.testing.assert_close(model(**batch)["logits"], model.base_model(**batch).logits)


def test_asym_gaussian_mass_mask_orientation_gradient_and_archived_parity():
    op = AsymGaussianRedistributor(.7, 1.9)
    values = torch.tensor([[0., 3., -2., 8., 0.]], requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 1]])
    actual = op(values, mask)
    positions = torch.arange(5, dtype=torch.float32)
    difference = positions[:, None] - positions[None, :]
    sigma = torch.where(difference < 0, op.sigma_left, op.sigma_right)
    kernel = torch.exp(-.5 * (difference / sigma)**2)
    kernel = kernel * (mask.bool().T & mask.bool())
    kernel /= kernel.sum(0).clamp_min(1)
    expected = (kernel @ (values[0] * mask[0])).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual.sum(1), (values * mask).sum(1))
    assert actual[0, 3] == 0 and actual[0, 4] != 0
    (actual * torch.arange(5)).sum().backward()
    assert op.log_sigma_left.grad.abs() > 0 and op.log_sigma_right.grad.abs() > 0
    assert values.grad[0, 3] == 0
    archive = ROOT / "artifacts/cleanup_20261029/original_workspace.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as saved:
            old_tree = ast.parse(saved.read("legacy_code/va_model_code/decoder_va/redistribution.py"))
        new_tree = ast.parse((ROOT / "holdout_experiments/redistribution.py").read_text())
        for name in ("_positive_float", "AsymGaussianRedistributor"):
            old = next(n for n in old_tree.body if getattr(n, "name", None) == name)
            new = next(n for n in new_tree.body if getattr(n, "name", None) == name)
            assert ast.dump(old) == ast.dump(new)


def test_redistribution_optimizer_has_own_fp32_zero_decay_group():
    settings = Settings()
    model, _ = tiny_model(settings, "redistribution")
    optimizer = optimizer_for(model, settings)
    widths = optimizer.param_groups[-1]
    assert widths["lr"] == 1e-3 and widths["weight_decay"] == 0
    assert len(widths["params"]) == 2 and all(p.dtype == torch.float32 for p in widths["params"])
    all_params = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(all_params) == len(set(all_params)) == len(list(model.parameters()))


@pytest.mark.parametrize("arm", ARMS)
def test_safetensors_full_state_dtype_roundtrip_and_hash_guard(tmp_path, arm, monkeypatch):
    settings = Settings()
    model, _ = tiny_model(settings, arm)
    before = {name: p.detach().clone() for name, p in model.state_dict().items()}
    manifest = save_model(model, tmp_path, settings, "protocol")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    restore_weights(model, tmp_path, expected_sha256=manifest["weights_sha256"])
    for name, tensor in model.state_dict().items():
        assert tensor.dtype == before[name].dtype and torch.equal(tensor, before[name])
    with pytest.raises(ValueError, match="SHA"):
        restore_weights(model, tmp_path, expected_sha256="bad")
    from holdout_experiments import model as model_module
    monkeypatch.setattr(model_module, "load_auto_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(model_module, "build_model", tiny_model)
    loaded, _, _ = model_module.load_saved(tmp_path, settings, "protocol", arm)
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, before[name])
    with pytest.raises(ValueError, match="belong"):
        model_module.load_saved(tmp_path, settings, "other-protocol", arm)
    from safetensors.torch import load_file, save_file
    weights_path = tmp_path / "model.safetensors"
    state = load_file(str(weights_path))
    first_key = next(iter(state))
    state[first_key] = state[first_key].double()
    save_file(state, str(weights_path))
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        restore_weights(model, tmp_path, expected_sha256=digest(weights_path))


def test_tiny_three_fit_control_flow_test_gate_final_mix_and_cpu_verifier(tmp_path, monkeypatch):
    run, lock, settings, _ = synthetic_run(tmp_path)
    from holdout_experiments import engine
    real_read = engine.read_partition
    accesses = []

    def train_only_reader(path):
        """Catch even one accidental test table read during a complete fit."""

        accesses.append(Path(path).name)
        assert Path(path).name in ("train.tsv", "val.tsv")
        return real_read(path)

    monkeypatch.setattr(engine, "read_partition", train_only_reader)
    monkeypatch.setattr(engine, "build_model", tiny_model)
    for arm in ARMS:
        if arm == "redistribution":
            from holdout_experiments.reporting import prepare_prediction_mixture
            prepared = prepare_prediction_mixture(run)
            assert prepared["actual_optimizer_updates"] == 0
            assert prepared["test_evaluated_at_preparation"] is False
            assert not (run / "redistribution/train_complete.json").exists()
            assert not (run / "evaluation_started.json").exists()
        output = run / arm
        output.mkdir()
        result = _fit(run, output, arm, lock, settings, {"fixture": "tiny CPU; not efficacy"},
                      time.perf_counter(), device=torch.device("cpu"))
        assert result["actual_optimizer_updates"] == lock["total_updates_per_fit"]
        assert result["test_evaluations_during_training"] == 0
        assert result["training_device"] == "cpu"
    assert accesses == ["train.tsv", "val.tsv"] * 3
    receipts = completion_receipts(run, lock)
    assert set(receipts) == set(ARMS)
    sealed = seal_selection(run)
    assert sealed == seal_selection(run)
    assert not (run / "evaluation_started.json").exists()
    with pytest.raises(RuntimeError, match="sealed"):
        engine.train(run, "baseline")
    from holdout_experiments import reporting

    def cpu_fixture_load(directory, config, protocol_sha256, arm):
        """Replace external pretrained loading with the saved tiny CPU model only."""

        if arm == "redistribution":
            assert (run / "prediction_mixture/condition_complete.json").exists()
        manifest = json.loads((Path(directory) / "model_manifest.json").read_text())
        model, tokenizer = tiny_model(config, arm)
        restore_weights(model, directory, expected_sha256=manifest["weights_sha256"])
        model.to = lambda *args, **kwargs: model
        return model.eval(), tokenizer, manifest

    def cpu_fixture_predict(model, loader, device):
        """Exercise real sequential prediction while excluding any CUDA execution."""

        assert device.type == "cuda"
        return predict(model, loader, torch.device("cpu"))

    monkeypatch.setattr(reporting, "runtime_check", lambda: {"fixture": "CPU only; not efficacy"})
    monkeypatch.setattr(reporting, "load_saved", cpu_fixture_load)
    monkeypatch.setattr(reporting, "predict", cpu_fixture_predict)
    report = reporting.evaluate(run)
    assert set(report["metrics"]) == {*ARMS, "prediction_mixture"}
    assert report["conditions"] == ["baseline", "companion", "prediction_mixture", "redistribution"]
    assert report["condition_count"] == 4 and report["fit_count"] == 3
    assert report["condition_results"]["prediction_mixture"]["actual_optimizer_updates"] == 0
    verified = reporting.verify_results(run)
    assert verified["status"] == "passed" and verified["verified_prediction_rows"] == report["n_test"] * 4
    with pytest.raises(FileExistsError, match="already started"):
        reporting.evaluate(run)
    with (run / "test_results/baseline.npz").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="archive SHA"):
        reporting.verify_results(run)


def test_cannot_seal_missing_models(tmp_path):
    run, _, _, _ = synthetic_run(tmp_path)
    with pytest.raises(FileNotFoundError):
        seal_selection(run)


def test_fit_source_has_no_test_reader_or_evaluation_selection_argument():
    from holdout_experiments import engine
    import inspect
    tree = ast.parse(inspect.getsource(engine._fit))
    file_literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "splits/test.tsv" not in file_literals
    assert "splits/train.tsv" in file_literals and "splits/val.tsv" in file_literals
