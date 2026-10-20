"""Verify paired controls, unchanged prefixes, provenance gates, and immutable output."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from decoder_va.gaze_relation import TRTRelativeRotation, gaze_relation_contract
from decoder_va.gaze_relation_inputs import shuffle_relation_trt
from decoder_va.gaze_relation_diagnostics import (
    ARMS, PhaseAccumulator, describe_batch, evaluate_paired_batches,
    paired_metrics, paired_prediction_frame, relation_arm, validate_heldout_data,
    write_diagnostic_results,
)


class FakeProvider:
    """Generate fixed raw TRT for active, non-special text tokens."""

    feature_indices = (3,)

    def compute(self, ids, attention_mask):
        mask = (ids >= 2) & attention_mask.bool()
        return torch.where(mask, ids.float() / 5., 0.).unsqueeze(-1), mask


class FakeModel(nn.Module):
    """Use production rotation and shuffle controls while retaining a raw-prefix trace."""

    def __init__(self):
        super().__init__()
        self.rotation = TRTRelativeRotation(2)
        self.rotation.alpha.data.copy_(torch.tensor([.1, -.2]))
        self.dropout = nn.Dropout(.5)
        self.gaze_provider = FakeProvider()
        self.gaze_relation = gaze_relation_contract("trt-relative-rotation")
        self.config = SimpleNamespace(gaze_relation=deepcopy(self.gaze_relation))
        self.prefixes = []
        self.fail = False

    def forward(self, input_ids, attention_mask):
        if self.fail:
            raise RuntimeError("deliberate inference failure")
        raw, mask = self.gaze_provider.compute(input_ids, attention_mask)
        self.prefixes.append(raw.detach().clone())
        trt = raw[..., 0]
        if self.gaze_relation.get("input_mode") == "shuffled":
            trt = shuffle_relation_trt(trt, input_ids, attention_mask, mask, seed=self.gaze_relation["shuffle_seed"])
        ids = input_ids.float()
        query = torch.stack((ids / 10., ids / 7., ids / 3. + 1., ids / 4.), -1)[:, None].repeat(1, 2, 1, 1)
        key = query.flip(-1)
        correction = self.rotation(query, key, trt, mask)
        causal = torch.ones_like(input_ids[0, :, None] * input_ids[0, None, :], dtype=torch.bool).tril()
        score = (correction * causal).sum((-1, -2)) * .01
        logits = .5 + .1 * torch.tanh(score)
        return SimpleNamespace(logits=logits)


def batch():
    """Include zero, singleton, and multiple-anchor rows with padding."""

    ids = torch.tensor([[1, 0, 0, 0], [2, 1, 0, 0], [2, 4, 5, 1], [2, 3, 4, 5]])
    return {"input_ids": ids, "attention_mask": (ids != 0).long()}


def frame():
    """Give four stable row identities and both dataset grouping classes."""

    return pd.DataFrame({"index": [10, 11, 12, 13], "text": ["x", "y", "z", "w"],
                         "dataset_of_origin": ["Emobank", "word ratings ENG", "Emobank", "word ratings ENG"],
                         "valence": [.2, .4750, .6075, .8], "arousal": [.4, .5, .6, .7]})


@pytest.mark.parametrize("arm", ARMS)
def test_arm_restores_alpha_contract_modes_gradients_even_on_error(arm):
    """Nested mode states and object identities survive exceptions inside every control."""

    model = FakeModel().train()
    model.dropout.eval()
    contract, config = model.gaze_relation, model.config.gaze_relation
    original = model.rotation.alpha.detach().clone()
    model.rotation.alpha.grad = torch.ones_like(model.rotation.alpha)
    with pytest.raises(RuntimeError, match="intentional"):
        with relation_arm(model, arm):
            assert not torch.is_grad_enabled()
            assert not model.training
            if arm == "zero_alpha":
                assert torch.count_nonzero(model.rotation.alpha) == 0
            if arm == "shuffled_trt":
                assert model.gaze_relation["input_mode"] == "shuffled"
            raise RuntimeError("intentional")
    assert model.gaze_relation is contract
    assert model.config.gaze_relation is config
    assert model.training and not model.dropout.training
    assert model.rotation.alpha.requires_grad
    torch.testing.assert_close(model.rotation.alpha, original, atol=0, rtol=0)
    torch.testing.assert_close(model.rotation.alpha.grad, torch.ones_like(original))


def test_paired_evaluation_has_identical_prefix_and_zero_alpha_reference():
    """Raw-prefix values never change; zero alpha reproduces the same-weights no-correction path."""

    model = FakeModel()
    original = model.rotation.alpha.detach().clone()
    predictions, descriptions, phases = evaluate_paired_batches(model, [batch()], device=torch.device("cpu"))
    assert len(model.prefixes) == 3
    for prefix in model.prefixes[1:]:
        torch.testing.assert_close(prefix, model.prefixes[0], atol=0, rtol=0)
    np.testing.assert_array_equal(predictions["zero_alpha"], .5)
    assert np.any(predictions["learned"] != predictions["zero_alpha"])
    assert descriptions.anchor_count.tolist() == [0, 1, 3, 4]
    assert descriptions.last_original_token_anchored.tolist() == [False, False, False, True]
    assert descriptions.shuffled_anchors_changed.iloc[:2].sum() == 0
    assert phases["learned"]["eligible_causal_pairs_excluding_diagonal"] == 9
    assert phases["zero_alpha"]["nonzero_phase_count"] == 0
    assert phases["zero_alpha"]["absolute_nonzero_phase_radian_quantiles"] == {}
    torch.testing.assert_close(model.rotation.alpha, original, rtol=0, atol=0)
    assert model.rotation.alpha.grad is None


def test_paired_predictions_and_shuffle_are_invariant_to_batch_partition():
    """Row-local shuffling and paired ordered inference survive rebatched inputs."""

    inputs = batch()
    full, descriptions, _ = evaluate_paired_batches(FakeModel(), [inputs], device=torch.device("cpu"))
    pieces = [{key: value[row:row + 1, :int(inputs["attention_mask"][row].sum())] for key, value in inputs.items()}
              for row in range(4)]
    individual, other_descriptions, _ = evaluate_paired_batches(FakeModel(), pieces, device=torch.device("cpu"))
    for arm in ARMS:
        np.testing.assert_allclose(individual[arm], full[arm], atol=1e-7, rtol=0)
    pd.testing.assert_frame_equal(descriptions, other_descriptions)


def test_phase_report_counts_virtual_readout_for_singleton_without_claiming_shuffle_changed():
    """A zero-valued readout adds one causal pair but cannot make singleton shuffling effective."""

    model = FakeModel()
    model.gaze_relation = {**model.gaze_relation, "method": "trt-readout-rotation"}
    accumulator = {arm: PhaseAccumulator([0.] if arm == "zero_alpha" else [.1]) for arm in ARMS}
    descriptions = describe_batch(model, {"input_ids": torch.tensor([[5, 1]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)},
                                  shuffle_seed=42, accumulators=accumulator)
    assert descriptions[0]["anchor_count"] == 1
    assert not descriptions[0]["shuffled_any_changed"]
    report = accumulator["learned"].report()
    assert report["eligible_causal_pairs_excluding_diagonal"] == 1
    assert report["nonzero_phase_count"] == 1
    assert report["maximum_absolute_phase_radians"] == pytest.approx(.1)


def test_phase_quantiles_count_causal_pairs_exactly_and_label_sampling():
    """Actual pair phases exclude diagonal/equal values and approximate quantiles are explicit."""

    accumulator = PhaseAccumulator([0., .5, -.25], sample_pairs=20)
    accumulator.update([0., 2., 2.])
    report = accumulator.report()
    assert report["eligible_causal_pairs_excluding_diagonal"] == 3
    assert report["nonzero_phase_count"] == 4
    assert report["quantiles_exact"]
    assert report["absolute_nonzero_phase_radian_quantiles"]["0.5"] == pytest.approx(.75)
    sampled = PhaseAccumulator([.5], sample_pairs=2)
    sampled.update(np.arange(8.))
    report = sampled.report()
    assert not report["quantiles_exact"]
    assert report["sampled_pairs"] == 2
    assert report["nonzero_phase_count"] == 28
    with pytest.raises(ValueError, match="finite"):
        sampled.update([0., np.nan])


def test_metrics_keep_stable_indices_and_report_shuffle_coverage(tmp_path):
    """Output keeps paired identities and reports conditional strata without saving text."""

    predictions, descriptions, _ = evaluate_paired_batches(FakeModel(), [batch()], device=torch.device("cpu"))
    output = paired_prediction_frame(frame(), predictions, descriptions, held_out_fold=1)
    assert output["index"].tolist() == [10, 11, 12, 13]
    assert "text" not in output
    assert output.dataset_group.tolist() == ["sentence", "word_or_other", "sentence", "word_or_other"]
    metrics = paired_metrics(output)
    assert set(metrics.grouping) >= {"overall", "held_out_fold", "dataset_of_origin", "anchor_group", "fold_dataset"}
    singleton = metrics[(metrics.grouping == "anchor_group") & (metrics.group == "1")]
    assert singleton.shuffle_changed_row_fraction.eq(0).all()
    destination = write_diagnostic_results(tmp_path / "new", output, {"status": "completed"})
    assert sorted(path.name for path in destination.iterdir()) == ["diagnostics.json", "paired_metrics.tsv", "paired_predictions.tsv"]
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(FileExistsError):
        write_diagnostic_results(destination, output, {"status": "changed"})
    assert before == {path.name: path.read_bytes() for path in destination.iterdir()}


def member_fixture(tmp_path):
    """Create source files and a canonical stable held-out evidence table."""

    hashes = {}
    for fold in (1, 2):
        path = tmp_path / f"full_dataset_fold{fold}.csv"
        frame().to_csv(path, sep="\t", index=False)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    model_dir = tmp_path / "heldout_fold1" / "final_model"
    model_dir.mkdir(parents=True)
    saved = frame()
    saved[["valence", "arousal"]] = saved[["valence", "arousal"]].astype(np.float32).astype(np.float64)
    saved.to_csv(model_dir.parent / "predictions.tsv", sep="\t", index=False)
    return SimpleNamespace(name="heldout_fold1", held_out_fold=1, model_dir=model_dir,
                           run_manifest={"fold_sha256": hashes, "excluded_dataset_names": [], "evaluation_rows": 4})


def test_source_hash_and_row_identity_are_required(tmp_path):
    """Reject a changed source file and mismatched persisted metadata before inference."""

    member = member_fixture(tmp_path)
    selected, _ = validate_heldout_data([member], tmp_path)
    assert selected[1]["index"].tolist() == [10, 11, 12, 13]
    predictions = member.model_dir.parent / "predictions.tsv"
    frame().iloc[::-1].to_csv(predictions, sep="\t", index=False)
    with pytest.raises(ValueError, match="metadata order"):
        validate_heldout_data([member], tmp_path)
    saved = frame()
    saved[["valence", "arousal"]] = saved[["valence", "arousal"]].astype(np.float32).astype(np.float64)
    saved.to_csv(predictions, sep="\t", index=False)
    (tmp_path / "full_dataset_fold1.csv").write_text("changed source")
    with pytest.raises(ValueError, match="SHA256"):
        validate_heldout_data([member], tmp_path)


def test_disabled_or_shuffled_checkpoint_contract_is_rejected():
    """No diagnostic silently changes the original trained relation input semantics."""

    model = FakeModel()
    model.gaze_relation = gaze_relation_contract()
    with pytest.raises(ValueError, match="require"):
        with relation_arm(model, "learned"):
            pass
    model.gaze_relation = gaze_relation_contract("trt-relative-rotation", input_mode="shuffled", shuffle_seed=42)
    with pytest.raises(ValueError, match="aligned checkpoint"):
        with relation_arm(model, "learned"):
            pass


def cli_fixture(tmp_path, monkeypatch):
    """Stub only heavyweight loading while running the real CLI data and reporting path."""

    import evaluate_gaze_relation as cli

    member = member_fixture(tmp_path)
    member.run_manifest.update({"gaze_relation": gaze_relation_contract("trt-relative-rotation"),
                                "data_dir": str(tmp_path), "max_length": 20,
                                "eval_batch_size": 4, "dtype": "float32", "compute_dtype": "float32"})
    member.file_sha256 = {"model.safetensors": "fixture-only"}
    predictions, _, _ = evaluate_paired_batches(FakeModel(), [batch()], device=torch.device("cpu"))
    saved = frame()
    saved[["valence", "arousal"]] = saved[["valence", "arousal"]].astype(np.float32).astype(np.float64)
    saved["pred_valence"], saved["pred_arousal"] = predictions["learned"].T
    saved.to_csv(member.model_dir.parent / "predictions.tsv", sep="\t", index=False)
    features = [{key: value[index] for key, value in batch().items()} for index in range(4)]
    monkeypatch.setattr(cli, "discover_completed_run", lambda path: (member,))
    monkeypatch.setattr(cli, "load_saved_decoder_va_model", lambda *args, **kwargs: (FakeModel(), object()))
    monkeypatch.setattr(cli, "TokenizedTextDataset", lambda *args, **kwargs: features)
    monkeypatch.setattr(cli, "TextBatchCollator", lambda *args, **kwargs: lambda rows: {
        key: torch.stack([row[key] for row in rows]) for key in rows[0]})
    args = cli.build_parser().parse_args(["--run-dir", str(tmp_path), "--data-dir", str(tmp_path),
                                          "--output-dir", str(tmp_path / "diagnostics"), "--use-cpu"])
    return cli, args, member


def test_cli_runs_same_weights_and_refuses_overwrite(tmp_path, monkeypatch):
    """Complete reporting requires matching learned predictions and creates no training files."""

    cli, args, member = cli_fixture(tmp_path, monkeypatch)
    source_before = (member.model_dir.parent / "predictions.tsv").read_bytes()
    report = cli.run(args)
    assert report["status"] == "completed"
    assert report["members"][member.name]["reload_max_absolute_prediction_error"] < 1e-12
    assert source_before == (member.model_dir.parent / "predictions.tsv").read_bytes()
    with pytest.raises(FileExistsError):
        cli.run(args)


def test_cli_rejects_reconstruction_contract_mismatch_and_never_publishes(tmp_path, monkeypatch):
    """A stale reconstruction cannot silently be interpreted as the saved experiment."""

    cli, args, _ = cli_fixture(tmp_path, monkeypatch)
    wrong = FakeModel()
    wrong.gaze_relation = gaze_relation_contract("trt-relative-rotation", scale=2.)
    monkeypatch.setattr(cli, "load_saved_decoder_va_model", lambda *a, **kw: (wrong, object()))
    with pytest.raises(ValueError, match="contract disagrees"):
        cli.run(args)
    assert not Path(args.output_dir).exists()


def test_cli_rejects_reload_prediction_mismatch_and_never_publishes(tmp_path, monkeypatch):
    """Incorrect weights or an incompatible inference path fail the saved-prediction gate."""

    cli, args, member = cli_fixture(tmp_path, monkeypatch)
    path = member.model_dir.parent / "predictions.tsv"
    saved = pd.read_csv(path, sep="\t")
    saved["pred_valence"] = 0.
    saved.to_csv(path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="refusing"):
        cli.run(args)
    assert not Path(args.output_dir).exists()


def test_cli_dry_run_does_not_load_models_or_create_outputs(tmp_path, monkeypatch):
    """Preflight validates source identity without downloading models or modifying artifacts."""

    cli, args, _ = cli_fixture(tmp_path, monkeypatch)
    args.dry_run = True
    monkeypatch.setattr(cli, "load_saved_decoder_va_model", lambda *a, **kw: pytest.fail("dry-run loaded weights"))
    report = cli.run(args)
    assert report["source_fold_sha256"]
    assert not Path(args.output_dir).exists()


def test_publication_failure_cleans_staging_without_creating_output(tmp_path):
    """Nonfinite provenance never publishes an apparently complete report directory."""

    predictions, descriptions, _ = evaluate_paired_batches(FakeModel(), [batch()], device=torch.device("cpu"))
    output = paired_prediction_frame(frame(), predictions, descriptions, held_out_fold=1)
    with pytest.raises(ValueError):
        write_diagnostic_results(tmp_path / "failed", output, {"invalid": float("nan")})
    assert list(tmp_path.iterdir()) == []
