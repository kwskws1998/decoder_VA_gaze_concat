"""Two-fold CLI integration tests with production collation and paired rotation math."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import evaluate_gaze_relation as cli
from decoder_va.external_benchmarks import TextBatchCollator, TokenizedTextDataset
from decoder_va.gaze_relation import gaze_relation_contract
from decoder_va.gaze_relation_diagnostics import ARMS, evaluate_paired_batches
from decoder_va.gaze_relation_inputs import shuffle_relation_trt
from decoder_va.readout import readout_contract
from va_model_code.tests.test_gaze_relation_diagnostics import FakeModel, frame


class FixtureTokenizer:
    """Provide deterministic variable-length tokenization with real evaluator collation."""

    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"
    token_rows = {"x": [1], "y": [2, 1], "z": [2, 4, 5, 1], "w": [2, 3, 4, 5]}

    def __call__(self, text, *, max_length, truncation, padding):
        ids = self.token_rows[text][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, rows, *, padding, pad_to_multiple_of, return_tensors):
        length = max(len(row["input_ids"]) for row in rows)
        if pad_to_multiple_of:
            length = ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        return {key: torch.tensor([row[key] + [0] * (length - len(row[key])) for row in rows])
                for key in ("input_ids", "attention_mask")}


class FixtureReadoutModel(FakeModel):
    """Exercise production relation/shuffle with an appended zero-TRT reference query."""

    def __init__(self):
        super().__init__()
        self.gaze_relation = gaze_relation_contract("trt-readout-rotation")
        self.config.gaze_relation = deepcopy(self.gaze_relation)
        self.readout = readout_contract("learned-token")

    def forward(self, input_ids, attention_mask):
        raw, mask = self.gaze_provider.compute(input_ids, attention_mask)
        self.prefixes.append(raw.detach().clone())
        trt = raw[..., 0]
        if self.gaze_relation.get("input_mode") == "shuffled":
            trt = shuffle_relation_trt(trt, input_ids, attention_mask, mask,
                                       seed=self.gaze_relation["shuffle_seed"])
        ids = torch.cat((input_ids.float(), torch.full((len(input_ids), 1), 7., device=input_ids.device)), dim=1)
        trt = torch.cat((trt, torch.zeros(len(input_ids), 1, device=trt.device)), dim=1)
        mask = torch.cat((mask, torch.ones(len(input_ids), 1, dtype=torch.bool, device=mask.device)), dim=1)
        query = torch.stack((ids / 10., ids / 7., ids / 3. + 1., ids / 4.), -1)[:, None].repeat(1, 2, 1, 1)
        correction = self.rotation(query, query.flip(-1), trt, mask)
        score = correction[:, :, -1, :].sum(-1) * .01
        return SimpleNamespace(logits=.5 + .1 * torch.tanh(score))


def create_two_fold_run(tmp_path, monkeypatch, *, schema):
    """Create true source/float32 prediction evidence and replace only heavyweight discovery/loading."""

    source = tmp_path / "data"
    source.mkdir()
    root = tmp_path / "completed_run"
    root.mkdir()
    model_class = FakeModel if schema == 9 else FixtureReadoutModel
    hashes, frames = {}, {}
    for fold in (1, 2):
        current = frame()
        current["index"] += (fold - 1) * 100
        frames[fold] = current
        path = source / f"full_dataset_fold{fold}.csv"
        current.to_csv(path, sep="\t", index=False)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    members = []
    for fold in (1, 2):
        model_dir = root / f"heldout_fold{fold}" / "final_model"
        model_dir.mkdir(parents=True)
        model = model_class()
        tokenizer = FixtureTokenizer()
        loader = DataLoader(TokenizedTextDataset(frames[fold]["text"].tolist(), tokenizer, max_length=20),
                            batch_size=2, collate_fn=TextBatchCollator(tokenizer, pad_to_multiple_of=8))
        predictions, _, _ = evaluate_paired_batches(model, loader, device=torch.device("cpu"))
        persisted = frames[fold].copy()
        persisted[["valence", "arousal"]] = persisted[["valence", "arousal"]].astype(np.float32).astype(np.float64)
        persisted["pred_valence"], persisted["pred_arousal"] = predictions["learned"].T
        persisted.to_csv(model_dir.parent / "predictions.tsv", sep="\t", index=False)
        manifest = {"architecture_manifest_version": schema, "fold_sha256": hashes, "excluded_dataset_names": [],
                    "evaluation_rows": 4, "gaze_relation": model.gaze_relation, "data_dir": str(source),
                    "max_length": 20, "eval_batch_size": 2, "dtype": "float32", "compute_dtype": "float32"}
        if schema == 10:
            manifest["readout"] = model.readout
        provenance = {"fixture_checkpoint": hashlib.sha256(f"fold{fold}:schema{schema}".encode()).hexdigest()}
        members.append(SimpleNamespace(name=f"heldout_fold{fold}", held_out_fold=fold, model_dir=model_dir,
                                       file_sha256=provenance, run_manifest=manifest))
    monkeypatch.setattr(cli, "discover_completed_run", lambda path: tuple(members))
    monkeypatch.setattr(cli, "load_saved_decoder_va_model", lambda *args, **kwargs: (model_class(), FixtureTokenizer()))
    args = cli.build_parser().parse_args(["--run-dir", str(root), "--data-dir", str(source),
                                          "--output-dir", str(tmp_path / "paired_output"), "--use-cpu"])
    return args, members, frames


@pytest.mark.parametrize("schema", [9, 10])
def test_two_fold_cli_publishes_paired_evidence_and_exact_float32_targets(tmp_path, monkeypatch, schema):
    """Legacy/new readout checkpoints produce two-fold reports with stable identities and finite MSE."""

    args, members, original = create_two_fold_run(tmp_path, monkeypatch, schema=schema)
    originals = {member.name: (member.model_dir.parent / "predictions.tsv").read_bytes() for member in members}
    report = cli.run(args)
    output = tmp_path / "paired_output"
    predictions = pd.read_csv(output / "paired_predictions.tsv", sep="\t")
    metrics = pd.read_csv(output / "paired_metrics.tsv", sep="\t")
    assert report["status"] == "completed"
    assert set(report["members"]) == {"heldout_fold1", "heldout_fold2"}
    assert predictions.held_out_fold.tolist() == [1] * 4 + [2] * 4
    assert predictions["index"].tolist() == [10, 11, 12, 13, 110, 111, 112, 113]
    expected = pd.concat(list(original.values()))[["valence", "arousal"]].to_numpy(np.float32).astype(np.float64)
    np.testing.assert_allclose(predictions[["valence", "arousal"]].to_numpy(), expected, rtol=0, atol=1e-15)
    assert np.isfinite(metrics[["mse_mean", "rmse_valence", "rmse_arousal", "mse_delta_vs_learned"]]).all().all()
    assert set(metrics.loc[metrics.grouping == "overall", "arm"]) == set(ARMS)
    assert "text" not in predictions
    stored = json.loads((output / "diagnostics.json").read_text())
    assert stored["source_fold_sha256"] == members[0].run_manifest["fold_sha256"]
    for member in members:
        evidence = stored["members"][member.name]
        assert evidence["saved_file_sha256"] == member.file_sha256
        assert evidence["reload_max_absolute_prediction_error"] < 1e-12
        assert evidence["learned_alpha"]
        assert (member.model_dir.parent / "predictions.tsv").read_bytes() == originals[member.name]
        phase = evidence["phase_diagnostics"]["learned"]
        assert phase["eligible_causal_pairs_excluding_diagonal"] == (9 if schema == 9 else 17)
    singleton = predictions[predictions.anchor_group.astype(str) == "1"]
    assert singleton.shuffled_anchors_changed.eq(0).all()


def test_second_fold_failed_replay_does_not_publish_first_fold(tmp_path, monkeypatch):
    """A second-member reload failure cannot leave an apparently complete one-fold report."""

    args, members, _ = create_two_fold_run(tmp_path, monkeypatch, schema=10)
    second_path = members[1].model_dir.parent / "predictions.tsv"
    corrupt = pd.read_csv(second_path, sep="\t")
    corrupt["pred_valence"] = 0.
    corrupt.to_csv(second_path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="heldout_fold2.*refusing"):
        cli.run(args)
    assert not (tmp_path / "paired_output").exists()


@pytest.mark.parametrize("schema", [9, 10])
def test_two_fold_dry_run_checks_provenance_without_loading_weights(tmp_path, monkeypatch, schema):
    """A useful source/contract preflight works without pretrained assets or model downloads."""

    args, members, _ = create_two_fold_run(tmp_path, monkeypatch, schema=schema)
    args.dry_run = True
    monkeypatch.setattr(cli, "load_saved_decoder_va_model", lambda *a, **kw: pytest.fail("dry run loaded model weights"))
    report = cli.run(args)
    assert set(report["members"]) == {member.name for member in members}
    assert len(report["source_fold_sha256"]) == 2
    assert not (tmp_path / "paired_output").exists()


def test_requested_single_fold_is_explicit_and_does_not_silently_claim_oof(tmp_path, monkeypatch):
    """An explicitly requested subset reports only that held-out member."""

    args, _, _ = create_two_fold_run(tmp_path, monkeypatch, schema=9)
    args.held_out_folds = [2]
    report = cli.run(args)
    assert report["held_out_folds"] == [2]
    assert set(report["members"]) == {"heldout_fold2"}
    output = pd.read_csv(tmp_path / "paired_output" / "paired_predictions.tsv", sep="\t")
    assert output.held_out_fold.eq(2).all()
