"""Validate diagnostic isolation and the frozen-mean residual DDPM without remote models."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from diffusion_code.tests.test_diffusion import local_training, deterministic_cpu, inputs
from diffusion_code.config import DiffusionConfig
from diffusion_code.ddpm import GaussianDDPM
from diffusion_code.mean_guided import MeanCenteredDDPM, MeanGuidedDiffusionVA, load_mean_guided_model
from diffusion_code.model import load_diffusion_model
from diffusion_code.diagnostic_data import fixed_subset
from diffusion_code.probe import train_probe
from va_model_code.decoder_va.model import DecoderVARegressor
from va_model_code.tests.test_model_loss_evaluation import FakeCausalBackbone, FakeGazeProvider


def mean_model():
    """Keep the original direct head on a tiny Raw TRT model."""
    encoder = DecoderVARegressor(FakeCausalBackbone(), gaze_provider=FakeGazeProvider(),
                                 gaze_projection_dim=8, gaze_projection_dropout=(.2, .2), classifier_dropout=.2)
    return MeanGuidedDiffusionVA(encoder, DiffusionConfig(timesteps=8, hidden_dim=16, blocks=1, samples=4),
                                 reference={"held_out_fold": 1, "weights_sha256": "unit-test"})


def test_residual_forward_posterior_and_asymmetric_bounds():
    sampler = MeanCenteredDDPM(DiffusionConfig(timesteps=20))
    labels = torch.tensor([[0., 1.], [1., 0.]])
    mean = 1 - labels
    eps = torch.tensor([[.3, -.7], [.4, .8]])
    time = torch.tensor([8, 8])
    noisy, _, _ = sampler.noise_targets(labels, mean, timesteps=time, noise=eps)
    clean = 2 * (labels - mean)
    abar = sampler.alpha_bar[8]
    torch.testing.assert_close(noisy, abar.sqrt() * clean + (1 - abar).sqrt() * eps)
    posterior = sampler.reverse_step(noisy, eps, 8, torch.zeros_like(eps), mean)
    expected = sampler.posterior_clean_coefficient[8] * clean + sampler.posterior_noisy_coefficient[8] * noisy
    torch.testing.assert_close(posterior, expected)
    zero, _, _ = sampler.noise_targets(labels, mean, timesteps=torch.zeros(2, dtype=torch.long), noise=eps)
    torch.testing.assert_close(sampler.reverse_step(zero, eps, 0, eps * 1000, mean), clean)
    assert clean.abs().max() == 2
    midpoint = torch.full_like(mean, .5)
    legacy = GaussianDDPM(sampler.settings)
    shifted, _, _ = sampler.noise_targets(labels, midpoint, timesteps=time, noise=eps)
    original, _, _ = legacy.noise_targets(labels, timesteps=time, noise=eps)
    torch.testing.assert_close(shifted, original)


def test_residual_oracle_sampler_recovers_targets_and_zero_residual_mean():
    sampler = MeanCenteredDDPM(DiffusionConfig(timesteps=12, samples=4))
    mean = torch.tensor([[.1, .9], [.7, .2]])
    target = torch.tensor([[.9, .05], [.2, .8]])
    for labels in (target, mean):
        residual = 2 * (labels - mean)

        def oracle(noisy, time):
            abar = sampler.alpha_bar[time, None, None]
            return (noisy - abar.sqrt() * residual[:, None]) / (1 - abar).sqrt()

        samples = sampler.sample(oracle, mean=mean)
        torch.testing.assert_close(samples, labels[:, None].expand_as(samples), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("amp", [False, True])
def test_frozen_mean_backward_eval_and_label_isolation(amp):
    model = mean_model().train()
    before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        loss = model(**inputs(), labels=torch.tensor([[.1, .8], [.4, .6], [.9, .2]])).loss
    assert loss.dtype == torch.float32
    loss.backward()
    assert not model.encoder.training
    assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
    assert model.denoiser.mean_projection.weight.grad.abs().sum() > 0
    assert model.denoiser.trt_projection[0].weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    model.eval()
    rng = torch.get_rng_state()
    prediction = model(**inputs()).logits
    other = model(**inputs(), labels=torch.full((3, 2), float("nan"))).logits
    torch.testing.assert_close(prediction, other, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    separate = torch.cat([model(**{k: v[i:i+1] for k, v in inputs().items()}).logits for i in range(3)])
    torch.testing.assert_close(prediction, separate, rtol=2e-5, atol=2e-6)
    assert ((prediction >= 0) & (prediction <= 1)).all()


def test_subset_selection_ignores_labels_and_row_order():
    frame = pd.DataFrame({"index": range(40), "text": [f"text{i}" for i in range(40)],
                          "dataset_of_origin": ["a"] * 20 + ["b"] * 20, "valence": np.linspace(0, 1, 40)})
    args = dict(per_dataset=4, seed=42, fold=1, partition="heldout")
    first = fixed_subset(frame, **args)
    frame.valence = 1 - frame.valence
    second = fixed_subset(frame.sample(frac=1), **args)
    assert first["index"].tolist() == second["index"].tolist()
    assert first.groupby("dataset_of_origin").size().tolist() == [4, 4]


def test_probe_training_never_uses_evaluation_labels_for_updates():
    train, evaluation = torch.randn(12, 6), torch.randn(8, 6)
    labels, gold = torch.rand(12, 2), torch.rand(8, 2)
    options = dict(seed=42, epochs=1, learning_rate=.01, batch_size=4, weight_decay=.01, dropout=.1, device="cpu")
    first, _ = train_probe(train, labels, evaluation, gold, **options)
    second, _ = train_probe(train, labels, evaluation, 1 - gold, **options)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_diagnostics_complete_lifecycle_preserves_source(local_training, monkeypatch):
    from diffusion_code import diagnose
    from diffusion_code.package_diagnostics import package_diagnostics
    train, args, _ = local_training
    source = train.run(args)
    monkeypatch.setattr(diagnose, "resolve_run_directory", lambda name: source)
    original = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    config = diagnose.build_parser().parse_args([
        "--run-name", args.run_name, "--data-dir", args.data_dir, "--use-cpu", "--rows-per-dataset", "2",
        "--samples", "2", "3", "--sampling-seeds", "7", "8", "--memory-trt", "aligned", "zero",
        "--probe-epochs", "2", "--probe-batch-size", "2", "--batch-size", "2",
    ])
    output = diagnose.run(config)
    assert (output / "COMPLETED.json").is_file()
    comparison = pd.read_csv(output / "sampling_comparison.tsv", sep="\t")
    assert len(comparison) == 2 * 2 * 2 * 5
    original_cases = comparison.dropna(subset=["max_abs_difference_from_source_predictions"])
    assert original_cases.max_abs_difference_from_source_predictions.max() < 1e-5
    assert (output / "probe_oof/oof_predictions.tsv").is_file()
    assert original == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    archive = package_diagnostics(output)
    assert archive.is_file()
    (output / "sampling_comparison.tsv").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        package_diagnostics(output)


@pytest.fixture
def raw_training(local_training, monkeypatch):
    """Train a real baseline lifecycle with tiny local components for the new source contract."""
    from diffusion_code import matched_raw
    from va_model_code.decoder_va import model as module
    train, args, tokenizer = local_training
    source = train.run(args)
    root = source.parent
    monkeypatch.setattr(matched_raw, "resolve_run_directory", lambda name: root / name)
    monkeypatch.setattr(train.baseline, "resolve_run_directory", lambda name: root / name)
    monkeypatch.setattr(train.baseline, "load_auto_tokenizer", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(train.baseline, "build_qwen_va_model", module.build_qwen_va_model)
    options = matched_raw.build_parser().parse_args([
        "--reference-run", args.run_name, "--data-dir", args.data_dir, "--run-name", "raw_test",
    ])
    raw = matched_raw.run(options)
    return raw, args, tokenizer


def test_matched_raw_then_frozen_mean_train_reload_and_pair_reports(raw_training, monkeypatch):
    from diffusion_code import train_mean_guided
    from diffusion_code.mean_reference import RawMeanReference
    from diffusion_code.train import baseline
    from diffusion_code.package_results import package_results
    raw, source_args, tokenizer = raw_training
    reference = RawMeanReference(raw, source_args.data_dir)
    monkeypatch.setattr(train_mean_guided, "resolve_run_directory", lambda name: raw.parent / name)
    args = train_mean_guided.build_parser().parse_args([
        "--raw-run", raw.name, "--data-dir", source_args.data_dir, "--run-name", "mean_test",
        "--epochs", "1", "--max-steps", "2", "--diffusion-steps", "3", "--diffusion-width", "16",
        "--diffusion-blocks", "1", "--diffusion-samples", "2",
    ])
    output = train_mean_guided.run(args)
    model, _ = load_mean_guided_model(output / "heldout_fold1/final_model")
    frozen, _ = reference.load(1, dtype=torch.float32, et_cache_size=1)
    for name, value in frozen.state_dict().items():
        torch.testing.assert_close(value, model.encoder.state_dict()[name], rtol=0, atol=0)
    recorded = pd.read_csv(output / "heldout_fold1/predictions.tsv", sep="\t")
    batch = tokenizer(recorded.text.tolist(), padding=True, return_tensors="pt")
    np.testing.assert_allclose(model(**batch).logits.detach().numpy(), recorded[["pred_valence", "pred_arousal"]], rtol=1e-5, atol=1e-6)
    with pytest.raises(FileNotFoundError):
        load_diffusion_model(output / "heldout_fold1/final_model")
    comparison = pd.read_csv(output / "mean_comparison.tsv", sep="\t")
    assert len(comparison) == 6
    state = json.loads((output / "heldout_fold1/trainer_state.json").read_text())
    assert state["global_step"] == 2
    counts = json.loads((output / "heldout_fold1/parameter_counts.json").read_text())
    assert counts["trainable"] == counts["denoiser"] < counts["total"]
    assert package_results(output).is_file()
    marker = output / "heldout_fold1/final_model/mean_guided_architecture.json"
    saved = json.loads(marker.read_text())
    saved["mean_guidance"]["clip_clean_residual"] = "[-1,1]"
    marker.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="contract"):
        load_mean_guided_model(marker.parent)
    with pytest.raises(ValueError):
        package_results(output)


def test_mean_optimizer_contains_only_denoiser_and_frozen_predictor_is_exact(tmp_path):
    from transformers import TrainingArguments
    from diffusion_code.trainer import DiffusionTrainer
    model = mean_model().train()
    trainer = DiffusionTrainer(model=model, head_learning_rate=2e-4,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", learning_rate=6e-6))
    optimizer = trainer.create_optimizer()
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.denoiser.parameters()}
    assert all(group["lr"] == 2e-4 for group in optimizer.param_groups)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model.encoder(**inputs()).logits
        _, mean = model.condition(**inputs())
    torch.testing.assert_close(mean, expected, rtol=0, atol=0)


def test_one_encoder_call_per_sampling_chain():
    model = mean_model().eval()
    calls = []
    handle = model.encoder.backbone.register_forward_hook(lambda *args: calls.append(1))
    model(**inputs())
    handle.remove()
    assert len(calls) == 1


@pytest.mark.parametrize("corruption", ["incomplete", "data", "wrong_fold", "final_mean_error"])
def test_mean_reference_rejects_wrong_source(raw_training, corruption):
    from diffusion_code.mean_reference import RawMeanReference
    raw, args, _ = raw_training
    if corruption == "data":
        path = Path(args.data_dir) / "full_dataset_fold1.csv"
        path.write_text(path.read_text() + "\n")
    elif corruption == "wrong_fold":
        path = raw / "heldout_fold1/run_manifest.json"
        value = json.loads(path.read_text())
        value["held_out_fold"] = 2
        path.write_text(json.dumps(value))
    else:
        path = raw / "heldout_fold1/trainer_state.json"
        value = json.loads(path.read_text())
        if corruption == "incomplete":
            value["global_step"] = 0
        else:
            value["best_metric"] += .1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        RawMeanReference(raw, args.data_dir)


def test_reference_hash_blocks_changed_weights(raw_training):
    from diffusion_code.mean_reference import RawMeanReference
    raw, args, _ = raw_training
    source = RawMeanReference(raw, args.data_dir)
    path = raw / "heldout_fold1/final_model/model.safetensors"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="weights changed"):
        source.load(1, dtype=torch.float32, et_cache_size=1)


def test_diagnostics_sampler_ignores_labels_and_matches_direct_sampling():
    from diffusion_code.tests.test_diffusion import make_model
    from diffusion_code.diagnose import sampling_predictions
    model = make_model().eval()
    first = {**inputs(), "labels": torch.zeros(3, 2)}
    second = {**inputs(), "labels": torch.ones(3, 2)}
    kwargs = dict(cases=[(model.settings.samples, model.settings.sampling_seed)], memory_modes=["aligned"],
                  device=torch.device("cpu"), dtype=torch.float32, amp=False)
    original = model.diffusion.sampling_noise.clone()
    before = torch.get_rng_state()
    result = sampling_predictions(model, [first], **kwargs)
    altered = sampling_predictions(model, [second], **kwargs)
    key = ("aligned", model.settings.samples, model.settings.sampling_seed)
    np.testing.assert_array_equal(result[key][0], altered[key][0])
    np.testing.assert_allclose(result[key][0], model(**inputs()).logits.detach().numpy(), rtol=0, atol=0)
    torch.testing.assert_close(model.diffusion.sampling_noise, original, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())


def test_mean_rejects_unfrozen_source_and_invalid_residual_contract():
    model = mean_model()
    next(model.encoder.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen"):
        model(**inputs())
    mean = torch.ones(2, 2, requires_grad=True)
    sampler = MeanCenteredDDPM(DiffusionConfig(timesteps=3))
    with pytest.raises(ValueError, match="detached"):
        sampler.noise_targets(torch.ones(2, 2), mean)


@pytest.mark.parametrize("amp", [False, True])
def test_native_qwen_frozen_mean_head_backward(amp):
    from va_model_code.tests.test_alignment_training_integration import tiny_native_qwen_backbone
    encoder = DecoderVARegressor(tiny_native_qwen_backbone(torch.float32), gaze_provider=FakeGazeProvider(),
                                 gaze_projection_dim=8, gaze_projection_dropout=(.1, .1), classifier_dropout=.1)
    model = MeanGuidedDiffusionVA(encoder, DiffusionConfig(timesteps=4, samples=2, hidden_dim=16, blocks=1), reference={})
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        loss = model(**inputs(), labels=torch.rand(3, 2)).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.denoiser.mean_projection.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        result = model(**inputs()).logits
    assert result.dtype == torch.float32 and torch.isfinite(result).all()
