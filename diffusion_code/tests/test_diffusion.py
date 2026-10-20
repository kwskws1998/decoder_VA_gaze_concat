"""Check DDPM mathematics, conditioning, label isolation and training/reload lifecycle."""

import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, TrainingArguments

from diffusion_code.config import DiffusionConfig
from diffusion_code.ddpm import GaussianDDPM
from diffusion_code.model import ConditionalDenoiser, DiffusionVA, load_diffusion_model
from diffusion_code.package_results import package_results
from diffusion_code.trainer import DiffusionTrainer
from va_model_code.decoder_va.model import DecoderVARegressor, EncodedVAFeatures
from va_model_code.tests.test_model_loss_evaluation import FakeCausalBackbone, FakeGazeProvider


@pytest.fixture(autouse=True)
def deterministic_cpu():
    """Keep small CPU probes reproducible and avoid excessive BLAS thread startup."""

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(913)
    yield
    torch.set_num_threads(threads)


def make_model(conditioning="token-trt", native=False):
    """Construct a tiny causal encoder without any remote weights."""

    if native:
        from va_model_code.tests.test_alignment_training_integration import tiny_native_qwen_backbone
        backbone = tiny_native_qwen_backbone(torch.float32)
    else:
        backbone = FakeCausalBackbone()
    encoder = DecoderVARegressor(
        backbone, gaze_provider=FakeGazeProvider(), gaze_projection_dim=8,
        gaze_projection_dropout=(0, 0), classifier_dropout=0,
    )
    return DiffusionVA(encoder, DiffusionConfig(
        timesteps=8, samples=4, hidden_dim=16, blocks=2, conditioning=conditioning,
    ))


def inputs():
    """Include no-anchor, single-anchor, multi-anchor, punctuation/padding-like cases."""

    return {
        "input_ids": torch.tensor([[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]),
    }


@pytest.mark.parametrize("change", [
    {"timesteps": 1}, {"hidden_dim": 7}, {"hidden_dim": 9}, {"blocks": 0},
    {"samples": 0}, {"sampling_seed": -1}, {"version": True},
    {"objective": "mse"}, {"conditioning": "unknown"}, {"clip_clean": 1},
])
def test_bad_configuration_fails(change):
    with pytest.raises(ValueError):
        DiffusionConfig(**change)


def test_config_requires_complete_saved_semantics():
    saved = DiffusionConfig().to_dict()
    assert DiffusionConfig.from_dict(saved).to_dict() == saved
    del saved["sampling_seed"]
    with pytest.raises(ValueError):
        DiffusionConfig.from_dict(saved)


def test_forward_schedule_and_true_posterior_independently():
    ddpm = GaussianDDPM(DiffusionConfig(timesteps=20, clip_clean=False))
    assert torch.all(ddpm.alpha_bar[1:] < ddpm.alpha_bar[:-1])
    assert ddpm.alpha_bar[-1] < 1e-5
    labels = torch.tensor([[0.1, 0.8], [0.9, 0.3]])
    noise = torch.tensor([[0.4, -0.7], [-0.3, 1.1]])
    time = torch.tensor([4, 4])
    noisy, _, _ = ddpm.noise_targets(labels, timesteps=time, noise=noise)
    clean = 2 * labels - 1
    abar = ddpm.alpha_bar[4]
    previous = ddpm.alpha_bar[3]
    beta = 1 - abar / previous
    expected_noisy = abar.sqrt() * clean + (1 - abar).sqrt() * noise
    torch.testing.assert_close(noisy, expected_noisy)
    expected_mean = beta * previous.sqrt() / (1 - abar) * clean
    expected_mean += (1 - previous) * (1 - beta).sqrt() / (1 - abar) * noisy
    expected_var = beta * (1 - previous) / (1 - abar)
    torch.testing.assert_close(
        ddpm.reverse_step(noisy, noise, 4, torch.ones_like(noise)),
        expected_mean + expected_var.sqrt(), rtol=1e-5, atol=1e-6,
    )
    noisy0, _, noise0 = ddpm.noise_targets(labels, timesteps=torch.zeros(2, dtype=torch.long), noise=noise)
    torch.testing.assert_close(ddpm.reverse_step(noisy0, noise0, 0, noise * 1000), clean)
    assert ddpm.posterior_variance[0] == 0


def test_invalid_labels_and_final_clipping():
    ddpm = GaussianDDPM(DiffusionConfig(timesteps=3))
    for labels in (torch.tensor([[float("nan"), .5]]), torch.tensor([[1.01, .5]])):
        with pytest.raises(ValueError):
            ddpm.noise_targets(labels)
    final = ddpm.reverse_step(torch.tensor([[100., -100.]]), torch.zeros(1, 2), 0, torch.zeros(1, 2))
    torch.testing.assert_close(final, torch.tensor([[1., -1.]]))


def test_aligned_features_match_original_text_positions():
    model = make_model()
    encoder = model.encoder.eval()
    batch = inputs()
    encoded = encoder.encode(**batch, return_aligned_tokens=True)
    hidden = encoder.backbone.last_call["last_hidden_state"]
    for row, length in enumerate((1, 2, 4)):
        offset = length - 1 + 2
        torch.testing.assert_close(encoded.text_hidden[row, :length], hidden[row, offset:offset + length])
        torch.testing.assert_close(encoded.pooled[row], hidden[row, offset + length - 1])
        assert not encoded.text_hidden[row, length:].count_nonzero()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_training_backward_and_gradient_checkpoint_replay(native, bf16):
    model = make_model(native=native).train()
    if native:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        loss = model(**inputs(), labels=torch.tensor([[.2, .8], [.5, .4], [.8, .1]])).loss
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    for module in (model.denoiser, model.encoder.backbone, model.encoder.gaze_projector):
        gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    assert model.denoiser.trt_projection[0].weight.grad.abs().sum() > 0
    assert not any(name.startswith("encoder.regression_head.") for name in model.state_dict())


@pytest.mark.parametrize("conditioning", ["token-trt", "pooled"])
def test_eval_ignores_labels_preserves_rng_batching_and_order(conditioning):
    model = make_model(conditioning).eval()
    batch = inputs()
    before = torch.get_rng_state()
    with torch.no_grad():
        first = model(**batch).logits
        changed_labels = model(**batch, labels=torch.full((3, 2), 1000.)).logits
        individual = torch.cat([
            model(**{k: v[i:i+1] for k, v in batch.items()}).logits for i in range(3)
        ])
        reversed_rows = model(**{k: v.flip(0) for k, v in batch.items()}).logits.flip(0)
    assert torch.equal(torch.get_rng_state(), before)
    torch.testing.assert_close(first, changed_labels, rtol=0, atol=0)
    torch.testing.assert_close(first, individual, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(first, reversed_rows, rtol=2e-5, atol=2e-6)
    assert first.shape == (3, 2) and first.dtype == torch.float32
    assert ((first >= 0) & (first <= 1)).all()


def test_qwen_and_et_called_once_for_entire_reverse_chain():
    model = make_model().eval()
    calls = {"qwen": 0, "et": 0}
    original = model.encoder.gaze_provider.compute

    def compute(*args):
        calls["et"] += 1
        return original(*args)

    def hook(*args):
        calls["qwen"] += 1

    model.encoder.gaze_provider.compute = compute
    handle = model.encoder.backbone.register_forward_hook(hook)
    model(**inputs())
    handle.remove()
    assert calls == {"qwen": 1, "et": 1}


def test_single_anchor_trt_affects_value_and_has_gradient():
    head = ConditionalDenoiser(6, DiffusionConfig(hidden_dim=16, blocks=1))
    trt = torch.tensor([[[.2], [999.]]], requires_grad=True)
    features = EncodedVAFeatures(
        pooled=torch.zeros(1, 6), text_hidden=torch.ones(1, 2, 6),
        raw_gaze=trt, gaze_mask=torch.tensor([[True, False]]),
    )
    noisy, time = torch.zeros(1, 1, 2), torch.tensor([5])
    first = head(noisy, time, head.prepare(features))
    first.sum().backward()
    assert trt.grad[0, 0].abs().sum() > 0
    assert trt.grad[0, 1].abs().sum() == 0
    features.raw_gaze = torch.tensor([[[1.2], [float("nan")]]])
    second = head(noisy, time, head.prepare(features))
    assert torch.isfinite(second).all()
    assert not torch.allclose(first, second)
    features.raw_gaze = torch.tensor([[[.2], [float("nan")]]])
    torch.testing.assert_close(first, head(noisy, time, head.prepare(features)))


def test_denoiser_can_fit_planted_conditional_epsilon_targets():
    settings = DiffusionConfig(hidden_dim=16, blocks=2, conditioning="pooled", timesteps=10)
    head = ConditionalDenoiser(6, settings)
    features = EncodedVAFeatures(pooled=torch.randn(32, 6))
    noisy = torch.randn(32, 1, 2)
    time = torch.arange(32) % 10
    target = noisy * .5 + features.pooled[:, None, :2] * .3
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)
    losses = []
    for step in range(160):
        optimizer.zero_grad()
        loss = (head(noisy, time, head.prepare(features)) - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * .04


def test_optimizer_covers_every_weight_once_with_separate_lr(tmp_path):
    model = make_model()
    trainer = DiffusionTrainer(
        model=model, args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", learning_rate=6e-6),
        head_learning_rate=1e-4,
    )
    optimizer = trainer.create_optimizer()
    head = {id(p) for p in model.denoiser.parameters()}
    ids = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            ids.append(id(p))
            assert group["lr"] == (1e-4 if id(p) in head else 6e-6)
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}


@pytest.fixture
def local_training(monkeypatch, tmp_path):
    """Patch only downloads/model size; retain the real Trainer and reporting lifecycle."""

    import diffusion_code.train as train
    import va_model_code.decoder_va.model as encoder_module

    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "one": 3, "two": 4, "three": 5, "four": 6}
    core = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
    )

    class Provider(FakeGazeProvider):
        def __init__(self, tokenizer=None, repo_id="fake", revision="fake", filename="fake", feature_indices=(3,), cache_size=0):
            super().__init__(feature_indices)
            self.repo_id, self.revision, self.filename = repo_id, revision, filename

    monkeypatch.setattr(encoder_module, "load_qwen_backbone", lambda *a, **k: FakeCausalBackbone())
    monkeypatch.setattr(encoder_module, "ET2GazeProvider", Provider)
    monkeypatch.setattr(train, "load_auto_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(train, "resolve_run_directory", lambda name: tmp_path / name)
    data = tmp_path / "data"
    data.mkdir()
    for fold in (1, 2):
        pd.DataFrame({
            "index": [fold * 10 + i for i in range(4)], "text": ["one", "one two", "three four", "two three"],
            "dataset_of_origin": ["Emobank"] * 4, "valence": [.1, .3, .6, .9],
            "arousal": [.8, .2, .7, .4],
        }).to_csv(data / f"full_dataset_fold{fold}.csv", sep="\t", index=False)
    args = train.build_parser().parse_args([
        "--data-dir", str(data), "--run-name", "ddpm_test", "--use-cpu", "--precision", "fp32",
        "--train-batch-size", "2", "--eval-batch-size", "2", "--gradient-accumulation-steps", "1",
        "--epochs", "1", "--max-steps", "2", "--logging-steps", "1",
        "--diffusion-steps", "3", "--diffusion-width", "16", "--diffusion-blocks", "1",
        "--diffusion-samples", "2", "--no-gradient-checkpointing",
    ])
    return train, args, tokenizer


def test_full_two_fold_lifecycle_reload_and_results_only_zip(local_training, monkeypatch):
    train, args, tokenizer = local_training
    output = train.run(args)
    archive = package_results(output)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert not any(name.endswith((".safetensors", ".bin", ".pt", ".pth")) for name in zipped.namelist())
        assert sum(name.endswith("/predictions.tsv") for name in zipped.namelist()) == 2
    final = output / "heldout_fold1/final_model"
    model, _ = load_diffusion_model(final)
    frame = pd.read_csv(output / "heldout_fold1/predictions.tsv", sep="\t")
    tokenized = tokenizer(frame.text.tolist(), return_tensors="pt", padding=True)
    predictions = model(**tokenized).logits.detach().numpy()
    np.testing.assert_allclose(predictions, frame[["pred_valence", "pred_arousal"]], rtol=2e-5, atol=2e-6)
    assert not (final / "decoder_va_architecture.json").exists()
    state = json.loads((output / "heldout_fold1/trainer_state.json").read_text())
    assert state["best_metric"] is not None
    assert any("eval_mse_mean" in item for item in state["log_history"])
    from datetime import datetime
    import diffusion_code.package_results as package_module

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 10, 20)

    monkeypatch.setattr(package_module, "datetime", FixedDateTime)
    fixed_archive = package_results(output)
    original_zip = fixed_archive.read_bytes()
    with pytest.raises(FileExistsError):
        package_results(output)
    assert fixed_archive.read_bytes() == original_zip
    wrong = output / "COMPLETED.json"
    wrong.unlink()
    with pytest.raises(ValueError, match="missing"):
        package_results(output)


def test_dry_run_and_invalid_text_only_condition(local_training):
    train, args, _ = local_training
    args.dry_run = True
    assert train.run(args) is None
    assert not (Path(args.data_dir).parent / args.run_name).exists()
    args.gaze_fusion = "none"
    with pytest.raises(ValueError, match="token-trt"):
        train.run(args)


def test_resume_one_fold_preserves_other_fold_and_sampling_contract(local_training):
    train, args, _ = local_training
    args.max_steps, args.epochs, args.save_total_limit = 4, 2, 2
    output = train.run(args)
    original = pd.read_csv(output / "oof_predictions.tsv", sep="\t")
    other = (output / "heldout_fold2/predictions.tsv").read_bytes()
    resumed = copy.deepcopy(args)
    resumed.held_out_folds = [1]
    resumed.resume_from_checkpoint = str(output / "heldout_fold1/checkpoints/checkpoint-2")
    assert Path(resumed.resume_from_checkpoint).is_dir()
    train.run(resumed)
    assert (output / "heldout_fold2/predictions.tsv").read_bytes() == other
    actual = pd.read_csv(output / "oof_predictions.tsv", sep="\t")
    np.testing.assert_allclose(
        actual[["pred_valence", "pred_arousal"]], original[["pred_valence", "pred_arousal"]],
        rtol=1e-5, atol=1e-6,
    )
    assert json.loads((output / "COMPLETED.json").read_text())["held_out_folds"] == [1, 2]
    assert package_results(output).is_file()
    resumed.sampling_seed += 1
    with pytest.raises(ValueError, match="configuration/source"):
        train.run(resumed)


@pytest.mark.parametrize("corruption", ["metrics", "prediction", "incomplete", "contract"])
def test_packaging_rejects_inconsistent_or_partial_runs(local_training, corruption):
    train, args, _ = local_training
    args.held_out_folds = [1]
    output = train.run(args)
    fold = output / "heldout_fold1"
    if corruption == "metrics":
        path = fold / "metrics.json"
        value = json.loads(path.read_text())
        value["test_mse_mean"] += .1
        path.write_text(json.dumps(value))
    elif corruption == "prediction":
        with (fold / "predictions.tsv").open("a") as stream:
            stream.write("\n")
    elif corruption == "incomplete":
        path = fold / "trainer_state.json"
        value = json.loads(path.read_text())
        value["max_steps"] += 1
        path.write_text(json.dumps(value))
    else:
        path = fold / "final_model/diffusion_training_contract.json"
        value = json.loads(path.read_text())
        value["diffusion"]["sampling_seed"] += 1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        package_results(output)
    assert not list(output.glob("*.zip"))


def test_all_noise_bank_reconstruction_and_rng_independence():
    before = torch.get_rng_state()
    config = DiffusionConfig(timesteps=3, samples=4)
    first = GaussianDDPM(config)
    assert torch.equal(before, torch.get_rng_state())
    second = GaussianDDPM(DiffusionConfig.from_dict(config.to_dict()))
    assert torch.equal(first.sampling_noise, second.sampling_noise)
    other = GaussianDDPM(DiffusionConfig(timesteps=3, samples=4, sampling_seed=0))
    assert not torch.equal(first.sampling_noise, other.sampling_noise)


def test_text_only_pooled_training_and_inference_work():
    encoder = DecoderVARegressor(FakeCausalBackbone(), gaze_fusion="none", classifier_dropout=0)
    model = DiffusionVA(encoder, DiffusionConfig(
        timesteps=3, hidden_dim=16, samples=2, conditioning="pooled",
    ))
    loss = model(**inputs(), labels=torch.full((3, 2), .5)).loss
    loss.backward()
    assert torch.isfinite(model.eval()(**inputs()).logits).all()
