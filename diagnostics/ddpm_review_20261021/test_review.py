"""Independent numerical and Trainer checks for the October 21 DDPM review."""

import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import TrainingArguments

from diffusion_code.config import DiffusionConfig
from diffusion_code.ddpm import GaussianDDPM
from diffusion_code.trainer import DiffusionTrainer
from diffusion_code.tests.test_diffusion import make_model, inputs


@pytest.fixture(autouse=True)
def deterministic():
    """Keep review probes reproducible without altering other tests' thread settings."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(810)
    yield
    torch.set_num_threads(previous)


def test_epsilon_objective_is_snr_weighted_va_reconstruction():
    """Check the exact per-timestep identity before any clean-target clipping."""

    ddpm = GaussianDDPM(DiffusionConfig())
    time = torch.arange(100)
    labels = .1 + .8 * torch.rand(100, 2)
    noisy, _, noise = ddpm.noise_targets(labels, timesteps=time)
    prediction = noise + torch.randn_like(noise) * .03
    a = ddpm.alpha_bar[time, None]
    estimated_va = ((noisy - (1 - a).sqrt() * prediction) / a.sqrt() + 1) / 2
    torch.testing.assert_close(
        (prediction - noise).square(), 4 * a / (1 - a) * (estimated_va - labels).square(),
        rtol=3e-3, atol=1e-7,
    )


@pytest.mark.parametrize("nsteps", [100, 250])
def test_perfect_conditional_noise_recovers_clean_va_end_to_end(nsteps):
    """Run every reverse step using a known synthetic point-mass target."""

    ddpm = GaussianDDPM(DiffusionConfig(timesteps=nsteps))
    clean = torch.tensor([[[-.7, .8]], [[.6, -.2]]])

    def oracle(noisy, time):
        """Compute true epsilon from deliberately known clean synthetic states."""

        a = ddpm.alpha_bar[time, None, None]
        return (noisy - a.sqrt() * clean) / (1 - a).sqrt()

    samples = ddpm.sample(oracle, batch_size=2)
    torch.testing.assert_close(samples, ((clean + 1) / 2).expand_as(samples), atol=1e-6, rtol=0)


def test_real_trainer_accumulation_matches_one_full_batch(tmp_path):
    """Use the actual Trainer backward path with identical per-row corruption."""

    full = make_model()
    accumulated = copy.deepcopy(full)
    batch = {key: value[1:] for key, value in inputs().items()}
    batch["labels"] = torch.tensor([[.2, .7], [.8, .3]])

    def fix_corruption(model):
        """Preserve the production noising equation while eliminating random-draw differences."""

        original = model.diffusion.noise_targets

        def noise_targets(labels):
            """Choose identical t and epsilon for each row in either batching arrangement."""

            return original(
                labels, timesteps=torch.full((len(labels),), 4, dtype=torch.long),
                noise=labels * 2 - .1,
            )

        model.diffusion.noise_targets = noise_targets

    fix_corruption(full)
    fix_corruption(accumulated)
    full(**batch).loss.backward()
    trainer = DiffusionTrainer(
        model=accumulated,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to="none",
            gradient_accumulation_steps=2,
        ),
    )
    trainer.create_optimizer()
    trainer.current_gradient_accumulation_steps = 2
    for i in range(2):
        trainer.training_step(accumulated, {key: value[i:i+1] for key, value in batch.items()})
    assert trainer.model_accepts_loss_kwargs is False
    for (name, left), (_, right) in zip(full.named_parameters(), accumulated.named_parameters()):
        if left.grad is None:
            assert right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=5e-5, msg=name)
    norm = trainer._clip_grad_norm(accumulated)
    after = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(p.grad) for p in accumulated.parameters() if p.grad is not None
    ]))
    assert norm >= after
    assert after <= 1.00001


def test_scheduler_keeps_explicit_head_encoder_lr_ratio(tmp_path):
    """Verify that warmup and linear decay preserve the configured separate learning rates."""

    model = make_model()
    trainer = DiffusionTrainer(
        model=model, head_learning_rate=1e-4,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to="none",
            learning_rate=6e-6, warmup_steps=10,
        ),
    )
    trainer.create_optimizer_and_scheduler(100)
    head = {id(p) for p in model.denoiser.parameters()}
    for step in range(100):
        for group in trainer.optimizer.param_groups:
            expected_base = 1e-4 if id(group["params"][0]) in head else 6e-6
            multiplier = step / 10 if step < 10 else (100 - step) / 90
            assert group["lr"] == pytest.approx(expected_base * multiplier)
        trainer.optimizer.step()
        trainer.lr_scheduler.step()


def test_ideal_point_predictions_do_not_inherit_seed_bias():
    """Separate finite Monte Carlo error from an unavoidable error in the sampler."""

    target = torch.tensor([[[.4, -.6]]])
    predictions = []
    for seed in (2026, 2027):
        ddpm = GaussianDDPM(DiffusionConfig(sampling_seed=seed))

        def oracle(noisy, time):
            """Provide exact epsilon for the synthetic deterministic target."""

            a = ddpm.alpha_bar[time, None, None]
            return (noisy - a.sqrt() * target) / (1 - a).sqrt()

        predictions.append(ddpm.sample(oracle, batch_size=1).mean(1))
    torch.testing.assert_close(predictions[0], predictions[1], atol=1e-7, rtol=0)
