"""Run an offline synthetic trainability diagnostic, not a pretrained VA experiment.

All token sequences, TRT values, and two-dimensional targets are synthetic. A
randomly initialized tiny native Qwen3.5 with planted relation coefficients
generates the targets. No ET2 inference, pretrained weights, or downloads are
used. The held-out synthetic inputs are evaluated at the fixed final step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel, TrainingArguments

from decoder_va.gaze_relation import gaze_relation_contract
from decoder_va.model import DecoderVARegressor
from decoder_va.trainer import GAZE_RELATION_PARAMETER_SUFFIX, VARegressionTrainer


ARMS = ("raw", "relation", "relation_shuffled")
PLANTED_ALPHA = (0.35, -0.25, 0.20, -0.30)


class SyntheticTRTProvider:
    """Supply deterministic invented TRT features; this class performs no ET inference."""

    feature_indices = (3,)

    def compute(self, input_ids, attention_mask):
        position = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        trt = 0.3 + ((17 * input_ids + 7 * position.square()) % 29).float() / 8
        mask = attention_mask.bool()
        return trt.masked_fill(~mask, 0).unsqueeze(-1), mask


def tiny_qwen_config() -> Qwen3_5TextConfig:
    """Define a native hybrid linear/full-attention model without consulting the Hub."""

    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        attention_dropout=0.0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
    )
    config._attn_implementation = "eager"
    return config


def build_model(
    seed: int, *, relation: bool, input_mode: str = "aligned", shuffle_seed: int | None = None,
) -> DecoderVARegressor:
    """Construct matched FP32 models with all stochastic dropout disabled."""

    torch.manual_seed(seed)
    return DecoderVARegressor(
        Qwen3_5TextModel(tiny_qwen_config()),
        gaze_provider=SyntheticTRTProvider(),
        gaze_fusion="prefix-concat",
        gaze_relation=gaze_relation_contract(
            "trt-relative-rotation" if relation else "none",
            input_mode=input_mode, shuffle_seed=shuffle_seed,
        ),
        classifier_dropout=0,
        gaze_projection_dropout=(0, 0),
        gaze_projection_dim=8,
    )


def relation_parameters(model):
    """Identify trainable relation coefficients independently of model layer numbering."""

    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith(GAZE_RELATION_PARAMETER_SUFFIX)
    }


def load_common_state(model, state):
    """Copy every common weight and permit only newly introduced zero relation coefficients."""

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(
        not name.endswith(GAZE_RELATION_PARAMETER_SUFFIX) for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"Unexpected model mismatch: {incompatible}")
    for name, tensor in state.items():
        if not torch.equal(model.state_dict()[name], tensor):
            raise RuntimeError(f"Common initial weight differs: {name}")


def synthetic_inputs(seed: int, count: int):
    """Generate independent padded token sequences of length four through ten."""

    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(4, 11, (count,), generator=generator)
    tokens = torch.randint(3, 64, (count, 10), generator=generator)
    mask = torch.arange(10)[None, :] < lengths[:, None]
    return {"input_ids": tokens.masked_fill(~mask, 0), "attention_mask": mask.long()}


def tensor_digest(tensors) -> str:
    """Record deterministic tensor content hashes for paired data and initialization checks."""

    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict(model, inputs, *, batch_size: int):
    """Evaluate ordered synthetic rows without labels, selection, or parameter updates."""

    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(inputs["input_ids"]), batch_size):
            batch = {name: values[start : start + batch_size] for name, values in inputs.items()}
            pieces.append(model(**batch).logits.detach())
    predictions = torch.cat(pieces)
    if not bool(torch.isfinite(predictions).all()):
        raise RuntimeError("Nonfinite synthetic predictions.")
    return predictions


def run_seed(seed: int, args):
    """Run paired controls from identical weights and a prespecified final training step."""

    started = time.perf_counter()
    raw = build_model(seed, relation=False)
    initial_state = {name: tensor.detach().clone() for name, tensor in raw.state_dict().items()}
    models = {"raw": raw}
    for arm in ARMS[1:]:
        models[arm] = build_model(
            seed, relation=True,
            input_mode="shuffled" if arm == "relation_shuffled" else "aligned",
            shuffle_seed=seed + 4000 if arm == "relation_shuffled" else None,
        )
        load_common_state(models[arm], initial_state)
    teacher = build_model(seed, relation=True)
    load_common_state(teacher, initial_state)
    with torch.no_grad():
        for parameter in relation_parameters(teacher).values():
            parameter.copy_(torch.tensor(PLANTED_ALPHA))
    teacher.requires_grad_(False)
    train_inputs = synthetic_inputs(seed + 1000, args.train_examples)
    heldout_inputs = synthetic_inputs(seed + 2000, args.heldout_examples)
    train_sequences = {tuple(row.tolist()) for row in train_inputs["input_ids"]}
    heldout_sequences = {tuple(row.tolist()) for row in heldout_inputs["input_ids"]}
    if train_sequences.intersection(heldout_sequences):
        raise RuntimeError("Synthetic held-out input duplicates a training input.")
    train_labels = predict(teacher, train_inputs, batch_size=args.batch_size)
    heldout_labels = predict(teacher, heldout_inputs, batch_size=args.batch_size)
    del teacher
    initial_train = predict(raw, train_inputs, batch_size=args.batch_size)
    initial_heldout = predict(raw, heldout_inputs, batch_size=args.batch_size)
    initialization_differences = {}
    for arm in ARMS[1:]:
        arm_initial = predict(models[arm], heldout_inputs, batch_size=args.batch_size)
        difference = float((arm_initial - initial_heldout).abs().max())
        initialization_differences[arm] = difference
        if not torch.equal(arm_initial, initial_heldout):
            raise RuntimeError(f"Zero-alpha initial predictions differ for {arm}: {difference}")
    generator = torch.Generator().manual_seed(seed + 3000)
    batch_indices = torch.randint(
        args.train_examples, (args.steps, args.batch_size), generator=generator
    )
    seed_result = {
        "seed": seed,
        "initial_common_weights_sha256": tensor_digest(initial_state),
        "training_inputs_sha256": tensor_digest(train_inputs),
        "heldout_inputs_sha256": tensor_digest(heldout_inputs),
        "batch_order_sha256": tensor_digest({"indices": batch_indices}),
        "heldout_training_token_overlap": 0,
        "initial_prediction_absmax_difference": initialization_differences,
        "initial_training_mse": float((initial_train - train_labels).square().mean()),
        "initial_heldout_mse": float((initial_heldout - heldout_labels).square().mean()),
        "synthetic_target_mean": heldout_labels.mean(dim=0).tolist(),
        "synthetic_target_std": heldout_labels.std(dim=0).tolist(),
        "arms": {},
    }
    prediction_rows = []
    for arm, model in models.items():
        arm_started = time.perf_counter()
        model.requires_grad_(False)
        model.regression_head.requires_grad_(True)
        alphas = relation_parameters(model)
        for parameter in alphas.values():
            parameter.requires_grad_(True)
        trainer = VARegressionTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(args.output_dir / f"seed_{seed}" / arm),
                learning_rate=args.head_lr,
                weight_decay=0,
                report_to="none",
                use_cpu=True,
                disable_tqdm=True,
            ),
            loss_name="mse",
            gaze_relation_learning_rate=args.alpha_lr if alphas else None,
        )
        optimizer = trainer.create_optimizer()
        observed_gradient = {name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in alphas.items()}
        maximum_alpha_gradient = 0.0
        losses = []
        model.train()
        for indices in batch_indices:
            batch = {name: values[indices] for name, values in train_inputs.items()}
            batch["labels"] = train_labels[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = trainer.compute_loss(model, batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite loss for seed {seed}, arm {arm}.")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"Nonfinite gradient for {name}.")
            for name, parameter in alphas.items():
                if parameter.grad is None:
                    raise RuntimeError(f"Relation coefficient has no gradient: {name}")
                observed_gradient[name] |= parameter.grad != 0
                maximum_alpha_gradient = max(maximum_alpha_gradient, float(parameter.grad.abs().max()))
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise RuntimeError(f"Nonfinite updated parameter for seed {seed}, arm {arm}.")
            losses.append(float(loss.detach()))
        final_train = predict(model, train_inputs, batch_size=args.batch_size)
        final_heldout = predict(model, heldout_inputs, batch_size=args.batch_size)
        alpha_values = {name: parameter.detach().tolist() for name, parameter in alphas.items()}
        seed_result["arms"][arm] = {
            "gaze_relation": model.gaze_relation,
            "final_training_mse": float((final_train - train_labels).square().mean()),
            "final_heldout_mse": float((final_heldout - heldout_labels).square().mean()),
            "final_step": args.steps,
            "loss_history": losses,
            "all_losses_gradients_parameters_finite": True,
            "all_alpha_heads_received_nonzero_gradient": (
                all(bool(observed.all()) for observed in observed_gradient.values()) if alphas else None
            ),
            "maximum_absolute_alpha_gradient": maximum_alpha_gradient if alphas else None,
            "initial_alpha": {name: [0.0] * parameter.numel() for name, parameter in alphas.items()},
            "final_alpha": alpha_values,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "runtime_seconds": time.perf_counter() - arm_started,
        }
        for row, (target, estimate) in enumerate(zip(heldout_labels.tolist(), final_heldout.tolist())):
            prediction_rows.append(
                {"seed": seed, "arm": arm, "synthetic_row": row,
                 "target_0": target[0], "target_1": target[1],
                 "prediction_0": estimate[0], "prediction_1": estimate[1]}
            )
        del trainer, optimizer
    seed_result["runtime_seconds"] = time.perf_counter() - started
    return seed_result, prediction_rows


def parse_args():
    """Parse bounded diagnostic settings and reject accidental result replacement."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "gaze_relation_diagnostic_20261014")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--train-examples", type=int, default=128)
    parser.add_argument("--heldout-examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=5e-2)
    args = parser.parse_args()
    for name in ("steps", "train_examples", "heldout_examples", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    for name in ("head_lr", "alpha_lr"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct.")
    if (args.output_dir / "diagnostic.json").exists():
        parser.error("This output directory already contains diagnostic.json; choose another directory.")
    return args


def main():
    """Persist reproducibility details and paired synthetic results without efficacy claims."""

    args = parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, predictions = [], []
    for seed in args.seeds:
        result, rows = run_seed(seed, args)
        results.append(result)
        predictions.extend(rows)
        print(json.dumps({"seed": seed, "heldout_mse": {arm: result["arms"][arm]["final_heldout_mse"] for arm in ARMS}}, sort_keys=True), flush=True)
    report = {
        "evidence_type": "synthetic_planted_teacher_trainability_diagnostic",
        "pretrained_weights_used": False,
        "et2_inference_used": False,
        "downloads_used": False,
        "human_va_labels_used": False,
        "supports_va_efficacy_or_novelty_claim": False,
        "limitations": [
            "The teacher explicitly contains the tested relation, so this positive control is favorable by construction.",
            "Tokens, TRT, and target vectors are synthetic, not an emotion dataset or measured gaze.",
            "The common backbone and gaze projector are frozen; only the common head and optional alpha are trained.",
            "The diagnostic learning rates and tiny randomly initialized model are not the production training configuration.",
            "A successful planted-target fit does not establish useful VA generalization, cognitive validity, or novelty.",
        ],
        "environment": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "platform": platform.platform(), "device": "cpu", "dtype": "float32", "torch_num_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "config": {
            **{key: value for key, value in vars(args).items() if key != "output_dir"},
            "model": tiny_qwen_config().to_dict(),
            "gaze_relation": gaze_relation_contract("trt-relative-rotation"),
            "planted_teacher_alpha": PLANTED_ALPHA,
            "selection_policy": "fixed_final_step_no_checkpoint_or_hyperparameter_selection",
            "batch_sampling": "with_replacement_identical_index_schedule_for_all_arms",
            "shuffle_policy": "production_blake2b_rank_v1_content_keyed_relation_only_raw_prefix_unchanged",
            "shuffle_seed_policy": "experiment_seed_plus_4000",
            "optimizer": "AdamW_constant_learning_rates_zero_weight_decay",
        },
        "results": results,
        "mean_final_heldout_mse": {arm: float(np.mean([result["arms"][arm]["final_heldout_mse"] for result in results])) for arm in ARMS},
        "runtime_seconds": time.perf_counter() - started,
    }
    with (args.output_dir / "diagnostic.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    pd.DataFrame(predictions).to_csv(args.output_dir / "synthetic_heldout_predictions.tsv", sep="\t", index=False)
    print(json.dumps({"report": str(args.output_dir / "diagnostic.json"), "mean_final_heldout_mse": report["mean_final_heldout_mse"], "runtime_seconds": report["runtime_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
