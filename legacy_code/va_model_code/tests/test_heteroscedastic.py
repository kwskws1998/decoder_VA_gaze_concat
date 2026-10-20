"""Validate probabilistic VA math, unchanged means, stable variance and ordered exports."""

import math
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from va_model_code.decoder_va.evaluation import (
    calculate_va_metrics, metrics_from_prediction_frame, prediction_frame, write_oof_reports,
)
from va_model_code.decoder_va.losses import va_regression_loss
from va_model_code.decoder_va.model import DecoderVARegressor
from va_model_code.decoder_va.uncertainty import (
    MIN_VARIANCE, INITIAL_VARIANCE, log_variance_from_raw, prediction_contract,
    saved_prediction_contract, make_variance_head, UNCERTAINTY_TABLE_COLUMNS,
)
from va_model_code.tests.test_model_loss_evaluation import FakeCausalBackbone, FakeGazeProvider


def test_nll_matches_independent_gaussian_density_and_both_gradients():
    """Compare the objective and gradients with PyTorch's Normal probability distribution."""

    outputs = torch.tensor([[0.2, 0.7, -3., -2.], [0.8, 0.5, -4., -1.]], requires_grad=True)
    labels = torch.tensor([[0.3, 0.9], [0.4, 0.2]])
    actual = va_regression_loss(outputs, labels, "heteroscedastic")
    reference = -torch.distributions.Normal(outputs[:, :2], (outputs[:, 2:] / 2).exp()).log_prob(labels).mean()
    torch.testing.assert_close(actual.total, reference)
    torch.testing.assert_close(torch.autograd.grad(actual.total, outputs)[0], torch.autograd.grad(reference, outputs)[0])
    torch.testing.assert_close(actual.point, F.mse_loss(outputs[:, :2], labels))


def test_constant_variance_scales_mse_and_variance_gradient_tracks_residual():
    """Variance increases for underexplained errors and decreases for overestimated noise."""

    means = torch.tensor([[0.2, 0.8]], requires_grad=True)
    log_vars = torch.tensor([[math.log(0.1), math.log(0.1)]], requires_grad=True)
    labels = torch.tensor([[0.2, 0.2]])
    nll = va_regression_loss(torch.cat((means, log_vars), -1), labels, "heteroscedastic").total
    expected = 5 * F.mse_loss(means, labels) + 0.5 * math.log(2 * math.pi * 0.1)
    torch.testing.assert_close(nll, expected)
    nll.backward()
    assert log_vars.grad[0, 0] > 0
    assert log_vars.grad[0, 1] < 0
    assert means.grad[0, 1] > 0


@pytest.mark.parametrize("fusion", ["none", "prefix-concat"])
@pytest.mark.parametrize("bf16", [False, True])
def test_variance_head_preserves_initial_means_rng_and_receives_gradients(fusion, bf16):
    """Adding variance keeps the original VA/prefix initialization and CPU BF16 mean path intact."""

    def build(loss):
        torch.manual_seed(712)
        return DecoderVARegressor(
            FakeCausalBackbone(), gaze_fusion=fusion,
            gaze_provider=FakeGazeProvider() if fusion != "none" else None,
            prediction_distribution=prediction_contract(loss), classifier_dropout=0.0,
            gaze_projection_dropout=(0., 0.),
        )

    point = build("mse").eval()
    point_rng = torch.get_rng_state().clone()
    model = build("heteroscedastic").eval()
    assert torch.equal(point_rng, torch.get_rng_state())
    for name, value in point.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    batch = dict(input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]), attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        baseline = point(**batch).logits
        outputs = model(**batch).logits
        loss = va_regression_loss(outputs, torch.tensor([[0.1, 0.9], [0.8, 0.2]]), "heteroscedastic").total
    assert outputs.shape == (2, 4)
    assert outputs.dtype == loss.dtype == torch.float32
    torch.testing.assert_close(outputs[:, :2], baseline, atol=0, rtol=0)
    torch.testing.assert_close(outputs[:, 2:].exp(), torch.full((2, 2), INITIAL_VARIANCE))
    assert (outputs[:, 2:] < 0).all()
    assert model.readout_token is None
    assert model.gaze_relation["method"] == "none"
    loss.backward()
    for parameter in (model.variance_head.weight, model.regression_head[2].weight, model.backbone.embedding.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    if fusion != "none":
        assert torch.count_nonzero(model.gaze_projector[0].weight.grad) > 0


def test_extreme_raw_variance_stays_positive_finite_with_finite_backward():
    """The variance floor prevents zero-error NLL collapse and exponential overflow."""

    raw = torch.tensor([[-1e4, 1e4]], requires_grad=True)
    log_vars = log_variance_from_raw(raw)
    outputs = torch.cat((torch.tensor([[0.5, 0.5]]), log_vars), -1)
    loss = va_regression_loss(outputs, torch.tensor([[0.5, 0.6]]), "heteroscedastic").total
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()
    assert log_vars.exp().min().item() == pytest.approx(MIN_VARIANCE)


@pytest.mark.parametrize("width,loss", [(2, "heteroscedastic"), (4, "mse"), (4, "ccc"), (3, "heteroscedastic")])
def test_loss_rejects_output_contract_mismatch(width, loss):
    with pytest.raises(ValueError, match="logits must have shape"):
        va_regression_loss(torch.ones(2, width), torch.ones(2, 2), loss)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -100., 100.])
def test_invalid_variance_fails_in_loss_and_metrics(value):
    outputs = torch.tensor([[0.5, 0.5, value, -2.]])
    with pytest.raises(ValueError):
        va_regression_loss(outputs, torch.zeros(1, 2), "heteroscedastic")
    with pytest.raises(ValueError):
        calculate_va_metrics(np.zeros((1, 2)), outputs.numpy())


def test_variance_learns_two_noise_scales_without_uncertainty_labels():
    """A controlled fixed-mean probe recovers different conditional residual variances."""

    features = torch.eye(2).repeat_interleave(20, dim=0)
    signs = torch.tensor([-1., 1.]).repeat(20).unsqueeze(1)
    std = features @ torch.tensor([[0.05, 0.2], [0.2, 0.05]])
    labels = 0.5 + signs * std
    head = make_variance_head(2)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(300):
        optimizer.zero_grad()
        outputs = torch.cat((torch.full((40, 2), 0.5), log_variance_from_raw(head(features))), -1)
        loss = va_regression_loss(outputs, labels, "heteroscedastic").total
        loss.backward()
        optimizer.step()
    predicted = log_variance_from_raw(head(torch.eye(2))).exp().detach()
    torch.testing.assert_close(predicted, torch.tensor([[0.0025, 0.04], [0.04, 0.0025]]), rtol=0.02, atol=1e-5)


def test_metrics_export_roundtrip_preserves_means_and_variance_and_rejects_tampering(tmp_path):
    """OOF metrics must use means for VA and verify derived uncertainty columns on reload."""

    labels = np.array([[0.1, 0.8], [0.4, 0.6], [0.9, 0.2]])
    means = np.array([[0.2, 0.7], [0.5, 0.6], [0.8, 0.3]])
    predictions = np.column_stack((means, np.log(np.array([[0.01, 0.04], [0.02, 0.01], [0.03, 0.02]]))))
    metadata = pd.DataFrame({"index": [2, 0, 1], "text": ["a", "b", "c"], "dataset_of_origin": ["word", "sentence", "word"]})
    frame = prediction_frame(metadata, labels, predictions, fold=1)
    assert tuple(frame.columns[-6:]) == UNCERTAINTY_TABLE_COLUMNS
    point = calculate_va_metrics(labels, means)
    metrics = calculate_va_metrics(labels, predictions)
    for key in point:
        assert metrics[key] == pytest.approx(point[key])
    loss = va_regression_loss(torch.tensor(predictions), torch.tensor(labels), "heteroscedastic").total
    assert metrics["gaussian_nll_mean"] == pytest.approx(loss.item(), abs=1e-6)
    assert metrics["coverage_95_valence"] == 1.
    combined, written, _ = write_oof_reports([frame], tmp_path)
    loaded = pd.read_csv(tmp_path / "oof_predictions.tsv", sep="\t")
    pd.testing.assert_frame_equal(loaded, combined)
    assert metrics_from_prediction_frame(loaded)["gaussian_nll_mean"] == pytest.approx(written["gaussian_nll_mean"])
    loaded.loc[0, "pred_variance_valence"] *= 2
    with pytest.raises(ValueError, match="disagrees"):
        metrics_from_prediction_frame(loaded)
    with pytest.raises(ValueError, match="Incomplete"):
        metrics_from_prediction_frame(frame.drop(columns="pred_std_arousal"))


def test_contract_prevents_legacy_or_mismatched_loss_reinterpretation():
    for schema in (8, 9, 10):
        assert saved_prediction_contract({}, schema_version=schema, context="test") == prediction_contract()
        with pytest.raises(ValueError):
            saved_prediction_contract({"prediction_distribution": prediction_contract("heteroscedastic")}, schema_version=schema, context="test")
    with pytest.raises(ValueError, match="missing"):
        saved_prediction_contract({}, schema_version=11, context="test")
    with pytest.raises(ValueError, match="loss disagrees"):
        saved_prediction_contract({"prediction_distribution": prediction_contract(), "loss": "heteroscedastic"}, schema_version=11, context="test")
