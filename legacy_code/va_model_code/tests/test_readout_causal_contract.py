"""Independent native-Qwen tests for the new final-readout causal information path."""

import pytest
import torch

from va_model_code.decoder_va.gaze_relation import gaze_relation_contract
from va_model_code.decoder_va.model import DecoderVARegressor
from va_model_code.decoder_va.readout import readout_contract
from va_model_code.tests.test_gaze_attention import tiny_backbone


class SingleWordTRT:
    """Attach one positive TRT to the first token, leaving the last subtoken unmapped."""

    feature_indices = (3,)

    def compute(self, input_ids, attention_mask):
        """Generate deterministic raw gaze without pretrained downloads."""

        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        mask[:, 0] = True
        values = torch.zeros_like(input_ids, dtype=torch.float32)
        values[:, 0] = 2.0
        return values.unsqueeze(-1), mask


def readout_model(seed=103, head_dim=8, partial=1.0):
    """Build a hybrid native Qwen whose last layer is full attention."""

    torch.manual_seed(seed)
    return DecoderVARegressor(
        tiny_backbone(head_dim=head_dim, partial_rotary_factor=partial),
        gaze_provider=SingleWordTRT(),
        gaze_relation=gaze_relation_contract("trt-readout-rotation"),
        readout=readout_contract("learned-token"),
        gaze_projection_dim=8,
        gaze_projection_dropout=(0, 0),
        classifier_dropout=0,
    )


@pytest.mark.parametrize("bf16", [False, True])
def test_real_head_geometry_singleton_reaches_last_layer_and_predictions(bf16):
    """Exercise the actual 256-dimensional partial-RoPE geometry with one gaze anchor."""

    model = readout_model(head_dim=256, partial=0.25).eval()
    batch = {"input_ids": torch.tensor([[2, 9, 11]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        baseline = model(**batch).logits
    baseline.sum().backward()
    alpha = model.backbone.layers[-1].self_attn.gaze_relative_rotation.alpha
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
    assert torch.count_nonzero(alpha.grad) > 0
    assert model.readout_token.grad is not None and torch.count_nonzero(model.readout_token.grad) > 0
    with torch.no_grad():
        alpha.copy_(torch.tensor([0.2, -0.15, 0.4, -0.3]))
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            changed = model(**batch).logits
    assert torch.isfinite(changed).all()
    assert not torch.equal(changed, baseline)


def test_final_readout_never_changes_preceding_token_states():
    """Changing a future readout embedding must not leak into the gaze prefix or text."""

    model = readout_model().eval()
    batch = {"input_ids": torch.tensor([[2, 7, 5], [4, 0, 0]]), "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]])}
    captured = []

    def capture(module, arguments, output):
        """Record the hidden states before the regression readout selects its token."""

        captured.append(output.last_hidden_state.detach().clone())

    handle = model.backbone.register_forward_hook(capture)
    try:
        with torch.no_grad():
            model.backbone.layers[-1].self_attn.gaze_relative_rotation.alpha.fill_(0.2)
            first = model(**batch).logits
            model.readout_token.add_(torch.linspace(-0.3, 0.4, model.hidden_size))
            second = model(**batch).logits
    finally:
        handle.remove()
    for row, readout_position in enumerate((6, 4)):
        torch.testing.assert_close(captured[0][row, :readout_position], captured[1][row, :readout_position], rtol=0, atol=0)
    assert not torch.equal(first, second)
