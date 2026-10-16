"""Tests for pairvc.model: shapes, causality, and streaming == whole-clip."""

import pytest
import torch
from conftest import tiny_model_config

from pairvc.config import ModelConfig
from pairvc.model import PairVC


def random_model(config: ModelConfig, std: float = 0.1) -> PairVC:
    """Bigger random weights than the init, so that any mismatch shows up clearly."""
    torch.manual_seed(0)
    model = PairVC(config).eval()
    for p in model.parameters():
        p.data.normal_(0, std)
    return model


def test_output_shape_any_length() -> None:
    model = PairVC(tiny_model_config())
    for length in (64, 67, 100):  # also not a multiple of the hop (8)
        out = model(torch.randn(3, length), torch.tensor([0, 1, 1]))
        assert out.shape == (3, length)


def test_default_model_size() -> None:
    model = PairVC(ModelConfig())
    n = sum(p.numel() for p in model.parameters())
    assert 9e6 < n < 10e6  # ~9.6M parameters


def test_causal_future_input_cannot_change_past_output() -> None:
    model = random_model(tiny_model_config())
    x = torch.randn(1, 160)
    y = x.clone()
    y[:, 100:] = torch.randn(1, 60)  # change only the future
    speaker = torch.tensor([0])
    with torch.no_grad():
        torch.testing.assert_close(model(x, speaker)[:, :100], model(y, speaker)[:, :100])


def test_speaker_changes_the_output() -> None:
    model = random_model(tiny_model_config())
    x = torch.randn(1, 64).repeat(2, 1)
    with torch.no_grad():
        out = model(x, torch.tensor([0, 1]))
    assert not torch.allclose(out[0], out[1])


@pytest.mark.parametrize("chunk_frames", [1, 2, 5])
def test_streaming_equals_whole_clip(chunk_frames: int) -> None:
    model = random_model(tiny_model_config())
    hop = model.config.hop_length
    x, speaker = torch.randn(2, hop * 40), torch.tensor([1, 0])
    with torch.no_grad():
        full = model(x, speaker)
        states, outs = model.init_state(2), []
        for start in range(0, x.shape[-1], hop * chunk_frames):
            out, states = model.stream_step(x[:, start : start + hop * chunk_frames], speaker, states)
            outs.append(out)
    torch.testing.assert_close(torch.cat(outs, dim=-1), full, rtol=1e-4, atol=1e-4)


def test_streaming_equals_whole_clip_default_model() -> None:
    model = random_model(ModelConfig(), std=0.05)
    x, speaker = torch.randn(1, 320 * 12), torch.tensor([1])
    with torch.no_grad():
        full = model(x, speaker)
        states, outs = model.init_state(1), []
        for start in range(0, x.shape[-1], 320):
            out, states = model.stream_step(x[:, start : start + 320], speaker, states)
            outs.append(out)
    error = (torch.cat(outs, dim=-1) - full).abs().max() / full.abs().max()
    assert error < 1e-5  # float32 rounding only


def test_stream_step_rejects_partial_frames() -> None:
    model = PairVC(tiny_model_config())
    with pytest.raises(ValueError):
        model.stream_step(torch.randn(1, 5), torch.tensor([0]), model.init_state(1))


def test_convert_removes_the_lookahead_delay() -> None:
    model = random_model(tiny_model_config())
    delay = model.config.lookahead_samples
    x, speaker = torch.randn(1, 96), torch.tensor([0])
    with torch.no_grad():
        delayed = model(torch.nn.functional.pad(x, (0, delay)), speaker)
    aligned = model.convert(x, speaker)
    assert aligned.shape == x.shape
    torch.testing.assert_close(aligned, delayed[:, delay:])


def test_gradient_checkpointing_gives_the_same_gradients() -> None:
    model = random_model(tiny_model_config()).train()
    x, speaker = torch.randn(2, 64), torch.tensor([0, 1])
    grads = []
    for flag in (False, True):
        model.zero_grad()
        model.grad_checkpointing = flag
        model(x, speaker).pow(2).mean().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b)


def test_receptive_field_is_measured() -> None:
    rf = PairVC(ModelConfig()).receptive_field()
    assert 5_000 < rf < 20_000  # ~0.7 s of past audio at 16 kHz
