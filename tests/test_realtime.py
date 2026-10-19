"""Tests for pairvc.realtime: the ONNX live path equals offline conversion."""

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal import resample_poly
from test_convert import write_checkpoint

from pairvc.realtime import StreamResampler


@pytest.mark.parametrize(("up", "down", "block"), [(1, 3, 960), (3, 1, 317), (2, 3, 101)])
def test_resampler_blocks_equal_whole(up: int, down: int, block: int) -> None:
    x = np.random.default_rng(0).standard_normal(4000)
    whole = StreamResampler(up, down).process(x)
    r = StreamResampler(up, down)
    blocks = np.concatenate([r.process(x[i : i + block]) for i in range(0, len(x), block)])
    np.testing.assert_allclose(blocks, whole, atol=1e-6)
    assert abs(len(whole) - len(x) * up / down) <= 1


def test_resampler_keeps_a_tone() -> None:
    t = np.arange(48_000) / 48_000
    out = StreamResampler(1, 3).process(np.sin(2 * np.pi * 440 * t))
    delay = round(StreamResampler(1, 3).delay)
    expected = np.sin(2 * np.pi * 440 * (np.arange(len(out)) - delay) / 16_000)
    np.testing.assert_allclose(out[100:], expected[100:], atol=0.02)


@pytest.fixture
def onnx_model(tmp_path: Path) -> tuple[Path, torch.nn.Module]:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from pairvc.export import export_onnx

    model = write_checkpoint(tmp_path / "best.pt")
    export_onnx(model, tmp_path / "pairvc.onnx", chunk_samples=16, sample_rate=16_000)
    return tmp_path / "pairvc.onnx", model


def test_streamer_reads_metadata_and_matches_offline(onnx_model: tuple[Path, torch.nn.Module]) -> None:
    from pairvc.realtime import OnnxStreamer

    path, model = onnx_model
    streamer = OnnxStreamer(path, speaker=1)
    assert (streamer.chunk, streamer.sample_rate) == (16, 16_000)
    assert streamer.n_speakers == model.config.n_speakers
    assert streamer.lookahead == model.config.lookahead_samples
    audio = 0.1 * np.random.default_rng(1).standard_normal(16 * 20 + 5).astype(np.float32)
    out, times = streamer.convert(audio)
    with torch.inference_mode():
        reference = model.convert(torch.from_numpy(audio)[None], torch.tensor([1]))[0].numpy()
    assert len(out) == len(audio) and len(times) > 0
    np.testing.assert_allclose(out, reference, atol=1e-4)


def test_live_converter_frames_in_frames_out(onnx_model: tuple[Path, torch.nn.Module]) -> None:
    from pairvc.realtime import LiveConverter

    path, _ = onnx_model
    audio16 = 0.1 * np.random.default_rng(2).standard_normal(16 * 60).astype(np.float32)
    audio48 = resample_poly(audio16, 3, 1).astype(np.float32)
    live = LiveConverter(path, speaker=0, rate=48_000)
    frames = [live.process(audio48[i : i + 50]) for i in range(0, len(audio48), 50)]  # odd frame size
    assert [len(f) for f in frames] == [len(audio48[i : i + 50]) for i in range(0, len(audio48), 50)]
    assert len(live.stats.times) == len(audio16) // 16 and live.stats.rtf > 0
    assert np.abs(np.concatenate(frames)).max() > 0
    live.bypass = True
    np.testing.assert_array_equal(live.process(audio48[:50]), audio48[:50])
