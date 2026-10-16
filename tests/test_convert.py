"""Tests for pairvc.convert and pairvc.export: checkpoint -> converted audio / ONNX."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from conftest import tiny_model_config

from pairvc.config import DataConfig, config_to_dict
from pairvc.convert import convert_file, convert_streaming, load_model, run, save_generator_only
from pairvc.model import PairVC


def write_checkpoint(path: Path) -> PairVC:
    """Save a checkpoint with the keys convert.py reads (as train.py writes them)."""
    torch.manual_seed(0)
    model = PairVC(tiny_model_config()).eval()
    for p in model.parameters():
        p.data.normal_(0, 0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": model.state_dict(),
            "model_config": config_to_dict(model.config),
            "data_config": config_to_dict(DataConfig()),
        },
        path,
    )
    return model


def test_load_model_restores_the_same_weights(tmp_path: Path) -> None:
    original = write_checkpoint(tmp_path / "best.pt")
    loaded, sample_rate = load_model(tmp_path / "best.pt")
    assert sample_rate == 16_000 and not loaded.training
    x, speaker = torch.randn(1, 160), torch.tensor([1])
    torch.testing.assert_close(loaded.convert(x, speaker), original.convert(x, speaker))


def test_generator_only_file_loads_like_the_full_checkpoint(tmp_path: Path) -> None:
    original = write_checkpoint(tmp_path / "best.pt")
    slim = save_generator_only(tmp_path / "best.pt", tmp_path / "pairvc_generator.pt")
    loaded, _ = load_model(slim)
    x, speaker = torch.randn(1, 160), torch.tensor([0])
    torch.testing.assert_close(loaded.convert(x, speaker), original.convert(x, speaker))


def test_streaming_conversion_equals_offline(tmp_path: Path) -> None:
    model = write_checkpoint(tmp_path / "best.pt")
    audio = 0.1 * torch.randn(1003)  # not a whole number of chunks
    streamed, stats = convert_streaming(model, audio, 1, chunk_samples=16, sample_rate=16_000)
    offline = model.convert(audio[None], torch.tensor([1]))[0]
    torch.testing.assert_close(streamed, offline, rtol=1e-4, atol=1e-4)
    assert stats.rtf > 0 and stats.latency_ms == 1000 * (16 + 16) / 16_000  # chunk + lookahead


def test_convert_cli_writes_one_file_per_input(toy_data: Path, tmp_path: Path) -> None:
    write_checkpoint(tmp_path / "best.pt")
    inputs = sorted((toy_data / "val").glob("*_original.wav"))[:2]
    for extra in ([], ["--streaming"]):
        outputs = run([
            "--checkpoint", str(tmp_path / "best.pt"), "--speaker", "1",
            "--output-dir", str(tmp_path / "out"), "--input", *map(str, inputs), *extra,
        ])  # fmt: skip
        assert [p.name for p in outputs] == [f"{p.stem}_to_1.wav" for p in inputs]
        audio, sr = sf.read(outputs[0])
        assert sr == 16_000 and len(audio) == len(sf.read(inputs[0])[0])  # same length as input


def test_convert_rejects_unknown_speaker(toy_data: Path, tmp_path: Path) -> None:
    model = write_checkpoint(tmp_path / "best.pt")
    with pytest.raises(ValueError):
        convert_file(model, next((toy_data / "val").glob("*.wav")), 7, 16_000)


def test_onnx_export_matches_pytorch(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from pairvc.export import export_onnx, run_onnx_streaming

    model = write_checkpoint(tmp_path / "best.pt")
    export_onnx(model, tmp_path / "pairvc.onnx", chunk_samples=16)
    audio = 0.1 * np.random.default_rng(0).standard_normal(16 * 20).astype(np.float32)
    onnx_out = run_onnx_streaming(tmp_path / "pairvc.onnx", audio, speaker=1)
    with torch.inference_mode():
        reference = model(torch.from_numpy(audio)[None], torch.tensor([1]))[0].numpy()
    np.testing.assert_allclose(onnx_out, reference, rtol=1e-4, atol=1e-4)
