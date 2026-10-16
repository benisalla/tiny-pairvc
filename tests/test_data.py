"""Tests for pairvc.data: file pairing, audio loading, cropping."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pairvc.data import PairedVoiceDataset, crop_pair, find_pairs, load_audio


def test_find_pairs_matches_every_conversion(tmp_path: Path) -> None:
    silence = np.zeros(100, dtype=np.float32)
    for name in ["a_original.wav", "a_converted_0.wav", "a_converted_3.wav",
                 "b_original.wav", "c_converted_1.wav", "notes.wav"]:  # fmt: skip
        sf.write(tmp_path / name, silence, 16_000)
    pairs = find_pairs(tmp_path)
    # b has no conversion, c has no source, notes.wav is not part of a pair.
    assert [(p.source.name, p.target.name, p.speaker) for p in pairs] == [
        ("a_original.wav", "a_converted_0.wav", 0),
        ("a_original.wav", "a_converted_3.wav", 3),
    ]


def test_load_audio_handles_int16_stereo_and_resampling(tmp_path: Path) -> None:
    t = np.arange(8000) / 8000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    sf.write(tmp_path / "x.wav", np.stack([tone, tone], axis=1), 8000, subtype="PCM_16")
    audio = load_audio(tmp_path / "x.wav", 16_000)
    assert audio.dtype == torch.float32 and audio.shape == (16_000,)  # mono, 2x the samples
    assert 0.45 < audio.abs().max() < 0.55  # int16 scaled to [-1, 1]


def test_crop_pair_pads_short_clips_and_crops_long_ones() -> None:
    a, b = torch.arange(10.0), torch.arange(12.0)
    x, y = crop_pair(a, b, 16, random_crop=True)
    assert x.shape == y.shape == (16,) and x[10:].eq(0).all()
    x, y = crop_pair(torch.arange(100.0), torch.arange(100.0), 16, random_crop=True)
    assert torch.equal(x, y)  # the same window from both clips: alignment is kept
    x, _ = crop_pair(torch.arange(100.0), torch.arange(100.0), 16, random_crop=False)
    assert x[0] == 0  # validation always starts at the beginning


def test_dataset_items(toy_data: Path) -> None:
    dataset = PairedVoiceDataset(toy_data / "train", 16_000, 1600, train=True)
    assert len(dataset) == 16  # 8 sources x 2 voices
    assert dataset.speakers == [0, 1]
    item = dataset[0]
    assert item["source"].shape == item["target"].shape == (1600,)
    assert item["speaker"].dtype == torch.long
    assert len(PairedVoiceDataset(toy_data / "train", 16_000, 1600, limit=3)) == 3
