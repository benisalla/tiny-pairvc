"""Tests for pairvc.metrics, pairvc.digits (the DSP teacher) and pairvc.evaluate."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_convert import write_checkpoint

from pairvc.digits import phase_vocoder_stretch, pitch_shift
from pairvc.evaluate import SYSTEMS, evaluate_split, results_table, run
from pairvc.metrics import compare, estimate_f0, f0_metrics, mel_cepstral_distortion

SR = 16_000


def tone(f0: float, seconds: float = 0.5, brightness: float = 0.6) -> np.ndarray:
    """A buzzy vowel-like tone with 8 harmonics and soft edges."""
    t = np.arange(int(seconds * SR)) / SR
    audio = sum(brightness**h * np.sin(2 * np.pi * h * f0 * t) for h in range(1, 9))
    return (0.2 * np.minimum(1, 10 * np.sin(np.pi * t / seconds).clip(0)) * audio).astype(np.float32)


@pytest.mark.parametrize("f0", [90.0, 150.0, 280.0])
def test_f0_tracker_finds_the_pitch_of_a_tone(f0: float) -> None:
    track = estimate_f0(tone(f0), SR)
    voiced = track[track > 0]
    assert len(voiced) > 0.6 * len(track)
    assert abs(1200 * np.log2(np.median(voiced) / f0)) < 30  # within 30 cents


def test_identical_audio_scores_perfectly() -> None:
    audio = tone(150.0)
    scores = compare(audio, audio, SR)
    assert scores["mcd_db"] == 0 and scores["mel_l1"] == 0 and scores["f0_rmse_cents"] == 0
    assert scores["voicing_agreement"] == 1.0


def test_metrics_grow_with_the_difference() -> None:
    reference = tone(150.0)
    rng = np.random.default_rng(0)
    small = mel_cepstral_distortion(reference + 0.002 * rng.standard_normal(len(reference)), reference, SR)
    large = mel_cepstral_distortion(reference + 0.05 * rng.standard_normal(len(reference)), reference, SR)
    assert 0 < small < large
    assert mel_cepstral_distortion(tone(150.0, brightness=0.3), reference, SR) > 1.0  # other timbre
    # A semitone up is 100 cents of pitch error.
    assert abs(f0_metrics(tone(150.0 * 2 ** (1 / 12)), reference, SR)["f0_rmse_cents"] - 100) < 20


def test_phase_vocoder_changes_the_length_not_the_pitch() -> None:
    audio = tone(150.0, seconds=0.6)
    longer = phase_vocoder_stretch(audio, 0.5)
    assert abs(len(longer) - 2 * len(audio)) <= 1
    assert abs(1200 * np.log2(np.median(estimate_f0(longer, SR)[estimate_f0(longer, SR) > 0]) / 150)) < 30


@pytest.mark.parametrize("semitones", [5.0, -5.0])
def test_pitch_shift_keeps_the_length_and_moves_the_pitch(semitones: float) -> None:
    audio = tone(150.0, seconds=0.6)
    shifted = pitch_shift(audio, semitones)
    assert shifted.shape == audio.shape and shifted.dtype == np.float32
    assert abs(f0_metrics(shifted, audio, SR)["f0_rmse_cents"] - 100 * abs(semitones)) < 40


def test_evaluate_split_reports_every_system(toy_data: Path, tmp_path: Path) -> None:
    model = write_checkpoint(tmp_path / "best.pt")
    results = evaluate_split(model, toy_data / "val", SR, limit=2)
    assert list(results) == list(SYSTEMS)
    assert results["teacher"]["mcd_db"] == 0  # the reference against itself
    assert results["no conversion"]["mcd_db"] > 0
    table = results_table(results)
    assert table.count("\n") == 4 and "MCD dB" in table and "DNSMOS" not in table
    only_model = evaluate_split(model, toy_data / "val", SR, limit=2, baselines=False)
    assert list(only_model) == ["PairVC"] and only_model["PairVC"] == results["PairVC"]


def test_eval_cli_writes_json(toy_data: Path, tmp_path: Path) -> None:
    write_checkpoint(tmp_path / "best.pt")
    for extra in ([], ["--untrained"]):
        out = tmp_path / f"results{len(extra)}.json"
        run(["--checkpoint", str(tmp_path / "best.pt"), "--data-dir", str(toy_data),
             "--limit", "2", "--output", str(out), *extra])  # fmt: skip
        assert set(json.loads(out.read_text())) == set(SYSTEMS)
