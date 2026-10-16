"""Paired training data: source speech + the same speech in a target voice.

PairVC learns from *pairs*. A large, slow teacher voice-conversion model was
run offline on every source clip, once per target voice; PairVC is trained to
reproduce the teacher's output, sample for sample, while being small and causal
enough for real time (teacher -> student distillation on synthetic parallel
data). See docs/data.md for how to build such a dataset.

Files in one split directory:

    <id>_original.wav       source speech, any speaker
    <id>_converted_0.wav    the same words, converted to target voice 0
    <id>_converted_1.wav    ... target voice 1, and so on

Every (source, converted_k) couple is one training example with label k, so a
source with 2 conversions gives 2 examples.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor
from torch.utils.data import Dataset

SOURCE_SUFFIX = "_original.wav"
TARGET_PATTERN = re.compile(r"^(?P<id>.+)_converted_(?P<speaker>\d+)\.wav$")


@dataclass(frozen=True, slots=True)
class Pair:
    source: Path
    target: Path
    speaker: int


def find_pairs(split_dir: Path) -> list[Pair]:
    """Every (source, converted_k) couple in a directory, sorted (same order on every machine)."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_dir}")
    pairs = []
    for target in sorted(split_dir.glob("*_converted_*.wav")):
        match = TARGET_PATTERN.match(target.name)
        if match is None:
            continue
        source = split_dir / f"{match['id']}{SOURCE_SUFFIX}"
        if source.exists():
            pairs.append(Pair(source, target, int(match["speaker"])))
    return pairs


def load_audio(path: Path, sample_rate: int) -> Tensor:
    """Read a sound file as mono float32 in [-1, 1] at `sample_rate`: [T]."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)  # any bit depth -> float
    audio = audio.mean(axis=1)  # stereo -> mono
    if sr != sample_rate:
        g = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(audio))


def crop_pair(source: Tensor, target: Tensor, length: int, random_crop: bool) -> tuple[Tensor, Tensor]:
    """Cut the same window out of both clips, zero-padding clips that are too short."""
    n = min(source.shape[-1], target.shape[-1])  # the teacher may add or drop a few samples
    source, target = source[:n], target[:n]
    if n < length:
        return torch.nn.functional.pad(source, (0, length - n)), torch.nn.functional.pad(
            target, (0, length - n)
        )
    start = random.randint(0, n - length) if random_crop else 0
    return source[start : start + length], target[start : start + length]


class PairedVoiceDataset(Dataset):
    """Returns {"source": [T], "target": [T], "speaker": long} with T = segment_samples.

    Training takes a random window of each clip (new each epoch); validation
    always takes the start of the clip, so its numbers are comparable.
    """

    def __init__(
        self,
        split_dir: Path,
        sample_rate: int,
        segment_samples: int,
        train: bool = False,
        limit: int | None = None,
    ) -> None:
        self.pairs = find_pairs(split_dir)
        if limit is not None:
            self.pairs = self.pairs[:limit]
        if not self.pairs:
            raise ValueError(f"no '<id>_original.wav' + '<id>_converted_<k>.wav' pairs in {split_dir}")
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.train = train

    @property
    def speakers(self) -> list[int]:
        return sorted({p.speaker for p in self.pairs})

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        pair = self.pairs[index]
        source = load_audio(pair.source, self.sample_rate)
        target = load_audio(pair.target, self.sample_rate)
        source, target = crop_pair(source, target, self.segment_samples, random_crop=self.train)
        return {"source": source, "target": target, "speaker": torch.tensor(pair.speaker)}


def _toy_voice(
    f0: np.ndarray, envelope: np.ndarray, sample_rate: int, pitch_scale: float, brightness: float
) -> np.ndarray:
    """7 harmonics of a pitch contour; brightness sets how fast they fade (the timbre)."""
    phase = 2 * np.pi * np.cumsum(f0 * pitch_scale) / sample_rate
    harmonics = sum(np.sin(h * phase) * brightness**h for h in range(1, 8))
    return (0.15 * envelope * harmonics).astype(np.float32)


def write_toy_dataset(
    root: Path,
    n_clips: int = 16,
    seconds: float = 1.0,
    sample_rate: int = 16_000,
    n_speakers: int = 2,
    seed: int = 0,
) -> Path:
    """Write a tiny synthetic paired dataset: no real data needed.

    Each "source" is a buzzy vowel-like tone with a random pitch contour; voice
    k is the same contour with pitch scaled by (k + 2) / 2 and a different
    timbre. Useful to check the whole pipeline (tests, notebook smoke test);
    a model trained on it will not convert real voices.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    for split, n_split in (("train", n_clips), ("val", max(2, n_clips // 4))):
        split_dir = Path(root) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_split):
            f0 = rng.uniform(100, 220) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t))
            envelope = np.clip(np.sin(np.pi * t / seconds) * 1.5, 0, 1)
            source = _toy_voice(f0, envelope, sample_rate, pitch_scale=1.0, brightness=0.6)
            sf.write(split_dir / f"clip{i:04d}{SOURCE_SUFFIX}", source, sample_rate)
            for k in range(n_speakers):
                target = _toy_voice(f0, envelope, sample_rate, (k + 2) / 2, 0.4 + 0.2 * k)
                sf.write(split_dir / f"clip{i:04d}_converted_{k}.wav", target, sample_rate)
    return Path(root)


__all__ = [
    "Pair",
    "PairedVoiceDataset",
    "crop_pair",
    "find_pairs",
    "load_audio",
    "write_toy_dataset",
]
