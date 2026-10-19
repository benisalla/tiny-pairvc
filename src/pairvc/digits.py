"""A small *real speech* paired dataset, built in a minute: spoken digits + a DSP teacher.

The real recipe needs a big teacher VC model (docs/data.md). To show that PairVC
learns on real speech without one, this module plays the teacher with plain
signal processing:

- source speech: the Free Spoken Digit Dataset (FSDD, CC BY-SA 4.0), 3000 clips
  of 6 people saying "zero" ... "nine", 8 kHz, ~0.5 s each;
- target voice k: the same clip pitch-shifted by `SHIFTS[k]` semitones with a
  phase vocoder. Pitch *and* formants move together (a higher / deeper voice),
  and the timing is kept, which is what PairVC's loss needs.

Validation uses a speaker that is never seen in training ("theo"), so the
numbers say how well PairVC converts a voice it never heard.

    from pairvc.digits import write_digits_dataset
    write_digits_dataset("digits_data")      # downloads FSDD once (16 MB)
"""

from __future__ import annotations

import io
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from pairvc.data import SOURCE_SUFFIX

FSDD_URL = "https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/refs/heads/master"
SHIFTS = (5.0, -5.0)  # semitones: target voice 0 is higher, voice 1 is deeper
VAL_SPEAKER = "theo"  # held out: validation is on an unseen source voice


def phase_vocoder_stretch(audio: np.ndarray, rate: float, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Play `audio` `rate` times faster without changing its pitch (rate < 1 = longer).

    Classic phase vocoder: read the STFT at fractional frame positions, keep each
    bin's magnitude and advance its phase by the measured instantaneous frequency.
    """
    window = np.hanning(n_fft)
    pad = np.pad(audio, (n_fft // 2, n_fft // 2 + n_fft))
    n_frames = 1 + (len(pad) - n_fft) // hop
    frames = np.stack([pad[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    spec = np.fft.rfft(frames, axis=1)  # [frames, bins]

    expected = 2 * np.pi * hop * np.arange(n_fft // 2 + 1) / n_fft  # phase advance per hop
    positions = np.arange(0, n_frames - 1, rate)
    phase = np.angle(spec[0])
    out_frames = []
    for pos in positions:
        i, frac = int(pos), pos - int(pos)
        magnitude = (1 - frac) * np.abs(spec[i]) + frac * np.abs(spec[i + 1])
        out_frames.append(magnitude * np.exp(1j * phase))
        delta = np.angle(spec[i + 1]) - np.angle(spec[i]) - expected
        delta -= 2 * np.pi * np.round(delta / (2 * np.pi))  # wrap to [-pi, pi]
        phase = phase + expected + delta

    # Overlap-add with the same window, normalised by the summed squared window.
    frames_out = np.fft.irfft(np.stack(out_frames), n=n_fft, axis=1) * window
    length = n_fft + hop * (len(frames_out) - 1)
    out, norm = np.zeros(length), np.zeros(length)
    for j, frame in enumerate(frames_out):
        out[j * hop : j * hop + n_fft] += frame
        norm[j * hop : j * hop + n_fft] += window**2
    out = out / np.maximum(norm, 1e-3)
    return out[n_fft // 2 : n_fft // 2 + round(len(audio) / rate)]


def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Same duration, pitch (and formants) moved by `semitones`: the toy teacher."""
    ratio = 2 ** (semitones / 12)
    stretched = phase_vocoder_stretch(audio, 1 / ratio)  # ratio x longer, same pitch
    up, down = (round(1000 / ratio), 1000)
    g = math.gcd(up, down)
    shifted = resample_poly(stretched, up // g, down // g)  # back to the original length, pitch x ratio
    shifted = np.pad(shifted, (0, max(0, len(audio) - len(shifted))))[: len(audio)]
    return shifted.astype(np.float32)


def download_fsdd(cache_dir: Path) -> Path:
    """Download and unpack FSDD once; returns the folder holding the 3000 wav files."""
    recordings = Path(cache_dir) / "fsdd" / "recordings"
    if recordings.is_dir() and any(recordings.glob("*.wav")):
        return recordings
    recordings.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(FSDD_URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for name in archive.namelist():
        if "/recordings/" in name and name.endswith(".wav"):
            (recordings / Path(name).name).write_bytes(archive.read(name))
    return recordings


def write_digits_dataset(
    root: Path,
    sample_rate: int = 16_000,
    clips_per_speaker: int = 100,
    val_clips: int = 40,
    cache_dir: Path | None = None,
    shifts: tuple[float, ...] = SHIFTS,
    seed: int = 0,
) -> Path:
    """Write <root>/{train,val}/<id>_original.wav + <id>_converted_<k>.wav from FSDD.

    `clips_per_speaker` clips of each of the 5 training speakers go to train/,
    `val_clips` clips of the held-out speaker go to val/. Audio is resampled from
    8 kHz to `sample_rate`, peak-normalised, and padded with 0.1 s of silence.
    """
    recordings = download_fsdd(Path(cache_dir) if cache_dir else Path(root) / ".cache")
    by_speaker: dict[str, list[Path]] = {}
    for path in sorted(recordings.glob("*.wav")):
        by_speaker.setdefault(path.stem.split("_")[1], []).append(path)  # <digit>_<speaker>_<n>.wav

    rng = np.random.default_rng(seed)
    pad = np.zeros(sample_rate // 10, dtype=np.float32)
    for speaker, paths in sorted(by_speaker.items()):
        split, n = ("val", val_clips) if speaker == VAL_SPEAKER else ("train", clips_per_speaker)
        split_dir = Path(root) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in rng.permutation(paths)[:n]:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            g = math.gcd(sr, sample_rate)
            audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
            audio = np.concatenate([pad, 0.9 * audio / max(np.abs(audio).max(), 1e-6), pad])
            clip_id = Path(path).stem
            sf.write(split_dir / f"{clip_id}{SOURCE_SUFFIX}", audio, sample_rate)
            for k, semitones in enumerate(shifts):
                target = np.clip(pitch_shift(audio, semitones), -1, 1)
                sf.write(split_dir / f"{clip_id}_converted_{k}.wav", target, sample_rate)
    return Path(root)


__all__ = [
    "FSDD_URL",
    "SHIFTS",
    "VAL_SPEAKER",
    "download_fsdd",
    "phase_vocoder_stretch",
    "pitch_shift",
    "write_digits_dataset",
]
