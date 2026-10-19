"""Objective metrics: how close is PairVC's output to the teacher, and how natural is it?

All numbers compare a converted clip `x` with the teacher's clip `y` of the
same speech (they are time-aligned, so frames can be compared one to one):

- MCD (mel cepstral distortion, dB, lower is better): distance between the
  spectral envelopes, i.e. the *timbre*. Identical audio gives 0; values of
  ~4-6 dB are typical for good VC systems against their reference.
- F0 RMSE (cents, lower is better): pitch error on frames voiced in both clips.
  100 cents = 1 semitone; a model that does not change the pitch at all scores
  about 100 x the teacher's shift.
- Voicing agreement (%, higher is better): frames where both agree on "voiced"
  vs "unvoiced/silent".
- Mel L1: mean absolute log-mel difference (the training loss, at one scale).
- DNSMOS (optional, 1-5, higher is better): Microsoft's neural MOS predictor
  (P.835 SIG / BAK / OVRL), a stand-in for a listening test. Needs onnxruntime
  and downloads a 1.1 MB model once.

Everything here is plain numpy, no network (except DNSMOS), so it also runs in
the tests and on a laptop.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly

from pairvc.losses import mel_filterbank

DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------


def frames_of(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """[n_frames, frame] centred frames (zero-padded at both ends)."""
    audio = np.pad(np.asarray(audio, dtype=np.float64), (frame // 2, frame // 2))
    n = max(1, 1 + (len(audio) - frame) // hop)
    index = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return audio[index]


def log_mel(audio: np.ndarray, sample_rate: int, n_mels: int = 80) -> np.ndarray:
    """[n_frames, n_mels] natural-log mel spectrogram, 64 ms window, 16 ms hop."""
    n_fft = 2 ** round(math.log2(0.064 * sample_rate))
    spectrum = np.abs(np.fft.rfft(frames_of(audio, n_fft, n_fft // 4) * np.hanning(n_fft), axis=1))
    mel = spectrum @ mel_filterbank(sample_rate, n_fft, n_mels).numpy().T.astype(np.float64)
    return np.log(np.maximum(mel, 1e-5))


def energy_mask(audio: np.ndarray, sample_rate: int, threshold_db: float = -40.0) -> np.ndarray:
    """[n_frames] True where the frame is within `threshold_db` of the loudest one (speech)."""
    n_fft = 2 ** round(math.log2(0.064 * sample_rate))
    db = 10 * np.log10(np.mean(frames_of(audio, n_fft, n_fft // 4) ** 2, axis=1) + 1e-10)
    return db > db.max() + threshold_db


def mel_cepstrum(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """[n_frames, n_mels] real cepstrum of the log-mel spectrum: c_n = 1/N sum_k log S_k cos(...)."""
    log_spec = log_mel(audio, sample_rate)
    return dct(log_spec, type=2, axis=1) / (2 * log_spec.shape[1])  # scipy's DCT-II is 2 x the sum


def mel_cepstral_distortion(x: np.ndarray, y: np.ndarray, sample_rate: int, n_coeffs: int = 13) -> float:
    """MCD in dB between x and the reference y, on y's speech frames (c0, the loudness, excluded)."""
    n = min(len(x), len(y))
    cx, cy = mel_cepstrum(x[:n], sample_rate)[:, 1:n_coeffs], mel_cepstrum(y[:n], sample_rate)[:, 1:n_coeffs]
    mask = energy_mask(y[:n], sample_rate)
    per_frame = np.sqrt(2 * np.sum((cx - cy) ** 2, axis=1))
    return float(10 / math.log(10) * per_frame[mask].mean())


def mel_l1(x: np.ndarray, y: np.ndarray, sample_rate: int) -> float:
    """Mean |log10 mel(x) - log10 mel(y)|, the training loss at one scale."""
    n = min(len(x), len(y))
    return float(np.abs(log_mel(x[:n], sample_rate) - log_mel(y[:n], sample_rate)).mean() / math.log(10))


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------


def estimate_f0(
    audio: np.ndarray, sample_rate: int, fmin: float = 60.0, fmax: float = 600.0, threshold: float = 0.2
) -> np.ndarray:
    """[n_frames] pitch in Hz (0 = unvoiced or silent), YIN algorithm, 16 ms hop.

    YIN: for each lag, how different is the frame from itself shifted by that lag
    (cumulative-mean normalised); the first dip below `threshold` is the period.
    """
    frame = 2 ** math.ceil(math.log2(2.5 * sample_rate / fmin))  # >= 2 periods of the lowest pitch
    hop = 2 ** round(math.log2(0.064 * sample_rate)) // 4  # same hop as log_mel
    frames = frames_of(audio, frame, hop)
    frames = frames - frames.mean(axis=1, keepdims=True)
    half = frame // 2
    # Difference function via FFT autocorrelation: d(tau) = r_0 + r_tau_energy - 2 r(tau).
    spectrum = np.fft.rfft(frames, n=2 * frame, axis=1)
    acf = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, : half + 1]
    energy = np.cumsum(frames[:, ::-1] ** 2, axis=1)[:, ::-1]  # energy of x[tau:]
    diff = acf[:, :1] + energy[:, : half + 1] - 2 * acf
    diff[:, 0] = 0
    cmnd = diff[:, 1:] * np.arange(1, half + 1) / np.maximum(np.cumsum(diff[:, 1:], axis=1), 1e-12)

    lag_min, lag_max = int(sample_rate / fmax), min(half - 1, int(sample_rate / fmin))
    f0 = np.zeros(len(frames))
    voiced_energy = energy_mask(audio, sample_rate)[: len(frames)]
    for i, row in enumerate(cmnd):
        if i >= len(voiced_energy) or not voiced_energy[i]:
            continue
        below = np.nonzero(row[lag_min:lag_max] < threshold)[0]
        if len(below) == 0:
            continue
        lag = lag_min + below[0]
        while lag + 1 < lag_max and row[lag + 1] < row[lag]:  # walk down to the bottom of the dip
            lag += 1
        f0[i] = sample_rate / (lag + 1)  # row index 0 is lag 1
    return f0


def f0_metrics(x: np.ndarray, y: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Pitch error of x against the reference y: RMSE in cents and voicing agreement."""
    n = min(len(x), len(y))
    fx, fy = estimate_f0(x[:n], sample_rate), estimate_f0(y[:n], sample_rate)
    both = (fx > 0) & (fy > 0)
    cents = 1200 * np.log2(fx[both] / fy[both]) if both.any() else np.array([np.nan])
    return {
        "f0_rmse_cents": float(np.sqrt(np.mean(cents**2))),
        "voicing_agreement": float(np.mean((fx > 0) == (fy > 0))),
        "median_f0_hz": float(np.median(fx[fx > 0])) if (fx > 0).any() else 0.0,
    }


def compare(x: np.ndarray, y: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Every reference metric of x against y in one dict."""
    return {
        "mcd_db": mel_cepstral_distortion(x, y, sample_rate),
        **f0_metrics(x, y, sample_rate),
        "mel_l1": mel_l1(x, y, sample_rate),
    }


# ---------------------------------------------------------------------------
# DNSMOS: predicted listening-test score (optional)
# ---------------------------------------------------------------------------


class DNSMOS:
    """Microsoft DNSMOS P.835 (non-intrusive: no reference needed). Scores on a 1-5 scale.

    SIG = speech quality, BAK = background noise, OVRL = overall. The model wants
    9 s at 16 kHz; shorter clips are repeated to fill 9 s, as the official script does.
    """

    SAMPLE_RATE = 16_000
    INPUT_SAMPLES = 144_160  # 9.01 s
    # Polynomials mapping the raw network outputs to MOS (from the official dnsmos_local.py).
    POLY: ClassVar[dict[str, tuple[float, float, float]]] = {
        "sig": (-0.08397278, 1.22083953, 0.0052439),
        "bak": (-0.13166888, 1.60915514, -0.39604546),
        "ovrl": (-0.06766283, 1.11546468, 0.04602535),
    }

    def __init__(self, model_path: Path | None = None) -> None:
        import onnxruntime as ort  # optional dependency: pip install 'tiny-pairvc[onnx]'

        if model_path is None:
            model_path = Path.home() / ".cache" / "pairvc" / "dnsmos_sig_bak_ovr.onnx"
        model_path = Path(model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(DNSMOS_URL, model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

    def __call__(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        if sample_rate != self.SAMPLE_RATE:
            g = math.gcd(sample_rate, self.SAMPLE_RATE)
            audio = resample_poly(audio, self.SAMPLE_RATE // g, sample_rate // g)
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0 or not np.any(audio):
            return {"dnsmos_sig": 1.0, "dnsmos_bak": 1.0, "dnsmos_ovrl": 1.0}
        reps = math.ceil(self.INPUT_SAMPLES / len(audio))
        clip = np.tile(audio, reps)[: self.INPUT_SAMPLES][None]
        sig, bak, ovrl = self.session.run(None, {"input_1": clip})[0][0]
        scores = {}
        for name, raw in (("sig", sig), ("bak", bak), ("ovrl", ovrl)):
            a, b, c = self.POLY[name]
            scores[f"dnsmos_{name}"] = float(a * raw**2 + b * raw + c)
        return scores


__all__ = [
    "DNSMOS",
    "DNSMOS_URL",
    "compare",
    "energy_mask",
    "estimate_f0",
    "f0_metrics",
    "log_mel",
    "mel_cepstral_distortion",
    "mel_cepstrum",
    "mel_l1",
]
