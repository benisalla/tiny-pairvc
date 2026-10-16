"""Training losses.

Generator loss = lambda_adv * adversarial + lambda_fm * feature matching + lambda_mel * mel

- mel (the main signal): L1 between log-mel spectrograms of the generated and
  the teacher audio, at 7 resolutions (5 mels / 32-sample window ... 320 mels /
  2048-sample window), as in DAC. On its own it gives intelligible but buzzy audio.
- adversarial + feature matching (least-squares GAN, as HiFi-GAN): make the
  audio sound natural; feature matching = L1 between the discriminators'
  hidden activations on real and generated audio.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# GAN losses (least squares)
# ---------------------------------------------------------------------------


def discriminator_loss(real_scores: list[Tensor], fake_scores: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """D wants real -> 1 and fake -> 0. Returns (loss, mean real term, mean fake term)."""
    real = torch.stack([torch.mean((1 - r.float()) ** 2) for r in real_scores]).mean()
    fake = torch.stack([torch.mean(f.float() ** 2) for f in fake_scores]).mean()
    return real + fake, real.detach(), fake.detach()


def generator_adversarial_loss(fake_scores: list[Tensor]) -> Tensor:
    """G wants the discriminators to score its audio as real (1)."""
    return torch.stack([torch.mean((1 - f.float()) ** 2) for f in fake_scores]).mean()


def feature_matching_loss(real_fmaps: list[list[Tensor]], fake_fmaps: list[list[Tensor]]) -> Tensor:
    """Mean L1 between hidden activations; the final score layer of each D is left out."""
    losses = [
        torch.mean(torch.abs(f.float() - r.detach().float()))
        for real, fake in zip(real_fmaps, fake_fmaps, strict=True)
        for r, f in zip(real[:-1], fake[:-1], strict=True)
    ]
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Multi-scale mel loss
# ---------------------------------------------------------------------------


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney mel scale (librosa's default): linear below 1 kHz, log above."""
    hz = np.asarray(hz, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
    return np.where(
        hz < min_log_hz, hz / f_sp, min_log_mel + np.log(np.maximum(hz, 1e-10) / min_log_hz) / logstep
    )


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
    return np.where(mel < min_log_mel, mel * f_sp, min_log_hz * np.exp(logstep * (mel - min_log_mel)))


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> Tensor:
    """[n_mels, n_fft // 2 + 1] triangular filters, Slaney-normalised (same as librosa.filters.mel)."""
    freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2), n_mels + 2))
    widths = np.diff(edges)
    ramps = edges[:, None] - freqs[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (edges[2:] - edges[:-2]))[:, None]  # equal area per filter
    return torch.from_numpy(weights).float()


class MultiScaleMelLoss(nn.Module):
    """Sum over the 7 scales of L1(log10 mel(x), log10 mel(y))."""

    def __init__(
        self,
        sample_rate: int,
        n_mels: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320),
        window_lengths: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048),
        clamp_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.window_lengths = window_lengths
        self.clamp_eps = clamp_eps
        for i, (mels, win) in enumerate(zip(n_mels, window_lengths, strict=True)):
            # Not saved in checkpoints (persistent=False): they are rebuilt from the numbers above.
            self.register_buffer(f"window_{i}", torch.hann_window(win), persistent=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # a few top-resolution filters may be empty: harmless
                self.register_buffer(f"mel_{i}", mel_filterbank(sample_rate, win, mels), persistent=False)

    def log_mel(self, audio: Tensor, i: int) -> Tensor:
        win = self.window_lengths[i]
        spec = torch.stft(
            audio.float(), win, hop_length=win // 4, window=getattr(self, f"window_{i}"),
            center=True, pad_mode="reflect", return_complex=True,
        ).abs()  # fmt: skip
        mel = getattr(self, f"mel_{i}") @ spec  # [B, n_mels, frames]
        return torch.log10(mel.clamp(min=self.clamp_eps))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """x, y: [B, T] audio (T > 1024 for the largest window)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            losses = [
                F.l1_loss(self.log_mel(x, i), self.log_mel(y, i)) for i in range(len(self.window_lengths))
            ]
        return torch.stack(losses).sum()


__all__ = [
    "MultiScaleMelLoss",
    "discriminator_loss",
    "feature_matching_loss",
    "generator_adversarial_loss",
    "mel_filterbank",
]
