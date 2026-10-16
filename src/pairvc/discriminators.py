"""Discriminators: small networks that learn to tell teacher audio from PairVC audio.

They are only used for training (the GAN part). Each one looks at the audio in
a different way, so the generator cannot fool them all with the same artifact:

- MultiScaleDiscriminator      raw waveform at 1x, 1/2x, 1/4x rate  (HiFi-GAN)
- MultiPeriodDiscriminator     waveform folded by periods 2, 3, 5, 7, 11  (HiFi-GAN)
- MultiResolutionDiscriminator STFT magnitude at 3 resolutions  (UnivNet)
- MultiBandDiscriminator       complex STFT split into frequency bands  (DAC)

Every discriminator returns, for real audio y and generated audio y_hat:
    real_scores, fake_scores: one score map per sub-discriminator
    real_fmaps,  fake_fmaps:  the hidden activations, for feature matching
"""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from pairvc.config import DiscConfig

Scores = list[Tensor]
FeatureMaps = list[list[Tensor]]
DiscOutput = tuple[Scores, Scores, FeatureMaps, FeatureMaps]


def stft_fp32(x: Tensor, n_fft: int, hop: int, win: int, center: bool = True) -> Tensor:
    """Complex STFT [B, F, frames] computed in fp32 even under bf16 autocast."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        window = torch.hann_window(win, device=x.device)
        return torch.stft(
            x.float(), n_fft, hop_length=hop, win_length=win, window=window,
            center=center, return_complex=True,
        )  # fmt: skip


class _PairwiseDiscriminator(nn.Module):
    """Runs every sub-discriminator on real and fake audio and collects the outputs."""

    discriminators: nn.ModuleList

    def forward(self, y: Tensor, y_hat: Tensor) -> DiscOutput:
        real_scores, fake_scores, real_fmaps, fake_fmaps = [], [], [], []
        for d in self.discriminators:
            score, fmap = d(y)
            real_scores.append(score)
            real_fmaps.append(fmap)
            score, fmap = d(y_hat)
            fake_scores.append(score)
            fake_fmaps.append(fmap)
        return real_scores, fake_scores, real_fmaps, fake_fmaps


# ---------------------------------------------------------------------------
# Waveform discriminators
# ---------------------------------------------------------------------------


class ScaleDiscriminator(nn.Module):
    """Strided grouped 1D convs over the raw waveform [B, 1, T]."""

    def __init__(self, lrelu_slope: float, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])  # fmt: skip
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiScaleDiscriminator(_PairwiseDiscriminator):
    """ScaleDiscriminators on the waveform average-pooled 1x, 2x, 4x, ..."""

    def __init__(self, config: DiscConfig) -> None:
        super().__init__()
        # HiFi-GAN: spectral norm on the full-rate one, weight norm on the others.
        self.discriminators = nn.ModuleList(
            ScaleDiscriminator(config.lrelu_slope, use_spectral_norm=i == 0)
            for i in range(config.time_scales)
        )
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, y: Tensor, y_hat: Tensor) -> DiscOutput:
        real_scores, fake_scores, real_fmaps, fake_fmaps = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                y, y_hat = self.pool(y), self.pool(y_hat)
            for audio, scores, fmaps in ((y, real_scores, real_fmaps), (y_hat, fake_scores, fake_fmaps)):
                score, fmap = d(audio)
                scores.append(score)
                fmaps.append(fmap)
        return real_scores, fake_scores, real_fmaps, fake_fmaps


class PeriodDiscriminator(nn.Module):
    """Folds [B, 1, T] into [B, 1, T/p, p] and applies (k x 1) 2D convs."""

    def __init__(self, period: int, kernel_size: int, stride: int, lrelu_slope: float) -> None:
        super().__init__()
        self.period = period
        self.lrelu_slope = lrelu_slope
        pad = (kernel_size - 1) // 2
        channels = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(
                channels[i], channels[i + 1], (kernel_size, 1),
                (stride if i < 4 else 1, 1), padding=(pad, 0),
            ))
            for i in range(5)
        )  # fmt: skip
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        x = x.view(b, c, -1, self.period)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiPeriodDiscriminator(_PairwiseDiscriminator):
    def __init__(self, config: DiscConfig) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            PeriodDiscriminator(p, config.period_kernel_size, config.period_stride, config.lrelu_slope)
            for p in config.periods
        )


# ---------------------------------------------------------------------------
# Spectrogram discriminators
# ---------------------------------------------------------------------------


class ResolutionDiscriminator(nn.Module):
    """2D convs over one STFT magnitude [B, 1, F, frames]."""

    def __init__(self, resolution: tuple[int, int, int], channels: int, lrelu_slope: float) -> None:
        super().__init__()
        self.n_fft, self.hop, self.win = resolution
        self.lrelu_slope = lrelu_slope
        c = channels
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, c, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(c, c, (3, 3), padding=(1, 1))),
        ])  # fmt: skip
        self.conv_post = weight_norm(nn.Conv2d(c, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        pad = (self.n_fft - self.hop) // 2
        x = F.pad(x, (pad, pad), mode="reflect").squeeze(1)
        # center=False: we already padded (padding twice shifts the frames).
        x = stft_fp32(x, self.n_fft, self.hop, self.win, center=False).abs().unsqueeze(1)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiResolutionDiscriminator(_PairwiseDiscriminator):
    def __init__(self, config: DiscConfig) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            ResolutionDiscriminator(r, config.resolution_channels, config.lrelu_slope)
            for r in config.resolutions
        )


class BandDiscriminator(nn.Module):
    """Complex STFT (real + imag as 2 channels) split into frequency bands, one conv stack each."""

    def __init__(self, n_fft: int, config: DiscConfig) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.lrelu_slope = config.lrelu_slope
        n_bins = n_fft // 2 + 1
        edges = [int(e * n_bins) for e in config.band_edges]
        self.bands = list(pairwise(edges))
        c = config.band_channels

        def stack() -> nn.ModuleList:
            return nn.ModuleList([
                weight_norm(nn.Conv2d(2, c, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(c, c, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(c, c, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(c, c, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(c, c, (3, 3), (1, 1), padding=(1, 1))),
            ])  # fmt: skip

        self.band_convs = nn.ModuleList(stack() for _ in self.bands)
        self.conv_post = weight_norm(nn.Conv2d(c, 1, (3, 3), (1, 1), padding=(1, 1)))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = x.squeeze(1)
        x = x - x.mean(dim=-1, keepdim=True)  # remove DC offset
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True).values + 1e-9)  # peak-normalise
        spec = stft_fp32(x, self.n_fft, self.n_fft // 4, self.n_fft)
        spec = torch.view_as_real(spec).permute(0, 3, 2, 1)  # [B, 2, frames, F]
        fmap, outs = [], []
        for (lo, hi), convs in zip(self.bands, self.band_convs):
            band = spec[..., lo:hi]
            for i, conv in enumerate(convs):
                band = F.leaky_relu(conv(band), self.lrelu_slope)
                if i > 0:
                    fmap.append(band)
            outs.append(band)
        x = self.conv_post(torch.cat(outs, dim=-1))
        fmap.append(x)
        return x.flatten(1), fmap


class MultiBandDiscriminator(_PairwiseDiscriminator):
    def __init__(self, config: DiscConfig) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            BandDiscriminator(n_fft, config) for n_fft in config.band_fft_sizes
        )


# ---------------------------------------------------------------------------
# All of them together
# ---------------------------------------------------------------------------

_KINDS = {
    "time": MultiScaleDiscriminator,
    "period": MultiPeriodDiscriminator,
    "resolution": MultiResolutionDiscriminator,
    "band": MultiBandDiscriminator,
}


class Discriminators(nn.Module):
    """The discriminators named in `config.kinds`, run together; outputs are concatenated."""

    def __init__(self, config: DiscConfig) -> None:
        super().__init__()
        self.config = config
        self.discriminators = nn.ModuleList(_KINDS[kind](config) for kind in config.kinds)

    def forward(self, y: Tensor, y_hat: Tensor) -> DiscOutput:
        """y (target) and y_hat (generated): [B, T] or [B, 1, T]."""
        if y.dim() == 2:
            y = y.unsqueeze(1)
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(1)
        real_scores, fake_scores, real_fmaps, fake_fmaps = [], [], [], []
        for d in self.discriminators:
            rs, fs, rf, ff = d(y, y_hat)
            real_scores += rs
            fake_scores += fs
            real_fmaps += rf
            fake_fmaps += ff
        return real_scores, fake_scores, real_fmaps, fake_fmaps


__all__ = ["Discriminators"]
