"""Tests for pairvc.losses and pairvc.discriminators."""

import pytest
import torch

from pairvc.config import DiscConfig
from pairvc.discriminators import Discriminators
from pairvc.losses import (
    MultiScaleMelLoss,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    mel_filterbank,
)


def test_mel_filterbank_shape_and_order() -> None:
    fb = mel_filterbank(16_000, 1024, 80)
    assert fb.shape == (80, 513)
    assert (fb >= 0).all()
    peaks = fb.argmax(dim=1)
    assert (peaks[1:] >= peaks[:-1]).all()  # filters go from low to high frequencies


def test_mel_loss_is_zero_for_identical_audio_and_grows_with_noise() -> None:
    loss = MultiScaleMelLoss(16_000)
    x = torch.randn(2, 4000)
    assert loss(x, x).item() == pytest.approx(0.0, abs=1e-6)
    small = loss(x + 0.01 * torch.randn_like(x), x)
    big = loss(x + 0.5 * torch.randn_like(x), x)
    assert 0 < small < big


def test_gan_losses_least_squares_values() -> None:
    ones, zeros = [torch.ones(2, 5)], [torch.zeros(2, 5)]
    loss, real, fake = discriminator_loss(ones, zeros)  # perfect discriminator
    assert loss.item() == real.item() == fake.item() == 0.0
    assert generator_adversarial_loss(zeros).item() == 1.0  # fooled nobody
    assert generator_adversarial_loss(ones).item() == 0.0
    fmap = [[torch.ones(3), torch.ones(3), torch.zeros(1)]]
    assert feature_matching_loss(fmap, fmap).item() == 0.0


@pytest.mark.parametrize("kind", ["time", "period", "resolution", "band"])
def test_every_discriminator_scores_real_and_fake(kind: str) -> None:
    disc = Discriminators(DiscConfig(kinds=(kind,)))
    y, y_hat = torch.randn(2, 4000), torch.randn(2, 4000, requires_grad=True)
    real_scores, fake_scores, real_fmaps, fake_fmaps = disc(y, y_hat)
    assert len(real_scores) == len(fake_scores) == len(real_fmaps) == len(fake_fmaps) > 0
    assert all(s.shape[0] == 2 for s in real_scores + fake_scores)
    generator_adversarial_loss(fake_scores).backward()
    assert y_hat.grad is not None and y_hat.grad.abs().sum() > 0  # gradient reaches the audio
