"""PairVC: a small, fully causal waveform-to-waveform voice converter.

    source audio [B, T] --encoder--> latent frames [B, T/320, 64]
                       --decoder (+ target speaker)--> converted audio [B, T]

Every convolution is *causal*: output sample t only depends on input samples
<= t. That is what lets the same network run in real time, 20 ms at a time.

Each layer has two ways to run:

- `forward(x)`            the whole clip at once (training, offline conversion)
- `step(x, states, new)`  one chunk, carrying a small state between chunks
                          (streaming, ONNX export)

Both give the same numbers (tests/test_model.py checks it), so a model trained
with `forward` can be streamed without any conversion.

Streaming state convention: `init_state(batch, out)` appends this layer's
zero states to the list `out`; `step(x, states, new)` reads its states from the
iterator `states` in the same order and appends the updated ones to `new`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from pairvc.config import ModelConfig

States = Iterator[Tensor]  # states being read by step()
NewStates = list[Tensor]  # states being written by step()


# ---------------------------------------------------------------------------
# Causal building blocks
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Conv1d):
    """Conv1d that only looks at the past (left zero padding).

    With kernel k, dilation d and stride s, one output looks at
    `receptive = d * (k - 1) + 1` input samples. Padding the left by
    `receptive - s` makes an input of length T give exactly T / s outputs.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
        dilation: int = 1,
    ) -> None:  # fmt: skip
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.receptive = dilation * (kernel_size - 1) + 1
        self.left_pad = self.receptive - stride
        if self.left_pad < 0:
            raise ValueError("kernel too small for its stride: samples would be skipped")

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.left_pad, 0)))

    def init_state(self, batch_size: int, out: NewStates) -> None:
        # The last `left_pad` input samples not yet fully used; zeros = the left padding.
        out.append(torch.zeros(batch_size, self.in_channels, self.left_pad))

    def step(self, x: Tensor, states: States, new: NewStates) -> Tensor:
        x = torch.cat([next(states), x], dim=-1)
        stride = self.stride[0]
        n_out = (x.shape[-1] - self.receptive) // stride + 1
        used = n_out * stride  # where the next output window starts
        new.append(x[..., used:])
        return super().forward(x[..., : used - stride + self.receptive])


class CausalConvTranspose1d(nn.ConvTranspose1d):
    """Upsampling ConvTranspose1d trimmed so that output sample t never uses future input.

    Kernel 2s / stride s: T frames in, exactly T * s samples out.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride)
        self.trim = kernel_size - stride  # samples at the end that still wait for the next frame
        if self.trim < 0:
            raise ValueError("kernel_size must be >= stride")
        self.context = math.ceil(self.trim / stride)  # past frames a chunk needs to see

    def forward(self, x: Tensor) -> Tensor:
        y = super().forward(x)
        return y[..., : y.shape[-1] - self.trim]

    def init_state(self, batch_size: int, out: NewStates) -> None:
        out.append(torch.zeros(batch_size, self.in_channels, self.context))

    def step(self, x: Tensor, states: States, new: NewStates) -> Tensor:
        n_new = x.shape[-1]
        x = torch.cat([next(states), x], dim=-1)
        new.append(x[..., x.shape[-1] - self.context :])
        y = super().forward(x)
        start = self.context * self.stride[0]  # drop what the previous chunk already output
        return y[..., start : start + n_new * self.stride[0]]


class ResidualUnit(nn.Module):
    """x + ELU(conv1x1(ELU(dilated causal conv(x))))."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + F.elu(self.conv2(F.elu(self.conv1(x))))

    def init_state(self, batch_size: int, out: NewStates) -> None:
        self.conv1.init_state(batch_size, out)
        self.conv2.init_state(batch_size, out)

    def step(self, x: Tensor, states: States, new: NewStates) -> Tensor:
        y = F.elu(self.conv1.step(x, states, new))
        return x + F.elu(self.conv2.step(y, states, new))


class EncoderBlock(nn.Module):
    """Residual units at the current rate, then a strided conv (downsample by `stride`)."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, residual_units: int,
        kernel_size: int,
    ) -> None:  # fmt: skip
        super().__init__()
        self.residuals = nn.ModuleList(
            ResidualUnit(in_channels, dilation=3**i, kernel_size=kernel_size) for i in range(residual_units)
        )
        self.down = CausalConv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        for unit in self.residuals:
            x = unit(x)
        return F.elu(self.down(x))

    def init_state(self, batch_size: int, out: NewStates) -> None:
        for unit in self.residuals:
            unit.init_state(batch_size, out)
        self.down.init_state(batch_size, out)

    def step(self, x: Tensor, states: States, new: NewStates) -> Tensor:
        for unit in self.residuals:
            x = unit.step(x, states, new)
        return F.elu(self.down.step(x, states, new))


class DecoderBlock(nn.Module):
    """Transposed conv (upsample by `stride`), then residual units at the new rate."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, residual_units: int,
        kernel_size: int,
    ) -> None:  # fmt: skip
        super().__init__()
        self.up = CausalConvTranspose1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride)
        self.residuals = nn.ModuleList(
            ResidualUnit(out_channels, dilation=3**i, kernel_size=kernel_size) for i in range(residual_units)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.elu(self.up(x))
        for unit in self.residuals:
            x = unit(x)
        return x

    def init_state(self, batch_size: int, out: NewStates) -> None:
        self.up.init_state(batch_size, out)
        for unit in self.residuals:
            unit.init_state(batch_size, out)

    def step(self, x: Tensor, states: States, new: NewStates) -> Tensor:
        x = F.elu(self.up.step(x, states, new))
        for unit in self.residuals:
            x = unit.step(x, states, new)
        return x


class FiLM(nn.Module):
    """Feature-wise linear modulation: x * (1 + gamma(c)) + beta(c), per channel.

    Predicting 1 + gamma (not gamma) means a fresh layer starts as the identity
    instead of multiplying the signal by a random number.
    Paper: "FiLM: Visual Reasoning with a General Conditioning Layer".
    """

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta = nn.Linear(cond_dim, channels)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x [B, C, T], cond [B, cond_dim] -> gamma, beta [B, C, 1]
        gamma = self.to_gamma(cond).unsqueeze(-1)
        beta = self.to_beta(cond).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


def init_weights(module: nn.Module) -> None:
    """Small normal init for convs (as HiFi-GAN), Xavier for linear layers."""
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(module.weight, 0.0, 0.01)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class PairVC(nn.Module):
    """Any-to-N voice converter: source speech + target speaker id -> converted speech."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.grad_checkpointing = False  # set by train.py; saves memory, costs compute
        ch, k = config.channels, config.kernel_size
        widths = [ch * 2**i for i in range(len(config.strides) + 1)]  # 32, 64, 128, 256, 512

        # encoder: 1 -> 32 -> ... -> 512 channels, T -> T / hop frames
        self.enc_in = CausalConv1d(1, widths[0], k)
        self.enc_blocks = nn.ModuleList(
            EncoderBlock(widths[i], widths[i + 1], s, config.residual_units, k)
            for i, s in enumerate(config.strides)
        )
        self.enc_out = CausalConv1d(widths[-1], config.latent_dim, kernel_size=3)

        # decoder: mirror of the encoder, each block followed by speaker FiLM
        self.speaker_embedding = nn.Embedding(config.n_speakers, config.speaker_dim)
        self.dec_in = CausalConv1d(config.latent_dim, widths[-1], k)
        n = len(config.strides)
        self.dec_blocks = nn.ModuleList(
            DecoderBlock(widths[n - i], widths[n - i - 1], s, config.residual_units, k)
            for i, s in enumerate(reversed(config.strides))
        )
        self.films = nn.ModuleList(FiLM(widths[n - i - 1], config.speaker_dim) for i in range(n))
        self.dec_out = CausalConv1d(widths[0], 1, k)

        self.apply(init_weights)

    # --- whole clip -----------------------------------------------------------

    def _block(self, block: nn.Module, x: Tensor) -> Tensor:
        if self.grad_checkpointing and self.training:
            # Recompute the block's activations in backward instead of storing them.
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def encode(self, audio: Tensor) -> Tensor:
        """[B, T] audio -> [B, latent_dim, T / hop] latent frames."""
        x = F.elu(self.enc_in(audio.unsqueeze(1)))
        for block in self.enc_blocks:
            x = self._block(block, x)
        return self.enc_out(x)

    def decode(self, latent: Tensor, speaker: Tensor) -> Tensor:
        """[B, latent_dim, frames] + [B] speaker ids -> [B, frames * hop] audio."""
        cond = self.speaker_embedding(speaker)
        x = F.elu(self.dec_in(latent))
        for block, film in zip(self.dec_blocks, self.films):
            x = film(self._block(block, x), cond)
        return self.dec_out(x).squeeze(1)

    def forward(self, audio: Tensor, speaker: Tensor) -> Tensor:
        """Convert a batch of clips: [B, T] float audio, [B] long speaker ids -> [B, T].

        The output is *delayed* by `config.lookahead_samples` (see `convert`).
        T need not be a multiple of the hop: the end is zero-padded and cut off
        again (padding the end cannot change earlier outputs of a causal model).
        """
        length = audio.shape[-1]
        pad = -length % self.config.hop_length
        if pad:
            audio = F.pad(audio, (0, pad))
        return self.decode(self.encode(audio), speaker)[..., :length]

    @torch.inference_mode()
    def convert(self, audio: Tensor, speaker: Tensor) -> Tensor:
        """Offline conversion, time-aligned with the input: [B, T] -> [B, T].

        The model was trained to output sample t at position t + lookahead, so
        we feed `lookahead` extra zeros and drop the first `lookahead` outputs.
        """
        delay = self.config.lookahead_samples
        out = self(F.pad(audio, (0, delay)), speaker)
        return out[..., delay:]

    # --- streaming --------------------------------------------------------------

    def _stateful_layers(self) -> list[nn.Module]:
        """Every layer with a streaming state, in the order states are stored."""
        return [self.enc_in, *self.enc_blocks, self.enc_out, self.dec_in, *self.dec_blocks,
                self.dec_out]  # fmt: skip

    def init_state(self, batch_size: int = 1) -> list[Tensor]:
        """Zero streaming states: the model has heard nothing yet."""
        states: list[Tensor] = []
        for layer in self._stateful_layers():
            layer.init_state(batch_size, states)
        device = self.enc_in.weight.device
        return [s.to(device) for s in states]

    def stream_step(
        self, chunk: Tensor, speaker: Tensor, states: list[Tensor]
    ) -> tuple[Tensor, list[Tensor]]:
        """Convert one chunk: [B, n * hop] -> [B, n * hop], plus the next states.

        Feeding a clip chunk by chunk gives exactly `forward` on the whole clip
        (the output is delayed by `lookahead_samples`, as in `forward`).
        """
        if chunk.shape[-1] % self.config.hop_length:
            raise ValueError(f"chunk length must be a multiple of {self.config.hop_length}")
        it, new = iter(states), []
        x = F.elu(self.enc_in.step(chunk.unsqueeze(1), it, new))
        for block in self.enc_blocks:
            x = block.step(x, it, new)
        x = self.enc_out.step(x, it, new)

        cond = self.speaker_embedding(speaker)
        x = F.elu(self.dec_in.step(x, it, new))
        for block, film in zip(self.dec_blocks, self.films):
            x = film(block.step(x, it, new), cond)
        x = self.dec_out.step(x, it, new)
        return x.squeeze(1), new

    # --- info ---------------------------------------------------------------------

    def receptive_field(self) -> int:
        """How many input samples can influence one output sample (encoder + decoder).

        Measured, not derived: the gradient of the last output sample is non-zero
        exactly on the input samples it depends on.
        """
        audio = torch.randn(1, self.config.hop_length * 128, requires_grad=True)
        was_training = self.training
        self.eval()
        with torch.enable_grad():
            device = self.enc_in.weight.device
            out = self(audio.to(device), torch.zeros(1, dtype=torch.long, device=device))
            out[0, -1].backward()
        self.train(was_training)
        self.zero_grad(set_to_none=True)
        used = (audio.grad[0] != 0).nonzero()
        return int(audio.shape[-1] - used.min())


__all__ = ["PairVC", "init_weights"]
