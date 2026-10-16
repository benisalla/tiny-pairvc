"""All hyperparameters live here, grouped by what they control.

Configs are frozen dataclasses: they cannot be changed by accident after
creation, and they are saved inside every checkpoint so a run can be reproduced
and a trained model can be rebuilt without remembering any flag.
"""

from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path
from typing import Any

DISCRIMINATOR_KINDS = ("time", "period", "resolution", "band")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the paired clips are and how they are cut into training examples.

    Layout expected under `data_dir` (see docs/data.md):
        <data_dir>/<split>/<id>_original.wav      source speech (any speaker)
        <data_dir>/<split>/<id>_converted_<k>.wav  same speech in target voice k
    """

    data_dir: Path = Path("data")
    train_split: str = "train"
    val_split: str = "val"
    sample_rate: int = 16_000
    # Length of one training example: 16_000 = 1 s at 16 kHz. Must be a multiple of
    # the model hop (320). Training memory grows with batch_size x segment length:
    # ~0.8 GB of activations per second of audio in the batch (see docs/training.md).
    # The generator is fully convolutional, so it converts clips of any length later.
    segment_samples: int = 16_000

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.segment_samples <= 0:
            raise ValueError("segment_samples must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape of the generator (the voice converter itself)."""

    n_speakers: int = 2  # number of target voices
    channels: int = 32  # width of the first layer; doubles after every stride
    latent_dim: int = 64  # size of one latent frame between encoder and decoder
    speaker_dim: int = 64  # size of the target-speaker embedding
    strides: tuple[int, ...] = (2, 4, 5, 8)  # product = hop = samples per latent frame
    residual_units: int = 3  # residual units per block (dilations 1, 3, 9)
    kernel_size: int = 7
    # The output is compared with the target `lookahead_frames` frames *earlier*:
    # the model may hear a little future before it speaks (2 x 20 ms = 40 ms).
    # This is extra latency in streaming, traded for quality.
    lookahead_frames: int = 2

    def __post_init__(self) -> None:
        if self.n_speakers <= 0:
            raise ValueError("n_speakers must be positive")
        if not self.strides or any(s <= 0 for s in self.strides):
            raise ValueError("strides must be a non-empty tuple of positive ints")
        if self.lookahead_frames < 0:
            raise ValueError("lookahead_frames must be >= 0")

    @property
    def hop_length(self) -> int:
        """Audio samples per latent frame (320 = 20 ms at 16 kHz by default)."""
        return prod(self.strides)

    @property
    def lookahead_samples(self) -> int:
        return self.lookahead_frames * self.hop_length


@dataclass(frozen=True, slots=True)
class DiscConfig:
    """Which discriminators judge the generated audio, and their sizes.

    "time":       multi-scale waveform discriminator (HiFi-GAN MSD)
    "period":     multi-period discriminator (HiFi-GAN MPD)
    "resolution": multi-resolution STFT magnitude discriminator (UnivNet MRD)
    "band":       multi-band complex-STFT discriminator (DAC MBD)
    """

    kinds: tuple[str, ...] = ("time",)
    lrelu_slope: float = 0.1
    # time
    time_scales: int = 3
    # period
    periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    period_kernel_size: int = 5
    period_stride: int = 3
    # resolution: (n_fft, hop_length, win_length)
    resolutions: tuple[tuple[int, int, int], ...] = (
        (1024, 90, 450),  # ~28 ms window, 5.6 ms hop
        (2048, 180, 900),  # ~56 ms window, 11 ms hop
        (512, 36, 180),  # ~11 ms window, 2.3 ms hop
    )
    resolution_channels: int = 32
    # band
    band_fft_sizes: tuple[int, ...] = (2048, 1024, 512)
    band_channels: int = 32
    band_edges: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)  # fractions of Nyquist

    def __post_init__(self) -> None:
        unknown = set(self.kinds) - set(DISCRIMINATOR_KINDS)
        if not self.kinds or unknown:
            raise ValueError(f"kinds must be a non-empty subset of {DISCRIMINATOR_KINDS}")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation and bookkeeping settings for one training run."""

    output_dir: Path = Path("checkpoints/run")
    seed: int = 42
    max_steps: int = 500_000  # GAN vocoders need many steps; stop earlier any time
    batch_size: int = 16  # per process (per GPU)
    g_lr: float = 1e-4  # generator learning rate
    d_lr: float = 1e-4  # discriminator learning rate
    betas: tuple[float, float] = (0.8, 0.99)  # HiFi-GAN values
    weight_decay: float = 0.0
    lr_decay: float = 0.9999996  # multiply both LRs by this after every step
    grad_clip: float = 0.0  # 0 = no clipping
    lambda_adv: float = 1.0  # adversarial loss weight
    lambda_fm: float = 2.0  # feature-matching loss weight
    lambda_mel: float = 15.0  # multi-scale mel loss weight
    precision: str = "auto"  # "auto" | "fp32" | "bf16"
    grad_checkpointing: bool = False  # less GPU memory, ~30% slower
    num_workers: int = 4
    train_limit: int | None = None  # use only N training pairs (tiny overfit runs)
    val_limit: int | None = None
    log_every: int = 50  # steps between log lines
    eval_every: int = 1000  # steps between validations (+ checkpoint + audio sample)
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    resume: Path | None = None  # checkpoint to continue from

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.precision not in {"auto", "fp32", "bf16"}:
            # fp16 is left out on purpose: GAN losses overflow it easily.
            raise ValueError("precision must be auto, fp32 or bf16")
        if not 0.0 < self.lr_decay <= 1.0:
            raise ValueError("lr_decay must be in (0, 1]")


def config_to_dict(config: Any) -> dict[str, Any]:
    """Turn a config into plain types so it can be stored in a checkpoint."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}


__all__ = [
    "DISCRIMINATOR_KINDS",
    "DataConfig",
    "DiscConfig",
    "ModelConfig",
    "TrainConfig",
    "config_to_dict",
]
