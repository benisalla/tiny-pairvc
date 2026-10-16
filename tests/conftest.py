"""Shared helpers: tiny configs and a tiny synthetic paired dataset (CPU, offline)."""

from pathlib import Path

import pytest

from pairvc.config import DataConfig, DiscConfig, ModelConfig, TrainConfig
from pairvc.data import write_toy_dataset

SAMPLE_RATE = 16_000
SEGMENT = 3200  # 0.2 s: > 2 x 1024 so every mel-loss scale works after the lookahead cut


def tiny_model_config(**overrides) -> ModelConfig:
    """Same architecture, much smaller: hop 8 samples, 4 channels."""
    values = {"channels": 4, "latent_dim": 8, "speaker_dim": 8, "strides": (2, 4),
              "residual_units": 2, "lookahead_frames": 2}  # fmt: skip
    return ModelConfig(**{**values, **overrides})


def tiny_train_config(output_dir: Path, **overrides) -> TrainConfig:
    values = {"output_dir": output_dir, "max_steps": 4, "batch_size": 2, "num_workers": 0,
              "log_every": 1, "eval_every": 2, "device": "cpu", "precision": "fp32"}  # fmt: skip
    return TrainConfig(**{**values, **overrides})


def tiny_disc_config() -> DiscConfig:
    return DiscConfig(kinds=("resolution",))  # the smallest one: fast tests


@pytest.fixture(scope="session")
def toy_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """8 train + 2 val synthetic sources, each converted to 2 voices."""
    return write_toy_dataset(tmp_path_factory.mktemp("toy"), n_clips=8, seconds=0.25)


@pytest.fixture
def data_config(toy_data: Path) -> DataConfig:
    return DataConfig(data_dir=toy_data, segment_samples=SEGMENT)
