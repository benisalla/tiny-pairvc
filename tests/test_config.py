"""Invalid settings are rejected when the config is built, not hours into training."""

import pytest

from pairvc.config import DataConfig, DiscConfig, ModelConfig, TrainConfig, config_to_dict


def test_defaults_are_valid_and_hop_is_20ms() -> None:
    model = ModelConfig()
    assert model.hop_length == 320  # 20 ms at 16 kHz
    assert model.lookahead_samples == 640
    assert DataConfig().segment_samples % model.hop_length == 0
    DiscConfig()
    TrainConfig()


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (DataConfig, {"sample_rate": 0}),
        (DataConfig, {"segment_samples": -1}),
        (ModelConfig, {"n_speakers": 0}),
        (ModelConfig, {"strides": ()}),
        (ModelConfig, {"lookahead_frames": -1}),
        (DiscConfig, {"kinds": ("cqt",)}),
        (DiscConfig, {"kinds": ()}),
        (TrainConfig, {"precision": "fp16"}),
        (TrainConfig, {"batch_size": 0}),
        (TrainConfig, {"lr_decay": 1.5}),
    ],
)
def test_invalid_values_are_rejected(cls, kwargs) -> None:
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_config_to_dict_round_trips() -> None:
    config = DataConfig()
    values = config_to_dict(config)
    assert isinstance(values["data_dir"], str)  # plain types: safe in a checkpoint
    assert DataConfig(**values).sample_rate == config.sample_rate
    assert ModelConfig(**config_to_dict(ModelConfig())) == ModelConfig()
