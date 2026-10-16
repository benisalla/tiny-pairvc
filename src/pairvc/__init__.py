"""PairVC: a tiny causal voice converter trained on paired (teacher-converted) speech."""

from pairvc.config import DataConfig, DiscConfig, ModelConfig, TrainConfig
from pairvc.model import PairVC

__all__ = ["DataConfig", "DiscConfig", "ModelConfig", "PairVC", "TrainConfig"]
