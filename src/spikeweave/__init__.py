"""Canonical public API; implementation namespace triwave remains compatible."""
from triwave.config import ModelConfig
from triwave.models import SpikeWeaveModel
from triwave.training import Trainer, TrainingConfig
from triwave.types import ConditionBatch

__all__ = ["SpikeWeaveModel", "ModelConfig", "Trainer", "TrainingConfig", "ConditionBatch"]
