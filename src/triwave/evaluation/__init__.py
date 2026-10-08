"""Development-first EEG evaluation, explicit imports and locked-test gates."""

from .benchmark import G1Config, SmallEEGCNN, run_g0, run_g1
from .data import EvaluationData, evaluation_data_from_dataset, load_evaluation_data
from .gates import GateLedger, authorize_g3, record_g2
from .metrics import average_precision, patient_average_precision

__all__ = ["EvaluationData", "G1Config", "GateLedger", "SmallEEGCNN", "run_g0", "run_g1",
           "load_evaluation_data", "evaluation_data_from_dataset", "record_g2", "authorize_g3",
           "average_precision", "patient_average_precision"]
