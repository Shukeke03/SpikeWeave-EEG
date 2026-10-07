"""SpikeWeave-EEG; historical component imports remain available."""
from .denoiser import MCWave, WaveDiT
from .encoders import StructureEncoder, TemporalLeadEncoder, TextEncoder
from .fusion import ConditionFusion
from .model import SpikeWeaveModel, TriWaveModel
from .snn import SpikingEEGEncoder
from .vision_language import SharedVisionLanguageEncoder

__all__ = ["SpikeWeaveModel", "TriWaveModel", "SpikingEEGEncoder", "SharedVisionLanguageEncoder",
           "MCWave", "WaveDiT", "TemporalLeadEncoder", "StructureEncoder", "TextEncoder", "ConditionFusion"]
