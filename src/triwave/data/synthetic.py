"""Deterministic engineering smoke data; these are not medical examples or labels."""

from __future__ import annotations

import math

import torch

from triwave.config import ModelConfig

from .dataset import tokenize_texts


def make_synthetic_batch(config: ModelConfig, batch_size: int, seed: int = 0) -> dict:
    config.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    time = torch.arange(config.samples) / config.sample_rate
    classes = torch.arange(batch_size) % 3
    frequencies = torch.tensor([6.0, 10.0, 18.0])[classes]
    frequencies = frequencies.clamp(max=config.sample_rate / 4)
    names = ("engineering low-frequency sinusoid", "engineering medium-frequency sinusoid",
             "engineering higher-frequency sinusoid")
    target_phase = torch.rand(batch_size, config.channels, 1, generator=generator) * (2 * math.pi)
    reference_phase = torch.rand(batch_size, config.channels, 1, generator=generator) * (2 * math.pi)
    amplitude = 0.7 + 0.6 * torch.rand(batch_size, config.channels, 1, generator=generator)
    angular = 2 * math.pi * frequencies[:, None, None] * time[None, None, :]
    target = amplitude * torch.sin(angular + target_phase)
    reference = amplitude * torch.sin(angular + reference_phase)
    target += 0.08 * torch.randn(target.shape, generator=generator)
    reference += 0.08 * torch.randn(reference.shape, generator=generator)
    angle = torch.arange(config.channels) * (2 * math.pi / config.channels)
    positive = torch.stack([0.08 * angle.cos(), 0.08 * angle.sin(), torch.full_like(angle, 0.05)], dim=-1)
    negative = torch.roll(positive, shifts=1, dims=0)
    geometry = torch.cat([positive, negative], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)
    ids, masks = tokenize_texts([names[int(index)] for index in classes], config)
    attributes = torch.zeros(batch_size, config.attribute_dim)
    attributes_valid = torch.zeros_like(attributes, dtype=torch.bool)
    if config.attribute_dim:
        attributes[:, 0] = frequencies
        attributes_valid[:, 0] = True
    return {"target_waveform": target, "target_valid": torch.ones(batch_size, config.channels, dtype=torch.bool),
            "reference_waveform": reference,
            "reference_valid": torch.ones(batch_size, config.channels, dtype=torch.bool),
            "lead_geometry": geometry, "input_ids": ids, "text_valid": masks,
            "attribute_values": attributes, "attribute_valid": attributes_valid, "class_ids": classes,
            "alignment_metadata": [
                {"instance_id": f"engineering:{seed}:{i}", "parent_id": f"engineering-parent:{seed}:{i}",
                 "patient_id": f"engineering-only:{i}", "event_ids": [],
                 "semantic_group": str(int(classes[i])), "namespace": "engineering.sinusoid_band",
                 "scope": "engineering_window", "semantic_reliable": True, "negative_eligible": True}
                for i in range(batch_size)],
            "metadata": {"synthetic": True, "label_namespace": "engineering.sinusoid_band",
                         "unit": "engineering_arbitrary", "geometry_source": "synthetic circle endpoints",
                         "reference_source": "independent phase and noise; shared user-specified frequency",
                         "seed": seed}}
