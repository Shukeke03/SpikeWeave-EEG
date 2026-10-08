"""Condition provenance/visible information tests, not clinical performance evidence."""

from dataclasses import replace

import pytest
import torch

from triwave.attributes import ATTRIBUTE_SCHEMA, waveform_attributes
from triwave.config import ModelConfig
from triwave.data.synthetic import make_synthetic_batch
from triwave.losses import attribute_control_loss
from triwave.signal import CWT_MASK_PROTOCOL, strict_patch_valid
from triwave.training import Trainer, TrainingConfig
from triwave.types import ConditionBatch


def declared_batch(trainer, n=3):
    batch = make_synthetic_batch(trainer.config, n, seed=7)
    batch["reference_provenance"] = {"origin_id": "independent-engineering-reference",
                                    "target_derived": False, "inference_available": True,
                                    "protocol": "source-only"}
    batch["attribute_origin"] = {"origin_id": "engineering-user-request",
                                "target_derived": False, "inference_available": True}
    return batch


def observed_batch(trainer):
    batch = declared_batch(trainer)
    batch["attribute_valid"].zero_()
    batch.pop("condition_attributes", None)
    batch.pop("condition_attribute_valid", None)
    image, available = trainer.cwt(batch["target_waveform"], batch["target_valid"])
    patch_valid = strict_patch_valid(available, trainer.config.visual_patch)
    selected = torch.zeros_like(patch_valid)
    for i in range(len(image)):
        index = patch_valid[i].flatten().nonzero()[0, 0]
        selected[i].flatten()[index] = True
    visible = selected.repeat_interleave(trainer.config.visual_patch, -2).repeat_interleave(trainer.config.visual_patch, -1)
    batch.update(structure=image, structure_valid=available, structure_visible=visible,
                 structure_role="partial_observation", cwt_mask_protocol=CWT_MASK_PROTOCOL,
                 structure_provenance={"origin_id": "engineering-partial-view", "target_derived": True,
                                       "inference_available": True, "protocol": "observation-conditioned"})
    return batch


def test_partial_observation_hides_values_and_full_target_is_rejected():
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig())
    batch = observed_batch(trainer)
    first = trainer.conditions(batch)
    assert torch.equal(first.structure_valid, batch["structure_visible"])
    assert torch.equal(first.structure_available_valid, batch["structure_valid"])
    batch["structure"][~batch["structure_visible"]] = float("nan")
    second = trainer.conditions(batch)
    torch.testing.assert_close(first.structure, second.structure)
    assert (second.structure[~second.structure_valid] == 0).all()
    control, mask = trainer.structure_control(second)
    assert control is not None and torch.equal(mask, batch["structure_visible"])
    batch["structure_visible"] = batch["structure_valid"].clone()
    with pytest.raises(ValueError, match="full target"):
        trainer.conditions(batch)
    batch["structure_role"] = "target_teacher"
    with pytest.raises(ValueError, match="teachers are forbidden"):
        trainer.conditions(batch)


def test_external_reference_is_never_pointwise_truth_and_dropped_partial_has_no_loss():
    torch.set_num_threads(2)
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig(warmup_steps=0))
    reference = declared_batch(trainer)
    reference["attribute_valid"].zero_()
    reference.pop("condition_attributes", None)
    reference.pop("condition_attribute_valid", None)
    trainer._drop_conditions = lambda condition: condition
    condition = trainer.conditions(reference)
    assert condition.structure_role == ("reference_condition",) * 3
    assert trainer.structure_control(condition) == (None, None)
    assert trainer.train_step(reference)["constraint_control"] == 0
    partial = observed_batch(trainer)
    kept = trainer.train_step(partial)
    assert kept["provided_structure_control_bins"] > 0 and kept["constraint_control"] > 0
    trainer._drop_conditions = lambda condition: replace(
        condition, structure_valid=torch.zeros_like(condition.structure_valid),
        structure_visible=torch.zeros_like(condition.structure_visible)
    )
    dropped = trainer.train_step(partial)
    assert dropped["provided_structure_control_bins"] == 0 and dropped["constraint_control"] == 0


def test_attribute_loss_scores_only_post_dropout_given_controls():
    torch.set_num_threads(2)
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig(warmup_steps=0))
    batch = declared_batch(trainer)
    trainer._drop_conditions = lambda condition: condition
    assert trainer.train_step(batch)["constraint_control"] > 0
    trainer._drop_conditions = lambda condition: replace(
        condition, attribute_valid=torch.zeros_like(condition.attribute_valid)
    )
    result = trainer.train_step(batch)
    assert result["constraint_control"] == 0 and result["provided_attribute_count"] == 0


def test_source_teacher_attributes_are_explicit_training_inputs_not_query_inputs():
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig())
    batch = declared_batch(trainer)
    values, valid = waveform_attributes(batch["target_waveform"], batch["target_valid"], trainer.config.sample_rate)
    batch["attribute_valid"].zero_()
    batch.pop("condition_attributes", None)
    batch.pop("condition_attribute_valid", None)
    origin = {"origin_id": "source-training-window", "target_derived": True, "source_role": "train",
              "inference_available": False}
    batch.update(teacher_attributes=values, teacher_attribute_valid=valid, teacher_attribute_origin=origin)
    external = trainer.conditions(batch)
    assert not external.attribute_valid.any()
    training = trainer.conditions(batch, allow_teacher_attributes=True)
    assert training.attribute_valid.any()
    with pytest.raises(ValueError, match="teacher attributes cannot be deployed"):
        trainer.sample(training, steps=1)
    batch["metadata"] = [{"split": "validation", "role": "query"}] * 3
    with pytest.raises(ValueError, match="query/support"):
        trainer.conditions(batch, allow_teacher_attributes=True)
    with pytest.raises(ValueError, match="source/train targets"):
        trainer.train_step(batch)


def test_sparse_attributes_follow_nonnull_half_empty_uniform_one_to_four_recipe():
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig())
    n = 4096
    condition = ConditionBatch(
        n, torch.zeros(n, 4, 6), attribute_values=torch.ones(n, 8),
        attribute_valid=torch.ones(n, 8, dtype=torch.bool),
        attribute_origin={"origin_id": "user", "target_derived": False, "inference_available": True},
    )
    torch.manual_seed(42)
    state = torch.get_rng_state()
    dropped = trainer._drop_conditions(condition)
    counts = dropped.attribute_valid.sum(-1)
    torch.set_rng_state(state)
    nonnull = torch.rand(n) < 0.9
    assert (counts[~nonnull] == 0).all()
    assert abs(float((counts[nonnull] > 0).float().mean()) - .5) < .04
    positive = counts[counts > 0]
    assert positive.min() == 1 and positive.max() == 4
    for i in range(1, 5):
        assert abs(float((positive == i).float().mean()) - .25) < .05
    assert (dropped.attribute_values[~dropped.attribute_valid] == 0).all()


def test_attributes_share_physical_extractor_and_masks_with_differentiable_loss():
    physical = torch.randn(2, 3, 256)
    mask = torch.ones_like(physical, dtype=torch.bool)
    mask[0, 1] = False
    values, valid = waveform_attributes(physical, mask, 256)
    scaled = (physical / 3).requires_grad_()
    loss = attribute_control_loss(scaled, values, valid, ATTRIBUTE_SCHEMA, 256, 3,
                                  waveform_valid=mask)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-5, rtol=0)
    changed = values.clone()
    changed[:, 1] += .5
    attribute_control_loss(scaled, changed, valid, ATTRIBUTE_SCHEMA, 256, 3,
                           waveform_valid=mask).sum().backward()
    assert torch.isfinite(scaled.grad).all() and scaled.grad.abs().sum() > 0
    assert (scaled.grad[~mask] == 0).all()


def test_checkpoint_records_mask_attributes_and_requires_explicit_weights_only_migration(tmp_path):
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig())
    trainer.run_provenance = {"generator_training_seed": 17, "source_identity": ["source-a"],
                              "dataset_digest": "source-digest", "stage": "diffusion"}
    path = tmp_path / "current.pt"
    trainer.save(path)
    restored = Trainer.load(path)
    assert restored.run_provenance == trainer.run_provenance
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["condition_protocols"]["cwt_mask"] == CWT_MASK_PROTOCOL
    checkpoint.pop("condition_protocols")
    checkpoint["step"], checkpoint["microstep"] = 11, 11
    legacy = tmp_path / "legacy.pt"
    torch.save(checkpoint, legacy)
    with pytest.raises(ValueError, match="explicit protocol migration"):
        Trainer.load(legacy)
    migrated = Trainer.load(legacy, allow_protocol_migration=True)
    assert migrated.step == migrated.microstep == 0
    assert migrated.normalization is None and not migrated.hard_audit_approved
    assert migrated.protocol_migration["weights_only"]
    assert "generator_training_seed" not in migrated.run_provenance
    assert migrated.run_provenance["parent_run_provenance"] == trainer.run_provenance
    with pytest.raises(ValueError, match="optimizer resume is forbidden"):
        Trainer.load(legacy, resume_optimizer=True, allow_protocol_migration=True)


def test_explicit_empty_image_cannot_bypass_reference_waveform_provenance():
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig())
    batch = declared_batch(trainer)
    shape = (3, trainer.config.channels, trainer.config.cwt_frequencies, trainer.config.cwt_times)
    batch.update(structure=torch.zeros(shape), structure_valid=torch.zeros(shape, dtype=torch.bool),
                 structure_visible=torch.zeros(shape, dtype=torch.bool), structure_role="none",
                 cwt_mask_protocol=CWT_MASK_PROTOCOL)
    batch.pop("reference_provenance")
    with pytest.raises(ValueError, match="reference.*provenance"):
        trainer.conditions(batch)
    batch["reference_provenance"] = {"origin_id": "query-target", "target_derived": True,
                                     "inference_available": True}
    with pytest.raises(ValueError, match="reference.*provenance"):
        trainer.conditions(batch)
    # A validated generic manifest audit remains a legal migration source,
    # including when no image construction path is taken.
    batch.pop("reference_provenance")
    batch["metadata"] = [{"split": "train", "role": "source", "reference_audit": {
        "reference_id": f"source-reference-{i}", "reference_split": "train",
        "reference_role": "source", "context_overlap": False, "protocol": "source-only"}}
        for i in range(3)]
    condition = trainer.conditions(batch)
    assert condition.reference_provenance[0]["origin_id"] == "source-reference-0"
    condition.validate_for_inference()


def test_direct_condition_batch_and_inference_revalidate_reference_provenance():
    n = 2
    geometry = torch.zeros(n, 4, 6)
    reference = torch.randn(n, 4, 128)
    mask = torch.ones(n, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="reference.*provenance"):
        ConditionBatch(n, geometry, reference_waveform=reference, reference_valid=mask)
    origin = {"origin_id": "allowed-reference", "target_derived": False, "inference_available": True}
    condition = ConditionBatch(n, geometry, reference_waveform=reference,
                               reference_valid=mask, reference_provenance=origin)
    condition.reference_provenance = None
    with pytest.raises(ValueError, match="reference.*provenance"):
        condition.validate_for_inference()
    # Explicit null references are absent information; no provenance is needed.
    ConditionBatch(n, geometry, reference_waveform=reference,
                   reference_valid=torch.zeros_like(mask)).validate_for_inference()


def test_few_shot_reference_requires_declared_support_budget_and_context_isolation():
    kwargs = {"batch_size": 1, "lead_geometry": torch.zeros(1, 4, 6),
              "reference_waveform": torch.randn(1, 4, 128)}
    origin = {"origin_id": "support-1", "target_derived": False, "inference_available": True,
              "protocol": "few-shot-support", "context_overlap": False}
    with pytest.raises(ValueError, match="support budget"):
        ConditionBatch(**kwargs, reference_provenance=origin)
    origin = {**origin, "support_budget": {"support_ids": ["support-1"], "events": 1}}
    ConditionBatch(**kwargs, reference_provenance=origin).validate_for_inference()
    with pytest.raises(ValueError, match="support budget"):
        ConditionBatch(**kwargs, reference_provenance={**origin, "context_overlap": True})


@pytest.mark.parametrize("stage", ["pretrain", "align", "diffusion"])
@pytest.mark.parametrize("metadata", [{"split": "test"}, {"split": "validation"}, {"role": "query"}])
def test_declared_query_targets_never_fit_even_without_teacher_attributes(stage, metadata):
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig(stage=stage))
    batch = declared_batch(trainer)
    batch["attribute_valid"].zero_()
    batch["metadata"] = metadata
    with pytest.raises(ValueError, match="source/train targets"):
        trainer.train_step(batch)
    assert trainer.step == trainer.microstep == 0


def test_top_level_query_declaration_is_guarded_and_undeclared_engineering_remains_usable():
    torch.set_num_threads(2)
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig(warmup_steps=0))
    batch = declared_batch(trainer)
    batch.pop("metadata")
    batch["split"] = "test"
    with pytest.raises(ValueError, match="source/train targets"):
        trainer.train_step(batch)
    batch.pop("split")
    assert trainer.train_step(batch)["optimizer_updated"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_visible_control_and_attribute_backward_cuda():
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig(mixed_precision=True), device="cuda")
    batch = trainer.move_batch(observed_batch(trainer))
    trainer._drop_conditions = lambda condition: condition
    metrics = trainer.train_step(batch)
    assert metrics["provided_structure_control_bins"] > 0 and metrics["gradient_norm"] > 0
