"""Model contracts: missingness, gradient routes, and target/condition isolation."""

from dataclasses import replace

import pytest
import torch

from triwave.config import ModelConfig
from triwave.models import TextEncoder, TriWaveModel, WaveDiT
from triwave.models.attention import ShiftedWindowBlock
from triwave.types import ConditionBatch


@pytest.fixture
def setup():
    torch.manual_seed(71)
    config = ModelConfig.tiny()
    model = TriWaveModel(config)
    batch = 2
    geometry = torch.randn(batch, config.channels, 6)
    reference = torch.randn(batch, config.channels, config.samples)
    structure = torch.randn(batch, config.channels, config.cwt_frequencies, config.cwt_times)
    structure_valid = torch.rand_like(structure) > 0.2
    input_ids = torch.randint(0, config.text_vocab_size, (batch, 8))
    text_valid = torch.ones_like(input_ids, dtype=torch.bool)
    attributes = torch.randn(batch, config.attribute_dim)
    conditions = ConditionBatch(batch_size=batch, lead_geometry=geometry,
                                reference_waveform=reference,
                                reference_valid=torch.ones(batch, config.channels, dtype=torch.bool),
                                structure=structure, structure_valid=structure_valid,
                                input_ids=input_ids, text_valid=text_valid,
                                attribute_values=attributes,
                                attribute_valid=torch.ones_like(attributes, dtype=torch.bool))
    return config, model, conditions


def assert_fused_equal(left, right):
    torch.testing.assert_close(left.tokens, right.tokens, rtol=0, atol=0)
    torch.testing.assert_close(left.global_vector, right.global_vector, rtol=0, atol=0)
    assert torch.equal(left.valid, right.valid)
    assert len(left.structure_scales) == len(right.structure_scales)
    for x, y in zip(left.structure_scales, right.structure_scales):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_complete_coefficients_and_tiny_forward(setup):
    config, model, conditions = setup
    assert model.band_lengths == (16, 16, 32, 64)
    fused = model.encode_conditions(conditions)
    assert len(fused.structure_scales) == len(config.structure_depths)
    assert all(mask.dtype == torch.bool for mask in fused.structure_scale_valid)
    noisy = torch.randn_like(conditions.reference_waveform)
    output = model.denoiser(noisy, torch.rand(2), fused, conditions.lead_geometry)
    assert output.shape == noisy.shape
    assert torch.isfinite(output).all()
    # Zero heads are intentional AdaLN-Zero initialization, not lost coefficients.
    loss = (output - torch.randn_like(output)).square().mean()
    loss.backward()
    assert all(head.weight.grad.abs().sum() > 0 for head in model.denoiser.output_heads)


def test_all_missing_has_visible_null_but_no_alignment_pair(setup):
    config, model, conditions = setup
    missing = ConditionBatch(batch_size=2, lead_geometry=conditions.lead_geometry)
    fused = model.encode_conditions(missing)
    assert fused.valid.any(-1).all()
    assert torch.isfinite(fused.tokens).all()
    output = model(torch.randn(2, config.channels, config.samples), torch.rand(2), missing)
    assert torch.isfinite(output).all()
    # Completely masked tensors can contain arbitrary unknown values, including NaN.
    align = model.alignment(
        torch.full_like(conditions.reference_waveform, float("nan")),
        torch.full_like(conditions.structure, float("nan")),
        torch.zeros_like(conditions.structure_valid),
        torch.full_like(conditions.input_ids, -999), torch.zeros_like(conditions.text_valid),
        conditions.lead_geometry, torch.zeros_like(conditions.reference_valid))
    assert not align.eeg_valid.any()
    assert not align.structure_valid.any()
    assert not align.text_valid.any()
    assert all(torch.isfinite(x).all() for x in (align.eeg, align.structure, align.text))


def test_masked_waveform_and_structure_values_do_not_change_embeddings(setup):
    _, model, conditions = setup
    model.eval()
    valid = torch.ones_like(conditions.reference_waveform, dtype=torch.bool)
    valid[:, 0] = False
    valid[:, 1, :16] = False
    first = model.waveform_encoder(conditions.reference_waveform, conditions.lead_geometry, valid)
    changed = conditions.reference_waveform.clone()
    changed[~valid] = float("nan")
    second = model.waveform_encoder(changed, conditions.lead_geometry, valid)
    torch.testing.assert_close(first.tokens, second.tokens, rtol=0, atol=0)
    torch.testing.assert_close(first.pooled, second.pooled, rtol=0, atol=0)
    changed_structure = conditions.structure.clone()
    changed_structure[~conditions.structure_valid] = float("nan")
    first = model.structure_encoder(conditions.structure, conditions.structure_valid, conditions.lead_geometry)
    second = model.structure_encoder(changed_structure, conditions.structure_valid, conditions.lead_geometry)
    torch.testing.assert_close(first.tokens, second.tokens, rtol=0, atol=0)
    torch.testing.assert_close(first.pooled, second.pooled, rtol=0, atol=0)
    for scale, valid in zip(second.scales, second.scale_valid):
        assert torch.equal(scale[~valid], torch.zeros_like(scale[~valid]))


def test_text_dropout_replaces_film_and_all_fusion_routes(setup):
    _, model, conditions = setup
    model.eval()
    # Activate FiLM to make this check meaningful beyond its identity initialization.
    torch.nn.init.normal_(model.fusion.text_film.weight, std=0.2)
    torch.nn.init.normal_(model.fusion.text_film.bias, std=0.1)
    altered = replace(conditions, input_ids=torch.randint(0, model.config.text_vocab_size,
                                                        conditions.input_ids.shape))
    first = model.encode_conditions(conditions, drop_text=True)
    second = model.encode_conditions(altered, drop_text=True)
    assert_fused_equal(first, second)
    present_first = model.encode_conditions(conditions)
    present_second = model.encode_conditions(altered)
    assert not torch.allclose(present_first.global_vector, present_second.global_vector)
    assert any(not torch.allclose(x, y) for x, y in
               zip(present_first.structure_scales, present_second.structure_scales))


def test_other_dropped_modalities_and_invalid_attributes_cannot_leak(setup):
    _, model, conditions = setup
    model.eval()
    altered = replace(conditions, reference_waveform=torch.full_like(conditions.reference_waveform, float("nan")),
                      structure=torch.full_like(conditions.structure, float("nan")),
                      attribute_values=torch.full_like(conditions.attribute_values, float("nan")))
    flags = dict(drop_reference=True, drop_structure=True, drop_attributes=True)
    assert_fused_equal(model.encode_conditions(conditions, **flags),
                       model.encode_conditions(altered, **flags))
    valid = torch.ones_like(conditions.attribute_values, dtype=torch.bool)
    valid[:, 3:] = False
    altered = replace(conditions, attribute_valid=valid,
                      attribute_values=conditions.attribute_values.clone())
    altered.attribute_values[~valid] = float("nan")
    original = replace(conditions, attribute_valid=valid)
    assert_fused_equal(model.encode_conditions(original), model.encode_conditions(altered))


def test_alignment_is_independent_and_all_three_branches_receive_gradients(setup):
    _, model, conditions = setup
    arguments = (conditions.reference_waveform, conditions.structure, conditions.structure_valid,
                 conditions.input_ids, conditions.text_valid, conditions.lead_geometry)
    align = model.alignment(*arguments)
    changed_ids = torch.randint(0, model.config.text_vocab_size, conditions.input_ids.shape)
    changed = model.alignment(*arguments[:3], changed_ids, *arguments[4:])
    torch.testing.assert_close(align.eeg, changed.eeg, rtol=0, atol=0)
    torch.testing.assert_close(align.structure, changed.structure, rtol=0, atol=0)
    assert not torch.allclose(align.text, changed.text)
    loss = ((align.eeg - align.structure.detach()).square().mean()
            + (align.structure - align.text.detach()).square().mean()
            + (align.text - align.eeg.detach()).square().mean())
    loss.backward()
    for branch in (model.waveform_encoder, model.structure_encoder, model.text_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    assert all(p.grad is None for p in model.fusion.parameters())
    assert all(p.grad is None for p in model.denoiser.parameters())


def test_alignment_allows_absent_text_without_a_false_positive_pair(setup):
    _, model, conditions = setup
    align = model.alignment(conditions.reference_waveform, conditions.structure,
                            conditions.structure_valid, None, None, conditions.lead_geometry)
    assert align.eeg_valid.all() and align.structure_valid.all()
    assert not align.text_valid.any()
    assert torch.isfinite(align.text).all()


def test_target_alignment_does_not_cache_or_modify_generation_conditions(setup):
    _, model, conditions = setup
    model.eval()
    before = model.encode_conditions(conditions)
    model.alignment(torch.randn_like(conditions.reference_waveform),
                    torch.randn_like(conditions.structure), conditions.structure_valid,
                    conditions.input_ids, conditions.text_valid, conditions.lead_geometry)
    assert_fused_equal(before, model.encode_conditions(conditions))
    with pytest.raises(TypeError):
        ConditionBatch(batch_size=2, lead_geometry=conditions.lead_geometry,
                       target_waveform=conditions.reference_waveform)
    with pytest.raises(TypeError):
        model(torch.randn_like(conditions.reference_waveform), torch.rand(2),
              conditions.reference_waveform, conditions.lead_geometry)


def test_shifted_windows_are_real_and_do_not_connect_wrapped_regions():
    block = ShiftedWindowBlock(8, 2, window=4, shift=2)
    mask = block._shift_mask(8, 8, torch.device("cpu"))
    assert mask is not None and not mask.all()
    assert mask.diagonal(dim1=-2, dim2=-1).all()
    x = torch.randn(2, 7, 9, 8)
    weight = torch.ones(2, 7, 9)
    weight[0] = 0
    output = block(x, weight)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output[0], torch.zeros_like(output[0]))


def test_geometry_required_and_offline_text_is_not_medically_pretrained(setup):
    config, model, conditions = setup
    assert not model.text_encoder.is_medically_pretrained
    assert any(p.requires_grad for p in model.text_encoder.embedding.parameters())
    with pytest.raises(ValueError, match="lead_geometry"):
        model.waveform_encoder(conditions.reference_waveform, conditions.lead_geometry[..., :3])
    fused = model.encode_conditions(conditions)
    with pytest.raises(ValueError, match="geometry"):
        model(torch.randn_like(conditions.reference_waveform), torch.rand(2), fused)
    with pytest.raises(ValueError, match="sum"):
        WaveDiT(config, (16, 16, 32))


def test_clinicalbert_backend_loads_local_random_bert_freezes_and_handles_missing(tmp_path):
    """Validate the HF interface using random weights, not medical pretraining."""
    from transformers import BertConfig, BertModel

    torch.manual_seed(13)
    checkpoint = tmp_path / "random-bert-interface-fixture"
    bert_config = BertConfig(vocab_size=128, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=4, intermediate_size=64,
                             max_position_embeddings=32)
    BertModel(bert_config).save_pretrained(checkpoint)
    config = replace(ModelConfig.tiny(), text_backend="clinicalbert",
                     text_model_name=str(checkpoint), text_local_files_only=True,
                     freeze_text=True)
    encoder = TextEncoder(config)
    encoder.train()
    assert encoder.training
    assert not encoder.hf_model.training
    assert all(not parameter.requires_grad for parameter in encoder.hf_model.parameters())
    input_ids = torch.randint(0, bert_config.vocab_size, (2, 8))
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    output = encoder(input_ids, valid)
    assert output.tokens.shape == (2, 8, config.dim)
    assert torch.isfinite(output.pooled).all()
    # A nonconstant objective checks that freezing the HF backbone does not
    # accidentally disable the trainable input projection or contrastive head.
    loss = output.pooled[:, 0].sum() + output.tokens.square().mean()
    loss.backward()
    assert encoder.input_projection.weight.grad.abs().sum() > 0
    assert encoder.projection[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in encoder.hf_model.parameters())
    missing = encoder(torch.full_like(input_ids, -999), torch.zeros_like(valid))
    assert torch.isfinite(missing.tokens).all() and torch.isfinite(missing.pooled).all()
    assert missing.valid[:, 0].all()
    assert torch.equal(missing.valid.sum(-1), torch.ones(2, dtype=torch.long))
    assert not encoder.hf_model.training


def test_pilot_tensor_shapes_keep_all_bands():
    # Pilot data sizes, reduced width for routine CPU verification.
    config = replace(ModelConfig(), dim=32, encoder_depth=1, structure_base_dim=8,
                     structure_depths=(1, 1), projection_dim=32, text_depth=1,
                     denoiser_dim=32, denoiser_heads=4, denoiser_depth=1, fusion_depth=1)
    model = TriWaveModel(config).eval()
    assert model.band_lengths == (128, 128, 256, 512)
    conditions = ConditionBatch(batch_size=1, lead_geometry=torch.randn(1, 16, 6),
                                structure=torch.randn(1, 16, 64, 128),
                                structure_valid=torch.ones(1, 16, 64, 128, dtype=torch.bool),
                                input_ids=torch.randint(0, config.text_vocab_size, (1, 12)),
                                text_valid=torch.ones(1, 12, dtype=torch.bool))
    with torch.no_grad():
        fused = model.encode_conditions(conditions)
        output = model(torch.randn(1, 16, 1024), torch.rand(1), fused, conditions.lead_geometry)
    assert output.shape == (1, 16, 1024)
    assert fused.structure_scales[0].shape == (1, 128, 32)
    assert torch.isfinite(output).all()
