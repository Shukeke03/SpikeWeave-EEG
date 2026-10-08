from dataclasses import replace

import pytest
import torch

from triwave.config import ModelConfig
from triwave.models.vision_language import SharedVisionLanguageEncoder


def test_same_c_parameters_serve_all_three_modes():
    cfg = ModelConfig.tiny()
    c = SharedVisionLanguageEncoder(cfg).eval()
    g = torch.randn(2, 4, 6)
    image = torch.randn(2, 4, 16, 32)
    ids = torch.randint(1, 4096, (2, 12))
    names = {id(p) for p in c.parameters()}
    joint = c(image, input_ids=ids, lead_geometry=g)
    text = c(input_ids=ids, lead_geometry=g)
    visual = c(image, lead_geometry=g)
    assert names == {id(p) for p in c.parameters()}
    assert joint.semantic_valid.all() and joint.structural_valid.all()
    assert text.semantic_valid.all() and not text.structural_valid.any()
    assert visual.structural_valid.all() and not visual.semantic_valid.any()
    null = c(lead_geometry=g)
    assert null.valid.any(-1).all() and not null.present.any()
    assert all(torch.isfinite(out.tokens).all() for out in (joint, text, visual, null))
    (
        joint.semantic[:, 0].sum() + text.semantic[:, 1].sum() + visual.structural[:, 2].sum()
    ).backward()
    assert c.joint_blocks[0].attention.query.weight.grad.abs().sum() > 0
    assert not c.is_medically_pretrained


def test_medical_backend_fails_explicitly_without_weights():
    with pytest.raises(ValueError, match="biomedclip_local_path"):
        SharedVisionLanguageEncoder(replace(ModelConfig.tiny(), frontend_backend="biomedclip"))


def test_actual_openclip_hf_timm_checkpoint_interface_without_medical_claim(tmp_path):
    """A small RANDOM local checkpoint tests loading/masking, not pretraining."""
    import json

    pytest.importorskip("open_clip")
    from open_clip.model import CustomTextCLIP
    from transformers import BertConfig

    root = tmp_path / "random-interface"
    text = root / "text_config"
    text.mkdir(parents=True)
    BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
    ).save_pretrained(text)
    cfg = {
        "embed_dim": 64,
        "vision_cfg": {
            "timm_model_name": "vit_tiny_patch16_224",
            "timm_model_pretrained": False,
            "timm_pool": "",
            "timm_proj": "linear",
            "image_size": 224,
        },
        "text_cfg": {
            "hf_model_name": str(text),
            "hf_model_pretrained": False,
            "hf_proj_type": "mlp",
            "hf_pooler_type": "cls_last_hidden_state_pooler",
            "context_length": 16,
        },
    }
    model = CustomTextCLIP(**cfg)
    torch.save(model.state_dict(), root / "open_clip_pytorch_model.bin")
    (root / "open_clip_config.json").write_text(json.dumps({"model_cfg": cfg}))
    del model
    config = replace(
        ModelConfig.tiny(),
        frontend_backend="biomedclip",
        frontend_test_fixture=True,
        biomedclip_local_path=str(root),
        condition_dim=192,
        condition_heads=3,
        visual_patch=16,
    )
    c = SharedVisionLanguageEncoder(config).eval()
    assert not c.is_medically_pretrained
    geometry = torch.randn(2, 4, 6)
    image = torch.randn(2, 4, 16, 32)
    valid = torch.ones_like(image, dtype=torch.bool)
    valid[1] = False
    ids = torch.randint(1, 128, (2, 12))
    text_valid = torch.ones_like(ids, dtype=torch.bool)
    text_valid[1] = False
    output = c(image, valid, ids, text_valid, lead_geometry=geometry)
    assert torch.isfinite(output.tokens).all()
    assert output.present.tolist() == [True, False]
    (output.semantic[:, 0].sum() + output.tokens.square().mean()).backward()
    assert c.image_stem.weight.grad.abs().sum() > 0
    assert c.text_adapter.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in c.medical.parameters())
    # The numeric single-channel stem sums RGB weights, without RGB image scaling.
    torch.testing.assert_close(
        c.image_stem.weight, c.medical.visual.trunk.patch_embed.proj.weight.sum(1, keepdim=True)
    )


def test_numeric_adaptation_masks_teacher_pixels_before_stem():
    cfg = ModelConfig.tiny()
    encoder = SharedVisionLanguageEncoder(cfg).eval()
    image = torch.randn(2, 4, 16, 32)
    valid = torch.ones_like(image, dtype=torch.bool)
    mask = torch.rand(2, 4, 4, 8) < 0.4
    a = encoder.numeric_reconstruction(image, valid, mask)
    altered = image.clone()
    pixel = mask.repeat_interleave(4, -2).repeat_interleave(4, -1)
    altered[pixel] = float("nan")
    b = encoder.numeric_reconstruction(altered, valid, mask)
    torch.testing.assert_close(a, b)
    a.square().mean().backward()
    assert encoder.image_stem.weight.grad.abs().sum() > 0
    assert encoder.numeric_reconstruction_head.weight.grad.abs().sum() > 0


def test_incomplete_image_patches_are_absent_even_if_some_pixels_are_observed():
    cfg = ModelConfig.tiny()
    encoder = SharedVisionLanguageEncoder(cfg).eval()
    image = torch.randn(2, 4, 16, 32)
    valid = torch.ones_like(image, dtype=torch.bool)
    valid[:, :, 0, 0] = False
    geometry = torch.randn(2, 4, 6)
    before = encoder(image, valid, lead_geometry=geometry)
    changed = image.clone()
    # Every pixel of an incomplete patch is excluded, including its valid ones.
    changed[:, :, :4, :4] = 10000
    after = encoder(changed, valid, lead_geometry=geometry)
    torch.testing.assert_close(before.tokens, after.tokens, atol=0, rtol=0)
    valid[:, :, ::4, ::4] = False
    absent = encoder(image, valid, lead_geometry=geometry)
    assert not absent.structural_valid.any() and not absent.present.any()
    assert torch.isfinite(absent.tokens).all()
