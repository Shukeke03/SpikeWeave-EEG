import torch

from triwave.config import ModelConfig
from triwave.models.snn import ParametricLIF, SpikingEEGEncoder, TimeSharedDropPath


def test_plif_state_is_fp32_local_and_tau_learns():
    neuron = ParametricLIF(8)
    x = torch.randn(2, 12, 3, 8, requires_grad=True) * 2
    spike, membrane = neuron(x)
    assert membrane.dtype == torch.float32
    assert set(spike.unique().tolist()) <= {0.0, 1.0}
    (membrane.square().mean() + spike.mean()).backward()
    assert neuron.decay_logit.grad.abs().sum() > 0
    torch.testing.assert_close(neuron(x)[1], membrane)
    torch.testing.assert_close(neuron(x[:1])[1], membrane[:1])
    assert ((neuron.decay > 0) & (neuron.decay < 1)).all()
    torch.testing.assert_close(neuron.tau, torch.tensor([2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 16.0, 16.0]))


def test_masked_patch_values_never_enter_encoder_and_state_resets():
    cfg = ModelConfig.tiny()
    e = SpikingEEGEncoder(cfg).eval()
    x = torch.randn(2, 4, 128)
    g = torch.randn(2, 4, 6)
    valid = torch.ones_like(x, dtype=torch.bool)
    mask = e.sample_patch_mask(valid)
    assert mask[:, 0].sum(-1).tolist() == [6, 6]
    a = e(x, g, valid, mask)
    changed = x.reshape(2, 4, 16, 8).clone()
    changed[mask] = float("nan")
    b = e(changed.reshape_as(x), g, valid, mask)
    torch.testing.assert_close(a.reconstruction, b.reconstruction)
    torch.testing.assert_close(a.semantic, e(x, g, valid, mask).semantic)
    a.reconstruction.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in e.named_parameters()
        if "decay_logit" in n
    )


def test_drop_path_is_constant_across_physical_time():
    layer = TimeSharedDropPath(0.5).train()
    result = layer(torch.ones(16, 8, 4, 32))
    assert torch.equal(result, result[:, :1, :1, :1].expand_as(result))
