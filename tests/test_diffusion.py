import pytest
import torch

from triwave.diffusion import GaussianDiffusion, flow_pair, sample_flow
from triwave.losses import contrastive_pair, signal_regularizers


def test_v_prediction_recovers_signed_clean_signal():
    diffusion = GaussianDiffusion(32)
    clean, epsilon = torch.randn(4, 3, 128), torch.randn(4, 3, 128)
    indices = torch.tensor([0, 1, 15, 31])
    noisy, velocity = diffusion.noise(clean, epsilon, indices)
    torch.testing.assert_close(diffusion.clean_from_v(noisy, velocity, indices), clean)


@pytest.mark.parametrize("steps", [1, 3, 32])
def test_ddim_oracle_has_final_clean_state(steps):
    diffusion = GaussianDiffusion(32)
    clean = torch.randn(2, 2, 8)
    initial = torch.randn_like(clean)

    def oracle(state, time):
        indices = (time * 31).round().long()
        alpha, sigma = diffusion.factors(indices, state)
        epsilon = (state - alpha * clean) / sigma
        return alpha * epsilon - sigma * clean

    generated = diffusion.sample(oracle, initial, steps)
    torch.testing.assert_close(generated, clean, atol=2e-6, rtol=2e-5)


def test_flow_orientation_and_reverse_integration():
    clean, epsilon = torch.randn(2, 2, 8), torch.randn(2, 2, 8)
    state, velocity = flow_pair(clean, epsilon, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(state[0], clean[0])
    torch.testing.assert_close(state[1], epsilon[1])
    generated = sample_flow(lambda x, t: velocity, epsilon, 5)
    torch.testing.assert_close(generated, clean)


def test_missing_modalities_cannot_be_negative_candidates():
    left = torch.eye(4, requires_grad=True)
    right = torch.eye(4, requires_grad=True)
    valid = torch.tensor([True, True, False, False])
    loss = contrastive_pair(left, right, valid, valid)
    changed = right.detach().clone()
    changed[~valid] = 100
    torch.testing.assert_close(loss, contrastive_pair(left, changed, valid, valid))
    loss.backward()
    assert torch.equal(right.grad[~valid], torch.zeros_like(right.grad[~valid]))
    assert torch.isfinite(contrastive_pair(left, right, valid & False, valid))


def test_signal_penalties_are_zero_for_identity_and_ignore_missing_leads():
    target = torch.randn(2, 3, 128)
    valid = torch.ones(2, 3, dtype=torch.bool)
    valid[:, -1] = False
    prediction = target.clone()
    prediction[:, -1] = 1000
    losses = signal_regularizers(prediction, target, valid)
    assert all(value == 0 for value in losses.values())
