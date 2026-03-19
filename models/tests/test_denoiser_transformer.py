"""Tests for V3 TransformerDenoiser.

Validates zero-init output, shapes, causal masking, overfit, train/eval
consistency, batch independence, and gradient flow.
"""

import torch
import pytest

from models.denoiser_transformer import TransformerDenoiser

# Constants
B = 4
T_P = 16
AC_DIM = 10
D_MODEL = 256
S_OBS = 6  # robomimic: T_o=2, K=2 → 2*2+2=6


def _make_denoiser(**kwargs):
    defaults = dict(
        ac_dim=AC_DIM, d_model=D_MODEL, n_head=4, n_layers=2,
        T_pred=T_P, cond_dim=D_MODEL, p_drop_emb=0.01, p_drop_attn=0.01,
    )
    defaults.update(kwargs)
    return TransformerDenoiser(**defaults)


def _make_inputs(b=B, t_p=T_P, ac_dim=AC_DIM, s_obs=S_OBS, d_model=D_MODEL):
    noisy_actions = torch.randn(b, t_p, ac_dim)
    timestep = torch.randint(0, 100, (b,))
    obs_cond = {"tokens": torch.randn(b, s_obs, d_model)}
    return noisy_actions, timestep, obs_cond


class TestDenoiserInit:
    def test_small_init_output(self):
        """At initialization, output should be small (all weights normal(0, 0.02))."""
        net = _make_denoiser()
        net.eval()
        noisy, ts, cond = _make_inputs()
        with torch.no_grad():
            out = net(noisy, ts, cond)
        # With normal(0,0.02) init throughout, outputs should be small but not zero
        assert out.abs().mean() < 1.0, f"Output mean abs {out.abs().mean():.4f} too large"
        assert torch.isfinite(out).all()


class TestDenoiserShapes:
    def test_output_shape(self):
        net = _make_denoiser()
        noisy, ts, cond = _make_inputs()
        out = net(noisy, ts, cond)
        assert out.shape == (B, T_P, AC_DIM)

    def test_output_dtype(self):
        net = _make_denoiser()
        noisy, ts, cond = _make_inputs()
        out = net(noisy, ts, cond)
        assert out.dtype == torch.float32

    def test_output_finite(self):
        net = _make_denoiser()
        noisy, ts, cond = _make_inputs()
        out = net(noisy, ts, cond)
        assert torch.isfinite(out).all()

    def test_scalar_timestep(self):
        """Accepts a scalar timestep (broadcast to batch)."""
        net = _make_denoiser()
        noisy, _, cond = _make_inputs()
        out = net(noisy, torch.tensor(42), cond)
        assert out.shape == (B, T_P, AC_DIM)

    def test_rlbench_dims(self):
        """Works with RLBench dims: ac_dim=8, S_obs=10."""
        net = _make_denoiser(ac_dim=8)
        noisy, ts, cond = _make_inputs(ac_dim=8, s_obs=10)
        out = net(noisy, ts, cond)
        assert out.shape == (B, T_P, 8)

    def test_memory_length(self):
        """Memory is 1 (timestep) + S_obs tokens."""
        net = _make_denoiser()
        noisy, ts, cond = _make_inputs()
        # Trace through forward manually
        time_token = net.time_mlp(net.time_emb(ts)).unsqueeze(1)
        cond_obs = net.cond_obs_emb(cond["tokens"])
        memory = torch.cat([time_token, cond_obs], dim=1)
        assert memory.shape == (B, 1 + S_OBS, D_MODEL)


class TestDenoiserCausalMasking:
    def test_causal_different_from_noncausal(self):
        """Causal and non-causal produce different outputs."""
        torch.manual_seed(42)
        net_causal = _make_denoiser(causal_attn=True)
        torch.manual_seed(42)
        net_noncausal = _make_denoiser(causal_attn=False)

        # Copy weights from causal to noncausal so only masking differs
        net_noncausal.load_state_dict(net_causal.state_dict())

        net_causal.eval()
        net_noncausal.eval()

        noisy, ts, cond = _make_inputs()
        with torch.no_grad():
            out_c = net_causal(noisy, ts, cond)
            out_nc = net_noncausal(noisy, ts, cond)

        assert not torch.allclose(out_c, out_nc, atol=1e-3), \
            "Causal and non-causal should produce different outputs"


class TestDenoiserOverfit:
    def test_overfit_single_batch(self):
        """Overfit on a single batch — loss should decrease significantly."""
        torch.manual_seed(0)
        net = _make_denoiser(n_layers=2, p_drop_emb=0.0, p_drop_attn=0.0)
        net.train()

        noisy, ts, cond = _make_inputs(b=8)
        target = torch.randn(8, T_P, AC_DIM)

        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

        losses = []
        for step in range(300):
            pred = net(noisy, ts, cond)
            loss = torch.nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # First quarter average should be much larger than last quarter
        q1 = sum(losses[:75]) / 75
        q4 = sum(losses[-75:]) / 75
        assert q4 < q1 * 0.1, f"Overfit failed: q1={q1:.6f}, q4={q4:.6f}"


class TestDenoiserTrainEvalConsistency:
    def test_both_modes_finite(self):
        """Both train and eval modes produce finite outputs."""
        net = _make_denoiser(p_drop_emb=0.0, p_drop_attn=0.3)
        noisy, ts, cond = _make_inputs()

        net.eval()
        with torch.no_grad():
            out_eval = net(noisy, ts, cond)
        assert torch.isfinite(out_eval).all()

        net.train()
        with torch.no_grad():
            out_train = net(noisy, ts, cond)
        assert torch.isfinite(out_train).all()
        assert out_eval.shape == out_train.shape


class TestDenoiserBatchIndependence:
    def test_batch_independence(self):
        """First 4 samples of B=8 match B=4."""
        net = _make_denoiser()
        net.eval()

        noisy, ts, cond = _make_inputs(b=8)
        with torch.no_grad():
            out8 = net(noisy, ts, cond)
            out4 = net(noisy[:4], ts[:4], {"tokens": cond["tokens"][:4]})

        assert torch.allclose(out8[:4], out4, atol=1e-5)


class TestDenoiserGradients:
    def test_all_params_receive_gradients(self):
        """All trainable params receive gradients after one backward pass."""
        net = _make_denoiser()
        noisy, ts, cond = _make_inputs()
        out = net(noisy, ts, cond)
        loss = out.sum()
        loss.backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"
                assert not torch.all(p.grad == 0), f"Zero grad for {name}"
