"""Tests for DiTDenoiser (adaLN-Zero diffusion transformer).

Mirrors test_denoiser_transformer.py. Validates shapes, conditioning,
causal masking, overfit, gradient flow, and adaLN-Zero initialization.
"""

import torch
import pytest

from models.denoiser_dit import DiTDenoiser

# Constants
B = 4
T_P = 16
AC_DIM = 10
D_MODEL = 64   # small for tests
COND_DIM = 1033  # robomimic: 2*512 + 9


def _make_denoiser(**kwargs):
    defaults = dict(
        ac_dim=AC_DIM, d_model=D_MODEL, n_head=4, n_layers=2,
        T_pred=T_P, cond_dim=COND_DIM, p_drop_emb=0.0, p_drop_attn=0.0,
    )
    defaults.update(kwargs)
    return DiTDenoiser(**defaults)


def _make_inputs(b=B, t_p=T_P, ac_dim=AC_DIM, t_obs=2, cond_dim=COND_DIM):
    noisy_actions = torch.randn(b, t_p, ac_dim)
    timestep = torch.randint(0, 100, (b,))
    obs_cond = {"tokens": torch.randn(b, t_obs, cond_dim)}
    return noisy_actions, timestep, obs_cond


class TestDiTDenoiserInit:
    def test_adaln_zero_init(self):
        """adaLN_modulation output projection is zero-initialized."""
        net = _make_denoiser()
        for block in net.blocks:
            w = block.adaLN_modulation[-1].weight
            b = block.adaLN_modulation[-1].bias
            assert torch.all(w == 0), "adaLN_modulation weight should be zero at init"
            assert torch.all(b == 0), "adaLN_modulation bias should be zero at init"

    def test_final_modulation_zero_init(self):
        """final_modulation output projection is zero-initialized."""
        net = _make_denoiser()
        w = net.final_modulation[-1].weight
        b = net.final_modulation[-1].bias
        assert torch.all(w == 0), "final_modulation weight should be zero at init"
        assert torch.all(b == 0), "final_modulation bias should be zero at init"

    def test_output_near_zero_at_init(self):
        """At init, gates=0 so output should be small (identity residuals)."""
        net = _make_denoiser()
        net.eval()
        noisy, ts, cond = _make_inputs()
        with torch.no_grad():
            out = net(noisy, ts, cond)
        assert out.abs().mean() < 1.0, f"Output mean abs {out.abs().mean():.4f} too large at init"
        assert torch.isfinite(out).all()


class TestDiTDenoiserShapes:
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
        net = _make_denoiser()
        noisy, _, cond = _make_inputs()
        out = net(noisy, torch.tensor(42), cond)
        assert out.shape == (B, T_P, AC_DIM)

    def test_rlbench_dims(self):
        """Works with RLBench dims: ac_dim=8, cond_dim=2056."""
        net = _make_denoiser(ac_dim=8, cond_dim=2056)
        noisy, ts, cond = _make_inputs(ac_dim=8, cond_dim=2056)
        out = net(noisy, ts, cond)
        assert out.shape == (B, T_P, 8)


class TestDiTDenoiserCausalMasking:
    def test_causal_different_from_noncausal(self):
        """Causal and non-causal masks produce different outputs."""
        torch.manual_seed(42)
        net_causal = _make_denoiser(causal_attn=True)
        torch.manual_seed(42)
        net_noncausal = _make_denoiser(causal_attn=False)
        net_noncausal.load_state_dict(net_causal.state_dict())

        net_causal.eval()
        net_noncausal.eval()

        noisy, ts, cond = _make_inputs()
        with torch.no_grad():
            out_c = net_causal(noisy, ts, cond)
            out_nc = net_noncausal(noisy, ts, cond)

        assert not torch.allclose(out_c, out_nc, atol=1e-3), \
            "Causal and non-causal should differ"


class TestDiTDenoiserOverfit:
    def test_overfit_single_batch(self):
        """Loss decreases significantly when overfitting a single batch."""
        torch.manual_seed(0)
        net = _make_denoiser(n_layers=2)
        net.train()

        noisy, ts, cond = _make_inputs(b=8)
        target = torch.randn(8, T_P, AC_DIM)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

        losses = []
        for _ in range(300):
            pred = net(noisy, ts, cond)
            loss = torch.nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        q1 = sum(losses[:75]) / 75
        q4 = sum(losses[-75:]) / 75
        assert q4 < q1 * 0.1, f"Overfit failed: q1={q1:.6f}, q4={q4:.6f}"


class TestDiTDenoiserBatchIndependence:
    def test_batch_independence(self):
        """First 4 of B=8 match standalone B=4."""
        net = _make_denoiser()
        net.eval()

        noisy, ts, cond = _make_inputs(b=8)
        with torch.no_grad():
            out8 = net(noisy, ts, cond)
            out4 = net(noisy[:4], ts[:4], {"tokens": cond["tokens"][:4]})

        assert torch.allclose(out8[:4], out4, atol=1e-5)


class TestDiTDenoiserGradients:
    def test_all_params_receive_gradients(self):
        """All trainable parameters receive non-zero gradients after backward."""
        net = _make_denoiser()
        noisy, ts, cond = _make_inputs()
        out = net(noisy, ts, cond)
        loss = out.sum()
        loss.backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"
                assert not torch.all(p.grad == 0), f"Zero grad for {name}"

    def test_optim_groups_cover_all_params(self):
        """get_optim_groups() covers every parameter exactly once."""
        net = _make_denoiser()
        groups = net.get_optim_groups(weight_decay=1e-3)
        all_ids_from_groups = set()
        for g in groups:
            for p in g["params"]:
                pid = id(p)
                assert pid not in all_ids_from_groups, "Parameter appears in multiple groups"
                all_ids_from_groups.add(pid)
        all_param_ids = {id(p) for p in net.parameters()}
        assert all_param_ids == all_ids_from_groups, "Some params missing from optim groups"
