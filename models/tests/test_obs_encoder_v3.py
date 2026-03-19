"""Tests for V3 ObservationEncoder (Chi-matching: concat per timestep).

Output is (B, T_o, d_model) — one token per observation timestep, NOT
separate tokens per view. Memory = [timestep, obs_t0, obs_t1] = 3 tokens.
"""

import torch
import pytest

from models.obs_encoder_v3 import ObservationEncoder

# Constants
B = 4
T_O = 2
N_PATCHES = 196
ADAPTER_DIM = 512
D_MODEL = 256
PROPRIO_DIM = 9


def _make_inputs(b=B, t_o=T_O, k=4, n=N_PATCHES, d=ADAPTER_DIM, p=PROPRIO_DIM):
    adapted_tokens = torch.randn(b, t_o, k, n, d)
    proprio = torch.randn(b, t_o, p)
    view_present = torch.ones(b, k, dtype=torch.bool)
    return adapted_tokens, proprio, view_present


class TestObsEncoderShapes:
    def test_output_tokens_shape(self):
        """Output is (B, T_o, d_model) — one token per timestep."""
        enc = ObservationEncoder(num_views=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        out = enc(tokens, proprio, vp)
        assert out["tokens"].shape == (B, 2, D_MODEL)

    def test_output_shape_independent_of_K(self):
        """Same S_obs=T_o regardless of num_views (all concat into one vector)."""
        enc = ObservationEncoder(num_views=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False  # only 2 cameras active
        out = enc(tokens, proprio, vp)
        # Still T_o=2 tokens (not affected by number of active cameras)
        assert out["tokens"].shape == (B, 2, D_MODEL)

    def test_global_vector_shape(self):
        enc = ObservationEncoder(num_views=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        out = enc(tokens, proprio, vp)
        assert out["global"].shape == (B, D_MODEL)

    def test_output_keys(self):
        enc = ObservationEncoder()
        tokens, proprio, vp = _make_inputs()
        out = enc(tokens, proprio, vp)
        assert set(out.keys()) == {"tokens", "global"}


class TestObsEncoderMasking:
    def test_absent_views_affect_output(self):
        """Absent views are zeroed, producing different tokens than all-present."""
        enc = ObservationEncoder(num_views=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)

        out_all = enc(tokens, proprio, vp)

        vp_partial = vp.clone()
        vp_partial[:, 1:3] = False
        out_partial = enc(tokens, proprio, vp_partial)

        assert not torch.allclose(out_all["tokens"], out_partial["tokens"], atol=1e-3)


class TestObsEncoderGradients:
    def test_gradient_flow_obs_proj(self):
        enc = ObservationEncoder()
        tokens, proprio, vp = _make_inputs()
        out = enc(tokens, proprio, vp)
        loss = out["tokens"].sum()
        loss.backward()
        assert enc.obs_proj.weight.grad is not None
        assert not torch.all(enc.obs_proj.weight.grad == 0)

    def test_gradient_flow_global(self):
        enc = ObservationEncoder()
        tokens, proprio, vp = _make_inputs()
        out = enc(tokens, proprio, vp)
        loss = out["global"].sum()
        loss.backward()
        assert enc.global_proj[0].weight.grad is not None


class TestObsEncoderBatchIndependence:
    def test_batch_independence(self):
        enc = ObservationEncoder()
        enc.eval()
        t8, p8, vp8 = _make_inputs(b=8)
        with torch.no_grad():
            out8 = enc(t8, p8, vp8)
            out4 = enc(t8[:4], p8[:4], vp8[:4])
        assert torch.allclose(out8["tokens"][:4], out4["tokens"], atol=1e-5)

    def test_deterministic_eval(self):
        enc = ObservationEncoder()
        enc.eval()
        tokens, proprio, vp = _make_inputs()
        with torch.no_grad():
            out1 = enc(tokens, proprio, vp)
            out2 = enc(tokens, proprio, vp)
        assert torch.equal(out1["tokens"], out2["tokens"])
