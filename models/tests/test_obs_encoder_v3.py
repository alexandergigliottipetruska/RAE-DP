"""Tests for V3 ObservationEncoder (matching Chi: no LayerNorm, no projection).

Output is (B, T_o, concat_dim) — raw feature concat for denoiser's cond_obs_emb.
Only active camera features are concatenated (not zero-padded slots).
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
    def test_output_shape_robomimic(self):
        """2 active cameras → (B, T_o, 1033) raw concat."""
        enc = ObservationEncoder(n_active_cams=2, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False  # only slots 0 and 3 active
        out = enc(tokens, proprio, vp)
        assert out["tokens"].shape == (B, 2, 2 * 512 + 9)  # 1033

    def test_output_shape_rlbench(self):
        """4 active cameras → (B, T_o, 2057) raw concat."""
        enc = ObservationEncoder(n_active_cams=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        out = enc(tokens, proprio, vp)
        assert out["tokens"].shape == (B, 2, 4 * 512 + 9)  # 2057

    def test_output_dim_attribute(self):
        """output_dim matches expected concat dimension."""
        enc2 = ObservationEncoder(n_active_cams=2, proprio_dim=9)
        assert enc2.output_dim == 2 * 512 + 9  # 1033

        enc4 = ObservationEncoder(n_active_cams=4, proprio_dim=8)
        assert enc4.output_dim == 4 * 512 + 8  # 2056

    def test_global_vector_shape(self):
        enc = ObservationEncoder(n_active_cams=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False
        out = enc(tokens, proprio, vp)
        assert out["global"].shape == (B, D_MODEL)

    def test_output_keys(self):
        enc = ObservationEncoder(n_active_cams=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False
        out = enc(tokens, proprio, vp)
        assert set(out.keys()) == {"tokens", "global"}


class TestObsEncoderActiveCams:
    def test_only_active_cams_contribute(self):
        """Changing inactive slot data doesn't affect output."""
        enc = ObservationEncoder(n_active_cams=2)
        tokens1, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False  # slots 1,2 inactive

        tokens2 = tokens1.clone()
        tokens2[:, :, 1:3] = torch.randn_like(tokens2[:, :, 1:3]) * 100

        enc.eval()
        with torch.no_grad():
            out1 = enc(tokens1, proprio, vp)
            out2 = enc(tokens2, proprio, vp)

        assert torch.allclose(out1["tokens"], out2["tokens"], atol=1e-5)


class TestObsEncoderNoProjection:
    def test_no_obs_proj(self):
        """Encoder has no obs_proj — raw concat goes to denoiser."""
        enc = ObservationEncoder()
        assert not hasattr(enc, 'obs_proj')

    def test_no_feature_norm(self):
        """Encoder has no LayerNorm — matches Chi exactly."""
        enc = ObservationEncoder()
        assert not hasattr(enc, 'feature_norm')

    def test_tokens_are_raw_concat(self):
        """Output tokens are raw avg-pooled features + proprio concat."""
        enc = ObservationEncoder(n_active_cams=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False

        enc.eval()
        with torch.no_grad():
            out = enc(tokens, proprio, vp)

        # Manually compute expected: avg pool → select active → flatten → cat proprio
        pooled = tokens.mean(dim=3)  # (B, T_o, K, D)
        active = torch.stack([pooled[:, :, 0, :], pooled[:, :, 3, :]], dim=2)
        flat = active.reshape(B, T_O, 2 * ADAPTER_DIM)
        expected = torch.cat([flat, proprio], dim=-1)

        assert torch.allclose(out["tokens"], expected, atol=1e-5)


class TestObsEncoderGradients:
    def test_gradient_flow_through_global(self):
        """Gradients flow through global_proj."""
        enc = ObservationEncoder(n_active_cams=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False
        out = enc(tokens, proprio, vp)
        loss = out["global"].sum()
        loss.backward()
        assert enc.global_proj[0].weight.grad is not None

    def test_tokens_pass_gradients_to_inputs(self):
        """Token output passes gradients to adapted_tokens input."""
        enc = ObservationEncoder(n_active_cams=2)
        tokens = torch.randn(B, T_O, 4, N_PATCHES, ADAPTER_DIM, requires_grad=True)
        proprio = torch.randn(B, T_O, PROPRIO_DIM, requires_grad=True)
        vp = torch.ones(B, 4, dtype=torch.bool)
        vp[:, 1:3] = False
        out = enc(tokens, proprio, vp)
        loss = out["tokens"].sum()
        loss.backward()
        assert tokens.grad is not None
        assert proprio.grad is not None


class TestObsEncoderBatchIndependence:
    def test_batch_independence(self):
        enc = ObservationEncoder(n_active_cams=2)
        enc.eval()
        t8, p8, vp8 = _make_inputs(b=8)
        vp8[:, 1:3] = False
        with torch.no_grad():
            out8 = enc(t8, p8, vp8)
            out4 = enc(t8[:4], p8[:4], vp8[:4])
        assert torch.allclose(out8["tokens"][:4], out4["tokens"], atol=1e-5)

    def test_deterministic_eval(self):
        enc = ObservationEncoder(n_active_cams=2)
        enc.eval()
        tokens, proprio, vp = _make_inputs()
        vp[:, 1:3] = False
        with torch.no_grad():
            out1 = enc(tokens, proprio, vp)
            out2 = enc(tokens, proprio, vp)
        assert torch.equal(out1["tokens"], out2["tokens"])
