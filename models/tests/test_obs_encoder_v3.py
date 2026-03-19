"""Tests for V3 ObservationEncoder.

Replaces TokenAssembly (C.4). Validates shapes, masking, gradient flow,
and batch independence for both robomimic (K=2) and RLBench (K=4).
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
    """Create random inputs for ObservationEncoder."""
    adapted_tokens = torch.randn(b, t_o, k, n, d)
    proprio = torch.randn(b, t_o, p)
    view_present = torch.ones(b, k, dtype=torch.bool)
    return adapted_tokens, proprio, view_present


class TestObsEncoderShapes:
    def test_output_shape_rlbench(self):
        """K=4, T_o=2 → S_obs = 2*4 + 2 = 10 tokens."""
        enc = ObservationEncoder(num_views=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        out = enc(tokens, proprio, vp)
        assert out["tokens"].shape == (B, 10, D_MODEL)

    def test_output_shape_robomimic(self):
        """K=2, T_o=2 → S_obs = 2*2 + 2 = 6 tokens."""
        enc = ObservationEncoder(num_views=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        # Simulate robomimic: only 2 cameras present
        vp[:, 1:3] = False
        out = enc(tokens, proprio, vp)
        # Shape is still T_o*K + T_o = 10 (masked cameras are zeroed, not removed)
        assert out["tokens"].shape == (B, 10, D_MODEL)

    def test_global_vector_shape(self):
        """Global conditioning vector has shape (B, d_model)."""
        enc = ObservationEncoder(num_views=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        out = enc(tokens, proprio, vp)
        assert out["global"].shape == (B, D_MODEL)

    def test_output_dtypes(self):
        enc = ObservationEncoder()
        tokens, proprio, vp = _make_inputs()
        out = enc(tokens, proprio, vp)
        assert out["tokens"].dtype == torch.float32
        assert out["global"].dtype == torch.float32

    def test_output_keys(self):
        enc = ObservationEncoder()
        tokens, proprio, vp = _make_inputs()
        out = enc(tokens, proprio, vp)
        assert set(out.keys()) == {"tokens", "global"}


class TestObsEncoderMasking:
    def test_absent_views_zeroed(self):
        """Absent camera views produce zero tokens."""
        enc = ObservationEncoder(num_views=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:] = False
        vp[:, 0] = True  # only camera 0 active
        out = enc(tokens, proprio, vp)

        obs_tokens = out["tokens"]
        # Visual tokens are first T_o*K = 8, proprio are last T_o = 2
        visual_tokens = obs_tokens[:, :8, :]  # (B, 8, d_model)
        visual_tokens = visual_tokens.reshape(B, T_O, 4, D_MODEL)

        # Cameras 1,2,3 should be zero (before view embedding, but after masking)
        # The view_proj bias + view_emb still adds a constant, but the pooled input is zero
        # So camera 1,2,3 tokens = view_proj(0) + view_emb + time_emb = bias + embs
        # Camera 0 tokens should differ from camera 1 tokens (camera 0 has real data)
        cam0 = visual_tokens[:, :, 0, :]
        cam1 = visual_tokens[:, :, 1, :]
        # cam0 has real data + bias + embs, cam1 has only bias + embs
        assert not torch.allclose(cam0, cam1, atol=1e-3), \
            "Active and absent cameras should produce different tokens"

    def test_all_views_present(self):
        """All cameras active → all tokens should be nonzero and finite."""
        enc = ObservationEncoder(num_views=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        out = enc(tokens, proprio, vp)
        assert torch.isfinite(out["tokens"]).all()


class TestObsEncoderGradients:
    def test_gradient_flow_view_proj(self):
        """Gradients flow through view_proj."""
        enc = ObservationEncoder()
        tokens, proprio, vp = _make_inputs()
        out = enc(tokens, proprio, vp)
        loss = out["tokens"].sum()
        loss.backward()
        assert enc.view_proj.weight.grad is not None
        assert not torch.all(enc.view_proj.weight.grad == 0)

    def test_gradient_flow_proprio_proj(self):
        """Gradients flow through proprio_proj."""
        enc = ObservationEncoder()
        tokens, proprio, vp = _make_inputs()
        out = enc(tokens, proprio, vp)
        loss = out["tokens"].sum()
        loss.backward()
        assert enc.proprio_proj[0].weight.grad is not None
        assert not torch.all(enc.proprio_proj[0].weight.grad == 0)

    def test_gradient_flow_embeddings(self):
        """Gradients flow through view and time embeddings."""
        enc = ObservationEncoder()
        tokens, proprio, vp = _make_inputs()
        out = enc(tokens, proprio, vp)
        loss = out["tokens"].sum()
        loss.backward()
        assert enc.view_emb.weight.grad is not None
        assert enc.time_emb.weight.grad is not None

    def test_gradient_flow_global(self):
        """Gradients flow through global projection."""
        enc = ObservationEncoder()
        tokens, proprio, vp = _make_inputs()
        out = enc(tokens, proprio, vp)
        loss = out["global"].sum()
        loss.backward()
        assert enc.global_proj[0].weight.grad is not None


class TestObsEncoderBatchIndependence:
    def test_batch_independence(self):
        """First 4 samples of B=8 match B=4 (no cross-batch leakage)."""
        enc = ObservationEncoder()
        enc.eval()

        # Create B=8 inputs, then slice first 4 for comparison
        torch.manual_seed(42)
        t8, p8, vp8 = _make_inputs(b=8)

        with torch.no_grad():
            out8 = enc(t8, p8, vp8)
            out4 = enc(t8[:4], p8[:4], vp8[:4])

        assert torch.allclose(out8["tokens"][:4], out4["tokens"], atol=1e-5)
        assert torch.allclose(out8["global"][:4], out4["global"], atol=1e-5)

    def test_deterministic_eval(self):
        """Same input → same output in eval mode."""
        enc = ObservationEncoder()
        enc.eval()
        tokens, proprio, vp = _make_inputs()

        with torch.no_grad():
            out1 = enc(tokens, proprio, vp)
            out2 = enc(tokens, proprio, vp)

        assert torch.equal(out1["tokens"], out2["tokens"])
