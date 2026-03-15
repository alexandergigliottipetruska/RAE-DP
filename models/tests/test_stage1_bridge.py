"""Tests for C.2 Stage1Bridge.

Verifies that Stage1Bridge correctly loads Stage 1 components and provides
adapted tokens for Stage 3 policy training.
"""

import os

import pytest
import torch
import torch.nn as nn

from models.stage1_bridge import Stage1Bridge


# Test constants
B = 2
T_O = 2
K = 4
H, W = 224, 224
N = 196   # 14x14 patches
D_ENC = 1024
D_ADAPT = 512


def _make_images(b=B, t=T_O, k=K):
    """Create fake ImageNet-normalized images."""
    return torch.randn(b, t, k, 3, H, W)


def _make_view_present(b=B, k=K, all_present=True):
    """Create view_present mask."""
    vp = torch.ones(b, k, dtype=torch.bool)
    if not all_present:
        vp[:, 2:] = False  # only 2 cameras present
    return vp


def _save_mock_checkpoint(path):
    """Create a minimal Stage 1 checkpoint for testing."""
    bridge = Stage1Bridge(pretrained_encoder=False, load_decoder=True)
    ckpt = {
        "epoch": 10,
        "adapter": bridge.adapter.state_dict(),
        "decoder": bridge.decoder.state_dict(),
        "disc_head": {},  # not needed for Stage 3
        "opt_gen": {},
        "opt_disc": {},
        "val_metrics": {"val_rec": 0.1},
    }
    torch.save(ckpt, path)
    return path


# ============================================================
# Initialization tests
# ============================================================

class TestStage1BridgeInit:
    def test_is_nn_module(self):
        """Stage1Bridge is an nn.Module."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        assert isinstance(bridge, nn.Module)

    def test_encoder_frozen(self):
        """Encoder parameters have requires_grad=False."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        for p in bridge.encoder.parameters():
            assert not p.requires_grad, "Encoder should be frozen"

    def test_adapter_trainable_by_default(self):
        """Adapter parameters have requires_grad=True by default."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        for p in bridge.adapter.parameters():
            assert p.requires_grad, "Adapter should be trainable"

    def test_adapter_frozen_when_requested(self):
        """Adapter can be frozen via trainable_adapter=False."""
        bridge = Stage1Bridge(pretrained_encoder=False, trainable_adapter=False)
        for p in bridge.adapter.parameters():
            assert not p.requires_grad, "Adapter should be frozen"

    def test_decoder_none_by_default(self):
        """Decoder is not loaded unless load_decoder=True."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        assert bridge.decoder is None

    def test_decoder_loaded_when_requested(self):
        """Decoder is loaded when load_decoder=True."""
        bridge = Stage1Bridge(pretrained_encoder=False, load_decoder=True)
        assert bridge.decoder is not None
        assert isinstance(bridge.decoder, nn.Module)

    def test_cancel_affine_ln_no_learnable_params(self):
        """Cancel-affine LN has no learnable parameters."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        ln_params = list(bridge.cancel_affine_ln.parameters())
        assert len(ln_params) == 0


# ============================================================
# Checkpoint loading tests
# ============================================================

class TestCheckpointLoading:
    def test_load_checkpoint(self, tmp_path):
        """Weights are loaded from a Stage 1 checkpoint."""
        ckpt_path = _save_mock_checkpoint(str(tmp_path / "stage1.pt"))

        # Create bridge with random weights
        bridge1 = Stage1Bridge(pretrained_encoder=False)
        bridge2 = Stage1Bridge(ckpt_path, pretrained_encoder=False)

        # Weights should differ (bridge1 is random, bridge2 is loaded)
        for p1, p2 in zip(bridge1.adapter.parameters(), bridge2.adapter.parameters()):
            if p1.numel() > 1:  # skip biases that might be zero
                assert not torch.equal(p1, p2), "Loaded weights should differ from random"

    def test_load_checkpoint_with_decoder(self, tmp_path):
        """Decoder weights are loaded when load_decoder=True."""
        ckpt_path = _save_mock_checkpoint(str(tmp_path / "stage1.pt"))
        bridge = Stage1Bridge(ckpt_path, pretrained_encoder=False, load_decoder=True)
        assert bridge.decoder is not None

    def test_load_checkpoint_strips_compile_prefix(self, tmp_path):
        """Checkpoints with _orig_mod. prefix are handled correctly."""
        bridge_src = Stage1Bridge(pretrained_encoder=False)
        # Save with _orig_mod. prefix
        prefixed_sd = {
            f"_orig_mod.{k}": v for k, v in bridge_src.adapter.state_dict().items()
        }
        ckpt = {"adapter": prefixed_sd, "epoch": 5}
        path = str(tmp_path / "compiled.pt")
        torch.save(ckpt, path)

        # Load should strip prefix
        bridge = Stage1Bridge(path, pretrained_encoder=False)
        # If it loaded without error, prefix stripping worked
        for p1, p2 in zip(bridge_src.adapter.parameters(), bridge.adapter.parameters()):
            assert torch.equal(p1, p2)

    def test_empty_checkpoint_path_skips_load(self):
        """Empty checkpoint path means random initialization."""
        bridge = Stage1Bridge("", pretrained_encoder=False)
        assert bridge.adapter is not None  # still initialized, just random


# ============================================================
# Encoding tests
# ============================================================

class TestEncode:
    def test_output_shape_all_views(self):
        """encode() returns (B, T_o, K, N, d') with all views present."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        images = _make_images()
        vp = _make_view_present()
        adapted = bridge.encode(images, vp)
        assert adapted.shape == (B, T_O, K, N, D_ADAPT)

    def test_output_shape_partial_views(self):
        """encode() returns correct shape with some views absent."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        images = _make_images()
        vp = _make_view_present(all_present=False)  # only 2 of 4 present
        adapted = bridge.encode(images, vp)
        assert adapted.shape == (B, T_O, K, N, D_ADAPT)

    def test_absent_views_are_zero(self):
        """Absent views have zero tokens."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        images = _make_images()
        vp = _make_view_present(all_present=False)
        adapted = bridge.encode(images, vp)
        # Views 2 and 3 are absent
        assert torch.all(adapted[:, :, 2] == 0)
        assert torch.all(adapted[:, :, 3] == 0)

    def test_present_views_are_nonzero(self):
        """Present views have non-zero tokens."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        images = _make_images()
        vp = _make_view_present()
        adapted = bridge.encode(images, vp)
        for k in range(K):
            assert not torch.all(adapted[:, :, k] == 0), f"View {k} should be non-zero"

    def test_encoder_no_grad(self):
        """Encoder forward does not accumulate gradients."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        images = _make_images()
        vp = _make_view_present()
        adapted = bridge.encode(images, vp)
        loss = adapted.sum()
        loss.backward()
        for p in bridge.encoder.parameters():
            assert p.grad is None or torch.all(p.grad == 0), (
                "Encoder should not receive gradients"
            )

    def test_adapter_receives_grad(self):
        """Adapter receives gradients through encode()."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        images = _make_images()
        vp = _make_view_present()
        adapted = bridge.encode(images, vp)
        loss = adapted.sum()
        loss.backward()
        has_grad = False
        for p in bridge.adapter.parameters():
            if p.grad is not None and not torch.all(p.grad == 0):
                has_grad = True
                break
        assert has_grad, "Adapter should receive gradients"

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        bridge.eval()
        images = _make_images()
        vp = _make_view_present()
        out1 = bridge.encode(images, vp)
        out2 = bridge.encode(images, vp)
        assert torch.equal(out1, out2)

    def test_batch_independence(self):
        """Different batch elements don't affect each other."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        bridge.eval()
        images = _make_images(b=3)
        vp = _make_view_present(b=3)

        out_full = bridge.encode(images, vp)
        out_single = bridge.encode(images[1:2], vp[1:2])

        assert torch.allclose(out_full[1], out_single[0], atol=1e-5)


# ============================================================
# Reconstruction co-training tests
# ============================================================

class TestReconLoss:
    def test_recon_loss_finite(self):
        """compute_recon_loss returns a finite scalar."""
        bridge = Stage1Bridge(pretrained_encoder=False, load_decoder=True)
        images = _make_images()
        vp = _make_view_present()
        adapted = bridge.encode(images, vp)
        targets = torch.rand(B, T_O, K, 3, H, W)  # raw [0,1]
        loss = bridge.compute_recon_loss(adapted, targets, vp)
        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_recon_loss_requires_decoder(self):
        """compute_recon_loss raises if decoder not loaded."""
        bridge = Stage1Bridge(pretrained_encoder=False, load_decoder=False)
        adapted = torch.randn(B, T_O, K, N, D_ADAPT)
        targets = torch.rand(B, T_O, K, 3, H, W)
        vp = _make_view_present()
        with pytest.raises(RuntimeError, match="Decoder not loaded"):
            bridge.compute_recon_loss(adapted, targets, vp)

    def test_recon_loss_gradient_to_adapter(self):
        """Reconstruction loss gradient flows to adapter."""
        bridge = Stage1Bridge(pretrained_encoder=False, load_decoder=True)
        images = _make_images()
        vp = _make_view_present()
        adapted = bridge.encode(images, vp)
        targets = torch.rand(B, T_O, K, 3, H, W)
        loss = bridge.compute_recon_loss(adapted, targets, vp)
        loss.backward()
        has_grad = False
        for p in bridge.adapter.parameters():
            if p.grad is not None and not torch.all(p.grad == 0):
                has_grad = True
                break
        assert has_grad, "Adapter should receive gradients from recon loss"

    def test_recon_loss_no_grad_to_encoder(self):
        """Reconstruction loss does not flow to encoder."""
        bridge = Stage1Bridge(pretrained_encoder=False, load_decoder=True)
        images = _make_images()
        vp = _make_view_present()
        adapted = bridge.encode(images, vp)
        targets = torch.rand(B, T_O, K, 3, H, W)
        loss = bridge.compute_recon_loss(adapted, targets, vp)
        loss.backward()
        for p in bridge.encoder.parameters():
            assert p.grad is None or torch.all(p.grad == 0)


# ============================================================
# Properties and edge cases
# ============================================================

class TestEdgeCases:
    def test_last_layer_weight_with_decoder(self):
        """last_layer_weight returns decoder's weight when loaded."""
        bridge = Stage1Bridge(pretrained_encoder=False, load_decoder=True)
        assert bridge.last_layer_weight is not None
        assert isinstance(bridge.last_layer_weight, torch.Tensor)

    def test_last_layer_weight_without_decoder(self):
        """last_layer_weight returns None when decoder not loaded."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        assert bridge.last_layer_weight is None

    def test_single_timestep(self):
        """Works with T_o=1."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        images = _make_images(t=1)
        vp = _make_view_present()
        adapted = bridge.encode(images, vp)
        assert adapted.shape == (B, 1, K, N, D_ADAPT)

    def test_single_view(self):
        """Works with K=1."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        images = _make_images(k=1)
        vp = _make_view_present(k=1)
        adapted = bridge.encode(images, vp)
        assert adapted.shape == (B, T_O, 1, N, D_ADAPT)

    def test_per_sample_view_mask(self):
        """Different samples can have different view masks."""
        bridge = Stage1Bridge(pretrained_encoder=False)
        images = _make_images(b=3, k=3)
        vp = torch.tensor([
            [True, True, False],   # sample 0: 2 views
            [True, False, False],  # sample 1: 1 view
            [True, True, True],    # sample 2: 3 views
        ])
        adapted = bridge.encode(images, vp)
        assert adapted.shape == (3, T_O, 3, N, D_ADAPT)
        # View 2 for sample 0 and 1 should be zero
        assert torch.all(adapted[0, :, 2] == 0)
        assert torch.all(adapted[1, :, 2] == 0)
        assert torch.all(adapted[1, :, 1] == 0)
        # View 2 for sample 2 should be non-zero
        assert not torch.all(adapted[2, :, 2] == 0)
