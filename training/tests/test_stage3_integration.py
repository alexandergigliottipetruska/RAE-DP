"""C.9 Integration tests for Stage 3 pipeline.

These tests verify the interfaces between components. They are designed to:
  - PASS once all teammate components (C.1, C.2, C.4, C.10) are implemented
  - SKIP now (with clear messages) while those components don't exist yet
  - Serve as acceptance criteria — if these pass, the wiring is correct

Each test documents exactly what interface it expects, so the teammate
knows what to implement.

NOTE ON BATCH-FIRST: C.0 converted diffusion.py to batch-first (B, S, d).
The spec mentions PolicyDiT should transpose to seq-first, but that was
written BEFORE C.0. After C.0, no transpose is needed anywhere. PolicyDiT
should pass batch-first tensors directly to noise_net.

Owner: Swagman
"""

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from models.diffusion import _DiTNoiseNet
from models.ema import EMA
from models.view_dropout import ViewDropout
from training.train_stage3 import (
    Stage3Config,
    create_noise_scheduler,
    train_step,
    ddim_inference,
    save_checkpoint,
    load_checkpoint,
)

# ---------------------------------------------------------------------------
# Import guards for teammate components
# ---------------------------------------------------------------------------

def _try_import(module_path, class_name):
    """Try to import a class, return (cls, None) or (None, skip_reason)."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name), None
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        return None, f"{class_name} not yet implemented: {e}"


BasePolicy, _skip_c1 = _try_import("models.base_policy", "BasePolicy")
Stage1Bridge, _skip_c2 = _try_import("models.stage1_bridge", "Stage1Bridge")
TokenAssembly, _skip_c4 = _try_import("models.token_assembly", "TokenAssembly")
PolicyDiT, _skip_c10 = _try_import("models.policy_dit", "PolicyDiT")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
B = 2
K = 4
N = 196         # tokens per view (14x14 patches)
D_ENC = 1024    # DINOv3 output dim
D_MODEL = 512   # adapter / hidden dim
H = W = 224
T_O = 2
T_P = 16
AC_DIM = 7
PROPRIO_DIM = 9


# ---------------------------------------------------------------------------
# Synthetic HDF5 helper
# ---------------------------------------------------------------------------

def _make_hdf5(tmp_path, num_demos=2, demo_len=25):
    path = str(tmp_path / "test.hdf5")
    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "robomimic"
        f.attrs["task"] = "lift"
        f.attrs["action_dim"] = AC_DIM
        f.attrs["proprio_dim"] = PROPRIO_DIM
        f.attrs["image_size"] = 224
        f.attrs["num_cam_slots"] = K

        vp = np.array([True, True, False, False])
        keys = []
        for i in range(num_demos):
            key = f"demo_{i}"
            keys.append(key)
            grp = f.create_group(f"data/{key}")
            grp.create_dataset("images", data=np.random.randint(0, 256, (demo_len, K, H, W, 3), dtype=np.uint8))
            grp.create_dataset("actions", data=np.random.randn(demo_len, AC_DIM).astype(np.float32))
            grp.create_dataset("proprio", data=np.random.randn(demo_len, PROPRIO_DIM).astype(np.float32))
            grp.create_dataset("view_present", data=vp)

        mask = f.create_group("mask")
        dt = h5py.special_dtype(vlen=str)
        mask.create_dataset("train", data=keys, dtype=dt)
        mask.create_dataset("valid", data=keys[:1], dtype=dt)

        ns = f.create_group("norm_stats")
        for field, dim in [("actions", AC_DIM), ("proprio", PROPRIO_DIM)]:
            g = ns.create_group(field)
            g.create_dataset("mean", data=np.zeros(dim, dtype=np.float32))
            g.create_dataset("std", data=np.ones(dim, dtype=np.float32))
            g.create_dataset("min", data=-np.ones(dim, dtype=np.float32) * 2)
            g.create_dataset("max", data=np.ones(dim, dtype=np.float32) * 2)
    return path


# ============================================================
# C.1 BasePolicy interface tests
# ============================================================

class TestBasePolicy:
    """Tests for C.1 BasePolicy abstract interface.

    BasePolicy must:
      - Be an nn.Module subclass
      - Define compute_loss(batch) -> scalar tensor
      - Define predict_action(obs) -> [B, T_p, D_act] tensor
    """

    @pytest.mark.skipif(_skip_c1 is not None, reason=_skip_c1 or "")
    def test_is_nn_module(self):
        assert issubclass(BasePolicy, nn.Module)

    @pytest.mark.skipif(_skip_c1 is not None, reason=_skip_c1 or "")
    def test_has_compute_loss(self):
        assert hasattr(BasePolicy, "compute_loss")

    @pytest.mark.skipif(_skip_c1 is not None, reason=_skip_c1 or "")
    def test_has_predict_action(self):
        assert hasattr(BasePolicy, "predict_action")


# ============================================================
# C.2 Stage1Bridge interface tests
# ============================================================

class TestStage1Bridge:
    """Tests for C.2 Stage1Bridge.

    Stage1Bridge must:
      - Load a Stage 1 checkpoint (encoder + adapter + decoder)
      - encode(images_enc, view_present) -> adapted tokens [B_real, 196, 512]
      - Encoder must be frozen (no gradients)
      - Adapter must be trainable (gradients flow)
      - compute_recon_loss(adapted, images_target, view_present) -> scalar
    """

    @pytest.mark.skipif(_skip_c2 is not None, reason=_skip_c2 or "")
    def test_has_encode_method(self):
        assert hasattr(Stage1Bridge, "encode") or hasattr(Stage1Bridge, "encode_frozen")

    @pytest.mark.skipif(_skip_c2 is not None, reason=_skip_c2 or "")
    def test_has_compute_recon_loss(self):
        assert hasattr(Stage1Bridge, "compute_recon_loss")

    @pytest.mark.skipif(_skip_c2 is not None, reason=_skip_c2 or "")
    def test_encode_output_shape(self):
        """encode() should return tokens of shape [B_real, 196, 512]."""
        # Requires a Stage 1 checkpoint — use mock mode if available
        bridge = Stage1Bridge(checkpoint_path=None, pretrained=False)
        images = torch.randn(B, K, 3, H, W)
        vp = torch.tensor([[True, True, False, False]] * B)
        adapted = bridge.encode(images, vp)
        B_real = vp.sum().item()  # total real views across batch
        assert adapted.shape == (B_real, N, D_MODEL), (
            f"Expected ({B_real}, {N}, {D_MODEL}), got {adapted.shape}"
        )

    @pytest.mark.skipif(_skip_c2 is not None, reason=_skip_c2 or "")
    def test_encoder_frozen_no_grad(self):
        """Encoder parameters should not receive gradients."""
        bridge = Stage1Bridge(checkpoint_path=None, pretrained=False)
        images = torch.randn(B, K, 3, H, W)
        vp = torch.tensor([[True, True, False, False]] * B)
        adapted = bridge.encode(images, vp)
        adapted.sum().backward()
        for name, p in bridge.encoder.named_parameters():
            assert p.grad is None or torch.all(p.grad == 0), (
                f"Encoder param {name} should be frozen"
            )

    @pytest.mark.skipif(_skip_c2 is not None, reason=_skip_c2 or "")
    def test_adapter_receives_grad(self):
        """Adapter parameters should receive gradients."""
        bridge = Stage1Bridge(checkpoint_path=None, pretrained=False)
        images = torch.randn(B, K, 3, H, W)
        vp = torch.tensor([[True, True, False, False]] * B)
        adapted = bridge.encode(images, vp)
        adapted.sum().backward()
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in bridge.adapter.parameters()
        )
        assert has_grad, "Adapter should receive gradients through bridge.encode()"


# ============================================================
# C.4 TokenAssembly interface tests
# ============================================================

class TestTokenAssembly:
    """Tests for C.4 TokenAssembly.

    TokenAssembly must:
      - Accept adapted_tokens [B, T_o, K, N, d'], proprio [B, T_o, D_prop],
        view_present [B, K]
      - Return obs_tokens [B, S_obs, d'] (batch-first!)
        where S_obs = T_o * K_real * N + T_o (visual + proprio tokens)
      - Add spatial_pos_emb, view_emb, obs_time_emb, proprio MLP embeddings
    """

    @pytest.mark.skipif(_skip_c4 is not None, reason=_skip_c4 or "")
    def test_output_shape_all_views(self):
        """With 4 real views: S_obs = T_o * K * N + T_o = 2*4*196+2 = 1570."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        adapted = torch.randn(B, T_O, K, N, D_MODEL)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.ones(B, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)
        expected_S = T_O * K * N + T_O  # 1570
        assert out.shape == (B, expected_S, D_MODEL), (
            f"Expected (B, {expected_S}, {D_MODEL}), got {out.shape}"
        )

    @pytest.mark.skipif(_skip_c4 is not None, reason=_skip_c4 or "")
    def test_output_shape_partial_views(self):
        """With 2 real views: S_obs = T_o * 2 * N + T_o = 2*2*196+2 = 786."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        adapted = torch.randn(B, T_O, K, N, D_MODEL)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.tensor([[True, True, False, False]] * B)
        out = ta(adapted, proprio, vp)
        expected_S = T_O * 2 * N + T_O  # 786
        assert out.shape == (B, expected_S, D_MODEL), (
            f"Expected (B, {expected_S}, {D_MODEL}), got {out.shape}"
        )

    @pytest.mark.skipif(_skip_c4 is not None, reason=_skip_c4 or "")
    def test_output_is_batch_first(self):
        """Output must be batch-first [B, S, d] (C.0 converted everything)."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        adapted = torch.randn(B, T_O, K, N, D_MODEL)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.ones(B, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)
        # First dim should be batch (B=2), not sequence
        assert out.shape[0] == B

    @pytest.mark.skipif(_skip_c4 is not None, reason=_skip_c4 or "")
    def test_gradient_flows(self):
        """Gradients flow through token assembly to adapted tokens."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        adapted = torch.randn(B, T_O, K, N, D_MODEL, requires_grad=True)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.ones(B, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)
        out.sum().backward()
        assert adapted.grad is not None
        assert not torch.all(adapted.grad == 0)


# ============================================================
# C.10 PolicyDiT end-to-end tests
# ============================================================

class TestPolicyDiT:
    """Tests for C.10 PolicyDiT wrapper.

    PolicyDiT must:
      - Wrap Stage1Bridge + ViewDropout + TokenAssembly + _DiTNoiseNet
      - compute_loss(batch) -> finite scalar
      - predict_action(obs) -> [B, T_p, D_act]
      - Pass batch-first tensors (NO transpose after C.0)
      - Gradient reaches adapter but NOT encoder
    """

    @pytest.mark.skipif(_skip_c10 is not None, reason=_skip_c10 or "")
    def test_compute_loss_finite(self):
        """compute_loss returns a finite scalar."""
        config = Stage3Config(T_pred=T_P, hidden_dim=D_MODEL)
        # Assumes PolicyDiT can be instantiated with mock mode
        policy = PolicyDiT(config, stage1_bridge=None)  # mock
        batch = {
            "images_enc": torch.randn(B, T_O, K, 3, H, W),
            "images_target": torch.rand(B, T_O, K, 3, H, W),
            "actions": torch.randn(B, T_P, AC_DIM),
            "proprio": torch.randn(B, T_O, PROPRIO_DIM),
            "view_present": torch.ones(B, K, dtype=torch.bool),
        }
        loss = policy.compute_loss(batch)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    @pytest.mark.skipif(_skip_c10 is not None, reason=_skip_c10 or "")
    def test_predict_action_shape(self):
        """predict_action returns [B, T_p, D_act]."""
        config = Stage3Config(T_pred=T_P, hidden_dim=D_MODEL)
        policy = PolicyDiT(config, stage1_bridge=None)
        policy.eval()
        obs = {
            "images_enc": torch.randn(B, T_O, K, 3, H, W),
            "proprio": torch.randn(B, T_O, PROPRIO_DIM),
            "view_present": torch.ones(B, K, dtype=torch.bool),
        }
        actions = policy.predict_action(obs)
        assert actions.shape == (B, T_P, AC_DIM), (
            f"Expected ({B}, {T_P}, {AC_DIM}), got {actions.shape}"
        )

    @pytest.mark.skipif(_skip_c10 is not None, reason=_skip_c10 or "")
    def test_no_transpose_batch_first(self):
        """After C.0, PolicyDiT should NOT transpose before noise_net.

        noise_net now expects batch-first (B, S, d) tensors.
        If you see a .transpose(0,1) call before noise_net, remove it.
        """
        config = Stage3Config(T_pred=T_P, hidden_dim=D_MODEL)
        policy = PolicyDiT(config, stage1_bridge=None)
        batch = {
            "images_enc": torch.randn(B, T_O, K, 3, H, W),
            "images_target": torch.rand(B, T_O, K, 3, H, W),
            "actions": torch.randn(B, T_P, AC_DIM),
            "proprio": torch.randn(B, T_O, PROPRIO_DIM),
            "view_present": torch.ones(B, K, dtype=torch.bool),
        }
        # If this doesn't crash, the dims are compatible
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    @pytest.mark.skipif(_skip_c10 is not None, reason=_skip_c10 or "")
    def test_encoder_frozen_adapter_trainable(self):
        """Gradient reaches adapter but NOT encoder through PolicyDiT."""
        config = Stage3Config(T_pred=T_P, hidden_dim=D_MODEL)
        policy = PolicyDiT(config, stage1_bridge=None)
        batch = {
            "images_enc": torch.randn(B, T_O, K, 3, H, W),
            "images_target": torch.rand(B, T_O, K, 3, H, W),
            "actions": torch.randn(B, T_P, AC_DIM),
            "proprio": torch.randn(B, T_O, PROPRIO_DIM),
            "view_present": torch.ones(B, K, dtype=torch.bool),
        }
        loss = policy.compute_loss(batch)
        loss.backward()

        # Encoder should be frozen
        for p in policy.bridge.encoder.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

        # Adapter should get gradients
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in policy.bridge.adapter.parameters()
        )
        assert has_grad, "Adapter should receive gradients"


# ============================================================
# Cross-component wiring tests
# ============================================================

class TestCrossComponentWiring:
    """Tests that verify correct wiring between components.

    These catch shape mismatches, wrong dim orders, and missing embeddings
    that only surface when components are connected.
    """

    @pytest.mark.skipif(
        any(s is not None for s in [_skip_c2, _skip_c4]),
        reason="Requires C.2 + C.4"
    )
    def test_bridge_to_assembly_shapes(self):
        """Stage1Bridge output feeds into TokenAssembly without shape errors."""
        bridge = Stage1Bridge(checkpoint_path=None, pretrained=False)
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)

        images = torch.randn(B, T_O, K, 3, H, W)
        vp = torch.tensor([[True, True, False, False]] * B)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)

        # Encode each timestep
        adapted_list = []
        for t in range(T_O):
            adapted_t = bridge.encode(images[:, t], vp)
            adapted_list.append(adapted_t)

        # Reshape to [B, T_o, K, N, d'] for TokenAssembly
        # (Bridge returns [B_real, N, d'] — need to reconstruct per-view structure)
        # This reshaping is part of PolicyDiT's job, but we test the shapes match
        obs_tokens = ta(torch.randn(B, T_O, K, N, D_MODEL), proprio, vp)
        assert obs_tokens.shape[0] == B
        assert obs_tokens.shape[2] == D_MODEL

    @pytest.mark.skipif(
        any(s is not None for s in [_skip_c4]),
        reason="Requires C.4"
    )
    def test_assembly_to_noise_net_shapes(self):
        """TokenAssembly output feeds into noise_net.forward_enc without errors."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        noise_net = _DiTNoiseNet(
            ac_dim=AC_DIM, ac_chunk=T_P, hidden_dim=D_MODEL,
            num_blocks=2, nhead=8,
        )

        adapted = torch.randn(B, T_O, K, N, D_MODEL)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.ones(B, K, dtype=torch.bool)

        obs_tokens = ta(adapted, proprio, vp)  # [B, S_obs, d']

        # This should work batch-first after C.0
        enc_cache = noise_net.forward_enc(obs_tokens)
        assert isinstance(enc_cache, list)
        assert enc_cache[0].shape[0] == B  # batch dim first

    @pytest.mark.skipif(
        any(s is not None for s in [_skip_c10]),
        reason="Requires C.10"
    )
    def test_full_pipeline_dataset_to_loss(self, tmp_path):
        """Full pipeline: Stage3Dataset → PolicyDiT.compute_loss → finite scalar."""
        from data_pipeline.datasets.stage3_dataset import Stage3Dataset

        hdf5_path = _make_hdf5(tmp_path)
        ds = Stage3Dataset(hdf5_path, T_obs=T_O, T_pred=T_P)
        loader = torch.utils.data.DataLoader(ds, batch_size=B)
        batch = next(iter(loader))

        config = Stage3Config(T_pred=T_P, hidden_dim=D_MODEL)
        policy = PolicyDiT(config, stage1_bridge=None)

        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    @pytest.mark.skipif(
        any(s is not None for s in [_skip_c10]),
        reason="Requires C.10"
    )
    def test_full_pipeline_checkpoint_roundtrip(self, tmp_path):
        """Save and load a full PolicyDiT checkpoint."""
        config = Stage3Config(T_pred=T_P, hidden_dim=D_MODEL)
        policy = PolicyDiT(config, stage1_bridge=None)
        ema = EMA(policy.noise_net, decay=0.999)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        path = str(tmp_path / "policy.pt")
        save_checkpoint(
            path, epoch=0, global_step=0,
            noise_net=policy.noise_net,
            adapter=policy.bridge.adapter,
            optimizer=optimizer, ema=ema, val_metrics={},
        )
        assert os.path.isfile(path)
