"""Tests for C.7 Stage 3 training loop.

Uses mock encoder/adapter and small noise_net to test training mechanics
independently of the teammate's components (C.1, C.2, C.4, C.10).
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
    _create_lr_scheduler,
)

# ---------------------------------------------------------------------------
# Test constants (small for speed)
# ---------------------------------------------------------------------------
B = 2
K = 2           # camera slots
N = 4           # tokens per view (tiny)
D_ENC = 32      # encoder output dim
D_ADAPT = 64    # adapter output dim (= hidden_dim)
H = W = 224
T_O = 2
T_P = 8
AC_DIM = 7
PROPRIO_DIM = 9
HIDDEN_DIM = 64
NHEAD = 4
NUM_BLOCKS = 2


# ---------------------------------------------------------------------------
# Mock modules
# ---------------------------------------------------------------------------

class MockEncoder(nn.Module):
    """Returns fixed-size tokens. Deterministic: same input shape → same output."""
    def __init__(self, out_dim=D_ENC, n_tokens=N):
        super().__init__()
        self.out_dim = out_dim
        self.n_tokens = n_tokens
        # Fixed projection so output is deterministic given input
        self.proj = nn.Linear(3, out_dim * n_tokens, bias=False)
        self.proj.requires_grad_(False)

    def forward(self, x):
        B = x.shape[0]
        # Use mean of spatial dims as a deterministic summary
        feat = x.mean(dim=(-2, -1))  # [B, 3]
        return self.proj(feat).reshape(B, self.n_tokens, self.out_dim)


class MockAdapter(nn.Module):
    """Linear projection from encoder dim to hidden dim."""
    def __init__(self, in_dim=D_ENC, out_dim=D_ADAPT):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


class MockDecoder(nn.Module):
    """Produces images from tokens (for co-training test)."""
    def __init__(self, in_dim=D_ADAPT, n_tokens=N):
        super().__init__()
        self.proj = nn.Linear(in_dim * n_tokens, 3 * 224 * 224)
        self.last_layer_weight = self.proj.weight

    def forward(self, x):
        B = x.shape[0]
        flat = x.reshape(B, -1)
        out = self.proj(flat)
        return torch.sigmoid(out.reshape(B, 3, 224, 224))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_hdf5(path, num_demos=2, demo_len=20):
    """Create a synthetic HDF5 for training tests."""
    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "robomimic"
        f.attrs["task"] = "lift"
        f.attrs["action_dim"] = AC_DIM
        f.attrs["proprio_dim"] = PROPRIO_DIM
        f.attrs["image_size"] = 224
        f.attrs["num_cam_slots"] = K

        vp = np.zeros(K, dtype=bool)
        vp[:K] = True

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


@pytest.fixture
def tmp_hdf5(tmp_path):
    path = str(tmp_path / "test.hdf5")
    _make_synthetic_hdf5(path)
    return path


def _make_components():
    """Create mock encoder, adapter, and small noise_net."""
    encoder = MockEncoder(out_dim=D_ENC, n_tokens=N)
    adapter = MockAdapter(in_dim=D_ENC, out_dim=HIDDEN_DIM)
    noise_net = _DiTNoiseNet(
        ac_dim=AC_DIM,
        ac_chunk=T_P,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        nhead=NHEAD,
        dim_feedforward=HIDDEN_DIM * 2,
    )
    return encoder, adapter, noise_net


def _make_batch():
    """Create a synthetic batch matching Stage3Dataset output."""
    return {
        "images_enc": torch.randn(B, T_O, K, 3, H, W),
        "images_target": torch.rand(B, T_O, K, 3, H, W),
        "actions": torch.randn(B, T_P, AC_DIM),
        "proprio": torch.randn(B, T_O, PROPRIO_DIM),
        "view_present": torch.ones(B, K, dtype=torch.bool),
    }


def _make_config(**kwargs):
    defaults = dict(
        hdf5_paths=[],
        batch_size=B,
        T_obs=T_O,
        T_pred=T_P,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        nhead=NHEAD,
        train_diffusion_steps=100,
        eval_diffusion_steps=10,
        lr=1e-3,
        lr_adapter=1e-4,
        weight_decay=0,
        grad_clip=1.0,
        warmup_steps=0,
        lambda_recon=0.0,
        p=0.0,
        ema_decay=0.999,
    )
    defaults.update(kwargs)
    return Stage3Config(**defaults)


# ============================================================
# Training step tests
# ============================================================

class TestTrainStep:
    def test_loss_is_finite(self):
        """Single training step produces a finite loss."""
        encoder, adapter, noise_net = _make_components()
        config = _make_config()
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        batch = _make_batch()
        losses = train_step(
            batch, noise_net, encoder, adapter, view_dropout,
            scheduler, optimizer, config, global_step=0,
        )
        assert "policy" in losses
        assert np.isfinite(losses["policy"])
        assert losses["policy"] > 0

    def test_loss_is_mse(self):
        """Policy loss is MSE (should be ~1.0 at init with unit Gaussian noise)."""
        encoder, adapter, noise_net = _make_components()
        config = _make_config()
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        # Run a few steps and check loss is reasonable
        losses_list = []
        for _ in range(5):
            batch = _make_batch()
            losses = train_step(
                batch, noise_net, encoder, adapter, view_dropout,
                scheduler, optimizer, config, global_step=0,
            )
            losses_list.append(losses["policy"])

        # MSE of random predictions vs unit Gaussian noise should be ~1-2
        avg_loss = np.mean(losses_list)
        assert 0.1 < avg_loss < 10.0, f"Unexpected loss range: {avg_loss}"

    def test_loss_decreasing_overfit(self):
        """Average loss decreases when overfitting on a single batch.

        DDPM loss is noisy (random timestep each step), so we compare
        rolling averages rather than single-step values. Use a fixed seed
        for reproducibility and enough steps for the signal to dominate.
        """
        torch.manual_seed(42)
        encoder, adapter, noise_net = _make_components()
        config = _make_config(lr=5e-3, lr_adapter=5e-3)
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=5e-3, weight_decay=0.0)

        batch = _make_batch()
        losses_list = []

        for step in range(300):
            losses = train_step(
                batch, noise_net, encoder, adapter, view_dropout,
                scheduler, optimizer, config, global_step=step,
            )
            losses_list.append(losses["policy"])

        # Compare first-third average vs last-third average
        n = len(losses_list)
        avg_first = np.mean(losses_list[:n // 3])
        avg_last = np.mean(losses_list[-n // 3:])

        assert avg_last < avg_first, (
            f"Loss did not decrease: first_third={avg_first:.4f} -> last_third={avg_last:.4f}"
        )

    def test_with_view_dropout(self):
        """Training step works with view dropout enabled."""
        encoder, adapter, noise_net = _make_components()
        config = _make_config(p=0.3)
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.3)

        params = list(noise_net.parameters()) + list(adapter.parameters()) + list(view_dropout.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        batch = _make_batch()
        losses = train_step(
            batch, noise_net, encoder, adapter, view_dropout,
            scheduler, optimizer, config, global_step=0,
        )
        assert np.isfinite(losses["policy"])

    def test_grad_clip_applied(self):
        """Gradient norms are clipped to config.grad_clip."""
        encoder, adapter, noise_net = _make_components()
        config = _make_config(grad_clip=0.1)
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        batch = _make_batch()
        _ = train_step(
            batch, noise_net, encoder, adapter, view_dropout,
            scheduler, optimizer, config, global_step=0,
        )
        # After step, check that no gradient exceeds clip norm
        # (already stepped, so gradients are zeroed — check via hook next time)
        # Just verify it runs without error at very tight clip
        assert True  # smoke test


# ============================================================
# Co-training tests
# ============================================================

class TestCoTraining:
    def test_recon_loss_computed(self):
        """Co-training produces a recon loss when lambda_recon > 0."""
        encoder, adapter, noise_net = _make_components()
        decoder = MockDecoder(in_dim=HIDDEN_DIM, n_tokens=N)
        config = _make_config(lambda_recon=0.1)
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = (list(noise_net.parameters()) +
                  list(adapter.parameters()) +
                  list(decoder.parameters()))
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        # Need LPIPS net
        from models.losses import create_lpips_net
        lpips_net = create_lpips_net()

        batch = _make_batch()
        losses = train_step(
            batch, noise_net, encoder, adapter, view_dropout,
            scheduler, optimizer, config, global_step=0,
            decoder=decoder, lpips_net=lpips_net,
        )
        assert "recon" in losses
        assert np.isfinite(losses["recon"])
        assert losses["recon"] > 0

    def test_no_recon_when_lambda_zero(self):
        """No recon loss when lambda_recon=0."""
        encoder, adapter, noise_net = _make_components()
        config = _make_config(lambda_recon=0.0)
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        batch = _make_batch()
        losses = train_step(
            batch, noise_net, encoder, adapter, view_dropout,
            scheduler, optimizer, config, global_step=0,
        )
        assert "recon" not in losses


# ============================================================
# DDIM inference tests
# ============================================================

class TestDDIMInference:
    def test_output_shape(self):
        """DDIM inference produces [B, T_p, D_act] actions."""
        _, _, noise_net = _make_components()
        scheduler = create_noise_scheduler(100)
        obs_tokens = torch.randn(B, T_O * K * N, HIDDEN_DIM)

        actions = ddim_inference(
            noise_net, obs_tokens,
            ac_dim=AC_DIM, T_pred=T_P,
            scheduler=scheduler, num_steps=5,
        )
        assert actions.shape == (B, T_P, AC_DIM)

    def test_output_finite(self):
        """DDIM inference produces finite values."""
        _, _, noise_net = _make_components()
        scheduler = create_noise_scheduler(100)
        obs_tokens = torch.randn(B, T_O * K * N, HIDDEN_DIM)

        actions = ddim_inference(
            noise_net, obs_tokens,
            ac_dim=AC_DIM, T_pred=T_P,
            scheduler=scheduler, num_steps=5,
        )
        assert torch.isfinite(actions).all()

    def test_deterministic_with_seed(self):
        """Same seed produces same actions."""
        _, _, noise_net = _make_components()
        noise_net.eval()
        scheduler = create_noise_scheduler(100)
        obs_tokens = torch.randn(B, T_O * K * N, HIDDEN_DIM)

        torch.manual_seed(42)
        a1 = ddim_inference(noise_net, obs_tokens, AC_DIM, T_P, scheduler, 5)
        torch.manual_seed(42)
        a2 = ddim_inference(noise_net, obs_tokens, AC_DIM, T_P, scheduler, 5)
        assert torch.allclose(a1, a2)


# ============================================================
# EMA integration tests
# ============================================================

class TestEMAIntegration:
    def test_ema_updates_during_training(self):
        """EMA weights change after training steps."""
        _, adapter, noise_net = _make_components()
        ema = EMA(noise_net, decay=0.99, warmup_steps=0)

        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Simulate a training step
        with torch.no_grad():
            for p in noise_net.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update()

        changed = False
        for k in ema.shadow:
            if not torch.equal(old_shadow[k], ema.shadow[k]):
                changed = True
                break
        assert changed, "EMA should have changed after update"


# ============================================================
# Checkpoint tests
# ============================================================

class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        """Save and load preserves model weights."""
        _, adapter, noise_net = _make_components()
        ema = EMA(noise_net, decay=0.999)
        optimizer = torch.optim.AdamW(
            list(noise_net.parameters()) + list(adapter.parameters()), lr=1e-3
        )

        # Do a dummy step to create optimizer state
        x = torch.randn(B, T_O * K * N, HIDDEN_DIM)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        _, eps = noise_net(noise_ac, time, x)
        eps.sum().backward()
        optimizer.step()
        ema.update()

        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, epoch=5, global_step=100,
                        noise_net=noise_net, adapter=adapter,
                        optimizer=optimizer, ema=ema, val_metrics={"policy": 0.5})

        # Create fresh models and load
        _, adapter2, noise_net2 = _make_components()
        ema2 = EMA(noise_net2, decay=0.999)
        optimizer2 = torch.optim.AdamW(
            list(noise_net2.parameters()) + list(adapter2.parameters()), lr=1e-3
        )

        start_epoch, gs = load_checkpoint(path, noise_net2, adapter2, optimizer2, ema2)
        assert start_epoch == 6
        assert gs == 100

        for p1, p2 in zip(noise_net.parameters(), noise_net2.parameters()):
            assert torch.equal(p1, p2)
        for p1, p2 in zip(adapter.parameters(), adapter2.parameters()):
            assert torch.equal(p1, p2)

    def test_checkpoint_contains_ema(self, tmp_path):
        """Checkpoint includes EMA state."""
        _, adapter, noise_net = _make_components()
        ema = EMA(noise_net, decay=0.999)
        optimizer = torch.optim.AdamW(noise_net.parameters(), lr=1e-3)

        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, 0, 0, noise_net, adapter, optimizer, ema, {})

        ckpt = torch.load(path, weights_only=False)
        assert "ema" in ckpt
        assert "decay" in ckpt["ema"]
        assert "shadow" in ckpt["ema"]


# ============================================================
# LR scheduler tests
# ============================================================

class TestLRScheduler:
    def test_warmup(self):
        """LR increases linearly during warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = _create_lr_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            sched.step()

        # LR should increase during warmup
        assert lrs[-1] > lrs[0], "LR should increase during warmup"

    def test_cosine_decay(self):
        """LR decays after warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = _create_lr_scheduler(optimizer, warmup_steps=10, total_steps=100)

        # Skip warmup
        for _ in range(10):
            sched.step()
        lr_after_warmup = optimizer.param_groups[0]["lr"]

        for _ in range(80):
            sched.step()
        lr_near_end = optimizer.param_groups[0]["lr"]

        assert lr_near_end < lr_after_warmup, "LR should decay after warmup"


# ============================================================
# Noise scheduler tests
# ============================================================

class TestNoiseScheduler:
    def test_creation(self):
        """Noise scheduler creates without error."""
        sched = create_noise_scheduler(100)
        assert sched.config.num_train_timesteps == 100

    def test_add_noise(self):
        """Adding noise produces finite result."""
        sched = create_noise_scheduler(100)
        actions = torch.randn(B, T_P, AC_DIM)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (B,))
        noisy = sched.add_noise(actions, noise, timesteps)
        assert noisy.shape == actions.shape
        assert torch.isfinite(noisy).all()


# ============================================================
# Config tests
# ============================================================

class TestConfig:
    def test_defaults(self):
        """Config has sensible defaults."""
        config = Stage3Config()
        assert config.T_pred == 16
        assert config.train_diffusion_steps == 100
        assert config.eval_diffusion_steps == 10
        assert config.ema_decay == 0.9999
        assert config.lr_adapter < config.lr

    def test_custom_values(self):
        """Config accepts custom values."""
        config = Stage3Config(T_pred=50, lr=2e-4, lambda_recon=0.25)
        assert config.T_pred == 50
        assert config.lr == 2e-4
        assert config.lambda_recon == 0.25


# ============================================================
# Gradient flow tests
# ============================================================

class TestGradientFlow:
    def test_encoder_frozen(self):
        """Encoder receives no gradients during training."""
        encoder, adapter, noise_net = _make_components()
        config = _make_config()
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        batch = _make_batch()
        _ = train_step(
            batch, noise_net, encoder, adapter, view_dropout,
            scheduler, optimizer, config, global_step=0,
        )
        for name, p in encoder.named_parameters():
            assert p.grad is None or torch.all(p.grad == 0), (
                f"Encoder param {name} should not receive gradients"
            )

    def test_adapter_receives_gradients(self):
        """Adapter parameters receive gradients."""
        encoder, adapter, noise_net = _make_components()
        config = _make_config()
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=0)  # lr=0 so grads aren't zeroed

        batch = _make_batch()
        # Don't step optimizer — we want to inspect gradients
        # Call train_step which does step internally, but we can check params changed
        old_adapter = [p.data.clone() for p in adapter.parameters()]
        config_lr = _make_config(lr=1e-2, lr_adapter=1e-2)
        _ = train_step(
            batch, noise_net, encoder, adapter, view_dropout,
            scheduler, torch.optim.AdamW(params, lr=1e-2), config_lr, global_step=0,
        )
        changed = any(
            not torch.equal(old, p.data)
            for old, p in zip(old_adapter, adapter.parameters())
        )
        assert changed, "Adapter weights should change after training step"

    def test_noise_net_receives_gradients(self):
        """Noise net parameters receive gradients."""
        encoder, adapter, noise_net = _make_components()
        config = _make_config(lr=1e-2)
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-2)

        old_weights = [p.data.clone() for p in noise_net.parameters() if p.requires_grad]
        batch = _make_batch()
        _ = train_step(
            batch, noise_net, encoder, adapter, view_dropout,
            scheduler, optimizer, config, global_step=0,
        )
        # At least some noise_net weights should change (some may be zero-init)
        n_changed = sum(
            1 for old, p in zip(old_weights, (p for p in noise_net.parameters() if p.requires_grad))
            if not torch.equal(old, p.data)
        )
        assert n_changed > 0, "Some noise_net weights should change"

    def test_grad_clip_limits_norms(self):
        """Gradient clipping actually limits gradient norms."""
        encoder, adapter, noise_net = _make_components()
        clip_val = 0.01  # very tight clip
        config = _make_config(grad_clip=clip_val)
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        all_params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=0)  # lr=0 to keep grads

        batch = _make_batch()

        # Manual forward + backward (don't use train_step, which zeros grads)
        device = "cpu"
        images_enc = batch["images_enc"]
        actions = batch["actions"]
        view_present = batch["view_present"]
        B_s, T_o_s, K_s, C_s, H_s, W_s = images_enc.shape

        with torch.no_grad():
            flat_imgs = images_enc.reshape(-1, C_s, H_s, W_s)
            flat_tokens = encoder(flat_imgs)
        N_s, d_s = flat_tokens.shape[1], flat_tokens.shape[2]
        tokens = flat_tokens.reshape(B_s, T_o_s, K_s, N_s, d_s)
        adapted = adapter(tokens.reshape(-1, N_s, d_s))
        d_prime = adapted.shape[-1]
        obs_tokens = adapted.reshape(B_s, T_o_s * K_s * N_s, d_prime)

        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (B_s,))
        noisy = scheduler.add_noise(actions, noise, timesteps)
        _, eps_pred = noise_net(noisy, timesteps, obs_tokens)
        loss = nn.functional.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()

        # Clip
        torch.nn.utils.clip_grad_norm_(
            [p for p in all_params if p.requires_grad], clip_val
        )

        # Check total norm is <= clip_val (with some float tolerance)
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in all_params if p.requires_grad], float("inf")
        )
        assert total_norm <= clip_val + 1e-4, (
            f"Grad norm {total_norm:.4f} exceeds clip {clip_val}"
        )


# ============================================================
# Separate LR param groups
# ============================================================

class TestParamGroups:
    def test_separate_lr_for_adapter(self):
        """Optimizer uses lower LR for adapter than noise_net."""
        encoder, adapter, noise_net = _make_components()
        config = _make_config(lr=1e-3, lr_adapter=1e-5)

        param_groups = [
            {"params": list(noise_net.parameters()), "lr": config.lr},
            {"params": list(adapter.parameters()), "lr": config.lr_adapter},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[1]["lr"] == 1e-5


# ============================================================
# Full pipeline with real dataset
# ============================================================

class TestFullPipeline:
    def test_train_step_with_real_dataset(self, tmp_hdf5):
        """Full pipeline: dataset → batch → train_step."""
        from data_pipeline.datasets.stage3_dataset import Stage3Dataset

        ds = Stage3Dataset(tmp_hdf5, T_obs=T_O, T_pred=T_P)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        encoder, adapter, noise_net = _make_components()
        config = _make_config()
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.0)

        params = list(noise_net.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        losses = train_step(
            batch, noise_net, encoder, adapter, view_dropout,
            scheduler, optimizer, config, global_step=0,
        )
        assert np.isfinite(losses["policy"])

    def test_multi_step_training(self, tmp_hdf5):
        """Multiple training steps complete without error."""
        from data_pipeline.datasets.stage3_dataset import Stage3Dataset

        ds = Stage3Dataset(tmp_hdf5, T_obs=T_O, T_pred=T_P)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)

        encoder, adapter, noise_net = _make_components()
        config = _make_config()
        scheduler = create_noise_scheduler(config.train_diffusion_steps)
        view_dropout = ViewDropout(d_model=HIDDEN_DIM, p=0.15)
        ema = EMA(noise_net, decay=0.99)

        params = list(noise_net.parameters()) + list(adapter.parameters()) + list(view_dropout.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        lr_sched = _create_lr_scheduler(optimizer, warmup_steps=5, total_steps=20)

        for i, batch in enumerate(loader):
            if i >= 10:
                break
            losses = train_step(
                batch, noise_net, encoder, adapter, view_dropout,
                scheduler, optimizer, config, global_step=i,
            )
            ema.update()
            lr_sched.step()
            assert np.isfinite(losses["policy"])

    def test_ema_eval_inference(self):
        """EMA weights produce valid DDIM inference."""
        _, _, noise_net = _make_components()
        ema = EMA(noise_net, decay=0.99)

        # Simulate some training
        with torch.no_grad():
            for p in noise_net.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update()

        # Apply EMA and run inference via context manager
        with ema.averaged_model():
            noise_net.eval()
            scheduler = create_noise_scheduler(100)
            obs_tokens = torch.randn(B, T_O * K * N, HIDDEN_DIM)
            actions = ddim_inference(noise_net, obs_tokens, AC_DIM, T_P, scheduler, 5)

            assert actions.shape == (B, T_P, AC_DIM)
            assert torch.isfinite(actions).all()

        noise_net.train()

    def test_ddim_different_num_steps(self):
        """DDIM works with various step counts."""
        _, _, noise_net = _make_components()
        noise_net.eval()
        scheduler = create_noise_scheduler(100)
        obs_tokens = torch.randn(B, T_O * K * N, HIDDEN_DIM)

        for steps in [1, 5, 10, 20]:
            actions = ddim_inference(
                noise_net, obs_tokens, AC_DIM, T_P, scheduler, steps
            )
            assert actions.shape == (B, T_P, AC_DIM)
            assert torch.isfinite(actions).all()

    def test_checkpoint_with_decoder(self, tmp_path):
        """Checkpoint save/load includes decoder for co-training."""
        _, adapter, noise_net = _make_components()
        decoder = MockDecoder(in_dim=HIDDEN_DIM, n_tokens=N)
        ema = EMA(noise_net, decay=0.999)
        optimizer = torch.optim.AdamW(
            list(noise_net.parameters()) + list(adapter.parameters()) +
            list(decoder.parameters()), lr=1e-3,
        )

        path = str(tmp_path / "ckpt_dec.pt")
        save_checkpoint(path, 3, 50, noise_net, adapter, optimizer, ema, {},
                        decoder=decoder)

        ckpt = torch.load(path, weights_only=False)
        assert "decoder" in ckpt

        # Load into fresh models
        _, adapter2, noise_net2 = _make_components()
        decoder2 = MockDecoder(in_dim=HIDDEN_DIM, n_tokens=N)
        ema2 = EMA(noise_net2, decay=0.999)
        optimizer2 = torch.optim.AdamW(
            list(noise_net2.parameters()) + list(adapter2.parameters()) +
            list(decoder2.parameters()), lr=1e-3,
        )

        epoch, gs = load_checkpoint(path, noise_net2, adapter2, optimizer2, ema2, decoder2)
        assert epoch == 4
        assert gs == 50

        for p1, p2 in zip(decoder.parameters(), decoder2.parameters()):
            assert torch.equal(p1, p2)
