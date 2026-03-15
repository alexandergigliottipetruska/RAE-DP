"""Tests for C.5 EMA (Exponential Moving Average).

Tests the EMA class which maintains shadow parameters for diffusion policy evaluation.
Updated to match the new API: EMA(model, decay, warmup_steps), update() with no args,
averaged_model() context manager, shadow stored as state_dict on CPU.
"""

import copy

import pytest
import torch
import torch.nn as nn

from models.ema import EMA


def _make_model():
    """Simple 2-layer model for testing."""
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


class TestEMABasic:
    def test_init_copies_weights(self):
        """EMA shadow matches the model at initialization."""
        model = _make_model()
        ema = EMA(model, decay=0.999)
        model_state = model.state_dict()
        for k in ema.shadow:
            assert torch.equal(ema.shadow[k], model_state[k].cpu())

    def test_decay_stored(self):
        """Decay value is stored correctly."""
        model = _make_model()
        ema = EMA(model, decay=0.9999)
        assert ema.decay == 0.9999

    def test_update_moves_weights(self):
        """After an update step, EMA shadow differs from initial copy."""
        model = _make_model()
        ema = EMA(model, decay=0.9, warmup_steps=0)
        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        # Simulate optimizer step
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update()
        for k in ema.shadow:
            assert not torch.equal(ema.shadow[k], old_shadow[k]), "EMA should have changed"

    def test_decay_formula(self):
        """EMA update follows: shadow = decay * shadow + (1-decay) * param."""
        model = _make_model()
        decay = 0.9
        ema = EMA(model, decay=decay, warmup_steps=0)
        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update()

        model_state = model.state_dict()
        for k in ema.shadow:
            expected = decay * old_shadow[k] + (1 - decay) * model_state[k].cpu()
            assert torch.allclose(ema.shadow[k], expected, atol=1e-6)

    def test_weights_diverge_over_steps(self):
        """EMA shadow progressively diverges from model after many updates."""
        model = _make_model()
        ema = EMA(model, decay=0.99, warmup_steps=0)

        for _ in range(50):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema.update()

        model_state = model.state_dict()
        diffs = []
        for k in ema.shadow:
            diffs.append((ema.shadow[k] - model_state[k].cpu()).abs().mean().item())
        assert max(diffs) > 0.001, "EMA should diverge from fast-moving model"

    def test_warmup_ramps_decay(self):
        """During warmup, effective decay is lower than target decay."""
        model = _make_model()
        ema = EMA(model, decay=0.9999, warmup_steps=100)
        # At step 1: effective = min(0.9999, 2/11) ≈ 0.182
        assert ema._effective_decay() < 0.9999
        # After many steps, effective decay should approach target
        ema._step = 1000
        assert abs(ema._effective_decay() - 0.9999) < 0.01

    def test_shadow_on_cpu(self):
        """Shadow params are stored on CPU regardless of model device."""
        model = _make_model()
        ema = EMA(model, decay=0.999)
        for k, v in ema.shadow.items():
            assert v.device == torch.device("cpu")


class TestEMASaveLoad:
    def test_state_dict_roundtrip(self):
        """Save and load preserves EMA shadow exactly."""
        model = _make_model()
        ema = EMA(model, decay=0.9999, warmup_steps=0)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update()

        state = ema.state_dict()
        # Create fresh EMA and load
        model2 = _make_model()
        ema2 = EMA(model2, decay=0.9999)
        ema2.load_state_dict(state)

        for k in ema.shadow:
            assert torch.equal(ema.shadow[k], ema2.shadow[k])

    def test_state_dict_contains_decay(self):
        """State dict stores the decay value."""
        model = _make_model()
        ema = EMA(model, decay=0.9999)
        state = ema.state_dict()
        assert "decay" in state
        assert state["decay"] == 0.9999

    def test_state_dict_contains_step(self):
        """State dict stores the step counter."""
        model = _make_model()
        ema = EMA(model, decay=0.999, warmup_steps=50)
        ema.update()
        ema.update()
        state = ema.state_dict()
        assert state["_step"] == 2


class TestEMAApplyRestore:
    def test_apply_copies_ema_to_model(self):
        """apply_to() copies EMA weights into the model."""
        model = _make_model()
        ema = EMA(model, decay=0.9, warmup_steps=0)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update()

        ema.apply_to(model)
        model_state = model.state_dict()
        for k in ema.shadow:
            assert torch.allclose(model_state[k].cpu(), ema.shadow[k], atol=1e-6)

    def test_restore_recovers_original(self):
        """restore() puts back the original model weights after apply_to()."""
        model = _make_model()
        ema = EMA(model, decay=0.9, warmup_steps=0)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        original_state = copy.deepcopy(model.state_dict())
        ema.update()

        backup = copy.deepcopy(model.state_dict())
        ema.apply_to(model)
        ema.restore(model, backup)

        for k in original_state:
            assert torch.equal(model.state_dict()[k], original_state[k])

    def test_averaged_model_context_manager(self):
        """averaged_model() swaps in EMA weights and restores on exit."""
        model = _make_model()
        ema = EMA(model, decay=0.9, warmup_steps=0)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        original_state = copy.deepcopy(model.state_dict())
        ema.update()

        with ema.averaged_model():
            # Inside context: model has EMA weights
            for k in ema.shadow:
                assert torch.allclose(model.state_dict()[k].cpu(), ema.shadow[k], atol=1e-6)

        # After context: model restored to original
        for k in original_state:
            assert torch.equal(model.state_dict()[k], original_state[k])

    def test_apply_restore_cycle(self):
        """Multiple apply/restore cycles work correctly."""
        model = _make_model()
        ema = EMA(model, decay=0.99, warmup_steps=0)
        for _ in range(5):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema.update()

        original = copy.deepcopy(model.state_dict())

        # Cycle 1
        backup = copy.deepcopy(model.state_dict())
        ema.apply_to(model)
        ema.restore(model, backup)
        for k in original:
            assert torch.equal(model.state_dict()[k], original[k])

        # Cycle 2 (via context manager)
        with ema.averaged_model():
            pass
        for k in original:
            assert torch.equal(model.state_dict()[k], original[k])
