"""Tests for camera dropout robustness evaluation and robomimic wrapper.

Tier 1/2: Robustness eval tests use MockEnv/MockPolicy (no sim).
Tier 3: Robomimic wrapper test requires robosuite (marked with pytest.mark.slow).
"""

import numpy as np
import pytest
import torch

from data_pipeline.evaluation.robustness_eval import (
    DROPOUT_CONFIGS,
    CameraDropoutEnvWrapper,
    evaluate_robustness,
)
from data_pipeline.evaluation.rollout import evaluate_policy


# ---------------------------------------------------------------------------
# Mock objects (reused from test_eval_integration)
# ---------------------------------------------------------------------------

class MockPolicy:
    def __init__(self):
        self.received_view_presents = []

    def predict(self, images, proprio, view_present):
        self.received_view_presents.append(view_present.numpy().copy())
        return torch.zeros(50, 7)


class MockEnv:
    def __init__(self, episode_len=3):
        self.episode_len = episode_len
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.episode_len
        return {}, 0.0, done, {"success": done}

    def get_multiview_images(self):
        # Non-zero so we can verify dropout zeros them out
        return np.ones((1, 4, 3, 224, 224), dtype=np.float32) * 0.5

    def get_proprio(self):
        return np.ones((1, 9), dtype=np.float32)

    def get_view_present(self):
        return np.array([True, True, True, True])

    def close(self):
        pass


# ---------------------------------------------------------------------------
# CameraDropoutEnvWrapper tests
# ---------------------------------------------------------------------------

class TestCameraDropoutWrapper:
    def test_no_dropout(self):
        """Empty drop list → images and view_present unchanged."""
        env = MockEnv()
        env.reset()
        wrapped = CameraDropoutEnvWrapper(env, drop_slots=[])

        images = wrapped.get_multiview_images()
        vp = wrapped.get_view_present()

        assert np.all(images != 0)
        assert np.all(vp == True)

    def test_drop_single_slot(self):
        """Dropping slot 0 → slot 0 zeroed, view_present[0] = False."""
        env = MockEnv()
        env.reset()
        wrapped = CameraDropoutEnvWrapper(env, drop_slots=[0])

        images = wrapped.get_multiview_images()
        vp = wrapped.get_view_present()

        assert np.all(images[0, 0] == 0.0)  # slot 0 zeroed
        assert np.all(images[0, 1] != 0.0)  # slot 1 untouched
        assert vp[0] == False
        assert vp[1] == True

    def test_drop_multiple_slots(self):
        """Dropping slots [0, 3] → both zeroed."""
        env = MockEnv()
        env.reset()
        wrapped = CameraDropoutEnvWrapper(env, drop_slots=[0, 3])

        images = wrapped.get_multiview_images()
        vp = wrapped.get_view_present()

        assert np.all(images[0, 0] == 0.0)
        assert np.all(images[0, 3] == 0.0)
        assert np.all(images[0, 1] != 0.0)
        assert np.all(images[0, 2] != 0.0)
        assert vp.tolist() == [False, True, True, False]

    def test_passthrough_methods(self):
        """reset, step, get_proprio, close pass through to inner env."""
        env = MockEnv(episode_len=2)
        wrapped = CameraDropoutEnvWrapper(env, drop_slots=[1])

        wrapped.reset()
        _, _, done, info = wrapped.step(np.zeros(7))
        assert not done
        _, _, done, info = wrapped.step(np.zeros(7))
        assert done

        proprio = wrapped.get_proprio()
        assert proprio.shape == (1, 9)


# ---------------------------------------------------------------------------
# evaluate_robustness tests
# ---------------------------------------------------------------------------

class TestEvaluateRobustness:
    def test_all_configs_run(self):
        """All 7 default configs produce results."""
        env = MockEnv(episode_len=3)
        policy = MockPolicy()

        results = evaluate_robustness(
            policy, env,
            num_episodes=1, max_steps=10,
            exec_horizon=8, obs_horizon=2,
        )

        assert len(results) == len(DROPOUT_CONFIGS)
        for cfg in DROPOUT_CONFIGS:
            label = cfg["label"]
            assert label in results
            assert "success_rate" in results[label]
            assert "ci_lower" in results[label]
            assert "ci_upper" in results[label]

    def test_dropout_flags_passed_to_policy(self):
        """Verify that dropped slots show False in view_present passed to policy."""
        env = MockEnv(episode_len=1)
        policy = MockPolicy()

        # Test just one config
        configs = [{"drop": [0, 3], "label": "no_front_wrist"}]

        evaluate_robustness(
            policy, env, configs=configs,
            num_episodes=1, max_steps=10,
            exec_horizon=8, obs_horizon=2,
        )

        # Policy should have received view_present with [False, True, True, False]
        assert len(policy.received_view_presents) >= 1
        vp = policy.received_view_presents[0][0]  # [1, K] → [K]
        assert vp.tolist() == [False, True, True, False]

    def test_custom_configs(self):
        """Custom config list works."""
        env = MockEnv(episode_len=2)
        policy = MockPolicy()

        custom = [{"drop": [2], "label": "custom_no_right"}]

        results = evaluate_robustness(
            policy, env, configs=custom,
            num_episodes=1, max_steps=10,
            exec_horizon=8, obs_horizon=2,
        )

        assert "custom_no_right" in results
        assert len(results) == 1
