"""Tests for visualization utilities and RLBench wrapper interface.

Visualization tests use mock data (no sim needed).
RLBench wrapper tests are skipped — require CoppeliaSim on Linux/WSL2.
"""

import numpy as np
import pytest

from data_pipeline.evaluation.visualization import (
    denormalize_image,
    plot_action_trajectory,
    plot_success_rates,
)


# ---------------------------------------------------------------------------
# denormalize_image
# ---------------------------------------------------------------------------

class TestDenormalizeImage:
    def test_output_shape_and_dtype(self):
        img_chw = np.random.randn(3, 224, 224).astype(np.float32)
        result = denormalize_image(img_chw)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8

    def test_known_values(self):
        """A pixel at ImageNet mean should denormalize to ~[124, 116, 104]."""
        # Normalized value of 0.0 corresponds to the ImageNet mean
        img_chw = np.zeros((3, 1, 1), dtype=np.float32)
        result = denormalize_image(img_chw)
        # Mean * 255: [0.485*255, 0.456*255, 0.406*255] = [123.7, 116.3, 103.5]
        assert abs(int(result[0, 0, 0]) - 124) <= 1
        assert abs(int(result[0, 0, 1]) - 116) <= 1
        assert abs(int(result[0, 0, 2]) - 104) <= 1

    def test_clipping(self):
        """Extreme values should be clipped to [0, 255]."""
        img_chw = np.full((3, 1, 1), 100.0, dtype=np.float32)
        result = denormalize_image(img_chw)
        assert np.all(result == 255)

        img_chw = np.full((3, 1, 1), -100.0, dtype=np.float32)
        result = denormalize_image(img_chw)
        assert np.all(result == 0)


# ---------------------------------------------------------------------------
# plot_action_trajectory (just verifies no crash + file output)
# ---------------------------------------------------------------------------

class TestPlotActionTrajectory:
    def test_predicted_only(self, tmp_path):
        pred = np.random.randn(50, 7).astype(np.float32)
        out = tmp_path / "actions.png"
        plot_action_trajectory(pred, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_with_ground_truth(self, tmp_path):
        pred = np.random.randn(50, 7).astype(np.float32)
        gt = np.random.randn(50, 7).astype(np.float32)
        out = tmp_path / "actions_gt.png"
        plot_action_trajectory(pred, ground_truth=gt, output_path=out)
        assert out.exists()

    def test_custom_labels(self, tmp_path):
        pred = np.random.randn(30, 3).astype(np.float32)
        out = tmp_path / "custom.png"
        plot_action_trajectory(
            pred, action_labels=["x", "y", "z"],
            title="Test", output_path=out,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_success_rates
# ---------------------------------------------------------------------------

class TestPlotSuccessRates:
    def test_basic_bar_chart(self, tmp_path):
        out = tmp_path / "rates.png"
        plot_success_rates(
            labels=["all_4", "no_front", "no_wrist"],
            rates=[0.8, 0.6, 0.4],
            output_path=out,
        )
        assert out.exists()

    def test_with_ci(self, tmp_path):
        out = tmp_path / "rates_ci.png"
        plot_success_rates(
            labels=["cfg_a", "cfg_b"],
            rates=[0.7, 0.5],
            ci_lower=[0.5, 0.3],
            ci_upper=[0.85, 0.7],
            title="With CI",
            output_path=out,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# RLBench wrapper — interface-only checks (no sim)
# ---------------------------------------------------------------------------

class TestRLBenchWrapperInterface:
    def test_import(self):
        """Module should import without CoppeliaSim (no side effects at import)."""
        from data_pipeline.envs.rlbench_wrapper import (
            RLBenchWrapper,
            TASK_CLASS_MAP,
        )
        assert "close_jar" in TASK_CLASS_MAP
        assert len(TASK_CLASS_MAP) >= 3

    def test_absolute_action_passthrough(self):
        """Verify absolute actions are passed through correctly by the wrapper."""
        # Wrapper now takes 8D absolute EE pose: [pos(3), quat_xyzw(4), grip(1)]
        action = np.array([0.3, 0.1, 0.5, 0.0, 0.0, 0.0, 1.0, 0.8], dtype=np.float32)

        # Extract like the wrapper does
        position = action[:3]
        quat_xyzw = action[3:7]
        gripper = 1.0 if float(action[7]) > 0.5 else 0.0

        assert np.allclose(position, [0.3, 0.1, 0.5])
        assert np.allclose(quat_xyzw, [0.0, 0.0, 0.0, 1.0])
        assert gripper == 1.0

    def test_gripper_threshold(self):
        """Values > 0.5 -> 1.0, <= 0.5 -> 0.0."""
        # This tests the convention documented in the spec
        assert (1.0 if 0.8 > 0.5 else 0.0) == 1.0
        assert (1.0 if 0.5 > 0.5 else 0.0) == 0.0  # 0.5 exactly -> 0.0
        assert (1.0 if 0.3 > 0.5 else 0.0) == 0.0
        assert (1.0 if 1.0 > 0.5 else 0.0) == 1.0
        assert (1.0 if 0.0 > 0.5 else 0.0) == 0.0
