"""Tests for C.6 Stage3Dataset.

Stage3Dataset returns temporal windows: T_o observation frames + T_p action targets.
Used for DDPM diffusion policy training in Stage 3.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch

from data_pipeline.datasets.stage3_dataset import Stage3Dataset

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
K = 4           # camera slots
H = W = 224     # image size
T_O = 2         # observation horizon
T_P = 16        # prediction horizon
D_ACT = 7       # action dim (robomimic)
D_PROP = 9      # proprio dim (robomimic)
NUM_DEMOS = 3
DEMO_LEN = 30   # timesteps per demo


# ---------------------------------------------------------------------------
# Synthetic HDF5 fixtures
# ---------------------------------------------------------------------------

def _create_synthetic_hdf5(
    path,
    num_demos=NUM_DEMOS,
    demo_len=DEMO_LEN,
    action_dim=D_ACT,
    proprio_dim=D_PROP,
    num_cams=2,         # real cameras (rest are zero-padded)
    split_ratio=0.67,   # train/valid split
    benchmark="robomimic",
    task="lift",
):
    """Create a synthetic unified HDF5 matching the project schema."""
    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = benchmark
        f.attrs["task"] = task
        f.attrs["action_dim"] = action_dim
        f.attrs["proprio_dim"] = proprio_dim
        f.attrs["image_size"] = H
        f.attrs["num_cam_slots"] = K

        # View present: first num_cams slots are real
        view_present = np.zeros(K, dtype=bool)
        view_present[:num_cams] = True

        demo_keys = []
        for i in range(num_demos):
            key = f"demo_{i}"
            demo_keys.append(key)
            grp = f.create_group(f"data/{key}")

            # Images: uint8 [T, K, H, W, 3]
            imgs = np.random.randint(0, 256, (demo_len, K, H, W, 3), dtype=np.uint8)
            # Zero out absent camera slots
            imgs[:, num_cams:] = 0
            grp.create_dataset("images", data=imgs)

            # Actions: float32 [T, action_dim]
            actions = np.random.randn(demo_len, action_dim).astype(np.float32)
            grp.create_dataset("actions", data=actions)

            # Proprio: float32 [T, proprio_dim]
            proprio = np.random.randn(demo_len, proprio_dim).astype(np.float32)
            grp.create_dataset("proprio", data=proprio)

            grp.create_dataset("view_present", data=view_present)

        # Split mask
        n_train = max(1, int(num_demos * split_ratio))
        mask_grp = f.create_group("mask")
        dt = h5py.special_dtype(vlen=str)
        train_keys = demo_keys[:n_train]
        valid_keys = demo_keys[n_train:]
        mask_grp.create_dataset("train", data=train_keys, dtype=dt)
        mask_grp.create_dataset("valid", data=valid_keys, dtype=dt)

        # Norm stats (for minmax normalization)
        ns = f.create_group("norm_stats")
        for field, dim in [("actions", action_dim), ("proprio", proprio_dim)]:
            grp = ns.create_group(field)
            grp.create_dataset("mean", data=np.zeros(dim, dtype=np.float32))
            grp.create_dataset("std", data=np.ones(dim, dtype=np.float32))
            grp.create_dataset("min", data=-np.ones(dim, dtype=np.float32) * 2)
            grp.create_dataset("max", data=np.ones(dim, dtype=np.float32) * 2)


@pytest.fixture
def synthetic_hdf5(tmp_path):
    """Single synthetic HDF5 file."""
    path = str(tmp_path / "test.hdf5")
    _create_synthetic_hdf5(path)
    return path


@pytest.fixture
def synthetic_hdf5_rlbench(tmp_path):
    """Synthetic RLBench HDF5 with 4 cameras and 8D actions."""
    path = str(tmp_path / "rlbench.hdf5")
    _create_synthetic_hdf5(
        path, action_dim=8, proprio_dim=8, num_cams=4,
        benchmark="rlbench", task="close_jar",
    )
    return path


@pytest.fixture
def two_hdf5_files(tmp_path):
    """Two HDF5 files with different action dims (multi-task)."""
    p1 = str(tmp_path / "robomimic.hdf5")
    p2 = str(tmp_path / "rlbench.hdf5")
    _create_synthetic_hdf5(p1, action_dim=7, proprio_dim=9, benchmark="robomimic")
    _create_synthetic_hdf5(p2, action_dim=8, proprio_dim=8, benchmark="rlbench", num_cams=4)
    return [p1, p2]


# ============================================================
# Shape tests
# ============================================================

class TestStage3DatasetShapes:
    def test_images_enc_shape(self, synthetic_hdf5):
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        assert sample["images_enc"].shape == (T_O, K, 3, H, W)

    def test_images_target_shape(self, synthetic_hdf5):
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        assert sample["images_target"].shape == (T_O, K, 3, H, W)

    def test_actions_shape(self, synthetic_hdf5):
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        assert sample["actions"].shape == (T_P, D_ACT)

    def test_proprio_shape(self, synthetic_hdf5):
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        assert sample["proprio"].shape == (T_O, D_PROP)

    def test_view_present_shape(self, synthetic_hdf5):
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        assert sample["view_present"].shape == (K,)

    def test_dtypes(self, synthetic_hdf5):
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        assert sample["images_enc"].dtype == torch.float32
        assert sample["images_target"].dtype == torch.float32
        assert sample["actions"].dtype == torch.float32
        assert sample["proprio"].dtype == torch.float32
        assert sample["view_present"].dtype == torch.bool

    def test_output_keys(self, synthetic_hdf5):
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        expected_keys = {"images_enc", "images_target", "actions", "proprio", "view_present"}
        assert set(sample.keys()) == expected_keys


# ============================================================
# Indexing tests
# ============================================================

class TestStage3DatasetIndexing:
    def test_length_correct(self, synthetic_hdf5):
        """Length = sum over train demos of (T_demo - T_p)."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        # 2 train demos (67% of 3), each with DEMO_LEN timesteps
        # Valid range per demo: t in [0, T_demo - T_p - 1] → T_demo - T_p samples
        n_train = 2  # ceil(3 * 0.67)
        expected = n_train * (DEMO_LEN - T_P)
        assert len(ds) == expected

    def test_valid_split(self, synthetic_hdf5):
        """Valid split has correct length."""
        ds = Stage3Dataset(synthetic_hdf5, split="valid", T_obs=T_O, T_pred=T_P)
        n_valid = 1  # 3 - 2
        expected = n_valid * (DEMO_LEN - T_P)
        assert len(ds) == expected

    def test_all_indices_accessible(self, synthetic_hdf5):
        """Every index in [0, len) returns a valid sample."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["actions"].shape == (T_P, D_ACT)

    def test_end_trimming(self, synthetic_hdf5):
        """Last valid timestep has T_p actions remaining."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[len(ds) - 1]
        # Should have exactly T_P actions — no out-of-bounds
        assert sample["actions"].shape == (T_P, D_ACT)
        assert torch.isfinite(sample["actions"]).all()


# ============================================================
# Start padding tests
# ============================================================

class TestStage3DatasetPadding:
    def test_start_padding_t0(self, synthetic_hdf5):
        """At t=0 with T_o=2, first obs frame is duplicated from t=0."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=2, T_pred=T_P)
        sample = ds[0]
        # Both observation frames should be identical (first frame repeated)
        assert torch.equal(sample["images_enc"][0], sample["images_enc"][1]) or True
        # Actually: at t=0, we need frames at [t-1, t] = [-1, 0]
        # Frame -1 doesn't exist, so we repeat frame 0
        # Both frames should come from t=0
        imgs = sample["images_enc"]
        assert torch.allclose(imgs[0], imgs[1], atol=1e-6), (
            "At t=0, both obs frames should be the same (start padding)"
        )

    def test_start_padding_proprio_t0(self, synthetic_hdf5):
        """Proprio is also padded at t=0."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=2, T_pred=T_P)
        sample = ds[0]
        assert torch.allclose(sample["proprio"][0], sample["proprio"][1], atol=1e-6)

    def test_no_padding_after_start(self, synthetic_hdf5):
        """At t >= T_o-1, no padding needed — frames should differ."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=2, T_pred=T_P)
        # Index T_o-1 corresponds to t=1 in first demo (no padding needed)
        sample = ds[1]
        imgs = sample["images_enc"]
        # With random data, frames at t=0 and t=1 should differ
        assert not torch.allclose(imgs[0], imgs[1], atol=1e-6), (
            "At t=1, obs frames should be from t=0 and t=1 (different)"
        )

    def test_T_obs_1_no_padding(self, synthetic_hdf5):
        """With T_obs=1, no padding is ever needed."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=1, T_pred=T_P)
        sample = ds[0]
        assert sample["images_enc"].shape == (1, K, 3, H, W)


# ============================================================
# Image normalization tests
# ============================================================

class TestStage3DatasetImages:
    def test_images_enc_imagenet_normalized(self, synthetic_hdf5):
        """images_enc values are in ImageNet-normalized range (~[-2.2, 2.7])."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        imgs = sample["images_enc"]
        # ImageNet-normalized values are roughly in [-2.2, 2.7]
        assert imgs.min() >= -3.0
        assert imgs.max() <= 3.0

    def test_images_target_zero_one_range(self, synthetic_hdf5):
        """images_target values are in [0, 1] range (raw float)."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        imgs = sample["images_target"]
        assert imgs.min() >= 0.0
        assert imgs.max() <= 1.0

    def test_images_target_is_chw(self, synthetic_hdf5):
        """images_target is in CHW format."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        # Shape is [T_o, K, 3, H, W] — channel dim is 3
        assert sample["images_target"].shape[2] == 3


# ============================================================
# Action / proprio normalization tests
# ============================================================

class TestStage3DatasetNormalization:
    def test_actions_minmax_range(self, synthetic_hdf5):
        """With minmax normalization, actions should be in ~[-1, 1]."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P, norm_mode="minmax")
        sample = ds[0]
        # Synthetic data: raw actions are randn, min=-2, max=2
        # Values within [-2, 2] map to [-1, 1]; outliers can exceed
        actions = sample["actions"]
        assert actions.min() >= -3.0  # allow some outliers from randn
        assert actions.max() <= 3.0

    def test_proprio_normalized(self, synthetic_hdf5):
        """Proprio is normalized."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P, norm_mode="minmax")
        sample = ds[0]
        proprio = sample["proprio"]
        assert torch.isfinite(proprio).all()

    def test_zscore_normalization(self, synthetic_hdf5):
        """Z-score normalization also works."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P, norm_mode="zscore")
        sample = ds[0]
        assert torch.isfinite(sample["actions"]).all()


# ============================================================
# View present tests
# ============================================================

class TestStage3DatasetViewPresent:
    def test_robomimic_2_cameras(self, synthetic_hdf5):
        """Robomimic has 2 real cameras."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        vp = sample["view_present"]
        assert vp[:2].all()      # first 2 are real
        assert not vp[2:].any()  # rest are padding

    def test_rlbench_4_cameras(self, synthetic_hdf5_rlbench):
        """RLBench has 4 real cameras."""
        ds = Stage3Dataset(synthetic_hdf5_rlbench, T_obs=T_O, T_pred=T_P)
        sample = ds[0]
        vp = sample["view_present"]
        assert vp.all()  # all 4 are real

    def test_view_present_constant_across_samples(self, synthetic_hdf5):
        """view_present is the same for all samples in a file."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        vp0 = ds[0]["view_present"]
        vp_last = ds[len(ds) - 1]["view_present"]
        assert torch.equal(vp0, vp_last)


# ============================================================
# Multi-file tests
# ============================================================

class TestStage3DatasetMultiFile:
    def test_multi_file_length(self, two_hdf5_files):
        """Length is sum of valid samples across all files."""
        ds = Stage3Dataset(two_hdf5_files, T_obs=T_O, T_pred=T_P)
        # 2 files, each with 2 train demos, each (DEMO_LEN - T_P) samples
        expected = 2 * 2 * (DEMO_LEN - T_P)
        assert len(ds) == expected

    def test_multi_file_all_accessible(self, two_hdf5_files):
        """All indices across multiple files are accessible."""
        ds = Stage3Dataset(two_hdf5_files, T_obs=T_O, T_pred=T_P)
        for i in range(len(ds)):
            sample = ds[i]
            assert torch.isfinite(sample["actions"]).all()

    def test_multi_file_action_dim_varies(self, two_hdf5_files):
        """Different files can have different action dims."""
        ds = Stage3Dataset(two_hdf5_files, T_obs=T_O, T_pred=T_P)
        # First file: robomimic (7D), second file: rlbench (8D)
        # Samples from first file have 7D actions
        sample_first = ds[0]
        n_first = 2 * (DEMO_LEN - T_P)  # samples from first file
        sample_second = ds[n_first]
        assert sample_first["actions"].shape[-1] == 7
        assert sample_second["actions"].shape[-1] == 8


# ============================================================
# DataLoader tests
# ============================================================

class TestStage3DatasetDataLoader:
    def test_dataloader_batching(self, synthetic_hdf5):
        """DataLoader can batch samples."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["images_enc"].shape == (4, T_O, K, 3, H, W)
        assert batch["actions"].shape == (4, T_P, D_ACT)

    def test_dataloader_full_epoch(self, synthetic_hdf5):
        """Full epoch iteration works."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        total = 0
        for batch in loader:
            total += batch["actions"].shape[0]
        assert total == len(ds)

    def test_dataloader_num_workers(self, synthetic_hdf5):
        """Multi-worker loading works (per-call file opening)."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2)
        batch = next(iter(loader))
        assert batch["actions"].shape[0] == 4


# ============================================================
# Edge cases
# ============================================================

class TestStage3DatasetEdgeCases:
    def test_short_demo_excluded(self, tmp_path):
        """Demos shorter than T_p are excluded (0 valid samples)."""
        path = str(tmp_path / "short.hdf5")
        _create_synthetic_hdf5(path, num_demos=1, demo_len=T_P - 1, split_ratio=1.0)
        ds = Stage3Dataset(path, T_obs=T_O, T_pred=T_P)
        assert len(ds) == 0

    def test_exact_length_demo(self, tmp_path):
        """Demo with exactly T_p timesteps has 0 valid samples (need t+T_p <= T)."""
        path = str(tmp_path / "exact.hdf5")
        _create_synthetic_hdf5(path, num_demos=1, demo_len=T_P, split_ratio=1.0)
        ds = Stage3Dataset(path, T_obs=T_O, T_pred=T_P)
        assert len(ds) == 0

    def test_demo_len_tp_plus_one(self, tmp_path):
        """Demo with T_p+1 timesteps has exactly 1 valid sample."""
        path = str(tmp_path / "tp1.hdf5")
        _create_synthetic_hdf5(path, num_demos=1, demo_len=T_P + 1, split_ratio=1.0)
        ds = Stage3Dataset(path, T_obs=T_O, T_pred=T_P)
        assert len(ds) == 1

    def test_large_T_pred(self, synthetic_hdf5):
        """T_pred=50 works (fewer valid samples per demo)."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=50)
        # DEMO_LEN=30, so 30-50 < 0 → no valid samples
        assert len(ds) == 0

    def test_single_file_string_input(self, synthetic_hdf5):
        """Accepts a single string path (not just list)."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        assert len(ds) > 0


# ============================================================
# rot6d conversion tests (V3)
# ============================================================

def _create_synthetic_hdf5_rot6d(path, num_demos=NUM_DEMOS, demo_len=DEMO_LEN):
    """Create synthetic HDF5 with 7D absolute actions and 10D norm stats.

    Actions are 7D (pos3 + axis_angle3 + grip1) with realistic ranges.
    Norm stats are 10D (computed after rot6d conversion).
    """
    from data_pipeline.utils.rotation import convert_actions_to_rot6d

    action_dim_stored = 7  # stored as 7D in HDF5
    action_dim_rot6d = 10  # norm stats are 10D
    proprio_dim = D_PROP

    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "robomimic"
        f.attrs["task"] = "lift"
        f.attrs["action_dim"] = action_dim_stored
        f.attrs["proprio_dim"] = proprio_dim
        f.attrs["image_size"] = H
        f.attrs["num_cam_slots"] = K

        view_present = np.array([True, False, False, True], dtype=bool)

        all_actions_10d = []
        demo_keys = []
        for i in range(num_demos):
            key = f"demo_{i}"
            demo_keys.append(key)
            grp = f.create_group(f"data/{key}")

            imgs = np.random.randint(0, 256, (demo_len, K, H, W, 3), dtype=np.uint8)
            imgs[:, 1:3] = 0
            grp.create_dataset("images", data=imgs)

            # Realistic 7D abs actions: pos near workspace, rot near identity, grip ±1
            pos = np.random.uniform(-0.5, 0.5, (demo_len, 3)).astype(np.float32)
            rot_aa = np.random.uniform(-0.5, 0.5, (demo_len, 3)).astype(np.float32)
            grip = np.random.choice([-1.0, 1.0], (demo_len, 1)).astype(np.float32)
            actions_7d = np.concatenate([pos, rot_aa, grip], axis=-1)
            grp.create_dataset("actions", data=actions_7d)

            proprio = np.random.randn(demo_len, proprio_dim).astype(np.float32)
            grp.create_dataset("proprio", data=proprio)
            grp.create_dataset("view_present", data=view_present)

            all_actions_10d.append(convert_actions_to_rot6d(actions_7d))

        # Split
        n_train = max(1, int(num_demos * 0.67))
        mask_grp = f.create_group("mask")
        dt = h5py.special_dtype(vlen=str)
        mask_grp.create_dataset("train", data=demo_keys[:n_train], dtype=dt)
        mask_grp.create_dataset("valid", data=demo_keys[n_train:], dtype=dt)

        # 10D norm stats (computed from rot6d-converted actions)
        train_actions = np.concatenate(all_actions_10d[:n_train], axis=0)
        ns = f.create_group("norm_stats")

        act_grp = ns.create_group("actions")
        act_grp.create_dataset("mean", data=train_actions.mean(0).astype(np.float32))
        act_grp.create_dataset("std", data=np.clip(train_actions.std(0), 1e-6, None).astype(np.float32))
        act_grp.create_dataset("min", data=train_actions.min(0).astype(np.float32))
        act_grp.create_dataset("max", data=train_actions.max(0).astype(np.float32))

        prop_grp = ns.create_group("proprio")
        prop_grp.create_dataset("mean", data=np.zeros(proprio_dim, dtype=np.float32))
        prop_grp.create_dataset("std", data=np.ones(proprio_dim, dtype=np.float32))
        prop_grp.create_dataset("min", data=-np.ones(proprio_dim, dtype=np.float32) * 2)
        prop_grp.create_dataset("max", data=np.ones(proprio_dim, dtype=np.float32) * 2)


@pytest.fixture
def rot6d_hdf5(tmp_path):
    """HDF5 with 7D abs actions and 10D norm stats for rot6d testing."""
    path = str(tmp_path / "rot6d.hdf5")
    _create_synthetic_hdf5_rot6d(path)
    return path


class TestStage3DatasetRot6d:
    def test_getitem_returns_10d_actions(self, rot6d_hdf5):
        """With use_rot6d=True, actions are 10D (pos3+rot6d6+grip1)."""
        ds = Stage3Dataset(rot6d_hdf5, T_obs=T_O, T_pred=T_P, use_rot6d=True)
        sample = ds[0]
        assert sample["actions"].shape == (T_P, 10)

    def test_without_rot6d_returns_7d(self, rot6d_hdf5):
        """With use_rot6d=False (default), actions stay 7D."""
        # Need 7D norm stats for this — use the standard fixture instead
        pass  # covered by existing test_actions_shape

    def test_rot6d_actions_finite(self, rot6d_hdf5):
        """rot6d-converted actions are all finite."""
        ds = Stage3Dataset(rot6d_hdf5, T_obs=T_O, T_pred=T_P, use_rot6d=True)
        for i in range(min(5, len(ds))):
            sample = ds[i]
            assert torch.isfinite(sample["actions"]).all(), f"Non-finite actions at idx {i}"

    def test_rot6d_roundtrip(self, rot6d_hdf5):
        """Load 7D → rot6d → denorm → rot6d_to_aa matches original within tolerance."""
        from data_pipeline.utils.rotation import convert_actions_from_rot6d

        ds = Stage3Dataset(rot6d_hdf5, T_obs=T_O, T_pred=T_P, use_rot6d=True, norm_mode="minmax")
        sample = ds[0]
        actions_norm = sample["actions"].numpy()  # (T_P, 10) normalized

        # Denormalize
        norm = ds._norm_per_file[0]["actions"]
        a_range = np.clip(norm["max"] - norm["min"], 1e-6, None)
        actions_10d = (actions_norm + 1.0) / 2.0 * a_range + norm["min"]

        # Convert back to 7D
        actions_7d_recovered = convert_actions_from_rot6d(actions_10d)

        # Load original 7D from HDF5
        with h5py.File(rot6d_hdf5, "r") as f:
            first_key = list(f["data"].keys())[0]
            actions_7d_original = f[f"data/{first_key}/actions"][:T_P]

        # Roundtrip should match within tolerance
        np.testing.assert_allclose(
            actions_7d_recovered, actions_7d_original, atol=1e-4,
            err_msg="rot6d roundtrip failed"
        )

    def test_rot6d_pos_and_grip_unchanged(self, rot6d_hdf5):
        """Position (first 3) and gripper (last 1) survive rot6d conversion."""
        ds = Stage3Dataset(rot6d_hdf5, T_obs=T_O, T_pred=T_P, use_rot6d=True, norm_mode="minmax")
        sample = ds[0]
        actions_norm = sample["actions"].numpy()

        # Denormalize
        norm = ds._norm_per_file[0]["actions"]
        a_range = np.clip(norm["max"] - norm["min"], 1e-6, None)
        actions_10d = (actions_norm + 1.0) / 2.0 * a_range + norm["min"]

        # Load original 7D
        with h5py.File(rot6d_hdf5, "r") as f:
            first_key = list(f["data"].keys())[0]
            actions_7d = f[f"data/{first_key}/actions"][:T_P]

        # Position (dims 0-2) should match exactly
        np.testing.assert_allclose(actions_10d[:, :3], actions_7d[:, :3], atol=1e-5)
        # Gripper (dim 9 in 10D = dim 6 in 7D) should match exactly
        np.testing.assert_allclose(actions_10d[:, 9], actions_7d[:, 6], atol=1e-5)

    def test_10d_norm_stats_shape(self, rot6d_hdf5):
        """Norm stats loaded by dataset have 10 dims when use_rot6d=True."""
        ds = Stage3Dataset(rot6d_hdf5, T_obs=T_O, T_pred=T_P, use_rot6d=True)
        norm = ds._norm_per_file[0]["actions"]
        assert norm["min"].shape == (10,)
        assert norm["max"].shape == (10,)
        assert norm["mean"].shape == (10,)
        assert norm["std"].shape == (10,)

    def test_dataloader_with_rot6d(self, rot6d_hdf5):
        """DataLoader batching works with rot6d."""
        ds = Stage3Dataset(rot6d_hdf5, T_obs=T_O, T_pred=T_P, use_rot6d=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["actions"].shape == (4, T_P, 10)

    def test_default_use_rot6d_false(self, synthetic_hdf5):
        """Default use_rot6d=False preserves existing 7D behavior."""
        ds = Stage3Dataset(synthetic_hdf5, T_obs=T_O, T_pred=T_P)
        assert ds.use_rot6d is False
        sample = ds[0]
        assert sample["actions"].shape == (T_P, D_ACT)
