"""Diagnostic script to isolate Stage 3 eval failures.

Tests each component of the eval pipeline independently:
1. Denormalization round-trip (are norm stats correct?)
2. GT action replay through eval denormalization (do denormalized actions work?)
3. Policy prediction comparison (does the model predict reasonable actions?)

Usage:
  python training/debug_eval.py \
    --checkpoint checkpoints/stage3/best.pt \
    --stage1_checkpoint checkpoints/epoch_024.pt \
    --hdf5 data/unified/robomimic/lift/ph.hdf5 \
    --raw_hdf5 data/raw/robomimic/lift/ph/demo_v15.hdf5
"""

import argparse
import logging
import os
import sys
import warnings

import h5py
import numpy as np
import torch

warnings.filterwarnings("ignore", module="robosuite")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline.conversion.compute_norm_stats import load_norm_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def test_denorm_roundtrip(hdf5_path, norm_mode="minmax"):
    """Test 1: Verify normalization round-trip."""
    log.info("=" * 60)
    log.info("TEST 1: Normalization round-trip")
    norm = load_norm_stats(hdf5_path)
    a = norm["actions"]

    with h5py.File(hdf5_path, "r") as f:
        demo = sorted(f["data"].keys())[0]
        raw = f[f"data/{demo}/actions"][:5]  # first 5 actions

    log.info("Raw actions[0]: %s", np.array2string(raw[0], precision=4))
    log.info("Action min: %s", np.array2string(a["min"], precision=4))
    log.info("Action max: %s", np.array2string(a["max"], precision=4))

    if norm_mode == "minmax":
        a_range = np.clip(a["max"] - a["min"], 1e-6, None)
        normalized = 2.0 * (raw - a["min"]) / a_range - 1.0
        denormalized = (normalized + 1.0) / 2.0 * a_range + a["min"]
    else:
        normalized = (raw - a["mean"]) / np.clip(a["std"], 1e-6, None)
        denormalized = normalized * a["std"] + a["mean"]

    log.info("Normalized[0]: %s", np.array2string(normalized[0], precision=4))
    log.info("Denormalized[0]: %s", np.array2string(denormalized[0], precision=4))
    max_err = np.max(np.abs(raw - denormalized))
    log.info("Max round-trip error: %.2e", max_err)
    assert max_err < 1e-5, f"Round-trip error too large: {max_err}"
    log.info("PASSED")


def test_gt_replay_with_denorm(hdf5_path, raw_hdf5_path, norm_mode="minmax"):
    """Test 2: Replay GT actions through the full denormalization pipeline."""
    log.info("=" * 60)
    log.info("TEST 2: GT replay with eval-style denormalization")

    from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper

    norm = load_norm_stats(hdf5_path)
    a = norm["actions"]
    env = RobomimicWrapper("lift")

    with h5py.File(hdf5_path, "r") as uf, h5py.File(raw_hdf5_path, "r") as rf:
        demo_keys = sorted(uf["data"].keys())[:5]  # test 5 demos
        successes = 0

        for key in demo_keys:
            raw_actions = uf[f"data/{key}/actions"][:]

            # Normalize then denormalize (simulating what eval does)
            if norm_mode == "minmax":
                a_range = np.clip(a["max"] - a["min"], 1e-6, None)
                normalized = 2.0 * (raw_actions - a["min"]) / a_range - 1.0
                denormed = (normalized + 1.0) / 2.0 * a_range + a["min"]
            else:
                normalized = (raw_actions - a["mean"]) / np.clip(a["std"], 1e-6, None)
                denormed = normalized * a["std"] + a["mean"]

            # Restore initial state and replay
            initial_state = rf[f"data/{key}/states"][0]
            env.reset()
            env._env.sim.set_state_from_flattened(initial_state)
            env._env.sim.forward()
            env._last_obs = env._env._get_observations()

            success = False
            for t in range(len(denormed)):
                _, _, _, info = env.step(denormed[t])
                if info["success"]:
                    success = True
                    break

            successes += int(success)
            log.info("  %s: %s (steps=%d/%d)", key,
                     "SUCCESS" if success else "FAIL", t + 1, len(denormed))

        env.close()
    log.info("GT replay: %d/%d", successes, len(demo_keys))
    log.info("PASSED" if successes >= 4 else "FAILED")


def test_policy_predictions(checkpoint, stage1_checkpoint, hdf5_path,
                            norm_mode="minmax", device="cuda"):
    """Test 3: Compare policy predictions to training data."""
    log.info("=" * 60)
    log.info("TEST 3: Policy prediction analysis")

    from training.eval_stage3 import load_policy
    from data_pipeline.datasets.stage3_dataset import Stage3Dataset

    policy = load_policy(checkpoint, stage1_checkpoint, device)
    policy.eval()

    norm = load_norm_stats(hdf5_path)
    a = norm["actions"]

    # Load a few training samples
    ds = Stage3Dataset(hdf5_path, split="train", T_obs=2, T_pred=16, norm_mode=norm_mode)
    log.info("Dataset size: %d", len(ds))

    for i in [0, 100, 500]:
        if i >= len(ds):
            continue
        sample = ds[i]

        # Get GT normalized actions
        gt_norm = sample["actions"]  # (T_pred, ac_dim) normalized

        # Run policy prediction
        obs = {}
        if "cached_tokens" in sample:
            obs["cached_tokens"] = sample["cached_tokens"].unsqueeze(0).to(device)
        else:
            obs["images_enc"] = sample["images_enc"].unsqueeze(0).to(device)
        obs["proprio"] = sample["proprio"].unsqueeze(0).to(device)
        obs["view_present"] = sample["view_present"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = policy.predict_action(obs)[0].cpu()  # (T_pred, ac_dim)

        # Compare
        mse = ((pred_norm - gt_norm) ** 2).mean().item()
        log.info("  Sample %d: MSE(pred, gt) = %.4f", i, mse)
        log.info("    GT  norm[0]: %s", np.array2string(gt_norm[0].numpy(), precision=4))
        log.info("    Pred norm[0]: %s", np.array2string(pred_norm[0].numpy(), precision=4))

        # Denormalize both
        if norm_mode == "minmax":
            a_range = np.clip(a["max"] - a["min"], 1e-6, None)
            gt_raw = (gt_norm.numpy() + 1.0) / 2.0 * a_range + a["min"]
            pred_raw = (pred_norm.numpy() + 1.0) / 2.0 * a_range + a["min"]
        else:
            gt_raw = gt_norm.numpy() * a["std"] + a["mean"]
            pred_raw = pred_norm.numpy() * a["std"] + a["mean"]

        log.info("    GT  raw[0]: %s", np.array2string(gt_raw[0], precision=4))
        log.info("    Pred raw[0]: %s", np.array2string(pred_raw[0], precision=4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--hdf5", required=True, help="Unified HDF5")
    parser.add_argument("--raw_hdf5", default=None, help="Raw robomimic HDF5 (for GT replay test)")
    parser.add_argument("--norm_mode", default="minmax")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Test 1: Denorm round-trip
    test_denorm_roundtrip(args.hdf5, args.norm_mode)

    # Test 2: GT replay (only if raw HDF5 provided)
    if args.raw_hdf5:
        test_gt_replay_with_denorm(args.hdf5, args.raw_hdf5, args.norm_mode)
    else:
        log.info("Skipping GT replay test (no --raw_hdf5)")

    # Test 3: Policy predictions
    test_policy_predictions(
        args.checkpoint, args.stage1_checkpoint, args.hdf5,
        args.norm_mode, device,
    )


if __name__ == "__main__":
    main()
