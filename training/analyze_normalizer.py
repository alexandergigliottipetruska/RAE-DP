"""Compare our minmax normalization vs Chi's per-component normalization.

Loads 10D rot6d norm stats from the unified HDF5 and shows exactly how
each approach transforms the action dimensions.

Usage:
  PYTHONPATH=. python training/analyze_normalizer.py \
    --hdf5 data/unified/robomimic/lift/ph_abs_v15.hdf5
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_rot6d_stats(hdf5_path):
    """Load 10D rot6d norm stats from HDF5."""
    from data_pipeline.conversion.compute_norm_stats import compute_and_save_norm_stats
    from data_pipeline.conversion.unified_schema import read_mask

    with h5py.File(hdf5_path, "r") as f:
        # Check if rot6d stats exist
        if "norm_stats" in f and f["norm_stats/actions/mean"].shape[-1] == 10:
            stats = {
                "min": f["norm_stats/actions/min"][:],
                "max": f["norm_stats/actions/max"][:],
                "mean": f["norm_stats/actions/mean"][:],
                "std": f["norm_stats/actions/std"][:],
            }
            return stats

    # Compute on the fly
    print("Computing rot6d stats on the fly...")
    with h5py.File(hdf5_path, "r") as f:
        train_keys = read_mask(f, "train")
        from data_pipeline.utils.rotation import convert_actions_to_rot6d

        all_actions = []
        for key in train_keys:
            acts = f[f"data/{key}/actions"][:]
            acts_10d = convert_actions_to_rot6d(acts)
            all_actions.append(acts_10d)
        all_actions = np.concatenate(all_actions, axis=0)

        return {
            "min": all_actions.min(axis=0).astype(np.float32),
            "max": all_actions.max(axis=0).astype(np.float32),
            "mean": all_actions.mean(axis=0).astype(np.float32),
            "std": all_actions.std(axis=0).astype(np.float32),
        }


def our_minmax(x, stats):
    """Our normalization: all dims minmax to [-1, 1]."""
    a_range = np.clip(stats["max"] - stats["min"], 1e-6, None)
    return 2.0 * (x - stats["min"]) / a_range - 1.0


def our_minmax_denorm(x_norm, stats):
    """Our denormalization: [-1, 1] back to original."""
    a_range = np.clip(stats["max"] - stats["min"], 1e-6, None)
    return (x_norm + 1.0) / 2.0 * a_range + stats["min"]


def chi_normalize(x, stats):
    """Chi's normalization: pos=minmax, other=identity."""
    result = x.copy()
    # Position [0:3]: minmax to [-1, 1]
    pos_min = stats["min"][:3]
    pos_max = stats["max"][:3]
    pos_range = np.clip(pos_max - pos_min, 1e-7, None)
    result[..., :3] = 2.0 * (x[..., :3] - pos_min) / pos_range - 1.0
    # Other [3:10]: identity (no change)
    return result


def chi_denormalize(x_norm, stats):
    """Chi's denormalization: pos=minmax inverse, other=identity."""
    result = x_norm.copy()
    pos_min = stats["min"][:3]
    pos_max = stats["max"][:3]
    pos_range = np.clip(pos_max - pos_min, 1e-7, None)
    result[..., :3] = (x_norm[..., :3] + 1.0) / 2.0 * pos_range + pos_min
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    args = parser.parse_args()

    stats = load_rot6d_stats(args.hdf5)

    dim_names = [
        "pos_x", "pos_y", "pos_z",
        "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
        "gripper",
    ]

    print("\n" + "=" * 90)
    print("ACTION NORMALIZATION COMPARISON: Ours (minmax all) vs Chi (pos=minmax, other=identity)")
    print("=" * 90)

    print(f"\n{'Dim':<10} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} | {'Range':>10}")
    print("-" * 70)
    for i, name in enumerate(dim_names):
        rng = stats["max"][i] - stats["min"][i]
        print(f"{name:<10} {stats['min'][i]:>10.4f} {stats['max'][i]:>10.4f} "
              f"{stats['mean'][i]:>10.4f} {stats['std'][i]:>10.4f} | {rng:>10.4f}")

    # Compute scale factors for both approaches
    print(f"\n{'':=<90}")
    print("SCALE FACTORS (how much each dim gets stretched/compressed)")
    print(f"{'':=<90}")
    print(f"\n{'Dim':<10} {'Our scale':>12} {'Our offset':>12} | {'Chi scale':>12} {'Chi offset':>12} | {'Diff':>8}")
    print("-" * 80)

    for i, name in enumerate(dim_names):
        # Our minmax: scale = 2 / range, offset = -1 - 2*min/range
        our_range = max(stats["max"][i] - stats["min"][i], 1e-6)
        our_scale = 2.0 / our_range
        our_offset = -1.0 - 2.0 * stats["min"][i] / our_range

        # Chi: pos = minmax, other = identity
        if i < 3:
            chi_scale = our_scale  # same for position
            chi_offset = our_offset
        else:
            chi_scale = 1.0
            chi_offset = 0.0

        diff = abs(our_scale - chi_scale)
        marker = " <<<" if diff > 0.01 else ""
        print(f"{name:<10} {our_scale:>12.4f} {our_offset:>12.4f} | "
              f"{chi_scale:>12.4f} {chi_offset:>12.4f} | {diff:>8.4f}{marker}")

    # Show what happens to example values
    print(f"\n{'':=<90}")
    print("EXAMPLE: Normalize the MEAN action vector")
    print(f"{'':=<90}")

    mean_action = stats["mean"].copy()
    our_normed = our_minmax(mean_action, stats)
    chi_normed = chi_normalize(mean_action, stats)

    print(f"\n{'Dim':<10} {'Raw value':>12} | {'Ours':>12} {'Chi':>12} | {'Difference':>12}")
    print("-" * 75)
    for i, name in enumerate(dim_names):
        diff = our_normed[i] - chi_normed[i]
        marker = " <<<" if abs(diff) > 0.01 else ""
        print(f"{name:<10} {mean_action[i]:>12.4f} | "
              f"{our_normed[i]:>12.4f} {chi_normed[i]:>12.4f} | {diff:>12.4f}{marker}")

    # Show what happens to example values at extremes
    print(f"\n{'':=<90}")
    print("EXAMPLE: Normalize the MIN action vector")
    print(f"{'':=<90}")

    min_action = stats["min"].copy()
    our_normed = our_minmax(min_action, stats)
    chi_normed = chi_normalize(min_action, stats)

    print(f"\n{'Dim':<10} {'Raw value':>12} | {'Ours':>12} {'Chi':>12} | {'Difference':>12}")
    print("-" * 75)
    for i, name in enumerate(dim_names):
        diff = our_normed[i] - chi_normed[i]
        marker = " <<<" if abs(diff) > 0.01 else ""
        print(f"{name:<10} {min_action[i]:>12.4f} | "
              f"{our_normed[i]:>12.4f} {chi_normed[i]:>12.4f} | {diff:>12.4f}{marker}")

    print(f"\n{'':=<90}")
    print("EXAMPLE: Normalize the MAX action vector")
    print(f"{'':=<90}")

    max_action = stats["max"].copy()
    our_normed = our_minmax(max_action, stats)
    chi_normed = chi_normalize(max_action, stats)

    print(f"\n{'Dim':<10} {'Raw value':>12} | {'Ours':>12} {'Chi':>12} | {'Difference':>12}")
    print("-" * 75)
    for i, name in enumerate(dim_names):
        diff = our_normed[i] - chi_normed[i]
        marker = " <<<" if abs(diff) > 0.01 else ""
        print(f"{name:<10} {max_action[i]:>12.4f} | "
              f"{our_normed[i]:>12.4f} {chi_normed[i]:>12.4f} | {diff:>12.4f}{marker}")

    # DDPM noise impact
    print(f"\n{'':=<90}")
    print("DDPM NOISE IMPACT")
    print("At t=0 (min noise), scheduler adds ~0.01 noise std.")
    print("How does this compare to the signal range in each normalized space?")
    print(f"{'':=<90}")

    noise_std = 0.01  # approximate at t=0
    print(f"\n{'Dim':<10} {'Our range':>12} {'Chi range':>12} | {'Our SNR':>10} {'Chi SNR':>10}")
    print("-" * 70)

    for i, name in enumerate(dim_names):
        our_range_i = max(stats["max"][i] - stats["min"][i], 1e-6)
        our_normed_range = 2.0  # always [-1, 1]
        if i < 3:
            chi_normed_range = 2.0  # pos is also [-1, 1]
        else:
            chi_normed_range = stats["max"][i] - stats["min"][i]  # raw range

        our_snr = our_normed_range / noise_std
        chi_snr = chi_normed_range / noise_std

        marker = " <<<" if abs(our_snr - chi_snr) > 10 else ""
        print(f"{name:<10} {our_normed_range:>12.4f} {chi_normed_range:>12.4f} | "
              f"{our_snr:>10.1f} {chi_snr:>10.1f}{marker}")

    # Roundtrip test
    print(f"\n{'':=<90}")
    print("ROUNDTRIP TEST (normalize -> denormalize)")
    print(f"{'':=<90}")

    test_action = stats["mean"].copy()
    our_rt = our_minmax_denorm(our_minmax(test_action, stats), stats)
    chi_rt = chi_denormalize(chi_normalize(test_action, stats), stats)
    our_err = np.abs(our_rt - test_action).max()
    chi_err = np.abs(chi_rt - test_action).max()
    print(f"Our roundtrip max error:  {our_err:.2e}")
    print(f"Chi roundtrip max error:  {chi_err:.2e}")


if __name__ == "__main__":
    main()
