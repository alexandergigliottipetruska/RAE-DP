"""Visual verification for RLBench unified HDF5 files.

Usage:
  python data_pipeline/scripts/verify_rlbench.py [--task close_jar]

Generates camera_slots and action_trajectory plots for the specified task.
"""

import argparse
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "c:/Users/naqee/OneDrive/Desktop/CSC415 Project/RAEDiTRobotics")
from data_pipeline.conversion.compute_norm_stats import load_norm_stats

DATA_DIR = "c:/Users/naqee/OneDrive/Desktop/CSC415 Project/data/unified/rlbench"
ACTION_LABELS = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
SLOT_LABELS = ["front", "left_shoulder", "right_shoulder", "wrist"]


def verify_task(task):
    hdf5_path = f"{DATA_DIR}/{task}.hdf5"
    print(f"\n=== Verifying: {task} ===")

    with h5py.File(hdf5_path, "r") as f:
        demo_key = list(f["data"].keys())[0]
        images = f[f"data/{demo_key}/images"]
        first_frame = images[0]  # (4, 224, 224, 3) uint8
        view_present = f[f"data/{demo_key}/view_present"][:]
        actions = f[f"data/{demo_key}/actions"][:]
        T = actions.shape[0]

        print(f"Demo: {demo_key}, T={T}")
        print(f"Image dtype: {images.dtype}, shape: {images.shape}")
        print(f"view_present: {view_present}")
        print(f"Action shape: {actions.shape}")

    # --- Camera slots ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, ax in enumerate(axes):
        img = first_frame[i]
        ax.imshow(img)
        ax.set_title(f"Slot {i} - {SLOT_LABELS[i]}\n(present={view_present[i]})", fontsize=10)
        ax.axis("off")
    plt.suptitle(f"RLBench {task} - {demo_key}, t=0", fontsize=13)
    plt.tight_layout()
    out_cam = f"rlbench_cameras_{task}.png"
    plt.savefig(out_cam, dpi=100)
    plt.close()
    print(f"Saved: {out_cam}")

    # --- Action trajectory ---
    t = np.arange(T)
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, actions[:, i], linewidth=1.0)
        ax.set_ylabel(ACTION_LABELS[i], fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    plt.suptitle(f"RLBench {task} - 7D Delta Actions ({demo_key}, T={T})", fontsize=13)
    plt.tight_layout()
    out_act = f"rlbench_actions_{task}.png"
    plt.savefig(out_act, dpi=100)
    plt.close()
    print(f"Saved: {out_act}")

    # --- Norm stats ---
    stats = load_norm_stats(hdf5_path)
    print(f"Action mean: {np.round(stats['actions']['mean'], 5)}")
    print(f"Action std:  {np.round(stats['actions']['std'], 5)}")
    print(f"Proprio mean: {np.round(stats['proprio']['mean'], 5)}")
    print(f"Proprio std:  {np.round(stats['proprio']['std'], 5)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, help="Task name (default: all 3)")
    args = parser.parse_args()

    tasks = ["close_jar", "open_drawer", "slide_block_to_color_target"]
    if args.task:
        tasks = [args.task]

    for task in tasks:
        verify_task(task)

    print("\nDone. Check the generated PNG files.")
