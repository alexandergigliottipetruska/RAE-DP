"""Visualization utilities for evaluation rollouts.

Provides:
- save_rollout_video: Stitch multi-view rollout frames into MP4.
- plot_action_trajectory: Plot predicted vs ground-truth action dimensions.
- plot_success_rates: Bar chart of success rates across configs/checkpoints.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no Tk/Tcl needed
import numpy as np


# ImageNet denormalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_image(img_chw: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalization: [3, H, W] float32 -> [H, W, 3] uint8."""
    img_hwc = np.moveaxis(img_chw, 0, -1)  # [H, W, 3]
    img = img_hwc * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def save_rollout_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: int = 10,
    view_labels: list[str] | None = None,
) -> None:
    """Save multi-view rollout frames as an MP4 video.

    Args:
        frames: List of [K, 3, H, W] float32 ImageNet-normalized arrays,
                one per timestep.
        output_path: Path for the output MP4.
        fps: Frames per second.
        view_labels: Optional labels for each camera slot.
    """
    import imageio.v3 as iio

    if not frames:
        return

    video_frames = []
    for frame in frames:
        # frame: [K, 3, H, W] -> denormalize each view, concatenate horizontally
        views = []
        for k in range(frame.shape[0]):
            img_uint8 = denormalize_image(frame[k])
            views.append(img_uint8)
        # Horizontal stack: [H, K*W, 3]
        video_frames.append(np.concatenate(views, axis=1))

    output_path = str(output_path)
    iio.imwrite(output_path, video_frames, fps=fps, codec="libx264")


def plot_action_trajectory(
    predicted: np.ndarray,
    ground_truth: np.ndarray | None = None,
    action_labels: list[str] | None = None,
    title: str = "Action Trajectory",
    output_path: str | Path | None = None,
) -> None:
    """Plot action dimensions over time.

    Args:
        predicted: [T, D] predicted actions.
        ground_truth: [T, D] ground truth actions (optional overlay).
        action_labels: Labels for each action dimension.
        title: Plot title.
        output_path: If provided, save figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt

    D = predicted.shape[1]
    if action_labels is None:
        action_labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"][:D]

    fig, axes = plt.subplots(D, 1, figsize=(10, 2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d, (ax, label) in enumerate(zip(axes, action_labels)):
        ax.plot(predicted[:, d], label="predicted", alpha=0.8)
        if ground_truth is not None:
            ax.plot(ground_truth[:, d], label="ground truth", alpha=0.5,
                    linestyle="--")
        ax.set_ylabel(label)
        if d == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(title)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_success_rates(
    labels: list[str],
    rates: list[float],
    ci_lower: list[float] | None = None,
    ci_upper: list[float] | None = None,
    title: str = "Success Rates",
    output_path: str | Path | None = None,
) -> None:
    """Bar chart of success rates with optional Wilson CI error bars.

    Args:
        labels: Config/checkpoint labels.
        rates: Success rates [0, 1].
        ci_lower, ci_upper: Lower/upper CI bounds.
        title: Plot title.
        output_path: If provided, save instead of showing.
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))

    if ci_lower is not None and ci_upper is not None:
        yerr_lo = [r - lo for r, lo in zip(rates, ci_lower)]
        yerr_hi = [hi - r for r, hi in zip(rates, ci_upper)]
        ax.bar(x, rates, yerr=[yerr_lo, yerr_hi], capsize=4, alpha=0.8)
    else:
        ax.bar(x, rates, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
