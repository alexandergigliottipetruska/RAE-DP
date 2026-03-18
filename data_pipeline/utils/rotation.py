"""Rotation representation conversions (matches Chi et al. diffusion_policy).

Converts between axis-angle (3D) and rotation_6d (6D) via rotation matrix.
Uses the same approach as diffusion_policy/model/common/rotation_transformer.py
but without the pytorch3d dependency.

6D rotation (Zhou et al. 2019) = first two columns of the rotation matrix, flattened.
Continuous, singularity-free, proven superior for learning.

Usage:
    # Convert actions: 7D (pos3 + aa3 + grip1) → 10D (pos3 + rot6d + grip1)
    actions_10d = convert_actions_to_rot6d(actions_7d)

    # Convert back: 10D → 7D (for robosuite)
    actions_7d = convert_actions_from_rot6d(actions_10d)
"""

import numpy as np


def axis_angle_to_rot6d(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (*, 3) → 6D rotation (*, 6).

    axis_angle → rotation_matrix → first two columns flattened.
    """
    orig_shape = aa.shape[:-1]
    aa_flat = aa.reshape(-1, 3).astype(np.float64)

    # Rodrigues formula: aa → rotation matrix
    theta = np.linalg.norm(aa_flat, axis=1, keepdims=True)  # (N, 1)
    safe_theta = np.where(theta < 1e-8, np.ones_like(theta), theta)
    k = aa_flat / safe_theta  # unit axis

    K = np.zeros((len(aa_flat), 3, 3), dtype=np.float64)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    eye = np.eye(3, dtype=np.float64)[None]
    sin_t = np.sin(theta)[:, :, None]
    cos_t = np.cos(theta)[:, :, None]
    R = eye + sin_t * K + (1 - cos_t) * (K @ K)

    # Near-zero angle: R ≈ I
    near_zero = (theta.squeeze(-1) < 1e-8)
    R[near_zero] = np.eye(3, dtype=np.float64)

    # First two columns → 6D
    rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)  # (N, 6)
    return rot6d.reshape(*orig_shape, 6).astype(np.float32)


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation (*, 6) → axis-angle (*, 3).

    Gram-Schmidt orthogonalization → rotation matrix → axis-angle.
    """
    orig_shape = rot6d.shape[:-1]
    rot6d_flat = rot6d.reshape(-1, 6).astype(np.float64)

    # Gram-Schmidt: recover orthonormal rotation matrix
    a1 = rot6d_flat[:, :3]
    a2 = rot6d_flat[:, 3:]

    b1 = a1 / np.clip(np.linalg.norm(a1, axis=1, keepdims=True), 1e-8, None)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)

    # Rotation matrix → axis-angle (Rodrigues inverse)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_angle)  # (N,)

    axis = np.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], axis=1)  # (N, 3)

    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    safe_norm = np.where(norm < 1e-8, np.ones_like(norm), norm)
    axis = axis / safe_norm

    # Near-zero angle: return zeros
    near_zero = (theta < 1e-8)
    aa = axis * theta[:, None]
    aa[near_zero] = 0.0

    return aa.reshape(*orig_shape, 3).astype(np.float32)


def convert_actions_to_rot6d(actions: np.ndarray) -> np.ndarray:
    """Convert 7D actions (pos3 + axis_angle3 + grip1) → 10D (pos3 + rot6d6 + grip1).

    Matches Chi et al.'s _convert_actions() in robomimic_replay_image_dataset.py.
    """
    pos = actions[..., :3]
    rot_aa = actions[..., 3:6]
    gripper = actions[..., 6:]
    rot_6d = axis_angle_to_rot6d(rot_aa)
    return np.concatenate([pos, rot_6d, gripper], axis=-1).astype(np.float32)


def convert_actions_from_rot6d(actions: np.ndarray) -> np.ndarray:
    """Convert 10D actions (pos3 + rot6d6 + grip1) → 7D (pos3 + axis_angle3 + grip1).

    Inverse of convert_actions_to_rot6d. Used at eval time before passing to robosuite.
    """
    pos = actions[..., :3]
    rot_6d = actions[..., 3:9]
    gripper = actions[..., 9:]
    rot_aa = rot6d_to_axis_angle(rot_6d)
    return np.concatenate([pos, rot_aa, gripper], axis=-1).astype(np.float32)
