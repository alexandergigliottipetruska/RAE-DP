"""Convert RLBench PerAct demos to unified schema.

Camera mapping:
  front_rgb          -> slot 0
  left_shoulder_rgb  -> slot 1
  right_shoulder_rgb -> slot 2
  wrist_rgb          -> slot 3

Actions: absolute EE pose -> 7D delta (pos + axis-angle + gripper).
Quaternion convention: RLBench stores wxyz, scipy expects xyzw.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def compute_delta_actions(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    grippers: np.ndarray,
) -> np.ndarray:
    """Absolute EE poses -> delta actions [T-1, 7]."""
    raise NotImplementedError


def convert_task(task_dir: str, output_path: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
