"""Convert ManiSkill3 trajectory HDF5 to unified schema.

ManiSkill3's replay tool handles action space conversion before this script runs:
  python -m mani_skill.trajectory.replay_trajectory --traj-path ... -c pd_ee_delta_pos

Camera configuration: 4 cameras matching RLBench layout (to be configured).
"""

import h5py
import numpy as np
from pathlib import Path


def convert_task(replayed_traj_path: str, output_path: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
