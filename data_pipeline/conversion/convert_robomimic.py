"""Convert robomimic HDF5 demos to unified schema.

Camera mapping:
  agentview            -> slot 0 (front)
  robot0_eye_in_hand   -> slot 3 (wrist)
  slots 1-2            -> zeros, view_present=False

Actions are copied directly (already 7D delta-EE).
Images are resized 84 -> 224.
"""

import h5py
import numpy as np
import yaml
from pathlib import Path


def load_paths(config_path: str) -> dict:
    raise NotImplementedError


def convert_task(raw_hdf5_path: str, output_path: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
