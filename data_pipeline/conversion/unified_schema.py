"""Unified HDF5 schema. SINGLE SOURCE OF TRUTH for data format.
Both conversion scripts and the dataset class import from here.
"""

import h5py
import numpy as np

NUM_CAMERA_SLOTS = 4
IMAGE_SIZE = (224, 224)
ACTION_DIM = 7
VALID_BENCHMARKS = ["robomimic", "rlbench", "maniskill", "kitchen"]


def create_demo_group(hdf5_file, demo_key, T, D_prop, compress=True):
    """Create a demo group with correct dataset shapes."""
    raise NotImplementedError
