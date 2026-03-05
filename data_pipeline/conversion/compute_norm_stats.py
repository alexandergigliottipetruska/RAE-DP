"""Compute per-dimension action normalization statistics (mean, std).

Rules:
  - Computed on TRAINING split only. Never val/test.
  - Saved into the unified HDF5 file under attrs or a dedicated group.
  - Used by the dataset class for z-score normalization at load time.
"""

import h5py
import numpy as np
from pathlib import Path


def compute_and_save_norm_stats(unified_hdf5_path: str, train_demo_keys: list) -> dict:
    """Compute per-dim mean/std from training demos and save to file."""
    raise NotImplementedError


def load_norm_stats(unified_hdf5_path: str) -> dict:
    raise NotImplementedError
