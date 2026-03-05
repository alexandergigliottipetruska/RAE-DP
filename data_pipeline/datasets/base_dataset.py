"""PyTorch dataset class for unified multi-view manipulation HDF5 files.

Benchmark-agnostic. Reads unified HDF5, builds flat index of
(demo_key, timestep) pairs, opens HDF5 per-call for multi-worker safety.

Output per sample:
  images:       (T_o, NUM_CAMERA_SLOTS, 3, H, W)  float32, ImageNet-normalized
  actions:      (T_p, ACTION_DIM)                  float32, z-score normalized
  proprio:      (T_o, D_prop)                      float32
  view_present: (NUM_CAMERA_SLOTS,)                bool
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiViewManipulationDataset(Dataset):

    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        T_obs: int = 2,
        T_pred: int = 50,
    ):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError
