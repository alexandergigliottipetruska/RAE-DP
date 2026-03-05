"""Data utilities: normalization, denormalization, demo split helpers."""

import numpy as np


def normalize_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def denormalize_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def get_demo_split(all_demo_keys: list, val_frac: float = 0.1, seed: int = 42) -> tuple:
    """Return (train_keys, val_keys) with no leakage."""
    raise NotImplementedError
