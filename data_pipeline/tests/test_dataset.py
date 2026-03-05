"""Smoke tests for MultiViewManipulationDataset.

Checks output shapes and value ranges after loading a unified HDF5.
"""

import pytest
import torch
from data_pipeline.datasets.base_dataset import MultiViewManipulationDataset


def test_output_shapes(unified_hdf5_path):
    raise NotImplementedError


def test_image_value_range(unified_hdf5_path):
    """Images should be in ~[-2.1, 2.6] after ImageNet normalization."""
    raise NotImplementedError


def test_view_present_flags(unified_hdf5_path):
    raise NotImplementedError
