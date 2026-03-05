"""Unit tests for unified HDF5 schema and conversion outputs.

Run with: pytest data_pipeline/tests/test_schema.py
"""

import pytest
import numpy as np
import h5py


def test_action_shape(unified_hdf5_path):
    raise NotImplementedError


def test_view_flags(unified_hdf5_path):
    raise NotImplementedError


def test_padded_cameras_are_zeros(unified_hdf5_path):
    raise NotImplementedError


def test_real_cameras_nonzero(unified_hdf5_path):
    raise NotImplementedError


def test_action_ranges(unified_hdf5_path):
    raise NotImplementedError


def test_normalization_roundtrip(unified_hdf5_path):
    raise NotImplementedError


def test_no_data_leakage(train_keys, val_keys):
    raise NotImplementedError
