"""Image transforms: resize, ImageNet normalization, augmentation stubs."""

import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def resize_image(img: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    raise NotImplementedError


def imagenet_normalize(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC [0,255] -> float32 CHW ImageNet-normalized."""
    raise NotImplementedError
