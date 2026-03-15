import h5py
import torch
import numpy as np
import tqdm
from models.encoder import FrozenMultiViewEncoder
from data_pipeline.conversion.unified_schema import read_mask
import os

# 1. Define device at the top level
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from data_pipeline.datasets.stage1_dataset import _IMAGENET_MEAN, _IMAGENET_STD

# Convert the numpy constants to torch tensors for the GPU
mean = torch.from_numpy(_IMAGENET_MEAN).view(1, 1, 3, 1, 1).to(device)
std = torch.from_numpy(_IMAGENET_STD).view(1, 1, 3, 1, 1).to(device)

def extract_features(hdf5_path, device="cuda"):
    # 1. Initialize the REAL encoder (not in cache mode) to do the work
    encoder = FrozenMultiViewEncoder(pretrained=True, use_cache=False).to(device)
    encoder.eval()

    out_path = hdf5_path.replace(".hdf5", "_features.hdf5").replace(".h5", "_features.h5")
    
    with h5py.File(hdf5_path, "r") as src, h5py.File(out_path, "w") as dst:
        # We process ALL keys in the file (train + valid)
        # The split logic remains in the Dataset class which reads the mask
        all_keys = list(src["data"].keys())
        
        for key in tqdm.tqdm(all_keys, desc=f"Processing {os.path.basename(hdf5_path)}"):
            imgs_uint8 = src[f"data/{key}/images"][:] # (T, K, H, W, 3)
            T, K, H, W, _ = imgs_uint8.shape
            
            # Prepare storage in new HDF5
            # Shape: (T, K, 196, 1024)
            dst_ds = dst.create_dataset(
                f"data/{key}/dino_features", 
                shape=(T, K, 196, 1024), 
                dtype='f2', 
                chunks=(1, K, 196, 1024), # Chunking per timestep for fast loading
                compression="gzip",
                compression_opts=4,
                shuffle=True
            )
            
            for t in range(T):
                # Convert to tensor and normalize: (K, 3, H, W)
                img_t = torch.from_numpy(imgs_uint8[t]).to(device).float() / 255.0
                img_t = img_t.permute(0, 3, 1, 2) # HWC -> CHW
                img_t = (img_t - mean.squeeze(0)) / std.squeeze(0)
                
                with torch.no_grad():
                    # Get tokens: (K, 196, 1024)
                    tokens = encoder(img_t)
                    # When saving, cast to half precision
                    dst_ds[t] = tokens.cpu().half().numpy()

if __name__ == "__main__":
    # Add your list of HDF5 files here
    files = [
        # "data/complete_unified_data/can.hdf5",
        "data/unified/close_jar.hdf5"
        ]

    for f in files:
        extract_features(f)