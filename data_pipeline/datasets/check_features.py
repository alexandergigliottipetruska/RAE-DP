import sys
import os
import torch

# Add project root to path so we can find 'models' and 'data_pipeline'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data_pipeline.datasets.stage1_dataset import Stage1Dataset
from models.encoder import FrozenMultiViewEncoder

def verify_cache(hdf5_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Encoder in both modes
    # pretrained=True, use_cache=False (The slow, live DINO path)
    encoder_live = FrozenMultiViewEncoder(pretrained=True, use_cache=False).to(device)
    # use_cache=True (The fast, passthrough path)
    encoder_cache = FrozenMultiViewEncoder(pretrained=True, use_cache=True).to(device)

    # 2. Initialize Datasets
    ds_live = Stage1Dataset(hdf5_path, use_cache=False)
    ds_cache = Stage1Dataset(hdf5_path, use_cache=True)

    # 3. Get sample index 0
    sample_live = ds_live[0]
    sample_cache = ds_cache[0]

    # 4. Compare results
    with torch.no_grad():
        # Shape: (1, K, 196, 1024)
        z_live = encoder_live(sample_live["images_enc"].unsqueeze(0).to(device))
        z_cache = encoder_cache(sample_cache["images_enc"].unsqueeze(0).to(device))

    diff = torch.abs(z_live - z_cache).max().item()
    print(f"\n--- Checking: {os.path.basename(hdf5_path)} ---")
    print(f"Max absolute difference: {diff:.8e}")

    if diff < 1e-2:
        print("✅ PASS: Cache matches Live DINO output.")
    else:
        print("❌ FAIL: Difference too large. Check normalization in extraction script.")

if __name__ == "__main__":
    # Point this to the absolute or relative path of a file you processed
    test_file = "data/complete_unified_data/can.hdf5" 
    verify_cache(test_file)