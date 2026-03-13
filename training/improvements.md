Strategic Optimization Report: Distributed HPO Pipeline
This document outlines high-impact improvements to reduce epoch duration and increase hardware efficiency across the lab PC swarm.

1. Architectural Optimization: Feature Pre-Caching
Current State: The frozen DINOv3-L16 encoder is executed on every batch, every epoch.
The Bottleneck: Training is "IO/Compute bound" by a 300M+ parameter frozen backbone that produces identical outputs for the same inputs repeatedly.

Improvement: Decouple the Encoder from the Training Loop.

Action: Create a preprocessing script to pass the HDF5 datasets through the FrozenMultiViewEncoder once. Save the resulting (196, 1024) tokens to disk.

Impact: Estimated 50-70% reduction in epoch time and significantly lower VRAM usage, allowing for larger batch sizes.

2. Training Loop & Math Optimizations
These "Quick-Wins" leverage PyTorch 2.0+ features to accelerate the training script (train_stage1_hp_distributed.py).

3. Storage & I/O Enhancements
Since storage limits on the lab PCs have been increased, we can trade disk space for training speed.

Asynchronous Checkpointing: Use a background thread for torch.save and Hugging Face uploads. This prevents the GPU from idling while waiting for disk/network I/O.

Local Data Striping: Move the HDF5 files from network storage to the local /tmp of each lab PC. This eliminates network latency bottlenecks during data loading.

DataLoader Prefetching: Increase num_workers and prefetch_factor in swarm_config.yaml to ensure the next batch is always ready in RAM before the GPU finishes the current one.

4. Implementation Priority Matrix
High Priority (Immediate): Enable TF32 and foreach=True in the optimizer. These are single-line changes with zero risk.

Medium Priority (Structural): Implement torch.compile. Requires testing to ensure compatibility with your specific environment.

Critical Priority (Scaling): Pre-compute DINOv3 tokens. This is the only way to significantly scale the number of trials processed per hour.