"""Precompute frozen DINOv3 encoder tokens for fast Stage 3 training.

Runs frozen DINOv3-L + Cancel-Affine LN on all images once and saves
the (196, 1024) tokens to a cache HDF5.  Training then skips the encoder
entirely and reads cached tokens directly (see Stage3Dataset).

The training dataloader casts every dtype to float32 at load time
(`tokens_raw.astype(np.float32)`), so any storage dtype or compression
scheme is fully plug-and-play — no training code changes needed.

Storage presets (--preset):
  ┌──────────────┬────────┬─────────────┬──────────────────────────────────┐
  │ Preset       │ Dtype  │ Compression │ Notes                           │
  ├──────────────┼────────┼─────────────┼──────────────────────────────────┤
  │ fp16-none    │ fp16   │ none        │ NOT recommended. 14.5 GB.       │
  │              │        │             │ Fastest reads, but fp16 clips   │
  │              │        │             │ small values — causes overfitting│
  ├──────────────┼────────┼─────────────┼──────────────────────────────────┤
  │ bf16-none    │ bf16   │ none        │ Same precision issues as fp16   │
  │              │        │             │ (bf16→fp16 for numpy compat).   │
  │              │        │             │ Prefer fp32-lzf instead.        │
  ├──────────────┼────────┼─────────────┼──────────────────────────────────┤
  │ fp32-lzf     │ fp32   │ lzf         │ RECOMMENDED. Full precision,    │
  │              │        │             │ fast lossless decompression.     │
  │              │        │             │ ~22-25 GB.                      │
  ├──────────────┼────────┼─────────────┼──────────────────────────────────┤
  │ fp32-gzip    │ fp32   │ gzip (lvl1) │ Full precision, better ratio    │
  │              │        │             │ but slower reads. ~20-23 GB.    │
  ├──────────────┼────────┼─────────────┼──────────────────────────────────┤
  │ fp32-shuffle │ fp32   │ shuffle +   │ Best compression for floats.    │
  │              │        │ gzip (lvl4) │ ~16-20 GB. Moderate read speed. │
  ├──────────────┼────────┼─────────────┼──────────────────────────────────┤
  │ fp32-none    │ fp32   │ none        │ Full precision, fastest reads.  │
  │              │        │             │ ~29 GB. Use if disk is free.    │
  └──────────────┴────────┴─────────────┴──────────────────────────────────┘

Usage examples:
  # Recommended: fp32 with fast LZF decompression (lossless, ~22-25 GB)
  python training/precompute_tokens.py \\
    --hdf5 data/unified/robomimic/lift/ph.hdf5 \\
    --preset fp32-lzf

  # Best compression ratio (~16-20 GB, slightly slower reads)
  python training/precompute_tokens.py \\
    --hdf5 data/unified/robomimic/lift/ph.hdf5 \\
    --preset fp32-shuffle

  # Maximum read speed, no compression (~29 GB)
  python training/precompute_tokens.py \\
    --hdf5 data/unified/robomimic/lift/ph.hdf5 \\
    --preset fp32-none

  # Legacy behaviour (fp16 — NOT recommended, causes precision loss)
  python training/precompute_tokens.py \\
    --hdf5 data/unified/robomimic/lift/ph.hdf5 \\
    --preset fp16-none
"""

import argparse
import logging
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.encoder import FrozenMultiViewEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Storage presets ──────────────────────────────────────────────────────
# Each preset is (numpy_dtype, torch_dtype, hdf5_kwargs).
# hdf5_kwargs are passed directly to h5py create_dataset().

PRESETS = {
    "fp16-none": {
        "np_dtype": np.float16,
        "torch_dtype": torch.float16,
        "h5_kwargs": {},                       # no compression
    },
    "bf16-none": {
        "np_dtype": np.float16,                # stored as float16 bytes
        "torch_dtype": torch.bfloat16,         # converted via bfloat16 first
        "h5_kwargs": {},
    },
    "fp32-none": {
        "np_dtype": np.float32,
        "torch_dtype": torch.float32,
        "h5_kwargs": {},
    },
    "fp32-lzf": {
        "np_dtype": np.float32,
        "torch_dtype": torch.float32,
        "h5_kwargs": {"compression": "lzf"},
    },
    "fp32-gzip": {
        "np_dtype": np.float32,
        "torch_dtype": torch.float32,
        "h5_kwargs": {"compression": "gzip", "compression_opts": 1},
    },
    "fp32-shuffle": {
        "np_dtype": np.float32,
        "torch_dtype": torch.float32,
        "h5_kwargs": {"shuffle": True, "compression": "gzip", "compression_opts": 4},
    },
}


def _to_numpy(tensor: torch.Tensor, preset: dict) -> np.ndarray:
    """Convert a GPU tensor to a numpy array in the preset's storage dtype.

    For bfloat16: torch casts fp32→bf16 on GPU (preserving range), then we
    cast bf16→fp16 in PyTorch (proper numerical conversion, not raw bit
    reinterpret) for numpy/HDF5 storage.

    WARNING on bf16 storage: numpy has no native bfloat16 dtype, so bf16
    values are cast bf16→fp16 in PyTorch (proper numerical conversion, not
    bit reinterpret). This means bf16-none suffers the same fp16 precision
    loss that causes degraded training (fp32→bf16→fp16→fp32 is double-lossy).
    Prefer fp32-lzf for lossless storage with fast reads.
    """
    torch_dtype = preset["torch_dtype"]
    np_dtype = preset["np_dtype"]

    if torch_dtype == torch.bfloat16:
        # fp32 → bf16 (keeps range) → fp16 (for numpy compatibility)
        return tensor.cpu().to(torch.bfloat16).to(torch.float16).numpy()
    else:
        return tensor.cpu().to(torch_dtype).numpy()


def precompute(
    hdf5_path: str,
    output_path: str,
    preset_name: str = "fp32-lzf",
    batch_size: int = 128,
    device: str = "cuda",
    rot6d: bool = False,
):
    """Precompute encoder tokens and save to cache HDF5.

    The output HDF5 mirrors the input structure but replaces `images` with
    `tokens` of shape (T, K, 196, 1024). Actions, proprio, view_present,
    masks, and norm_stats are copied verbatim.

    Args:
        hdf5_path:   Path to the source unified HDF5 (must contain `images`).
        output_path: Where to write the token cache HDF5.
        preset_name: Storage preset name (see PRESETS dict / docstring table).
        batch_size:  Images per forward pass through the encoder.
        device:      Torch device string ('cuda', 'cuda:0', 'cpu').
        rot6d:       If True, recompute norm_stats with 10D rot6d actions.
    """
    preset = PRESETS[preset_name]
    np_dtype = preset["np_dtype"]
    h5_kwargs = preset["h5_kwargs"]

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info(
        "Device: %s | batch_size: %d | preset: %s (dtype=%s, compression=%s)",
        device, batch_size, preset_name,
        np.dtype(np_dtype).name,
        h5_kwargs.get("compression", "none"),
    )

    # ── Load frozen encoder ──────────────────────────────────────────────
    encoder = FrozenMultiViewEncoder(pretrained=True).to(device).eval()
    cancel_affine_ln = nn.LayerNorm(1024, elementwise_affine=False).to(device)

    with h5py.File(hdf5_path, "r") as src, h5py.File(output_path, "w") as dst:
        # Copy file-level attributes
        for attr in src.attrs:
            dst.attrs[attr] = src.attrs[attr]
        dst.attrs["has_cached_tokens"] = True
        dst.attrs["token_preset"] = preset_name

        # Copy masks and norm_stats verbatim
        if "mask" in src:
            src.copy("mask", dst)
        if "norm_stats" in src:
            src.copy("norm_stats", dst)

        # Copy data-level attributes (e.g. env_args for robomimic eval)
        if "data" not in dst:
            dst.create_group("data")
        if "data" in src:
            for attr in src["data"].attrs:
                dst["data"].attrs[attr] = src["data"].attrs[attr]

        all_keys = sorted(src["data"].keys())
        log.info("Processing %d demos from %s", len(all_keys), hdf5_path)

        # Store original num_cam_slots so the loader knows full K
        first_grp = src[f"data/{all_keys[0]}"]
        K_full = first_grp["images"].shape[1]
        dst.attrs["num_cam_slots"] = K_full

        for key in tqdm(all_keys, desc="Precomputing tokens"):
            grp = src[f"data/{key}"]
            T = grp["images"].shape[0]       # timesteps
            view_present = grp["view_present"][:]

            # Only store active views (saves 50% for robomimic 2/4 cameras)
            active_indices = np.where(view_present)[0]
            K_active = len(active_indices)
            tokens_all = np.zeros((T, K_active, 196, 1024), dtype=np_dtype)

            for i, k in enumerate(active_indices):
                # Load all T images for this view: (T, H, W, 3) uint8
                imgs_raw = grp["images"][:, k]

                # Normalize: uint8 → [0,1] → [-1,1] → ImageNet stats
                imgs_float = imgs_raw.astype(np.float32) / 255.0
                imgs_neg11 = imgs_float * 2.0 - 1.0
                imgs_norm = (imgs_neg11 - _IMAGENET_MEAN) / _IMAGENET_STD
                imgs_chw = np.ascontiguousarray(
                    np.moveaxis(imgs_norm, -1, -3)
                )  # (T, 3, H, W)
                imgs_tensor = torch.from_numpy(imgs_chw)

                # Encode in batches
                view_tokens = []
                for j in range(0, T, batch_size):
                    batch = imgs_tensor[j : j + batch_size].to(device)
                    with torch.no_grad():
                        raw = encoder(batch)           # (B, 196, 1024) fp32
                        normed = cancel_affine_ln(raw) # (B, 196, 1024) fp32
                    view_tokens.append(_to_numpy(normed, preset))

                tokens_all[:, i] = np.concatenate(view_tokens, axis=0)

            # ── Write demo to output ─────────────────────────────────────
            dst_grp = dst.create_group(f"data/{key}")

            # Chunking helps compressed datasets; harmless for uncompressed
            chunks = (min(T, 16), K_active, 196, 1024) if h5_kwargs else None
            dst_grp.create_dataset(
                "tokens", data=tokens_all, chunks=chunks, **h5_kwargs,
            )
            # Map from compact index → original camera slot
            dst_grp.create_dataset("active_cam_indices", data=active_indices)

            # Copy non-image datasets verbatim
            for ds_name in ["actions", "proprio", "view_present"]:
                if ds_name in grp:
                    dst_grp.create_dataset(ds_name, data=grp[ds_name][:])

    # ── Optional: recompute norm_stats with rot6d ────────────────────────
    if rot6d:
        from data_pipeline.conversion.unified_schema import read_mask
        from data_pipeline.conversion.compute_norm_stats import compute_and_save_norm_stats
        with h5py.File(output_path, "a") as dst:
            train_keys = read_mask(dst, "train")
            compute_and_save_norm_stats(dst, train_keys, rot6d=True)
        log.info("Recomputed 10D rot6d norm stats")

    # ── Report final sizes ───────────────────────────────────────────────
    src_size = os.path.getsize(hdf5_path) / (1024**3)
    dst_size = os.path.getsize(output_path) / (1024**3)
    log.info(
        "Done! Source: %.2f GB → Cache: %.2f GB (%.1f%% of source)",
        src_size, dst_size, 100.0 * dst_size / src_size,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Precompute DINOv3 encoder tokens for Stage 3 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Presets:\n"
            "  fp16-none    fp16, no compression        (~14.5 GB, fastest reads)\n"
            "  bf16-none    bfloat16→fp16, no compress   (~14.5 GB, better range)\n"
            "  fp32-none    fp32, no compression         (~29 GB, full precision)\n"
            "  fp32-lzf     fp32 + LZF (RECOMMENDED)     (~22-25 GB, fast decompress)\n"
            "  fp32-gzip    fp32 + gzip level 1          (~20-23 GB, moderate speed)\n"
            "  fp32-shuffle fp32 + byte-shuffle + gzip 4 (~16-20 GB, best ratio)\n"
        ),
    )
    parser.add_argument("--hdf5", required=True, help="Input unified HDF5 with images")
    parser.add_argument(
        "--output", default=None,
        help="Output cache HDF5 path (default: auto-generated from preset name)",
    )
    parser.add_argument(
        "--preset", default="fp32-lzf", choices=list(PRESETS.keys()),
        help="Storage preset: dtype + compression combo (default: fp32-lzf)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rot6d", action="store_true",
        help="Recompute norm stats with 10D rot6d actions (for V3 training)",
    )
    args = parser.parse_args()

    # Auto-generate output filename from preset
    if args.output is None:
        base, ext = os.path.splitext(args.hdf5)
        suffix = f"_tokens_{args.preset.replace('-', '_')}"
        args.output = f"{base}{suffix}{ext}"

    precompute(
        args.hdf5, args.output,
        preset_name=args.preset,
        batch_size=args.batch_size,
        device=args.device,
        rot6d=args.rot6d,
    )


if __name__ == "__main__":
    main()
