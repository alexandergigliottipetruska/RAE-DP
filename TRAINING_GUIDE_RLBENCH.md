# RLBench Training Guide

## Setup (UTM dh2026pcXX machines)

All machines share `/virtual/csc415user/` via NFS. Each machine has an RTX 4080 (16GB).

See `docs/rlbench_setup.md` for CoppeliaSim + PyRep + RLBench installation.

```bash
cd /virtual/csc415user/RAEDiTRobotics
source ~/venv_rlbench/bin/activate

# CoppeliaSim env vars (required for RLBench)
export COPPELIASIM_ROOT=/virtual/csc415user/coppeliasim/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

---

## Step 1: Collect demos

```bash
# Start virtual display (headless rendering)
Xvfb :99 -screen 0 1024x768x24 &

# Collect demos (100 train + 25 valid per task)
DISPLAY=:99 PYTHONPATH=. python data_pipeline/scripts/collect_rlbench_demos.py \
    --tasks open_drawer close_jar \
    --train_episodes 100 --valid_episodes 25 \
    --save_path data/raw/rlbench

# Output: data/raw/rlbench/{train,valid}/{task}/all_variations/episodes/episode{N}/
```

## Step 2: Convert to unified HDF5

```bash
python data_pipeline/conversion/convert_rlbench.py \
    --raw_dir data/raw/rlbench \
    --task open_drawer \
    --output data/raw/rlbench/open_drawer.hdf5
```

## Step 3: Precompute tokens

```bash
python training/precompute_tokens.py \
    --hdf5 data/raw/rlbench/open_drawer.hdf5 \
    --preset fp32-none --rot6d

# Output: data/raw/rlbench/open_drawer_tokens_fp32_none.hdf5
```

## Step 4: Train

### With L1 Flow + spatial tokens (recommended)

```bash
python -m training.train_v3_script \
    --hdf5 data/raw/rlbench/open_drawer_tokens_fp32_none.hdf5 \
    --save_dir checkpoints/v3_rlbench_open_drawer \
    --eval_task open_drawer --eval_mode rlbench \
    --no_amp --no_compile --norm_mode chi \
    --action_space joint \
    --use_flow_matching \
    --spatial_pool_size 7 --n_cond_layers 4 \
    --n_active_cams 4 \
    --num_epochs 3000 --batch_size 64 --seed 42 \
    --eval_full_every_epoch 50 --eval_full_episodes 25 \
    --val_ratio 0.02
```

### Standard DDPM training

Remove `--use_flow_matching` from the command above.

## Step 5: Standalone eval

```bash
# Requires CoppeliaSim + Xvfb running
DISPLAY=:99 python -m training.eval_v3_rlbench \
    --checkpoint checkpoints/v3_rlbench_open_drawer/best.pt \
    --hdf5 data/raw/rlbench/open_drawer_tokens_fp32_none.hdf5 \
    --task open_drawer --num_episodes 25
```

---

## RLBench-specific notes

- **4 cameras**: front, left_shoulder, right_shoulder, wrist (use `--n_active_cams 4`)
- **Joint-space actions**: Use `--action_space joint` which sets `ac_dim=8`, `eval_mode=rlbench`, `norm_mode=chi`, disables rot6d
- **Image size**: RLBench collects at 128x128, resized to 224x224 during conversion (PIL LANCZOS)
- **CoppeliaSim**: Required for both demo collection and eval. Not needed for training (precomputed tokens)

## Available tasks

`open_drawer`, `close_jar`, `meat_off_grill`, `place_wine_at_rack_location`,
`push_buttons`, `slide_block_to_color_target`, `sweep_to_dustpan_of_size`, `turn_tap`

---

## GT Replay (validation)

Replay ground-truth actions to verify the pipeline:

```bash
DISPLAY=:99 python training/gt_replay_rlbench.py \
    --hdf5 data/raw/rlbench/open_drawer.hdf5 \
    --task open_drawer --num_episodes 5
```

Expected: 100% success rate with GT actions.

---

## tmux Cheat Sheet

| Action | Keys / Command |
|--------|---------------|
| Detach | `Ctrl+B`, then `D` |
| Reattach | `tmux attach -t train` |
| List sessions | `tmux ls` |
| Kill session | `tmux kill-session -t train` |
| Scroll up | `Ctrl+B`, then `[`, arrow keys (`q` to exit) |
