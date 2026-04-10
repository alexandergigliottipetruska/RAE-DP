# Distributed Hyperparameter Tuning Guide

Optuna-based HP sweep across UTM lab PCs for Stage 3 policy training (Square task, bf16 tokens).

## Architecture

- **PostgreSQL DB** on `csc415user@dh2026pc02` (port 5433) — coordinates all workers via Optuna
- **Swarm Manager** — SSHes into lab PCs, launches/stops workers
- **Workers** — each independently pulls trials from Optuna, trains for 300 epochs, reports eval success rate
- **Optuna Dashboard** — web UI for monitoring trials, hosted on pc02
- **HuggingFace** — trial checkpoints uploaded after completion, local files cleaned up

## Prerequisites

### 1. Account setup (one-time per account)

Each lab PC that will run trials needs:

```bash
# Clone repo and checkout the distributed branch
git clone https://github.com/alexandergigliottipetruska/RAE-DP.git ~/RAEDiTRobotics
cd ~/RAEDiTRobotics
git checkout distributed_v3_rlbench

# Virtual environment
pip install virtualenv
virtualenv ~/venv_rlbench
source ~/venv_rlbench/bin/activate

# PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Dependencies
pip install -r requirements.txt
pip install robosuite==1.5.2
pip install -e "git+https://github.com/ARISE-Initiative/robomimic.git@e10526b#egg=robomimic"
pip install transformers>=5.0.0 diffusers>=0.37.0 huggingface-hub>=1.0.0
pip install optuna psycopg2-binary
```

### 2. Data (one-time per account)

```bash
# Data lives OUTSIDE the repo (same layout as csc415user accounts)
mkdir -p ~/data/robomimic/square

# bf16 tokens (~22.5 GB)
scp csc415user@dh2026pc03.utm.utoronto.ca:/virtual/csc415user/data/robomimic/square/ph_abs_v15_tokens_bf16_none.hdf5 \
    ~/data/robomimic/square/

# Unified HDF5 for eval (~1.7 GB)
scp csc415user@dh2026pc03.utm.utoronto.ca:/virtual/csc415user/data/robomimic/square/ph_abs_v15.hdf5 \
    ~/data/robomimic/square/

# Stage 1 checkpoint (~314 MB)
mkdir -p ~/RAEDiTRobotics/checkpoints/stage1_full_rtx5090_0312_0400
scp csc415user@dh2026pc01.utm.utoronto.ca:/virtual/csc415user/RAEDiTRobotics/checkpoints/stage1_full_rtx5090_0312_0400/epoch_024.pt \
    ~/RAEDiTRobotics/checkpoints/stage1_full_rtx5090_0312_0400/
```

### 3. Secrets file (one-time per account)

Create `configs/secrets.yaml` (git-ignored):

```yaml
huggingface_token: "hf_YOUR_TOKEN_HERE"
db_url: "postgresql://csc415user@142.1.46.5:5433/optuna_db"
ssh_password: "YOUR_LAB_PASSWORD"   # optional, only if not using SSH keys
```

### 4. SSH keys (recommended, one-time)

If your home directory is on NFS (shared across lab PCs), this gives passwordless auth everywhere:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

## Starting the PostgreSQL Database

The Optuna DB runs on `csc415user@dh2026pc02`. PostgreSQL is installed at `/virtual/csc415user/pgsql/`.

**Start the DB** (from any machine with the csc415user password):

```bash
ssh csc415user@dh2026pc02.utm.utoronto.ca \
    "/virtual/csc415user/pgsql/bin/pg_ctl -D ~/optuna_pg_data -l ~/optuna_pg_data/pg.log start"
```

**Check status:**

```bash
psql -h 142.1.46.5 -p 5433 -U csc415user -d optuna_db -c "SELECT 1;"
```

**Stop the DB** (when done with all sweeps):

```bash
ssh csc415user@dh2026pc02.utm.utoronto.ca \
    "/virtual/csc415user/pgsql/bin/pg_ctl -D ~/optuna_pg_data stop"
```

## Running the Sweep

### Option A: Swarm Manager (launch on many PCs at once)

From any machine with SSH access to the worker nodes:

```bash
cd ~/RAEDiTRobotics
source ~/venv_rlbench/bin/activate

# Launch workers on all nodes listed in configs/swarm_stage3_config.yaml
python training/swarm_manager_stage3.py start

# Check which nodes are running
python training/swarm_manager_stage3.py status

# Restart crashed/idle nodes
python training/swarm_manager_stage3.py restart_idle

# Stop all workers
python training/swarm_manager_stage3.py stop
```

Edit `configs/swarm_stage3_config.yaml` to choose which PCs to use (uncomment nodes).
Set `ssh_user` in the config or `ssh_password` in secrets.yaml for authentication.

### Option B: Single Worker (manual, for testing)

```bash
cd ~/RAEDiTRobotics
source ~/venv_rlbench/bin/activate
python training/swarm_worker_v3.py
```

This runs `n_trials_per_worker` trials (default: 5) sequentially on the current machine.

## Monitoring

### Optuna Dashboard

**Start the dashboard** on pc02:

```bash
ssh csc415user@dh2026pc02.utm.utoronto.ca \
    "source ~/venv_rlbench/bin/activate && nohup optuna-dashboard postgresql://csc415user@142.1.46.5:5433/optuna_db --port 8080 > /tmp/optuna_dashboard.log 2>&1 &"
```

**SSH tunnel** from your laptop:

```bash
ssh -N -L 9090:localhost:8080 csc415user@dh2026pc02.utm.utoronto.ca
```

Open `http://localhost:9090` in your browser. If port 9090 is taken, try another (9091, etc.).

### Worker Logs

Logs are written to `/tmp/swarm_logs/` on each worker machine:

```bash
# Check a specific worker's log
ssh alinaqee@dh2026pc07.utm.utoronto.ca "tail -50 /tmp/swarm_logs/worker_dh2026pc07_stage3.log"

# Check multiple workers
for pc in 04 05 06 07 08; do
    echo "=== pc$pc ==="
    ssh alinaqee@dh2026pc${pc}.utm.utoronto.ca "tail -5 /tmp/swarm_logs/worker_dh2026pc${pc}_stage3.log" 2>/dev/null
done
```

### Trial Results

Completed trials are uploaded to HuggingFace: https://huggingface.co/swagman8008/RAE-DP-stage3-sweeps

## Search Space

The current sweep searches over 6 hyperparameters (defined in `configs/swarm_stage3_config.yaml`):

| Parameter | Range | Type | Rationale |
|-----------|-------|------|-----------|
| `lr` | [5e-5, 5e-4] | log-uniform | lr=2e-4 >> 1e-4 in our tests |
| `lr_adapter` | [1e-6, 1e-4] | log-uniform | adapter is pre-trained, may want lower lr |
| `d_model` | {256, 384, 512} | categorical | biggest architectural choice |
| `n_layers` | {6, 8, 10, 12} | discrete | denoiser depth |
| `n_cond_layers` | {2, 4, 6} | discrete | conditioning encoder depth |
| `p_drop_attn` | [0.0, 0.3] | continuous | regularization strength |

Fixed settings: spatial_pool_size=7, use_flow_matching=True, batch_size=64, 300 epochs (cosine LR), bf16 tokens, cache_in_ram.

## Storage Budget (65 GB per account)

| Item | Size |
|------|------|
| bf16 tokens | ~22.5 GB |
| Unified HDF5 | ~1.7 GB |
| venv | ~10 GB |
| Stage 1 checkpoint | ~0.3 GB |
| **Available for trials** | **~30 GB** |

Trial scratch goes to `/tmp` (local SSD, not counted). Checkpoints are uploaded to HuggingFace and deleted locally after each trial.

## Maintenance

### Clean slate (delete study and start over)

```bash
python -c "
import optuna, yaml
with open('configs/secrets.yaml') as f:
    db_url = yaml.safe_load(f)['db_url']
optuna.delete_study(study_name='v3_square_bf16_hp_sweep', storage=db_url)
print('Study deleted')
"
```

### Kill zombie processes on all nodes

```bash
for pc in 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20; do
    ssh alinaqee@dh2026pc${pc}.utm.utoronto.ca \
        "pkill -9 -u \$USER python; rm -rf /tmp/optuna_trials/* /tmp/swarm_logs/*" 2>/dev/null
    echo "Cleaned pc$pc"
done
```
