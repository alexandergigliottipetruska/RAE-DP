🐝 Robotics Stage 1: Distributed Swarm Trainer
This setup allows us to run Optuna Hyperparameter Sweeps across multiple UTM Lab PCs simultaneously. It bypasses the strict 10GB university disk quota by using /tmp for local high-speed caching and securely saves all successful results (weights + logs) directly to Hugging Face.

🏗️ Architecture
Primary Node (Head PC): Runs the PostgreSQL database and the Optuna Dashboard.

Worker Nodes (Lab PCs): Connect to the Primary Node, pull a trial, and train for 15 epochs.

Smart Pruning (DDP-Safe): If Optuna detects a trial is performing poorly, the lead GPU signals the swarm to abort early, saving massive amounts of compute time.

Fail-Safe Cleanup: Whether a trial succeeds, gets pruned, or crashes (OOM), the workers automatically wipe their /tmp scratch space to prevent quota violations. Only fully completed 15-epoch trials are uploaded to Hugging Face.

🛠️ Setup (One-Time)
1. Environment & Dependencies
Ensure the shared virtual environment is active and you have the required database/dashboard tools:

Bash
source ~/sweep_env/bin/activate
pip install optuna-dashboard psycopg2-binary
2. Secrets Configuration
Create configs/secrets.yaml (Do NOT git commit this file):

YAML
ssh_password: "YOUR_LAB_PASSWORD"
db_url: "postgresql://csc415user@142.1.46.5:5433/optuna_db"
hf_token: "your_huggingface_write_token"
(Note: The swarm manager uses the SSHPASS environment variable under the hood so your password never shows up in active process monitors).

🚀 How to Run
1. Start the Database (Head PC Only)
Bash
pg_ctl -D ~/optuna_pg_data -l ~/optuna_pg_data/pg.log start
2. Manage the Swarm (From any PC)
Use swarm_manager.py to orchestrate the lab machines via secure SSH:

Start the workers: python3 swarm_manager.py start

Check who is training: python3 swarm_manager.py status
(A healthy idle PC uses ~74 MiB of VRAM for the OS display. Active training will show 8,000+ MiB).

Graceful Stop: python3 swarm_manager.py stop



📊 Monitoring & The Dashboard
You can watch the hyperparameter learning curves and pruning events live.

1. Start the Dashboard (On the Head PC):

Bash
optuna-dashboard "postgresql://csc415user@142.1.46.5:5433/optuna_db"
Leave this terminal running.

2. Create the SSH Tunnel (On your Personal Laptop):
Open a new terminal on your personal computer and run:

Bash
ssh -N -L 8080:localhost:8080 csc415user@dh2026pc02.utm.utoronto.ca
3. View the UI:
Open your laptop's browser and navigate to http://localhost:8080.



🧹 Maintenance & Clean Slate Protocol
If you need to completely restart a sweep from Trial 0, or if the GPUs get stuck, follow these steps:

1. Kill PyTorch Zombies:
If swarm_manager.py stop leaves GPUs stuck with high VRAM usage, bring out the sledgehammer:

Bash
for node in dh2020pc10 dh2020pc11 dh2020pc13; do
    ssh $node.utm.utoronto.ca "pkill -9 -u \$USER python"
done
2. Delete the Old Study:
Run this to wipe the Optuna memory so the next run starts at 0:

Python
python3 -c "import optuna, yaml; db_url = yaml.safe_load(open('configs/secrets.yaml'))['db_url']; optuna.delete_study(study_name='robotics_stage1_phased_test', storage=db_url)"
3. Clear Leftover Temp Files:

Bash
for node in dh2020pc10 dh2020pc11 dh2020pc13; do
    ssh $node.utm.utoronto.ca "rm -rf /tmp/denassau_stage1/*"
done

4. for checking the logs:
for node in dh2020pc10 dh2020pc11 dh2020pc13; do
    echo "=== LOGS FOR $node ==="
    ssh -o ConnectTimeout=5 $node.utm.utoronto.ca "tail -n 50 /tmp/swarm_logs/worker_$node.log"
    echo ""
done
⚠️ Critical Rules for Experimenting
Don't touch the Shared Drive during training: Workers save intermediate checkpoints to /tmp. If you want to see logs, SSH into a specific PC and check /tmp/swarm_logs/ or check the central manager logs.

Batch Size Limits: Keep batch_size at 4-10. Any higher will cause a CUDA Out of Memory (OOM) error on these specific lab GPUs. If an OOM happens, the worker will safely catch it, wipe the drive, and report a failed trial.

Nightly Cleanup: Always run python3 swarm_manager.py stop before leaving the lab so we don't hog the GPUs indefinitely.

📂 File Structure
configs/swarm_config.yaml: Edit this to add/remove PC hostnames and define network domains.

configs/secrets.yaml: Private passwords and DB connections (Git-ignored).

training/swarm_worker.py: The Optuna objective function that defines hyperparameter ranges and handles the /tmp to Hugging Face pipeline.

training/train_stage1_hp_distributed.py: The core DDP PyTorch training loop equipped with trial.report() and trial.should_prune() logic.

swarm_manager.py: The orchestrator script used to launch, status-check, and stop the swarm.