import optuna
import yaml

# Load your DB URL
with open("configs/secrets.yaml", 'r') as f:
    secrets = yaml.safe_load(f)

study = optuna.load_study(
    study_name="robotics_stage1_phased_test", 
    storage=secrets["db_url"]
)

# Get all failed trials
failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

for t in failed_trials:
    print(f"Trial {t.number} failed.")
    # This prints the traceback recorded in the DB
    print(f"Error: {t.system_attrs.get('fail_reason', 'No reason recorded')}")
    print("-" * 30)
    