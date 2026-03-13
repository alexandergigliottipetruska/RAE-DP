import optuna
import yaml

with open("configs/secrets.yaml", 'r') as f:
    secrets = yaml.safe_load(f)

study = optuna.load_study(
    study_name="robotics_stage1_phased_can_meat_off_grill",
    storage=secrets["db_url"]
)

print("=== HYPERPARAMETERS IN DATABASE ===")
for trial in study.trials:
    if trial.state.name != "FAIL":
        print(f"Trial {trial.number}: {trial.params}")