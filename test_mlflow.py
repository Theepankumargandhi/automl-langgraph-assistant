# test_mlflow.py
import mlflow
import os

print("MLflow version:", mlflow.__version__)
print("Current directory:", os.getcwd())

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create experiment
exp_name = "test-experiment"
try:
    exp_id = mlflow.create_experiment(exp_name)
    print(f"Created experiment: {exp_name} with ID: {exp_id}")
except:
    exp = mlflow.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id
    print(f"Using existing experiment: {exp_name} with ID: {exp_id}")

# Start run
with mlflow.start_run(experiment_id=exp_id):
    mlflow.log_param("test_param", "test_value")
    mlflow.log_metric("test_metric", 0.95)
    print("Logged test data successfully")

print("Test completed - check MLflow UI")