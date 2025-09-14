# mlflow_config.py
import os
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

def initialize_mlflow():
    """Initialize MLflow tracking and create necessary directories."""
    # Set tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create artifact directory if it doesn't exist
    artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "./mlflow-artifacts")
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    
    # Set experiment
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "automl-assistant-experiments")
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_root
        )
    except mlflow.exceptions.MlflowException:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def start_automl_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Start an MLflow run for AutoML pipeline."""
    default_tags = {
        "mlflow.runName": run_name or f"automl-run-{mlflow.utils.time.get_current_time_millis()}",
        "pipeline.type": "automl",
        "pipeline.version": "1.0"
    }
    
    if tags:
        default_tags.update(tags)
    
    run = mlflow.start_run(tags=default_tags)
    return run

def log_dataset_profile(profile: Dict[str, Any]):
    """Log dataset profiling information."""
    # Log dataset metrics
    schema = profile.get("schema", {})
    if schema.get("n_rows"):
        mlflow.log_metric("dataset.n_rows", schema["n_rows"])
    if schema.get("n_cols"):
        mlflow.log_metric("dataset.n_cols", schema["n_cols"])
    
    # Log target information
    mlflow.log_param("target.column", profile.get("target"))
    mlflow.log_param("target.type", profile.get("target_type"))
    if profile.get("target_cardinality"):
        mlflow.log_metric("target.cardinality", profile["target_cardinality"])
    
    # Log feature counts
    numeric_cols = schema.get("numeric_cols", [])
    categorical_cols = schema.get("categorical_cols", [])
    mlflow.log_metric("features.numeric_count", len(numeric_cols))
    mlflow.log_metric("features.categorical_count", len(categorical_cols))
    
    # Log missing value percentage
    missing_pct = schema.get("missing_pct", {})
    if missing_pct:
        avg_missing = sum(missing_pct.values()) / len(missing_pct)
        mlflow.log_metric("data_quality.avg_missing_pct", avg_missing)

def log_pipeline_steps(steps: list):
    """Log the generated pipeline steps."""
    mlflow.log_param("pipeline.steps_count", len(steps))
    
    # Log steps as a text artifact
    steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
    mlflow.log_text(steps_text, "pipeline_steps.txt")

def log_model_with_registry(model: Any, model_name: str, origin: str = "unknown"):
    """Log model and register it in MLflow model registry."""
    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=f"automl-{model_name.lower()}",
        signature=None  # Will be inferred
    )
    
    # Log model metadata
    mlflow.log_param("model.origin", origin)
    mlflow.log_param("model.algorithm", model_name)
    
    # Log baseline selector info if available
    baseline_info = getattr(model, "_baseline_selector", None)
    if baseline_info:
        mlflow.log_param("baseline.problem_type", baseline_info.get("problem"))
        mlflow.log_param("baseline.primary_metric", baseline_info.get("primary_metric"))
        mlflow.log_param("baseline.cv_folds", baseline_info.get("cv_folds"))
        mlflow.log_param("baseline.winner", baseline_info.get("winner"))
        
        # Log CV results
        results = baseline_info.get("results", {})
        for model_type, metrics in results.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"cv.{model_type}.{metric_name}", value)
    
    return model_info

def log_evaluation_metrics(metrics: Dict[str, Any]):
    """Log evaluation metrics from the evaluation agent."""
    # Log numeric metrics
    numeric_metrics = ["accuracy", "f1_score", "f1_macro", "rmse", "mae", "r2_score"]
    for metric in numeric_metrics:
        if metric in metrics and metrics[metric] is not None:
            mlflow.log_metric(f"eval.{metric}", float(metrics[metric]))
    
    # Log split information
    if "n_train" in metrics:
        mlflow.log_metric("eval.n_train", metrics["n_train"])
    if "n_test" in metrics:
        mlflow.log_metric("eval.n_test", metrics["n_test"])
    
    # Log confusion matrix if available
    if "confusion_matrix" in metrics:
        import json
        cm_data = {
            "confusion_matrix": metrics["confusion_matrix"],
            "labels": metrics.get("labels", [])
        }
        mlflow.log_text(json.dumps(cm_data, indent=2), "confusion_matrix.json")

def log_cost_information(cost_tracker):
    """Log API cost information."""
    current_cost = cost_tracker.get_current_run_total()
    breakdown = cost_tracker.get_breakdown()
    
    mlflow.log_metric("cost.total_usd", current_cost)
    
    # Log cost breakdown
    for operation, cost in breakdown.items():
        mlflow.log_metric(f"cost.{operation}_usd", cost)

def log_artifacts_from_paths(artifact_paths: list, artifact_type: str = "plot"):
    """Log files as MLflow artifacts."""
    for path in artifact_paths:
        if os.path.exists(path):
            mlflow.log_artifact(path, f"{artifact_type}s")

def get_experiment_info():
    """Get current experiment information for UI display."""
    try:
        experiment = mlflow.get_experiment_by_name(
            os.getenv("MLFLOW_EXPERIMENT_NAME", "automl-assistant-experiments")
        )
        return {
            "experiment_id": experiment.experiment_id,
            "experiment_name": experiment.name,
            "artifact_location": experiment.artifact_location
        }
    except Exception:
        return None

def get_mlflow_ui_url():
    """Get MLflow UI URL for navigation."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    ui_port = os.getenv("MLFLOW_UI_PORT", "5000")
    ui_host = os.getenv("MLFLOW_UI_HOST", "localhost")
    
    if tracking_uri.startswith("sqlite"):
        return f"http://{ui_host}:{ui_port}"
    else:
        return tracking_uri
    
def safe_log_param(key, value):
    """Safely log parameter, avoiding duplicates"""
    try:
        mlflow.log_param(key, value)
    except mlflow.exceptions.MlflowException as e:
        if "already logged" in str(e):
            pass  # Skip duplicate parameters
        else:
            raise

def cleanup_old_artifacts(max_artifacts: int = 50):
    """Clean up old artifact files to prevent disk space issues."""
    artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "./mlflow-artifacts")
    artifact_path = Path(artifact_root)
    
    if artifact_path.exists():
        # Get all run directories sorted by modification time
        run_dirs = [d for d in artifact_path.iterdir() if d.is_dir()]
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent runs
        for old_dir in run_dirs[max_artifacts:]:
            try:
                import shutil
                shutil.rmtree(old_dir)
            except Exception:
                pass  # Ignore cleanup errors