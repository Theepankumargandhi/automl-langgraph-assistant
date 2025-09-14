# agents/evaluation_agent.py
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import json
import tempfile
import os


def evaluate_model(task_type: str, model, df: pd.DataFrame, target_column: str) -> Optional[Dict[str, Any]]:
    """
    Evaluate a fitted model on a fresh train/test split of df (no leakage).
    Returns a metrics dict. Includes a confusion matrix for classification.
    Returns None only if model/inputs are invalid or predict() fails.
    """
    # Basic guards
    if model is None or df is None or target_column not in df.columns:
        return None

    # Split features/target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Normalize / infer task type
    tt = (task_type or "").lower()
    if tt not in {"regression", "binary_classification", "multi_class_classification"}:
        if pd.api.types.is_numeric_dtype(y) and pd.Series(y).nunique() > 20:
            tt = "regression"
        else:
            tt = "binary_classification" if pd.Series(y).nunique() == 2 else "multi_class_classification"

    # Safer split (stratify for classification when possible)
    stratify = y if tt != "regression" and pd.Series(y).nunique() <= 20 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        # Retry without stratify (e.g., very small classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

    # Predict on hold-out
    try:
        y_pred = model.predict(X_test)
    except Exception:
        return None

    # Compute metrics
    metrics: Dict[str, Any] = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "task_type": tt,
        "split": {"test_size": 0.2, "stratified": bool(stratify is not None)},
    }

    if tt == "regression":
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["r2_score"] = float(r2_score(y_test, y_pred))
    elif tt == "binary_classification":
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_score"]   = float(f1_score(y_test, y_pred))
        # Confusion matrix (serialize to lists for Streamlit/JSON)
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["labels"] = np.unique(y_test).tolist()
    else:  # multi-class
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["labels"] = np.unique(y_test).tolist()

    # Log to MLflow if available (optional, non-breaking)
    try:
        _log_evaluation_to_mlflow(metrics, y_test, y_pred, tt)
    except Exception:
        pass  # Silently skip MLflow logging if it fails

    return metrics


def _log_evaluation_to_mlflow(metrics: Dict[str, Any], y_test, y_pred, task_type: str):
    """Log evaluation metrics and artifacts to MLflow"""
    try:
        import mlflow
        
        # Log numeric metrics
        numeric_metrics = ["accuracy", "f1_score", "f1_macro", "rmse", "mae", "r2_score"]
        for metric in numeric_metrics:
            if metric in metrics and metrics[metric] is not None:
                mlflow.log_metric(f"eval.{metric}", float(metrics[metric]))
        
        # Log dataset split information
        mlflow.log_metric("eval.n_train", metrics["n_train"])
        mlflow.log_metric("eval.n_test", metrics["n_test"])
        mlflow.log_param("eval.test_size", metrics["split"]["test_size"])
        mlflow.log_param("eval.stratified", metrics["split"]["stratified"])
        mlflow.log_param("eval.task_type", task_type)
        
        # Log confusion matrix for classification tasks
        if "confusion_matrix" in metrics and task_type != "regression":
            _log_confusion_matrix_artifact(metrics["confusion_matrix"], metrics.get("labels", []))
        
        # Log prediction distribution for regression
        if task_type == "regression":
            _log_regression_artifacts(y_test, y_pred)
            
        # Log classification report for classification tasks
        if task_type != "regression":
            _log_classification_artifacts(y_test, y_pred, metrics.get("labels", []))
            
    except ImportError:
        # MLflow not available, skip logging
        pass
    except Exception as e:
        print(f"MLflow evaluation logging error: {e}")


def _log_confusion_matrix_artifact(confusion_matrix: list, labels: list):
    """Log confusion matrix as JSON artifact"""
    try:
        import mlflow
        
        cm_data = {
            "confusion_matrix": confusion_matrix,
            "labels": labels,
            "matrix_shape": [len(confusion_matrix), len(confusion_matrix[0]) if confusion_matrix else 0]
        }
        
        # Save as temporary file and log as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cm_data, f, indent=2)
            temp_path = f.name
        
        mlflow.log_artifact(temp_path, "evaluation")
        os.unlink(temp_path)  # Clean up temp file
        
    except Exception as e:
        print(f"Error logging confusion matrix: {e}")


def _log_regression_artifacts(y_test, y_pred):
    """Log regression-specific artifacts"""
    try:
        import mlflow
        import matplotlib.pyplot as plt
        
        # Create residuals plot
        residuals = y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted')
        
        plt.tight_layout()
        
        # Save plot as artifact
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=300, bbox_inches='tight')
            temp_path = f.name
        plt.close()
        
        mlflow.log_artifact(temp_path, "evaluation")
        os.unlink(temp_path)
        
        # Log statistical metrics
        mlflow.log_metric("eval.residuals_mean", float(np.mean(residuals)))
        mlflow.log_metric("eval.residuals_std", float(np.std(residuals)))
        
    except Exception as e:
        print(f"Error logging regression artifacts: {e}")


def _log_classification_artifacts(y_test, y_pred, labels: list):
    """Log classification-specific artifacts"""
    try:
        import mlflow
        from sklearn.metrics import classification_report
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=labels if labels else None, output_dict=True)
        
        # Log per-class metrics
        for label, metrics in report.items():
            if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"eval.class_{label}.{metric_name}", float(value))
        
        # Log macro and weighted averages
        if 'macro avg' in report:
            for metric_name, value in report['macro avg'].items():
                mlflow.log_metric(f"eval.macro_avg.{metric_name}", float(value))
        
        if 'weighted avg' in report:
            for metric_name, value in report['weighted avg'].items():
                mlflow.log_metric(f"eval.weighted_avg.{metric_name}", float(value))
        
        # Save detailed classification report as artifact
        report_text = classification_report(y_test, y_pred, target_names=labels if labels else None)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(report_text)
            temp_path = f.name
        
        mlflow.log_artifact(temp_path, "evaluation")
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"Error logging classification artifacts: {e}")