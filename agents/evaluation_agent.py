# agents/evaluation_agent.py
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


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

    return metrics
