# profile_agent.py
from langchain.tools import tool
import pandas as pd
from sklearn.utils.multiclass import type_of_target

@tool
def profile_dataset(df: pd.DataFrame, target_column: str) -> dict:
    """
    Build a robust profile of the dataset for downstream planning/codegen.

    Returns a dict with:
      - target (str)
      - shape (tuple)
      - dtypes (dict[str,str])
      - missing_values (dict[col,int])
      - schema:
          n_rows, n_cols, dtypes, missing_pct, numeric_cols, categorical_cols
      - target_type_detailed: 'binary_classification' | 'multi_class_classification'
                              | 'multilabel_classification' | 'regression'
      - target_type: 'classification' | 'regression'   (generic for selectors/prompts)
      - target_cardinality (int)
      - class_balance (dict[value, float]) when classification
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in DataFrame columns")

    # Base profile (ensure JSON-friendly types)
    missing_values = df.isnull().sum()
    try:
        missing_values = missing_values.astype(int)
    except Exception:
        # best effort; cast elementwise if needed
        missing_values = missing_values.apply(lambda x: int(x) if pd.notnull(x) else 0)

    profile = {
        "shape": (int(df.shape[0]), int(df.shape[1])),
        "missing_values": missing_values.to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target": target_column,
    }

    # Feature-only schema (exclude target)
    feat_df = df.drop(columns=[target_column], errors="ignore")
    numeric_cols = feat_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in feat_df.columns if c not in numeric_cols]

    schema = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "dtypes": profile["dtypes"],
        "missing_pct": df.isna().mean().round(4).to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    profile["schema"] = schema

    # Target analysis
    y = df[target_column]
    tgt_kind = type_of_target(y)

    if tgt_kind == "binary":
        detailed = "binary_classification"
    elif tgt_kind == "multiclass":
        detailed = "multi_class_classification"
    elif tgt_kind in ("multilabel-indicator", "multiclass-multioutput"):
        detailed = "multilabel_classification"
    else:
        # 'continuous', 'continuous-multioutput', 'unknown' → treat as regression by default
        detailed = "regression"

    profile["target_type_detailed"] = detailed
    profile["target_type"] = "classification" if "classification" in detailed else "regression"
    profile["target_cardinality"] = int(pd.Series(y).nunique())

    # Class balance (only for classification)
    if profile["target_type"] == "classification":
        try:
            # normalize=True returns floats; keep keys as-is
            profile["class_balance"] = pd.Series(y).value_counts(normalize=True).round(4).to_dict()
        except Exception:
            profile["class_balance"] = None
    else:
        profile["class_balance"] = None

    return profile
