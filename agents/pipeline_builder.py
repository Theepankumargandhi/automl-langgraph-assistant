# agents/pipeline_builder.py
from typing import Dict, Tuple
import pandas as pd
import traceback
import re
import os

# ---- Module-global: store detected plot types by step index ----
plot_info: Dict[int, str] = {}


def build_pipeline(df: pd.DataFrame, profile: dict, code_map: Dict[str, str]) -> Tuple[pd.DataFrame, object]:
    """
    Executes LLM-generated code steps safely, captures any trained model,
    and guarantees a trained baseline if LLM code doesn't produce one.

    Returns
    -------
    (df_out, model)
      df_out : possibly transformed DataFrame (post-LLM code)
      model  : trained estimator or Pipeline
    """
    # Reset plot info each run (avoid stale state and NameError)
    global plot_info
    plot_info = {}

    # --- Lazy imports (keep module import time low, avoid global state issues) ---
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import (
        train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_validate
    )
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    # -------- Runtime toggles (SAFE defaults unless set via env) --------
    ALLOW_IO = os.getenv("ALLOW_IO", "0") == "1"          # allow pd.read_* / open()
    ALLOW_TUNING = os.getenv("ALLOW_TUNING", "0") == "1"  # allow Grid/Random/Optuna
    ALLOWED_DATA_DIR = os.getenv("ALLOWED_DATA_DIR", "")  # whitelist directory for file reads

    # ---- Resolve target column from profile ----
    target_column = profile.get("target")
    if not target_column:
        raise ValueError("Profile does not include 'target'. Ensure your profiler sets profile['target'].")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    # ---- Working copy ----
    working_df = df.copy()

    # -------- Safe readers (used only when ALLOW_IO=1) --------
    def _is_allowed_path(p: str) -> bool:
        if not ALLOWED_DATA_DIR:
            return False
        ap = os.path.abspath(p)
        base = os.path.abspath(ALLOWED_DATA_DIR)
        return ap.startswith(base + os.sep) or ap == base

    def safe_read_csv(path, *args, **kwargs):
        p = str(path)
        if p.startswith(("http://", "https://")) or not _is_allowed_path(p):
            print(f" blocked read_csv('{p}') — using in-memory df instead")
            return working_df.copy()
        try:
            import pandas as _pd
            return _pd.read_csv(path, *args, **kwargs)
        except Exception as e:
            print(f" read_csv failed ({e}); falling back to in-memory df")
            return working_df.copy()

    def safe_read_excel(path, *args, **kwargs):
        p = str(path)
        if p.startswith(("http://", "https://")) or not _is_allowed_path(p):
            print(f" blocked read_excel('{p}') — using in-memory df instead")
            return working_df.copy()
        try:
            import pandas as _pd
            return _pd.read_excel(path, *args, **kwargs)
        except Exception as e:
            print(f" read_excel failed ({e}); falling back to in-memory df")
            return working_df.copy()

    # ---- Exec environment ----
    local_vars = {
        "df": working_df,
        "data": None,   # alias some LLM code expects
        "pd": pd, "np": np, "plt": plt,

        # sklearn
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "classification_report": classification_report,
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "SVC": SVC,
        "LabelEncoder": LabelEncoder,
        "StandardScaler": StandardScaler,
        "ColumnTransformer": ColumnTransformer,
        "Pipeline": Pipeline,
        "SimpleImputer": SimpleImputer,
        "OneHotEncoder": OneHotEncoder,
        "GridSearchCV": GridSearchCV,

        # toggles & helpers
        "safe_read_csv": safe_read_csv,
        "safe_read_excel": safe_read_excel,

        # model placeholders
        "model": None, "best_model": None, "classifier": None, "clf": None,
        "pipeline": None, "estimator": None,

        # common vars
        "target_column": target_column,
    }
    local_vars["data"] = working_df.copy()

    # Make seaborn available if the LLM uses sns.* (no-op if not installed)
    try:
        import seaborn as sns  # type: ignore
        local_vars["sns"] = sns
    except Exception:
        pass

    model = None
    error_log = []

    # -------- Dynamic skip list --------
    skip_patterns = [r"\bdeploy\b", r"\bflask\b", r"\bfastapi\b", r"\bstreamlit\b", r"\bmlflow\b"]
    if not ALLOW_TUNING:
        skip_patterns += [r"\bgrid\s*search", r"\bgridsearch", r"\brandomized\s*search", r"\boptuna\b", r"\bhyperparam"]
    if not ALLOW_IO:
        skip_patterns += [
            r"\bread_csv\s*\(", r"\bread_excel\s*\(", r"\bopen\s*\(",
            r"\b(joblib|pickle)\s*\.(dump|load)", r"\brequests?\.", r"\burllib\."
        ]
    def should_skip(step_text: str, code_text: str) -> bool:
        t = f"{step_text}\n{code_text}".lower()
        return any(re.search(p, t) for p in skip_patterns)

    # ---- Sanitizer & patcher ----
    def _sanitize_and_patch(step: str, code: str, idx: int) -> str:
        c = code.strip()

        # strip fences
        if c.startswith("```python") and c.endswith("```"):
            c = c[len("```python"):-len("```")].strip()
        elif c.startswith("```") and c.endswith("```"):
            c = c[3:-3].strip()

        # File reads: safe vs permissive
        if not ALLOW_IO:
            c = re.sub(r"\bpd\s*\.\s*read_csv\s*\([^)]*\)", "df.copy()", c)
            c = re.sub(r"\bpd\s*\.\s*read_excel\s*\([^)]*\)", "df.copy()", c)
            c = re.sub(r"\bpd\s*\.\s*read_(json|parquet|table|html|pickle|feather|orc|sas|stata|hdf|gbq)\s*\([^)]*\)", "df.copy()", c)
        else:
            c = re.sub(r"\bpd\s*\.\s*read_csv\s*\(", "safe_read_csv(", c)
            c = re.sub(r"\bpd\s*\.\s*read_excel\s*\(", "safe_read_excel(", c)

        # Save plots to files (avoid plt.show())
        if "plt.show()" in c:
            plot_filename = f"step_{idx}_plot.png"
            c = c.replace("plt.show()", f'plt.savefig("{plot_filename}"); plt.close()')
            print(f"🖼️ Patched plt.show() → saved '{plot_filename}'")

            # 🔎 Detect common plot types and record for this step
            plot_type = None
            # Matplotlib
            if "plt.bar" in c or ".plot(kind='bar'" in c or '.plot(kind="bar"' in c:
                plot_type = "Bar Chart"
            elif "plt.hist" in c or ".plot(kind='hist'" in c or '.plot(kind="hist"' in c:
                plot_type = "Histogram"
            elif "plt.scatter" in c or ".plot(kind='scatter'" in c or '.plot(kind="scatter"' in c:
                plot_type = "Scatter Plot"
            elif "plt.boxplot" in c:
                plot_type = "Box Plot"
            elif "plt.plot" in c:
                plot_type = "Line Plot"
            # Seaborn
            elif "sns.heatmap" in c:
                plot_type = "Heatmap"
            elif "sns.pairplot" in c:
                plot_type = "Pair Plot"
            elif "sns.boxplot" in c:
                plot_type = "Box Plot"

            if plot_type:
                plot_info[idx] = plot_type
                print(f"🖼️ Detected visualization type for step {idx}: {plot_type}")

        # Stable LR default
        c = c.replace("LogisticRegression()", "LogisticRegression(max_iter=200)")

        # IQR patches (numeric-only)
        if "quantile" in c and "IQR" in c:
            c = re.sub(r"df\.quantile\(.*?\)", "df.select_dtypes(include=[np.number]).quantile(0.25)", c)
            c = c.replace(
                "df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]",
                """
num_df = df.select_dtypes(include=[np.number])
Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)]
"""
            )

        # Numeric-only EDA coercions
        c = re.sub(r"\b([A-Za-z_]\w*)\s*\.\s*corr\s*\(\s*\)", r"\1.select_dtypes(include=[np.number]).corr()", c)
        c = re.sub(r"\b([A-Za-z_]\w*)\s*\.\s*corr\s*\(\s*[^)]*\)", r"\1.select_dtypes(include=[np.number]).corr()", c)
        c = re.sub(r"sns\s*\.\s*heatmap\s*\(\s*([A-Za-z_]\w*)\s*\.\s*corr\s*\(\s*\)\s*\)", r"sns.heatmap(\1.select_dtypes(include=[np.number]).corr())", c)
        c = re.sub(r"sns\s*\.\s*pairplot\s*\(\s*([A-Za-z_]\w*)\s*\)", r"sns.pairplot(\1.select_dtypes(include=[np.number]))", c)
        c = re.sub(r"\b([A-Za-z_]\w*)\s*\.\s*describe\s*\(\s*\)", r"\1.select_dtypes(include=[np.number]).describe()", c)

        # Don't scale target
        if "StandardScaler" in c and "fit_transform" in c:
            c = re.sub(rf"df\[['\"]{re.escape(target_column)}['\"]\]\s*=\s*scaler\.fit_transform\(.*?\)",
                       f"# do not scale target '{target_column}'", c, flags=re.DOTALL)

        # 2D->1D for single-col fit_transform
        c = re.sub(r"(fit_transform\(\s*[A-Za-z_]\w*\s*\[\s*\[\s*['\"][^'\"]+['\"]\s*\]\s*\]\s*\))", r"\1.ravel()", c)

        # Ensure categorical_cols exists when LabelEncoder is used without selection
        if ("LabelEncoder" in c) and ("categorical_cols" in c) and ("select_dtypes" not in c):
            prefix = "categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()\n"
            c = prefix + c

        # ---- Auto-wrap direct estimator training in a preprocessing Pipeline ----
        def _wrap_fit_with_pipeline(text: str) -> str:
            estros = r"(LogisticRegression|RandomForestClassifier|SVC)"
            pattern_train = rf"(?s)\b([A-Za-z_]\w*)\s*=\s*{estros}\s*\((.*?)\)\s*?\n\s*\1\s*\.\s*fit\s*\(\s*X_train\s*,\s*y_train\s*\)"
            repl_train = (
                "num_cols = X_train.select_dtypes(include=['number','bool']).columns.tolist()\n"
                "cat_cols = [c for c in X_train.columns if c not in num_cols]\n"
                "pre = ColumnTransformer(\n"
                "    transformers=[\n"
                "        ('num', SimpleImputer(strategy='median'), num_cols),\n"
                "        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),\n"
                "                                ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),\n"
                "    ]\n"
                ")\n"
                r"\1 = Pipeline(steps=[('pre', pre), ('classifier', \2(\3))])\n"
                r"\1.fit(X_train, y_train)"
            )
            pattern_full = rf"(?s)\b([A-Za-z_]\w*)\s*=\s*{estros}\s*\((.*?)\)\s*?\n\s*\1\s*\.\s*fit\s*\(\s*X\s*,\s*y\s*\)"
            repl_full = (
                "num_cols = X.select_dtypes(include=['number','bool']).columns.tolist()\n"
                "cat_cols = [c for c in X.columns if c not in num_cols]\n"
                "pre = ColumnTransformer(\n"
                "    transformers=[\n"
                "        ('num', SimpleImputer(strategy='median'), num_cols),\n"
                "        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),\n"
                "                                ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),\n"
                "    ]\n"
                ")\n"
                r"\1 = Pipeline(steps=[('pre', pre), ('classifier', \2(\3))])\n"
                r"\1.fit(X, y)"
            )
            text = re.sub(pattern_train, repl_train, text)
            text = re.sub(pattern_full, repl_full, text)
            return text
        c = _wrap_fit_with_pipeline(c)

        # Alias common trained names to model (extended with more variants)
        c = re.sub(r"(?m)^(?P<var>rf_model|lr_model|svm_model|classifier|clf|pipeline|estimator)\s*=\s*.+$",
                   r"\g<0>\nmodel = \g<var>", c)
        c = re.sub(r"(?m)^(?P<var>final_model|trained_model|best_estimator|best_pipeline)\s*=\s*.+$",
                   r"\g<0>\nmodel = \g<var>", c)

        # GridSearchCV fixer (prefix classifier__*, add n_jobs, set model=best_estimator_)
        def _prefix_paramgrid_for_pipeline(txt: str) -> str:
            def _inject_njobs(m):
                inner = m.group(1)
                if "n_jobs" not in inner:
                    inner = inner.rstrip()
                    if inner and not inner.endswith(","): inner += ", "
                    inner += "n_jobs=-1"
                return f"GridSearchCV({inner})"
            txt = re.sub(r"GridSearchCV\(([^)]*)\)", _inject_njobs, txt)

            def _prefix_keys_in_dict(d: str) -> str:
                known = ["n_estimators","max_depth","min_samples_split","min_samples_leaf",
                         "criterion","class_weight","C","gamma","kernel","degree",
                         "max_iter","solver","penalty"]
                for k in known:
                    d = re.sub(rf"(['\"])({k})(\1)\s*:", rf"\1classifier__\2\1:", d)
                return d

            def _prefix_param_grid_block(m):
                prefix, body = m.group(1), m.group(2)
                def _fix_each_dict(md): return "{ " + _prefix_keys_in_dict(md.group(1)) + " }"
                fixed = re.sub(r"\{\s*(.*?)\s*\}", _fix_each_dict, body, flags=re.DOTALL)
                return f"{prefix}{fixed}"

            txt = re.sub(r"(param_grid\s*=\s*)(\{.*?\}|\[.*?\])", _prefix_param_grid_block, txt, flags=re.DOTALL)
            txt = re.sub(r"(?m)^(?P<g>(grid|search|gs|gscv))\s*\.\s*fit\s*\(\s*X_(?:train|TRAIN|Train)\s*,\s*y_(?:train|TRAIN|Train)\s*\)\s*$",
                         r"\g<0>\nmodel = \g<g>.best_estimator_", txt)
            txt = re.sub(r"(?m)^(?P<g>(grid|search|gs|gscv))\s*\.\s*fit\s*\(\s*X\s*,\s*y\s*\)\s*$",
                         r"\g<0>\nmodel = \g<g>.best_estimator_", txt)
            return txt
        if "GridSearchCV" in c or "gridsearchcv" in c.lower():
            c = _prefix_paramgrid_for_pipeline(c)

        # Neutralize persistence / external load & open()
        c = re.sub(r"\b(joblib|pickle)\s*\.\s*dump\s*\([^)]*\)\s*;?", "# skipped model persistence (offline run)", c)
        c = re.sub(r"(?m)^(?P<lhs>[A-Za-z_]\w*)\s*=\s*(?:joblib|pickle)\s*\.\s*load\s*\([^)]*\)\s*$",
                   r"\g<lhs> = model  # skipped external load; keep current in-memory model", c)
        c = re.sub(r"\b(?:joblib|pickle)\s*\.\s*load\s*\([^)]*\)", "model  # skipped external load", c)
        if not ALLOW_IO:
            c = re.sub(r"\bopen\s*\([^)]*\)", "None  # skipped file I/O", c)

        # --- Prevent toy-DataFrame overwrites of the real df ---
        c = re.sub(r"\bdf\s*=\s*pd\s*\.\s*DataFrame\s*\(", "toy_df = pd.DataFrame(", c)

        # Promote df_encoded -> df so preprocessing sticks
        if re.search(r"\bdf_encoded\s*=", c) and "df =" not in c:
            c += "\n# promote encoded frame back to df\ndf = df_encoded\n"

        return c

    # ---- Execute steps ----
    for idx, (step, raw_code) in enumerate(code_map.items(), start=1):
        if should_skip(step, raw_code):
            print(f"⏭️ Skipping step {idx}: {step}")
            continue

        print(f"\n➡️ Executing step {idx}: {step}\n")
        try:
            code = _sanitize_and_patch(step, raw_code, idx)
            print(f"🔧 Code to execute:\n{code}\n" + "-" * 60)

            # --- Preflight on a small sample (with one-shot self-repair) ---
            repaired_once = False
            while True:
                try:
                    sample = working_df.head(min(100, len(working_df))).copy()
                    prelocal = dict(local_vars)
                    prelocal["df"] = sample
                    prelocal["data"] = sample.copy()
                    compile(code, f"<preflight_step_{idx}>", "exec")
                    exec(code, {"__builtins__": __builtins__}, prelocal)
                    break  # preflight ok
                except Exception as pf_err:
                    msg = str(pf_err)
                    print(f" Preflight warning (step {idx}): {type(pf_err).__name__}: {pf_err}")
                    if repaired_once:
                        break

                    # ---- One-shot auto-repair (no extra LLM call) ----
                    patched = code
                    did_patch = False

                    # 1) NameError: sns -> inject seaborn import
                    if "name 'sns' is not defined" in msg.lower() and "import seaborn as sns" not in patched:
                        patched = "import seaborn as sns\n" + patched
                        did_patch = True
                        print(" Self-repair: added `import seaborn as sns`.")

                    # 2) ValueError around stratify (insufficient class counts)
                    if "stratify" in msg.lower():
                        patched2 = re.sub(
                            r"(train_test_split\s*\([^)]*?)stratify\s*=\s*[^,\)]+",
                            r"\1stratify=None",
                            patched,
                            flags=re.DOTALL,
                        )
                        if patched2 != patched:
                            patched = patched2
                            did_patch = True
                            print("🛠️ Self-repair: replaced `stratify=...` with `stratify=None` in train_test_split().")

                    # 3) Ensure X/y are set if code forgot them
                    if "name 'y' is not defined" in msg.lower() or "y is not defined" in msg.lower():
                        header = "X = df.drop(columns=[target_column]); y = df[target_column]\n"
                        if "X =" not in patched or " y =" not in patched:
                            patched = header + patched
                            did_patch = True
                            print("🛠️ Self-repair: injected `X,y` definition from df/target_column.")

                    if did_patch:
                        code = patched
                        repaired_once = True
                        continue  # retry preflight once with the patched code
                    else:
                        break  # nothing to repair

            # --- Full execution ---
            exec(code, {"__builtins__": __builtins__}, local_vars)
            print("✅ Step succeeded")

            # capture a trained model if present
            for candidate in (
                "best_model","model","classifier","clf","pipeline","estimator",
                "rf_model","lr_model","svm_model","final_model","trained_model",
                "best_estimator","best_pipeline"
            ):
                if local_vars.get(candidate) is not None:
                    model = local_vars[candidate]
                    # mark as AI-generated
                    try:
                        setattr(model, "_origin", "ai")
                    except Exception:
                        pass
                    break

        except Exception as e:
            msg = f"Step '{step}' failed: {type(e).__name__}: {e}"
            print(f"❌ {msg}")
            traceback.print_exc()
            error_log.append(msg)

    # ---- Finalize df ----
    df_out = local_vars.get("df", working_df)

    # --- Guard against tiny/toy frames or invalid targets ---
    try:
        n_rows = len(df_out)
        n_classes = df_out[target_column].nunique()
    except Exception:
        # If df_out got corrupted, revert to the original
        n_rows, n_classes = len(working_df), working_df[target_column].nunique()
        df_out = working_df
        local_vars["df"] = df_out

    # If a step accidentally shrank df to a toy sample, revert to the original dataset
    if (n_rows < 50) or (n_classes < 2):
        print("⚠️ Detected toy/invalid DataFrame from an LLM step; reverting to original dataset.")
        df_out = working_df
        local_vars["df"] = df_out

    # ---- Baseline fallback if needed ----
    needs_fallback = model is None
    if not needs_fallback:
        try:
            _ = model.predict(df_out.drop(columns=[target_column]).head(1))
        except Exception:
            needs_fallback = True

    if needs_fallback:
        print("ℹ️ No usable trained model from generated code; running 2-model BaselineSelector...")
        X = df_out.drop(columns=[target_column])
        y = df_out[target_column]

        # ---- Shared preprocessing for candidates ----
        num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        def _make_pre():
            return ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="median"), num_cols),
                    ("cat", Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore"))
                    ]), cat_cols),
                ]
            )

        def _detect_problem_type(y_series: pd.Series, profile: dict) -> str:
            t = (profile or {}).get("target_type", "").lower()
            if t in {"classification", "regression"}:
                return t
            if pd.api.types.is_numeric_dtype(y_series):
                # numeric: treat low-cardinality numeric as classification
                return "classification" if y_series.nunique() <= max(20, int(0.05 * len(y_series))) else "regression"
            return "classification"

        problem = _detect_problem_type(y, profile)

        # ---- Candidates ----
        if problem == "classification":
            candidates = {
                "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42),
            }
            # CV setup (respect smallest class count)
            cv_folds = 5 if len(y) >= 200 else 3
            min_class = y.value_counts().min() if y.nunique() > 1 else 1
            cv_folds = max(2, min(cv_folds, int(min_class)))
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = {"accuracy": "accuracy", "f1": "f1_weighted"}
            primary = "accuracy"
        else:
            candidates = {
                "Ridge": Ridge(alpha=1.0),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42),
            }
            cv_folds = 5 if len(y) >= 200 else 3
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}
            primary = "rmse"

        # ---- CV for each candidate ----
        results = {}
        for name, est in candidates.items():
            pipe = Pipeline(steps=[("pre", _make_pre()), ("clf", est)])
            try:
                cvres = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
                fit_time = float(cvres["fit_time"].sum())
                if problem == "classification":
                    acc = float(cvres["test_accuracy"].mean())
                    f1w = float(cvres["test_f1"].mean())
                    results[name] = {"accuracy": acc, "f1": f1w, "fit_time_s": fit_time}
                    print(f"  • {name}: acc={acc:.4f}, f1={f1w:.4f}, fit_time={fit_time:.2f}s")
                else:
                    rmse = float((-cvres["test_rmse"]).mean())  # convert from negative to positive
                    r2 = float(cvres["test_r2"].mean())
                    results[name] = {"rmse": rmse, "r2": r2, "fit_time_s": fit_time}
                    print(f"  • {name}: rmse={rmse:.4f}, r2={r2:.4f}, fit_time={fit_time:.2f}s")
            except Exception as e:
                print(f"  • {name}: CV failed ({type(e).__name__}: {e})")
                # Graceful degradation: simple holdout
                try:
                    strat = y if (problem == "classification" and y.nunique() > 1) else None
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
                    pipe.fit(Xtr, ytr)
                    if problem == "classification":
                        from sklearn.metrics import accuracy_score, f1_score
                        preds = pipe.predict(Xte)
                        acc = float(accuracy_score(yte, preds))
                        f1w = float(f1_score(yte, preds, average="weighted"))
                        results[name] = {"accuracy": acc, "f1": f1w, "fit_time_s": 0.0}
                    else:
                        from sklearn.metrics import mean_squared_error, r2_score
                        preds = pipe.predict(Xte)
                        rmse = float(mean_squared_error(yte, preds, squared=False))
                        r2 = float(r2_score(yte, preds))
                        results[name] = {"rmse": rmse, "r2": r2, "fit_time_s": 0.0}
                except Exception as e2:
                    print(f"    ↳ holdout also failed ({type(e2).__name__}: {e2})")

        # ---- Winner selection (primary → secondary → fit_time) ----
        def _winner_name(res: dict) -> str:
            if not res:
                return "RandomForestClassifier" if problem == "classification" else "RandomForestRegressor"
            if problem == "classification":
                # higher accuracy, then higher f1, then lower time
                items = []
                for n, m in res.items():
                    items.append((-m.get("accuracy", -1e9), -m.get("f1", -1e9), m.get("fit_time_s", 1e9), n))
                return sorted(items)[0][-1]
            else:
                # lower rmse, then higher r2, then lower time
                items = []
                for n, m in res.items():
                    items.append((m.get("rmse", 1e9), -m.get("r2", -1e9), m.get("fit_time_s", 1e9), n))
                return sorted(items)[0][-1]

        winner_name = _winner_name(results)
        print(f"🏆 BaselineSelector winner: {winner_name}")

        # ---- Fit winner on final split (keep your original split behavior) ----
        def _safe_split(X, Y, test_size, stratify):
            try:
                return train_test_split(X, Y, test_size=test_size, random_state=42, stratify=stratify)
            except ValueError:
                return train_test_split(X, Y, test_size=test_size, random_state=42, stratify=None)

        stratify = y if (problem == "classification" and y.nunique() <= 20 and y.value_counts().min() >= 2) else None
        X_train, X_test, y_train, y_test = _safe_split(X, y, 0.2, stratify)

        winner_est = (
            LogisticRegression(max_iter=1000, class_weight="balanced")
            if winner_name == "LogisticRegression" else
            RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
            if winner_name == "RandomForestClassifier" else
            Ridge(alpha=1.0)
            if winner_name == "Ridge" else
            RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
        )
        model = Pipeline(steps=[("pre", _make_pre()), ("clf", winner_est)])
        model.fit(X_train, y_train)
        local_vars["X_test"], local_vars["y_test"] = X_test, y_test
        try:
            local_vars["y_pred"] = model.predict(X_test)
        except Exception:
            pass

        # Attach summary (non-breaking)
        try:
            setattr(model, "_origin", "baseline")
            setattr(model, "_baseline_selector", {
                "problem": problem,
                "primary_metric": "accuracy" if problem == "classification" else "rmse",
                "winner": winner_name,
                "cv_folds": cv_folds,
                "results": results,
            })
        except Exception:
            pass

        print("✅ BaselineSelector trained.")

    # ---- Summary of any issues ----
    if error_log:
        print("\n--- Skipped/Errored Steps (summary) ---")
        for m in error_log:
            print("•", m)

    # Attach plot_info to model so Streamlit can show names without changing return signature
    try:
        if model is not None and hasattr(model, "__dict__"):
            setattr(model, "_plot_info", dict(plot_info))
    except Exception:
        pass

    return df_out, model
