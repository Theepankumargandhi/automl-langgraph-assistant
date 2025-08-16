# agents/summary_agent.py
from __future__ import annotations

import os
import re
import base64
import mimetypes
from typing import Dict, Any, List

# -------- helpers to clean LLM code blocks --------

_BULLET_LINE = re.compile(r'^\s*[-•]\s+')

def _clean_code(snippet: str) -> str:
    """Normalize LLM code for markdown:
    - remove stray ``` fences
    - un-bullet accidental lines ("- import ...")
    - normalize newlines / trim
    """
    if not isinstance(snippet, str):
        return str(snippet)

    s = snippet.replace("\r\n", "\n").replace("\r", "\n").strip()
    # strip triple backtick fences if present
    s = re.sub(r"^```(?:python)?\n?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n```$", "", s)

    lines: List[str] = []
    for ln in s.split("\n"):
        ln = _BULLET_LINE.sub("", ln.rstrip())
        lines.append(ln)
    s = "\n".join(lines).strip()

    # final trailing newline helps code fences render nicely
    if not s.endswith("\n"):
        s += "\n"
    return s

# -------- helpers for plots (optional inline embedding) --------

def _render_plots_inline_html(plots: List[Dict[str, Any]]) -> str:
    """
    Embed plots inline using <img src="data:...;base64,...">.
    This renders in most markdown viewers if HTML is allowed.
    """
    if not plots:
        return "## Generated Visualizations\n\n_No plots produced._\n"

    out: List[str] = ["## Generated Visualizations\n"]
    for p in plots:
        label = p.get("label") or (f"Step {p.get('step')}" if p.get("step") is not None else "Visualization")
        path = p.get("path")
        if path and os.path.exists(path):
            mime, _ = mimetypes.guess_type(path)
            mime = mime or "image/png"
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                out.append(f"**{label}**  ")
                out.append(
                    f'<img src="data:{mime};base64,{b64}" alt="{label}" '
                    'style="max-width:720px;border:1px solid #333;border-radius:8px;margin:8px 0;">\n'
                )
            except Exception:
                out.append(f"- {label} — `{os.path.basename(path)}`")
        else:
            fname = os.path.basename(path) if path else ""
            out.append(f"- {label} — `{fname}`")
    return "\n".join(out) + "\n"

# -------- helper for model algo name --------

def _algo_name(model: Any) -> str:
    """Try to determine the final estimator's class name from a sklearn Pipeline or plain estimator."""
    if model is None:
        return "None"
    try:
        # sklearn Pipeline-like
        if hasattr(model, "steps") and getattr(model, "steps"):
            est = getattr(model, "final_estimator", None) or getattr(model, "_final_estimator", None)
            if est is None:
                est = model.steps[-1][1]
            return type(est).__name__
        if hasattr(model, "named_steps") and getattr(model, "named_steps"):
            try:
                last = list(model.named_steps.values())[-1]
                return type(last).__name__
            except Exception:
                pass
        return type(model).__name__
    except Exception:
        return type(model).__name__

# -------- main summary function --------

def generate_run_summary(state: Dict[str, Any]) -> str:
    """Return a professional markdown report. Code artifacts are shown
    as proper ```python blocks, cleaned to match the UI, and plots are embedded inline."""
    profile = state.get("profile") or {}
    metrics = state.get("metrics") or state.get("evaluation") or {}
    model = state.get("model")
    origin = state.get("origin") or getattr(model, "_origin", "unknown")

    dataset_name = state.get("dataset_name") or "dataset"
    target = state.get("target") or profile.get("target")

    # Schema / shape
    schema = profile.get("schema") or {}
    n_rows = state.get("n_rows") or schema.get("n_rows") or profile.get("n_rows")
    n_cols = state.get("n_cols") or schema.get("n_cols") or profile.get("n_columns")

    task = (profile.get("target_type") or profile.get("target_type_detailed") or "unknown")

    # Quick EDA bits
    num_cols = schema.get("numeric_cols") or profile.get("numeric_cols") or []
    cat_cols = schema.get("categorical_cols") or profile.get("categorical_cols") or []
    mv = profile.get("missing_values") or profile.get("null_summary") or {}
    mv_cols = [k for k, v in mv.items() if v]
    class_balance = profile.get("class_balance") or {}

    # Prefer model_name from payload; fallback to inferring from the object.
    algo = (state.get("model_name") or "").strip()
    if not algo or algo.lower() == "none":
        algo = _algo_name(model)

    lines: List[str] = []
    lines.append("# AutoML Run Summary\n")

    # Dataset
    lines.append("## Dataset")
    lines.append(f"- **Source:** {dataset_name}")
    if n_rows and n_cols:
        lines.append(f"- **Rows × Cols:** {n_rows} × {n_cols}")
    if target:
        lines.append(f"- **Target:** `{target}`")
    if task:
        lines.append(f"- **Inferred task:** {task}")

    # Quick EDA
    lines.append("\n## Quick EDA")
    lines.append(f"- **Numeric columns:** {len(num_cols)} • **Categorical columns:** {len(cat_cols)}.")
    if mv_cols:
        mv_text = "; ".join(f"{k}: {v}" for k, v in mv.items() if v)
        lines.append(f"- **Missing-value summary:** {mv_text}")
    else:
        lines.append("- **Missing-value summary:** no columns with >0% missing.")
    if class_balance:
        parts = []
        for k, v in class_balance.items():
            try:
                parts.append(f"{k}: {round(float(v) * 100, 2)}%")
            except Exception:
                parts.append(f"{k}: {v}")
        lines.append(f"- **Class balance:** {', '.join(parts)}")

    # Final model
    lines.append("\n## Final Model")
    lines.append(f"- **Origin:** {origin}")
    lines.append(f"- **Algorithm:** {algo}")

    # Hold-out metrics
    lines.append("\n## Hold-out Metrics")
    if metrics:
        shown = False
        for k in ("accuracy", "f1_score", "f1_macro", "rmse", "mae", "r2_score"):
            if k in metrics and metrics[k] is not None:
                val = metrics[k]
                try:
                    val = float(val)
                    lines.append(f"- **{k}:** {val:.4f}")
                except Exception:
                    lines.append(f"- **{k}:** {val}")
                shown = True
        if not shown:
            lines.append("_No numeric metrics available._")
    else:
        lines.append("_No metrics available._")

    # Code artifacts (cleaned like the UI)
    lines.append("\n## Code Artifacts")
    code_map = state.get("code_map") or {}
    if code_map:
        for idx, (step, code) in enumerate(code_map.items(), start=1):
            code = _clean_code(code or "")
            lines.append(f"### Step {idx}: {step}")
            lines.append("```python")
            lines.append(code.rstrip())
            lines.append("```")
    else:
        lines.append("_No code artifacts._")

    # Visualizations (inline via base64 HTML <img>) – only renders where HTML is allowed.
    lines.append("")  # ensure a blank line before plots block
    plots = state.get("plots") or []
    lines.append(_render_plots_inline_html(plots))

    # Repro notes
    lines.append("## Reproducibility Notes")
    lines.append("- Heavy hyperparameter tuning and network/file I/O were disabled for safety.")
    lines.append("- Pipelines use sensible defaults (median impute for numeric; most_frequent + OHE for categorical).")
    lines.append("- The LLM pipeline is evaluated on a fresh hold-out; if unusable, a deterministic baseline is trained.")

    return "\n".join(lines).strip() + "\n"
