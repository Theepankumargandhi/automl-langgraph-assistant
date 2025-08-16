# prompts.py
from langchain.prompts import PromptTemplate

# =========================
# Planner Prompt (kept simple and strict)
# =========================
PLAN_PROMPT = PromptTemplate(
    input_variables=["rules"],
    template="""
You are an AI data science planner.

Given the retrieved rules below, produce an ordered list of concrete, codeable steps
for working with an in-memory DataFrame named `df`. The data is already loaded.

------------------------
{rules}
------------------------

STRICTLY EXCLUDE:
- File/network I/O (read_csv/read_excel/open/requests/urllib, downloads, uploads)
- Model persistence/registries (pickle/joblib, MLflow)
- Hyperparameter tuning / GridSearch / RandomizedSearch / Optuna
- Deployment/app code (Flask/Streamlit/FastAPI)
- Long explanations, reports, or documentation

GUIDELINES:
- Start with concise EDA and target inspection.
- Define preprocessing using a scikit-learn ColumnTransformer:
  numeric → SimpleImputer(median)
  categorical → SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")
- Use an appropriate split strategy (stratified if classification and feasible).
- Train a sensible baseline model for the inferred task and specify evaluation metrics.
- End with evaluation on a hold-out split; optional simple visuals are fine.
- Each step must be directly implementable in 1–2 short code snippets.
- Keep it short: 5–8 bullet points max.

Output: plain bullet points only (no numbering or extra prose).
"""
)

# =========================
# Code Generator Prompt (v1 - used by current execution_agent)
# =========================
CODEGEN_PROMPT = PromptTemplate(
    input_variables=["step", "target", "task"],
    template="""
You are a helpful data science assistant. Generate clean, executable Python code for the step below.

Task Type: {task}
Target Column: {target}
Step: {step}

Return ONLY runnable Python code — no markdown fences or explanations.

CODE CONTRACT:
- Do NOT read/write files or make any network calls. Use the in-memory DataFrame `df`.
- If you need features/target, define:
    X = df.drop(columns=['{target}'])
    y = df['{target}']
  (Keep y 1-D; do NOT scale or encode the target.)
- If this step is about model training or evaluation, build a scikit-learn Pipeline with a ColumnTransformer:
    numeric → SimpleImputer(strategy='median')
    categorical → SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
- Only create a trained model variable named `model` if this step actually trains a model.
- If you create `model`, add a final self-check line:
    _ = model.predict(df.drop(columns=['{target}']).head(1))

IMPLEMENTATION NOTES:
- Import everything you use (e.g., from sklearn..., import numpy as np, import matplotlib.pyplot as plt, import seaborn as sns if used).
- Keep the snippet self-contained; avoid undefined variables.
- If you make plots, standard matplotlib/seaborn calls are fine (showing with plt.show() is acceptable).
"""
)

# =========================
# Code Generator Prompt (v2 - richer context; optional to adopt later)
# =========================
CODEGEN_PROMPT_V2 = PromptTemplate(
    input_variables=["step", "target", "task", "profile", "rules", "examples"],
    template="""
You are a senior ML engineer. Generate robust, executable Python code for the step below,
grounded ONLY in the provided facts and rules. Return ONLY Python code — no prose.

Task Type: {task}
Target Column: {target}
Step: {step}

FACTS (schema/profile):
------------------------
{profile}
------------------------

RETRIEVED RULES / PATTERNS:
------------------------
{rules}
------------------------

REFERENCE EXAMPLES:
------------------------
{examples}
------------------------

CODE CONTRACT:
- No file I/O or network access. Use the in-memory DataFrame `df`.
- If you need features/target:
    X = df.drop(columns=['{target}'])
    y = df['{target}']
  (Keep y 1-D; do NOT transform the target.)
- For modeling/evaluation steps, build a scikit-learn Pipeline with a ColumnTransformer:
    numeric → SimpleImputer(strategy='median')
    categorical → SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
- Prefer sensible defaults; set random_state=42 when available.
- Only create a trained model variable named `model` if the step trains a model.
- If you create `model`, end with:
    _ = model.predict(df.drop(columns=['{target}']).head(1))

IMPLEMENTATION NOTES:
- Import all modules you use (sklearn components, numpy as np, matplotlib.pyplot as plt, seaborn as sns if used).
- Keep the code block self-contained and concise.
- Standard matplotlib/seaborn calls are fine if you visualize.
"""
)


