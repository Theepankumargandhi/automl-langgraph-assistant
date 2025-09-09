# planning_agent.py
from typing import List, Optional, Dict
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def _format_profile(profile: Optional[Dict]) -> str:
    """Compact, readable block of dataset/profile facts for the planner LLM."""
    if not profile:
        return "(no profile provided)"

    try:
        tgt = profile.get("target")
        ttype = (
            profile.get("target_type")
            or profile.get("target_type_detailed")
            or "unknown"
        )
        schema = profile.get("schema", {}) or {}
        n_rows = schema.get("n_rows")
        n_cols = schema.get("n_cols")
        dtypes = schema.get("dtypes", {}) or {}
        missing_pct = schema.get("missing_pct", {}) or {}
        num_cols = schema.get("numeric_cols", []) or []
        cat_cols = schema.get("categorical_cols", []) or []
        card = profile.get("target_cardinality")
        balance = profile.get("class_balance")

        # Short summaries
        dtypes_summary = ", ".join(sorted({str(v) for v in dtypes.values()}))[:300]
        top_missing = sorted(missing_pct.items(), key=lambda kv: kv[1], reverse=True)[:8]
        missing_summary = (
            ", ".join(f"{k}:{v:.2%}" for k, v in top_missing) if top_missing else "none >0%"
        )
        balance_summary = (
            ", ".join(
                f"{k}:{v:.2%}"
                for k, v in sorted(balance.items(), key=lambda kv: kv[1], reverse=True)[:8]
            )
            if isinstance(balance, dict)
            else "n/a"
        )

        return (
            f"- target: {tgt}\n"
            f"- inferred_task: {ttype}\n"
            f"- rows/cols: {n_rows}/{n_cols}\n"
            f"- target_cardinality: {card}\n"
            f"- dtypes_present: {dtypes_summary}\n"
            f"- numeric_cols: {len(num_cols)} • categorical_cols: {len(cat_cols)}\n"
            f"- top_missing_pct: {missing_summary}\n"
            f"- class_balance (if classification): {balance_summary}"
        )
    except Exception:
        # Never block planning if profile is malformed
        return "(profile provided, but could not be summarized)"


def plan_from_rules(
    retrieved_rules: List[str],
    model_name: str = "gpt-4o-mini",
    profile: Optional[Dict] = None,
) -> List[str]:
    """
    Turn retrieved rules (+ optional dataset profile) into a clean, codeable, concise action plan.

    - Excludes heavy/irrelevant steps (tuning, deploy, persistence, external I/O).
    - De-duplicates similar lines.
    - Caps the number of steps to keep runs fast and tidy.
    - If `profile` is provided, the LLM is grounded on concrete schema facts
      (rows/cols, dtypes, missingness, target type/cardinality), which reduces vague steps.
    """
    rules_text = "\n\n".join(retrieved_rules or [])
    profile_text = _format_profile(profile)

    prompt_template = PromptTemplate(
        input_variables=["rules", "facts"],
        template="""
You are an AI data science planner.

Use ONLY the information below (retrieved rules + dataset facts) to produce an ordered
list of concrete, codeable steps for a supervised ML pipeline on an in-memory DataFrame `df`.

DATASET FACTS (read-only):
------------------------
{facts}
------------------------

RETRIEVED RULES:
------------------------
{rules}
------------------------

STRICTLY EXCLUDE:
- Hyperparameter tuning, fine-tuning, GridSearch/RandomizedSearch/Optuna
- Model persistence (save/export/load pickle/joblib), registries, MLflow
- Deployment/app code (Flask/Streamlit/FastAPI)
- External file/network I/O (download/upload/read_csv/read_excel/open/requests/urllib)
- Reports, presentations, monitoring, long prose

PLANNING GUIDELINES:
- Begin with concise EDA & target inspection geared to {facts}.
- Define preprocessing with a scikit-learn ColumnTransformer:
  numeric → SimpleImputer(median)
  categorical → SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")
- Split strategy appropriate for the task (e.g., stratified if classification and feasible).
- Specify a baseline model suitable for the inferred task (classification vs regression) and metrics.
- End with a clear evaluation on a hold-out set; optional simple visuals.
- Each step must be directly codeable in 1–2 short functions/snippets.

OUTPUT FORMAT:
- Plain bullet points (one per line). No numbering, no extra prose.
""",
    )

    llm = ChatOpenAI(model=model_name, temperature=0)
    chain: RunnableSequence = prompt_template | llm
    response = chain.invoke({"rules": rules_text, "facts": profile_text})
    
    # Track API cost
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and 'cost_tracker' in st.session_state:
            st.session_state.cost_tracker.track_openai_call(response, "planning")
    except Exception:
        pass
    
    text = getattr(response, "content", str(response)).strip()

    # Split into lines, strip bullets, and drop empties
    raw_steps = [line.strip().lstrip("•-").strip() for line in text.splitlines() if line.strip()]

    # Exclusion filters (broader safety net)
    exclude_keywords = [
        # vague/non-codeable
        "define objective", "objective", "goal", "interpret", "document", "monitor",
        "conclusion", "report", "insight", "communication", "presentation",
        # heavy/irrelevant
        "fine-tune", "fine tune", "finetune", "hyperparameter", "gridsearch", "grid search",
        "randomizedsearch", "randomized search", "optuna", "bayes", "bayesian optimization",
        "save model", "export model", "serialize", "serialization", "registry", "mlflow",
        "pickle", "joblib", "load model",
        "deploy", "deployment", "flask", "fastapi", "streamlit",
        # file/network I/O
        "read_csv", "read csv", "read excel", "read_excel", "download", "upload", "open(",
        "requests.", "urllib.", "os.system", "subprocess",
    ]

    filtered = [s for s in raw_steps if not any(k in s.lower() for k in exclude_keywords)]

    # De-duplicate while preserving order (case-insensitive)
    seen = set()
    deduped: List[str] = []
    for s in filtered:
        key = s.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    # Keep the plan short and actionable
    MAX_STEPS = 8
    plan = deduped[:MAX_STEPS]

    # Safe default skeleton if nothing useful came back
    if not plan:
        plan = [
            "Perform quick EDA (shape, dtypes, missingness, target distribution)",
            "Build preprocessing with ColumnTransformer (num: median impute; cat: most_frequent + OHE ignore_unknown)",
            "Split data into train/test (stratify if classification and feasible)",
            "Train a suitable baseline estimator for the task",
            "Evaluate on the test set with appropriate metrics"
        ]

    return plan