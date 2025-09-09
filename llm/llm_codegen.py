# llm_codegen.py
from typing import Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Default LLMs (env-overridable)
_DEFAULT_MODEL_PRIMARY = "gpt-4o"
_DEFAULT_MODEL_SECONDARY = "gpt-4o"  # you can set to "gpt-4o-mini" via env if you prefer

# ---------- Prompt with Code Contract ----------
# Backward-compatible: callers may only pass `step`. We rely on `target_column` in exec env.
_CODEGEN_TEMPLATE = """
You are a senior ML engineer. Generate robust, runnable Python code for the step below.
Return ONLY Python code — no markdown fences or explanations.

STEP:
{step}

CONTEXT (facts are available at runtime):
- In-memory DataFrame: df
- Target column name variable: target_column
- Do NOT read or write files. Do NOT make network calls.

CODE CONTRACT (must follow)
- Define: X = df.drop(columns=[target_column]); y = df[target_column]  (keep y 1-D; do NOT transform target)
- Build a scikit-learn Pipeline with a ColumnTransformer:
    * numeric → SimpleImputer(strategy='median')
    * categorical → SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
- Choose a sensible estimator for the task (classification vs regression if the step implies it)
- Name the final trained estimator variable exactly: model
- End with a self-check:
    _ = model.predict(df.drop(columns=[target_column]).head(1))

IMPLEMENTATION NOTES
- Import everything you use (e.g., from sklearn..., import numpy as np, import matplotlib.pyplot as plt, import seaborn as sns if used)
- Keep the code self-contained. Avoid undefined names/variables.
"""

# A slightly different variant to produce diversity (tree vs linear hint)
_CODEGEN_TEMPLATE_ALT = _CODEGEN_TEMPLATE + """
HINT: Prefer a different reasonable estimator family than your first instinct
(e.g., if you'd normally pick a tree-based model, consider a linear model; if linear, consider a tree).
"""

def _track_api_cost(response, operation: str):
    """Track API cost if Streamlit session exists"""
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and 'cost_tracker' in st.session_state:
            st.session_state.cost_tracker.track_openai_call(response, operation)
    except Exception:
        pass  # Fail silently if tracking unavailable

def _build_chain(model_name: str, temperature: float, template: str) -> LLMChain:
    prompt = PromptTemplate(input_variables=["step"], template=template)
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Wrap the run method to track costs
    original_run = chain.run
    def tracked_run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        # Note: LLMChain.run() returns a string, not the raw response object
        # So we can't track tokens directly, but we can estimate
        try:
            _track_api_cost(result, "llm_codegen")
        except Exception:
            pass
        return result
    chain.run = tracked_run
    
    return chain

def _clean_code_blocks(text: str) -> str:
    """Extract raw Python code if fenced; otherwise return as-is."""
    t = text.strip()
    if "```python" in t:
        try:
            return t.split("```python", 1)[1].split("```", 1)[0].strip()
        except Exception:
            pass
    if "```" in t:
        try:
            return t.split("```", 1)[1].split("```", 1)[0].strip()
        except Exception:
            pass
    return t

# Simple static check to reduce obvious failures before execution
_BANNED_PATTERNS = [
    "pd.read_csv(", "pd.read_excel(", "open(", "requests.", "urllib.", "os.system(", "subprocess."
]

def _static_ok(code: str) -> bool:
    text = code.lower()
    if any(p in text for p in (bp.lower() for bp in _BANNED_PATTERNS)):
        return False
    # Require a Pipeline + ColumnTransformer and a 'model =' assignment
    has_pipeline = "pipeline(" in text
    has_ct = "columntransformer(" in text
    has_model_assign = "model =" in text
    return has_pipeline and has_ct and has_model_assign

def _ensure_imports(code: str) -> str:
    """Light auto-repair: add imports if the snippet forgot them but uses the symbols."""
    needed_lines = []
    lower = code.lower()

    def missing(snippet: str) -> bool:
        return snippet not in lower

    # sklearn core pieces
    if "columntransformer(" in lower and ("from sklearn.compose import columntransformer" not in lower):
        needed_lines.append("from sklearn.compose import ColumnTransformer")
    if "pipeline(" in lower and ("from sklearn.pipeline import pipeline" not in lower):
        needed_lines.append("from sklearn.pipeline import Pipeline")
    if "simpleimputer(" in lower and ("from sklearn.impute import simpleimputer" not in lower):
        needed_lines.append("from sklearn.impute import SimpleImputer")
    if "onehotencoder(" in lower and ("from sklearn.preprocessing import onehotencoder" not in lower):
        needed_lines.append("from sklearn.preprocessing import OneHotEncoder")
    if "train_test_split(" in lower and ("from sklearn.model_selection import train_test_split" not in lower):
        needed_lines.append("from sklearn.model_selection import train_test_split")
    # common libs
    if "np." in lower and "import numpy as np" not in code:
        needed_lines.append("import numpy as np")
    if "plt." in lower and "import matplotlib.pyplot as plt" not in code:
        needed_lines.append("import matplotlib.pyplot as plt")
    if "sns." in lower and "import seaborn as sns" not in code:
        needed_lines.append("import seaborn as sns")

    if needed_lines:
        # Prepend imports just once at the top
        return "\n".join(needed_lines) + "\n" + code
    return code

def generate_python_code(step: str,
                         target: Optional[str] = None,
                         task: Optional[str] = None,
                         profile: Optional[Dict[str, Any]] = None,
                         rules: Optional[str] = None,
                         examples: Optional[str] = None) -> str:
    """
    Generate robust Python code for a pipeline step.
    Backwards compatible: callers may pass only `step`.

    Strategy:
      - Ask two slightly different prompts (different temps / hints)
      - Clean/strip code fences
      - Run a static check (no I/O; requires Pipeline+ColumnTransformer+model)
      - Pick the first passing candidate
      - Light auto-repair for missing imports if symbols are used
    """
    # Candidate 1: deterministic / primary template
    chain1 = _build_chain(_DEFAULT_MODEL_PRIMARY, temperature=0.0, template=_CODEGEN_TEMPLATE)
    resp1 = chain1.run({"step": step}).strip()
    code1 = _clean_code_blocks(resp1)

    # Candidate 2: slight diversity / alternate template
    chain2 = _build_chain(_DEFAULT_MODEL_SECONDARY, temperature=0.3, template=_CODEGEN_TEMPLATE_ALT)
    resp2 = chain2.run({"step": step}).strip()
    code2 = _clean_code_blocks(resp2)

    # Pick the first that passes static checks
    selected = None
    if _static_ok(code1):
        selected = code1
    elif _static_ok(code2):
        selected = code2
    else:
        # Neither passes — return the first, but still try to add missing imports
        selected = code1

    # Light auto-repair: add imports if referenced but missing
    selected = _ensure_imports(selected)

    return selected