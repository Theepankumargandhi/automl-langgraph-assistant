# execution_agent.py
from typing import List, Dict
import re
import time
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from prompts import CODEGEN_PROMPT

load_dotenv()

# You can override via env if needed
MODEL_NAME_PRIMARY = "gpt-4o-mini"
MODEL_NAME_SECONDARY = "gpt-4o-mini"

# --- Helpers -----------------------------------------------------------------

def _retry_call(fn, retries=3, backoff=1.5):
    for i in range(retries):
        try:
            return fn()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(backoff ** i)

def extract_python_code(text: str) -> str:
    """
    Extracts clean Python code from the LLM response.
    - Accepts ```python ... ``` or plain ``` ... ```
    - Strips markdown and removes inplace=True to avoid pandas warnings.
    """
    # Try ```python ... ```
    matches = re.findall(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not matches:
        # Fallback to plain ```
        matches = re.findall(r"```(.*?)```", text, re.DOTALL)

    if matches:
        cleaned_code = matches[0].strip()
    else:
        # Fall back to line filter
        lines = text.splitlines()
        valid_lines = [
            line for line in lines
            if not line.strip().startswith(("```", "# ", "Assuming", "Please", "In this", "-", "```python"))
            and line.strip() != ""
        ]
        cleaned_code = "\n".join(valid_lines).strip()

    return cleaned_code.replace("inplace=True", "")

_BANNED_PATTERNS = [
    "pd.read_csv(", "pd.read_excel(", "open(", "requests.", "urllib.",
    "os.system(", "subprocess.", "mlflow", "flask", "fastapi", "streamlit"
]

def _static_ok(code: str) -> bool:
    """Lightweight static checks to avoid obvious bad generations."""
    t = code.lower()
    if any(p in t for p in (bp.lower() for bp in _BANNED_PATTERNS)):
        return False
    has_pipeline = "pipeline(" in t
    has_ct = "columntransformer(" in t
    has_model = re.search(r"\bmodel\s*=", t) is not None
    return has_pipeline and has_ct and has_model

def _ensure_imports_and_selfcheck(code: str) -> str:
    """
    Add missing imports for symbols used; ensure the final self-check line exists.
    Non-intrusive: only prepends safe imports if referenced and missing.
    """
    lower = code.lower()
    imports = []

    # sklearn core pieces
    if "columntransformer(" in lower and "from sklearn.compose import columntransformer" not in lower:
        imports.append("from sklearn.compose import ColumnTransformer")
    if "pipeline(" in lower and "from sklearn.pipeline import pipeline" not in lower:
        imports.append("from sklearn.pipeline import Pipeline")
    if "simpleimputer(" in lower and "from sklearn.impute import simpleimputer" not in lower:
        imports.append("from sklearn.impute import SimpleImputer")
    if "onehotencoder(" in lower and "from sklearn.preprocessing import onehotencoder" not in lower:
        imports.append("from sklearn.preprocessing import OneHotEncoder")
    if "train_test_split(" in lower and "from sklearn.model_selection import train_test_split" not in lower:
        imports.append("from sklearn.model_selection import train_test_split")

    # Common libs
    if "np." in lower and "import numpy as np" not in code:
        imports.append("import numpy as np")
    if "plt." in lower and "import matplotlib.pyplot as plt" not in code:
        imports.append("import matplotlib.pyplot as plt")
    if "sns." in lower and "import seaborn as sns" not in code:
        imports.append("import seaborn as sns")

    # Prepend imports if any
    if imports:
        code = "\n".join(imports) + "\n" + code

    # Ensure the self-check exists (uses `target_column` available at runtime)
    # Keep this minimal to avoid side effects; just a shape/prob call
    if "_ = model.predict(" not in code and "model.predict(" not in code:
        code += "\n_ = model.predict(df.drop(columns=[target_column]).head(1))\n"

    return code

def _context_from_state(state: Dict) -> Dict[str, str]:
    """
    Prepare optional context (ignored by the prompt if not referenced):
    - rules_text: joined rules from retrieval
    - fewshot: pre-defined pattern, if provided upstream
    - profile_json: compact JSON of profile facts
    """
    rules_list = state.get("rules") or []
    if isinstance(rules_list, (list, tuple)):
        rules_text = "\n\n".join([str(r) for r in rules_list if r])
    else:
        rules_text = str(rules_list)

    fewshot = state.get("fewshot", "")  # safe if caller injects a snippet
    profile_json = json.dumps(state.get("profile", {}), default=str)

    return {
        "rules_text": rules_text,
        "fewshot": fewshot,
        "profile_json": profile_json,
    }

def _track_api_cost(response, operation: str):
    """Track API cost if Streamlit session exists"""
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and 'cost_tracker' in st.session_state:
            st.session_state.cost_tracker.track_openai_call(response, operation)
    except Exception:
        pass  # Fail silently if tracking unavailable

# --- Main entrypoint ----------------------------------------------------------

def generate_code_for_steps(steps: List[str], state: Dict) -> Dict[str, str]:
    """
    Given a list of steps and global state containing target column and task type,
    generates Python code for each step using LLM (stable invoke + retry + static validation).
    Extra context (rules/few-shot/profile) is passed if your prompt supports it;
    otherwise it's harmlessly ignored.
    """
    llm_primary = ChatOpenAI(model=MODEL_NAME_PRIMARY, temperature=0.0)
    llm_secondary = ChatOpenAI(model=MODEL_NAME_SECONDARY, temperature=0.3)

    prompt: PromptTemplate = CODEGEN_PROMPT
    chain_primary = prompt | llm_primary
    chain_secondary = prompt | llm_secondary

    ctx = _context_from_state(state)

    code_map: Dict[str, str] = {}
    for step in steps:
        print(f"\n Generating code for step: {step}")

        def _invoke(chain, step_name):
            response = chain.invoke({
                "step": step,
                "target": state["target_column"],
                "task": state.get("profile", {}).get("target_type", "classification"),
                # Optional extras (ignored if your PromptTemplate doesn't reference them)
                "rules_text": ctx["rules_text"],
                "fewshot": ctx["fewshot"],
                "profile_json": ctx["profile_json"],
            })
            
            # Track API cost
            _track_api_cost(response, f"code_gen_{step_name}")
            return response

        try:
            # Two candidates
            resp1 = _retry_call(lambda: _invoke(chain_primary, step[:30]))
            txt1 = getattr(resp1, "content", str(resp1))
            code1 = extract_python_code(txt1)

            resp2 = _retry_call(lambda: _invoke(chain_secondary, step[:30]))
            txt2 = getattr(resp2, "content", str(resp2))
            code2 = extract_python_code(txt2)

            # Pick the first that passes static checks
            chosen = None
            if _static_ok(code1):
                chosen = code1
                print(" Candidate #1 passed static checks.")
            elif _static_ok(code2):
                chosen = code2
                print(" Candidate #2 passed static checks.")
            else:
                chosen = code1
                print(" Neither candidate passed static checks; using candidate #1 with light auto-repair.")

            # Light auto-repair (missing imports, ensure self-check)
            chosen = _ensure_imports_and_selfcheck(chosen)

            if not chosen.strip():
                print(f" Skipped step due to empty code: {step}")
                continue

            print(f" Code (post-check):\n{chosen}\n{'-'*60}\n")
            code_map[step] = chosen

        except Exception as e:
            print(f" Error generating code for step '{step}': {e}")
            continue

    return code_map