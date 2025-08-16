# main_runner.py

import os
import pathlib
import pandas as pd
from agents.graph_orchestrator import create_graph

# === Runtime flags (set defaults in code; no shell env needed) ===
# Allow LLM-generated code to read local files (only inside the whitelisted folder)
os.environ.setdefault("ALLOW_IO", "1")
# Skip expensive hyperparameter tuning by default
os.environ.setdefault("ALLOW_TUNING", "0")
# Whitelist the current project folder for any file reads
os.environ.setdefault("ALLOWED_DATA_DIR", str(pathlib.Path(__file__).parent.resolve()))

# === Load dataset ===
# Put your CSV inside the project folder (same folder as this file) or change the path below.
df = pd.read_csv("Wine.csv")  # Replace with your actual dataset path
target_column = "quality"

# === Create LangGraph application ===
app = create_graph()

# === Initialize graph input state (profile can be filled by the Profile node) ===
# To be extra safe with pipeline_builder, we include the target in profile too.
input_state = {
    "df": df,
    "target_column": target_column,
    "profile": {"target": target_column},  # ensure target is available downstream
    "rules": {},          # will be filled by RAG or fallback LLM
    "steps": [],          # filled by plan generator
    "fallback": False,    # set True if RAG fails
    "code_map": {},       # filled with generated code
    "model": None,        # trained model object
    "evaluation": {}      # evaluation report
}

# === Run the LangGraph pipeline ===
final_state = app.invoke(input_state)

# === Output Summary (null-safe, no truthiness checks) ===
print("\n===  FINAL OUTPUT ===")
fs_df = final_state.get("df", None)
print(f" Final DataFrame shape: {fs_df.shape if fs_df is not None else 'Unknown'}")

m = final_state.get("model", None)
model_name = type(m).__name__ if m is not None else "None"
print(f" Model trained: {model_name}")

metrics = final_state.get("evaluation", {})
print("\n Evaluation Report:")
if isinstance(metrics, dict) and metrics:
    for k, v in metrics.items():
        try:
            print(f"{k}: {float(v)}")
        except Exception:
            print(f"{k}: {v}")
else:
    print(" No evaluation available.")

# === Show Steps ===
steps = final_state.get("steps", [])
print("\n Generated Steps:")
if steps:
    for s in steps:
        print(f"• {s}")
else:
    print("• (no steps)")

# === Show Code Map (optional: prints the generated code for each step) ===
code_map = final_state.get("code_map", {})
print("\n Code Map:")
if code_map:
    for step, code in code_map.items():
        print(f"\n🔹 Step: {step}\n{code}\n{'-'*40}")
else:
    print("(empty)")
