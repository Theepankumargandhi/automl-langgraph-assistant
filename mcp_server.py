#!/usr/bin/env python3
"""
Simple Protocol Server for AutoML Assistant
Exposes the LangGraph AutoML pipeline via JSON-RPC style interface
"""

import json
import sys
import traceback
from typing import Dict, Any
import pandas as pd
from io import StringIO

from agents.graph_orchestrator import create_graph

def analyze_dataset(csv_data: str, target_column: str) -> Dict[str, Any]:
    """Profile a CSV dataset and return analysis."""
    try:
        df = pd.read_csv(StringIO(csv_data))
        
        from agents.profile_agent import profile_dataset
        profile = profile_dataset.invoke({
            "df": df,
            "target_column": target_column
        })
        
        return {
            "status": "success",
            "result": {
                "dataset_shape": profile.get("shape"),
                "target_type": profile.get("target_type"),
                "target_cardinality": profile.get("target_cardinality"),
                "missing_values": profile.get("missing_values"),
                "class_balance": profile.get("class_balance")
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def run_automl_pipeline(csv_data: str, target_column: str) -> Dict[str, Any]:
    """Run complete AutoML pipeline."""
    try:
        df = pd.read_csv(StringIO(csv_data))
        
        automl_app = create_graph()
        input_state = {
            "df": df,
            "target_column": target_column,
            "profile": {"target": target_column},
            "rules": {},
            "steps": [],
            "fallback": False,
            "code_map": {},
            "model": None,
            "evaluation": {}
        }
        
        final_state = automl_app.invoke(input_state)
        evaluation = final_state.get("evaluation", {})
        model = final_state.get("model")
        origin = getattr(model, "_origin", "unknown") if model else "unknown"
        
        return {
            "status": "success",
            "result": {
                "model_origin": origin,
                "evaluation_metrics": {
                    k: v for k, v in evaluation.items() 
                    if isinstance(v, (int, float, bool))
                },
                "dataset_info": {
                    "shape": df.shape,
                    "target": target_column
                }
            }
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def main():
    """Simple JSON-RPC style server."""
    print("AutoML Protocol Server Started")
    print("Available commands: analyze_dataset, run_automl_pipeline")
    print("Send JSON requests via stdin")
    
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            command = request.get("command")
            args = request.get("args", {})
            
            if command == "analyze_dataset":
                result = analyze_dataset(args["csv_data"], args["target_column"])
            elif command == "run_automl_pipeline":
                result = run_automl_pipeline(args["csv_data"], args["target_column"])
            else:
                result = {"status": "error", "error": f"Unknown command: {command}"}
            
            print(json.dumps(result))
            sys.stdout.flush()
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(json.dumps(error_result))
            sys.stdout.flush()

if __name__ == "__main__":
    main()