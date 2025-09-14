# agents/graph_orchestrator.py
# agents/graph_orchestrator.py
from typing import TypedDict
import pandas as pd
from langgraph.graph import StateGraph, END
from chromadb import Client

from agents.profile_agent import profile_dataset
from agents.retrieval_agent import retrieve_rules
from agents.planning_agent import plan_from_rules
from agents.execution_agent import generate_code_for_steps
from agents.pipeline_builder import build_pipeline
from agents.evaluation_agent import evaluate_model
from agents.summary_agent import generate_run_summary


class AgentState(TypedDict, total=False):
    df: pd.DataFrame
    target_column: str
    profile: dict
    rules: dict
    steps: list
    code_map: dict
    model: object
    evaluation: dict
    fallback: bool
    summary_md: str
    progress_callback: callable  # For progress tracking
    mlflow_run_id: str  # Track MLflow run


def _initialize_mlflow_run(state: AgentState) -> AgentState:
    """Initialize MLflow experiment and start run (optional, non-breaking)"""
    try:
        from mlflow_config import initialize_mlflow, start_automl_run, log_dataset_profile
        
        # Initialize MLflow
        initialize_mlflow()
        
        # Start run with dataset info for naming
        dataset_info = {
            "dataset_shape": state["df"].shape if state.get("df") is not None else (0, 0),
            "target": state.get("target_column", "unknown")
        }
        
        run_name = f"automl-{dataset_info['target']}-{dataset_info['dataset_shape'][0]}x{dataset_info['dataset_shape'][1]}"
        
        run = start_automl_run(
            run_name=run_name,
            tags={
                "pipeline.framework": "langgraph",
                "pipeline.agents": "7-agent-system"
            }
        )
        
        state["mlflow_run_id"] = run.info.run_id
        print(f"📊 Started MLflow run: {run.info.run_id}")
        
    except Exception as e:
        print(f"MLflow initialization warning: {e}")
        state["mlflow_run_id"] = None
    
    return state


def _finalize_mlflow_run(state: AgentState) -> AgentState:
    """Finalize MLflow run and log summary (optional, non-breaking)"""
    try:
        import mlflow
        
        if state.get("mlflow_run_id"):
            # Log final run summary
            mlflow.log_param("pipeline.completed", True)
            mlflow.log_param("pipeline.status", "success")
            
            # Log final model info
            model = state.get("model")
            if model:
                origin = getattr(model, "_origin", "unknown")
                mlflow.log_param("final.model_origin", origin)
            
            print(f"📊 Finalized MLflow run: {state['mlflow_run_id']}")
            
    except Exception as e:
        print(f"MLflow finalization warning: {e}")
    
    return state


def profile_node(state: AgentState) -> AgentState:
    if state.get("progress_callback"):
        state["progress_callback"]("Profiling Dataset")
    
    profile = profile_dataset.invoke({
        "df": state["df"],
        "target_column": state["target_column"]
    })
    state["profile"] = profile
    
    # Log dataset profile to MLflow (optional, non-breaking)
    try:
        from mlflow_config import log_dataset_profile
        log_dataset_profile(profile)
    except Exception:
        pass
    
    return state


def retrieval_node(state: AgentState) -> AgentState:
    if state.get("progress_callback"):
        state["progress_callback"]("Retrieving Rules")
    
    chroma_client = Client()
    rules, fallback = retrieve_rules(state["profile"], chroma_client)
    state["rules"] = rules
    state["fallback"] = fallback
    
    # Log retrieval info to MLflow (optional, non-breaking)
    try:
        import mlflow
        mlflow.log_param("rag.fallback_mode", fallback)
        mlflow.log_metric("rag.rules_retrieved", len(rules) if rules else 0)
    except Exception:
        pass
    
    return state


def planning_node(state: AgentState) -> AgentState:
    if state.get("progress_callback"):
        state["progress_callback"]("Planning Pipeline")
    
    if state.get("fallback"):
        steps = plan_from_rules(["Default plan for classification/regression on in-memory df."])
    else:
        steps = plan_from_rules(state.get("rules", []))
    state["steps"] = steps
    
    # Log planning info to MLflow (optional, non-breaking)
    try:
        from mlflow_config import log_pipeline_steps
        log_pipeline_steps(steps)
    except Exception:
        pass
    
    return state


def pipeline_node(state: AgentState) -> AgentState:
    if state.get("progress_callback"):
        state["progress_callback"]("Building Pipeline")
    
    steps = state.get("steps", [])
    code_map = generate_code_for_steps(steps, state)
    try:
        df_out, model = build_pipeline(state["df"], state["profile"], code_map)
        state["df"] = df_out
        state["model"] = model
        state["code_map"] = code_map
        
        # Log pipeline success to MLflow (optional, non-breaking)
        try:
            import mlflow
            mlflow.log_param("pipeline.execution_status", "success")
            mlflow.log_metric("pipeline.steps_executed", len(code_map))
        except Exception:
            pass
            
    except Exception as e:
        state["pipeline_error"] = str(e)
        
        # Log pipeline failure to MLflow (optional, non-breaking)
        try:
            import mlflow
            mlflow.log_param("pipeline.execution_status", "failed")
            mlflow.log_param("pipeline.error", str(e))
        except Exception:
            pass
    
    return state


def evaluation_node(state: AgentState) -> AgentState:
    if state.get("progress_callback"):
        state["progress_callback"]("Evaluating Model")
    
    profile = state.get("profile", {}) or {}
    task_type = profile.get("target_type")
    model = state.get("model", None)
    df = state.get("df", None)
    target_column = state.get("target_column")
    evaluation = None
    if model is not None and df is not None and target_column in df.columns:
        evaluation = evaluate_model(task_type, model, df, target_column)
    state["evaluation"] = evaluation
    
    # Note: evaluation_agent.py already handles MLflow logging internally
    
    return state


def summarize_node(state: AgentState) -> AgentState:
    if state.get("progress_callback"):
        state["progress_callback"]("Generating Summary")
    
    """
    We only stash a placeholder here; the rich payload and final markdown
    are built in app.py where we also know plot paths, etc.
    """
    try:
        # Minimal inline summary, useful if app.py opts to use it
        mini_payload = {
            "dataset_name": "dataset",
            "n_rows": int(state["df"].shape[0]) if state.get("df") is not None else None,
            "n_cols": int(state["df"].shape[1]) if state.get("df") is not None else None,
            "target": state.get("target_column"),
            "profile": state.get("profile") or {},
            "metrics": state.get("evaluation") or {},
            "plan": state.get("steps") or [],
            "code_map": state.get("code_map") or {},
            "origin": getattr(state.get("model", object()), "_origin", "unknown"),
        }
        state["summary_md"] = generate_run_summary(mini_payload)
        
        # Note: summary_agent.py already handles MLflow logging internally
        
    except Exception:
        state["summary_md"] = "# AutoML Run Report\n\n_Summary unavailable._"
    
    return state


def create_graph():
    workflow = StateGraph(AgentState)
    
    # Add initialization and finalization nodes
    workflow.add_node("InitMLflow", _initialize_mlflow_run)
    workflow.add_node("Profile", profile_node)
    workflow.add_node("RetrieveRules", retrieval_node)
    workflow.add_node("Plan", planning_node)
    workflow.add_node("BuildPipeline", pipeline_node)
    workflow.add_node("Evaluate", evaluation_node)
    workflow.add_node("Summarize", summarize_node)
    workflow.add_node("FinalizeMLflow", _finalize_mlflow_run)

    # Update workflow edges
    workflow.set_entry_point("InitMLflow")
    workflow.add_edge("InitMLflow", "Profile")
    workflow.add_edge("Profile", "RetrieveRules")
    workflow.add_edge("RetrieveRules", "Plan")
    workflow.add_edge("Plan", "BuildPipeline")
    workflow.add_edge("BuildPipeline", "Evaluate")
    workflow.add_edge("Evaluate", "Summarize")
    workflow.add_edge("Summarize", "FinalizeMLflow")
    workflow.add_edge("FinalizeMLflow", END)
    
    return workflow.compile()