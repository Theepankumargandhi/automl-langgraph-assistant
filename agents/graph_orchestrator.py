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
from agents.summary_agent import generate_run_summary  # <-- new


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
    summary_md: str  # <-- new


def profile_node(state: AgentState) -> AgentState:
    profile = profile_dataset.invoke({
        "df": state["df"],
        "target_column": state["target_column"]
    })
    state["profile"] = profile
    return state


def retrieval_node(state: AgentState) -> AgentState:
    chroma_client = Client()
    rules, fallback = retrieve_rules(state["profile"], chroma_client)
    state["rules"] = rules
    state["fallback"] = fallback
    return state


def planning_node(state: AgentState) -> AgentState:
    if state.get("fallback"):
        steps = plan_from_rules(["Default plan for classification/regression on in-memory df."])
    else:
        steps = plan_from_rules(state.get("rules", []))
    state["steps"] = steps
    return state


def pipeline_node(state: AgentState) -> AgentState:
    steps = state.get("steps", [])
    code_map = generate_code_for_steps(steps, state)
    try:
        df_out, model = build_pipeline(state["df"], state["profile"], code_map)
        state["df"] = df_out
        state["model"] = model
        state["code_map"] = code_map
    except Exception as e:
        state["pipeline_error"] = str(e)
    return state


def evaluation_node(state: AgentState) -> AgentState:
    profile = state.get("profile", {}) or {}
    task_type = profile.get("target_type")
    model = state.get("model", None)
    df = state.get("df", None)
    target_column = state.get("target_column")
    evaluation = None
    if model is not None and df is not None and target_column in df.columns:
        evaluation = evaluate_model(task_type, model, df, target_column)
    state["evaluation"] = evaluation
    return state


def summarize_node(state: AgentState) -> AgentState:
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
    except Exception:
        state["summary_md"] = "# AutoML Run Report\n\n_Summary unavailable._"
    return state


def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("Profile", profile_node)
    workflow.add_node("RetrieveRules", retrieval_node)
    workflow.add_node("Plan", planning_node)
    workflow.add_node("BuildPipeline", pipeline_node)
    workflow.add_node("Evaluate", evaluation_node)
    workflow.add_node("Summarize", summarize_node)

    workflow.set_entry_point("Profile")
    workflow.add_edge("Profile", "RetrieveRules")
    workflow.add_edge("RetrieveRules", "Plan")
    workflow.add_edge("Plan", "BuildPipeline")
    workflow.add_edge("BuildPipeline", "Evaluate")
    workflow.add_edge("Evaluate", "Summarize")
    workflow.add_edge("Summarize", END)
    return workflow.compile()
