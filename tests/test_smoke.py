# tests/test_smoke.py
import pytest
import pandas as pd
import os
from pathlib import Path

def test_imports():
    """Test that all critical modules can be imported."""
    try:
        import streamlit
        import langchain
        import langgraph
        import chromadb
        import sklearn
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_agents_importable():
    """Test that all agent modules can be imported."""
    try:
        from agents.graph_orchestrator import create_graph
        from agents.profile_agent import profile_dataset
        from agents.planning_agent import plan_from_rules
        from agents.pipeline_builder import build_pipeline
        assert True
    except ImportError as e:
        pytest.fail(f"Agent import failed: {e}")

def test_graph_creation():
    """Test that LangGraph can be created."""
    try:
        from agents.graph_orchestrator import create_graph
        app = create_graph()
        assert app is not None
    except Exception as e:
        pytest.fail(f"Graph creation failed: {e}")

def test_sample_data_processing():
    """Test basic data processing with sample data."""
    try:
        from agents.profile_agent import profile_dataset
        
        # Create sample data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Test profiling
        profile = profile_dataset.invoke({
            "df": df,
            "target_column": "target"
        })
        
        assert 'target_type' in profile
        assert 'shape' in profile
        assert profile['shape'] == (5, 3)
        
    except Exception as e:
        pytest.fail(f"Data processing test failed: {e}")

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
def test_rule_ingestion():
    """Test that rules can be ingested (requires OpenAI API key)."""
    try:
        from ingest_rules import fewshot_snippets
        
        sample_profile = {
            "target_type": "classification",
            "shape": (100, 5)
        }
        
        snippet = fewshot_snippets(sample_profile)
        assert snippet is not None
        assert len(snippet) > 0
        assert "Pipeline" in snippet
        
    except Exception as e:
        pytest.fail(f"Rule ingestion test failed: {e}")

def test_requirements_file_exists():
    """Test that requirements.txt exists and is valid."""
    req_file = Path("requirements.txt")
    assert req_file.exists(), "requirements.txt not found"
    
    content = req_file.read_text()
    required_packages = [
        "streamlit", "pandas", "scikit-learn", 
        "langchain", "langgraph", "chromadb"
    ]
    
    for package in required_packages:
        assert package in content, f"Missing required package: {package}"

if __name__ == "__main__":
    pytest.main([__file__])