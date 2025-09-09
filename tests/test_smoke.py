# tests/test_smoke.py
import pytest

def test_basic_imports():
    """Test that basic packages can be imported."""
    try:
        import pandas
        import numpy
        import sklearn
        import streamlit
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_requirements_exist():
    """Test that requirements.txt exists."""
    import pathlib
    req_file = pathlib.Path("requirements.txt")
    assert req_file.exists()

if __name__ == "__main__":
    pytest.main([__file__])