# tests/test_smoke.py
def test_always_passes():
    """Basic test that always passes to verify CI/CD pipeline works."""
    assert True

def test_python_works():
    """Test basic Python functionality."""
    x = 1 + 1
    assert x == 2