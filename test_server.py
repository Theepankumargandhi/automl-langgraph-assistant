#test_server.py
import requests
def test_run_automm():
    url = "http://localhost:8000/run_automm"
    payload = {"target_column": "label"}
    response = requests.post(url, json=payload)

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)
    assert "message" in data
    assert isinstance(data["message"], str)
