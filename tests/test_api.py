from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["dataset_records"] == 6


def test_run_ticket():
    response = client.post(
        "/api/run",
        json={
            "asset_type": "wind",
            "description": "Gearbox vibration increased with abnormal noise",
        },
    )
    assert response.status_code == 200
    assert response.json()["severity"] == "critical"
    assert response.json()["routing"]["priority_queue"] == "P1 - Immediate"


def test_reject_short_description():
    response = client.post("/api/run", json={"description": "bad"})
    assert response.status_code == 422
