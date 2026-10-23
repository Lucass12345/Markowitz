"""Exercise the real FastAPI request and exception boundary without persistence."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.routers import optimize
from src.optimization.errors import ConstraintFeasibilityError

PAYLOAD = {"model_type": "classic", "targets": {"AR": .2, "US": .5, "FUNDS_EQUITY": .1, "FUNDS_BONDS": .1, "GOLD": .1}}


def client():
    app = FastAPI()
    app.include_router(optimize.router, prefix="/optimize")
    return TestClient(app)


def test_invalid_targets_return_422_before_backend(monkeypatch):
    monkeypatch.setattr(optimize, "run_optimization", lambda *a, **k: pytest.fail("Invalid request reached backend"))
    response = client().post("/optimize", json={**PAYLOAD, "targets": {**PAYLOAD["targets"], "AR": -.1}})
    assert response.status_code == 422


def test_infeasible_capacity_returns_422(monkeypatch):
    def fail(*args, **kwargs):
        raise ConstraintFeasibilityError("Block AR target exceeds available capacity")
    monkeypatch.setattr(optimize, "run_optimization", fail)
    response = client().post("/optimize", json=PAYLOAD)
    assert response.status_code == 422
    assert "capacity" in response.json()["detail"]


def test_unexpected_runtime_failure_stays_500(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("numerical solver failed")
    monkeypatch.setattr(optimize, "run_optimization", fail)
    response = client().post("/optimize", json=PAYLOAD)
    assert response.status_code == 500
