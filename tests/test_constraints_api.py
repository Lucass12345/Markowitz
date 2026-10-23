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


def test_real_synthetic_optimization_and_capacity_contract(monkeypatch,tmp_path):
    import numpy as np
    import pandas as pd
    from src.optimization import optimizer_service as core
    from api.services.optimization_service import run_optimization
    data=pd.DataFrame(np.random.default_rng(22).normal(.001,.01,(30,3)),index=pd.bdate_range('2024-01-01',periods=30),columns=list('ABC'))
    data.attrs['dataset_version']='synthetic'
    monkeypatch.setattr(core,'load_returns',lambda **kwargs:data.copy())
    monkeypatch.setattr(core,'load_metadata',lambda **kwargs:pd.DataFrame({'ticker':list('ABC'),'block':['OTHER']*3}))
    monkeypatch.setenv('OPTIMIZATION_SNAPSHOT_DIR',str(tmp_path))
    monkeypatch.setattr(optimize,'run_optimization',lambda payload,**kwargs:run_optimization(payload,save_to_db=False,track_to_mlflow=False))
    payload={**PAYLOAD,'targets':{'OTHER':1},'min_history':20,'min_overlap':10,'max_weight':.6,'n_frontier_points':10}
    response=client().post('/optimize',json=payload)
    assert response.status_code==200,response.text
    result=response.json()['result']
    assert abs(sum(item['weight'] for item in result['blocks']['weights_df'])-1)<1e-7
    assert result['provenance']['snapshot_id']
    response=client().post('/optimize',json={**payload,'max_weight':.2})
    assert response.status_code==422
    assert 'capacity' in response.json()['detail']
