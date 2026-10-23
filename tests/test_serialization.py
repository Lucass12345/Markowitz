import importlib
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.utils.serialization import to_builtin


def test_non_finite_nested_numpy_pandas_and_json_contract():
    value={'frame':pd.DataFrame({'x':[np.nan,np.inf,-np.inf]}),'series':pd.Series({'a':np.float64(2.),'b':pd.NA}),
           'array':np.array([1,np.nan]),'bool':np.bool_(True),'date':pd.Timestamp('2024-01-01'),'nat':pd.NaT}
    converted=to_builtin(value)
    json.dumps(converted,allow_nan=False)
    assert converted['frame']==[{'x':None}]*3
    assert converted['bool'] is True and converted['nat'] is None
    app=FastAPI()
    @app.get('/data')
    def data():return to_builtin(value)
    response=TestClient(app).get('/data')
    assert response.status_code==200 and response.json()==converted


def test_legacy_service_import_has_no_side_effects():
    service=importlib.import_module('api.services.optimizer_service')
    assert callable(service.run_optimization)
