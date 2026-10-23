import numpy as np
import pandas as pd
from api.routers import diagnostics


def test_diagnostics_reports_actual_coverage_and_unconfirmed_services(monkeypatch):
    data=pd.DataFrame(np.random.default_rng(2).normal(0,.01,(260,2)),columns=['A','B'],index=pd.bdate_range('2024-01-01',periods=260))
    data.attrs['dataset_version']='v1'
    monkeypatch.setattr(diagnostics,'load_returns',lambda **kwargs:data)
    monkeypatch.setattr(diagnostics,'load_metadata',lambda **kwargs:pd.DataFrame({'ticker':['A','B'],'block':['OTHER','OTHER']}))
    from src.optimization import inputs_builder
    monkeypatch.setattr(inputs_builder,'_load_signals',lambda mode:pd.DataFrame())
    monkeypatch.delenv('MLFLOW_TRACKING_URI',raising=False);monkeypatch.delenv('MINIO_ENDPOINT',raising=False)
    report=diagnostics.serving_diagnostics()
    assert report['dataset']['version']=='v1' and report['dataset']['base_currency']=='USD'
    assert report['models']['lstm']['ai_assets']==0 and report['models']['lstm']['historical_assets']==2
    assert report['services']['mlflow']['status']=='not_configured'
    assert report['services']['github_actions']['status']=='not_probed'
