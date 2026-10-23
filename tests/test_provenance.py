import json
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from src.optimization.provenance import seal_snapshot, load_snapshot


def test_snapshot_survives_source_changes_and_detects_corruption(tmp_path):
    returns=pd.DataFrame({'A':[.01,-.02]},index=pd.bdate_range('2024-01-01',periods=2))
    returns.attrs['dataset_version']='v1'
    meta=pd.DataFrame({'ticker':['A'],'block':['OTHER']})
    result={'inputs':{'dataset_version':'v1','targets':{'OTHER':1}},'diagnostics':{'risk':{'signals':[]}},'metric':np.nan}
    prov=seal_snapshot(returns,meta,returns.mean(),returns.cov(),result,root=tmp_path)
    returns.iloc[:]=999
    assert load_snapshot(prov['snapshot_path'])['inputs']['dataset_version']=='v1'
    assert load_snapshot(prov['snapshot_path'])['metric'] is None
    path=tmp_path/prov['snapshot_id']/'mu.csv'
    path.write_text('overwritten')
    with pytest.raises(ValueError,match='integrity'):
        load_snapshot(prov['snapshot_path'])


def test_full_optimization_persists_full_config_and_service_links(tmp_path,monkeypatch):
    from src.optimization import optimizer_service as core
    from app import repository
    from app.db import metadata
    from api.services.optimization_service import run_optimization
    data=pd.DataFrame(np.random.default_rng(5).normal(.0005,.01,(30,3)),index=pd.bdate_range('2024-01-01',periods=30),columns=list('ABC'))
    data.attrs['dataset_version']='synthetic'
    monkeypatch.setattr(core,'load_returns',lambda **kwargs:data.copy())
    monkeypatch.setattr(core,'load_metadata',lambda **kwargs:pd.DataFrame({'ticker':list('ABC'),'block':['OTHER']*3}))
    monkeypatch.setenv('OPTIMIZATION_SNAPSHOT_DIR',str(tmp_path/'evidence'))
    monkeypatch.delenv('MLFLOW_TRACKING_URI',raising=False)
    monkeypatch.delenv('MINIO_ENDPOINT',raising=False)
    db=create_engine(f'sqlite:///{tmp_path}/history.db');metadata.create_all(db)
    monkeypatch.setattr(repository,'engine',db)
    monkeypatch.setattr(repository,'init_db',lambda:None)
    result=run_optimization({'model_type':'classic','targets':{'OTHER':1.},'max_weight':.6,'rf_annual_simple':0.,'n_frontier_points':10,'min_history':20,'min_overlap':10})
    row=repository.get_run_by_id(result['run_id'])
    provenance=row['provenance']['provenance_json']
    assert provenance['effective_configuration']['risk_config']['min_overlap']==10
    assert row['provenance']['tracking_json']['mlflow']['status']=='not_configured'
    assert provenance['code']['commit']
    assert load_snapshot(provenance['snapshot_path'])['blocks']['stats']['metrics_version']==2
    json.dumps(result,allow_nan=False)


def test_minio_failure_preserves_successful_mlflow_id(tmp_path,monkeypatch):
    from app import tracking
    class Run:
        info=SimpleNamespace(run_id='mlflow-123')
        def __enter__(self):return self
        def __exit__(self,*args):return False
    noop=lambda *args,**kwargs:None
    fake=SimpleNamespace(set_tracking_uri=noop,set_experiment=noop,start_run=lambda **kwargs:Run(),set_tags=noop,log_dict=noop,log_metrics=noop,log_artifacts=noop)
    monkeypatch.setitem(sys.modules,'mlflow',fake)
    monkeypatch.setenv('MLFLOW_TRACKING_URI','http://tracking.test')
    monkeypatch.setenv('MINIO_ENDPOINT','storage.test')
    (tmp_path/'manifest.json').write_text('{}')
    def fail(*args,**kwargs):raise ConnectionError('storage offline')
    monkeypatch.setattr(tracking,'upload_bytes',fail)
    result={key:{'stats':{'sharpe_daily':.1}} for key in ('minvar','maxsharpe','blocks')}
    result['provenance']={'snapshot_id':'immutable','snapshot_path':str(tmp_path)}
    status=tracking.track_run_details(4,None,{'mode':'classic'},result)
    assert status['mlflow']['run_id']=='mlflow-123' and status['mlflow']['status']=='completed'
    assert status['minio']['status']=='failed'
