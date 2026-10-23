"""Execute Streamlit scripts with deterministic backend replies, without network/services."""
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import requests
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def reply(payload):return SimpleNamespace(ok=True,status_code=200,json=lambda:payload,raise_for_status=lambda:None)


def test_optimizer_submits_and_displays_completed_job(monkeypatch):
    from app import db
    monkeypatch.setattr(db,'init_db',lambda:None)
    stats={'return_annual':.1,'vol_annual':.2,'sharpe_annual':.5}
    result={'frontier':[{'return_annual':.1,'vol_annual':.2}],
            **{name:{'stats':stats} for name in ['minvar','maxsharpe']},
            'blocks':{'stats':stats,'weights_df':[{'ticker':'A','weight':1.}],
                      'exposure_by_block':[{'block':'OTHER','target':1.,'achieved':1.,'diff':0.}]}}
    response={'result':result,'run_id':12,'mlflow_run_id':None,'mlflow_error':None}
    def get(url=None,**kwargs):
        if url.endswith('/optimize/metadata'):
            return reply({'dataset_version':'fixture','blocks':{'OTHER':1},'default_targets':{'OTHER':1.},'legacy_targets':{'OTHER':1.}})
        if '/jobs/' in url:return reply({'status':'completed','result_json':response})
        return reply({'status':'ok'})
    monkeypatch.setattr(requests,'get',get)
    submitted=[]
    monkeypatch.setattr(requests,'post',lambda url=None,**kwargs:submitted.append(kwargs['json']) or reply({'id':'job-123'}))
    app=AppTest.from_file(str(ROOT/'app/Optimizador_Markowitz.py')).run(timeout=15)
    assert not app.exception
    button=next(item for item in app.sidebar.button if item.label=='Correr optimización')
    button.click().run(timeout=15)
    assert not app.exception
    assert submitted[0]['targets']=={'OTHER':1.}
    assert app.session_state['last_run_id']==12
    assert any('sin registro confirmado' in value.value for value in app.info)


def test_diagnostic_page_reports_unconfirmed_sync(monkeypatch):
    monkeypatch.setattr(requests,'get',lambda *args,**kwargs:reply({'dataset':{'version':'v1'},'models':{},'services':{'mlflow':{'status':'not_configured'}},'tracking':None}))
    app=AppTest.from_file(str(ROOT/'app/pages/06_Diagnostico.py')).run(timeout=10)
    assert not app.exception
    assert len(app.json)==3
