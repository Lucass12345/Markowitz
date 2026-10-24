"""Unmocked Streamlit -> API -> worker verification inside the CI stack."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from app.api_client import PlatformAPI
from app.ui.components import SECTIONS


def run(report_path='/integration/ui-report.json'):
    api=PlatformAPI(timeout=30)
    active=api.get('/system')['active_dataset']['version']
    app=AppTest.from_file('app/Optimizador_Markowitz.py').run(timeout=60)
    for section in SECTIONS:
        app.sidebar.selectbox[0].select(section).run(timeout=60)
        assert not app.exception,[(item.message) for item in app.exception]
    app.sidebar.selectbox[0].select('Model Laboratory').run(timeout=60)
    def widget(kind,label):return next(item for item in getattr(app,kind) if item.label==label)
    widget('selectbox','Familia de modelo').select('mlp').run(timeout=60)
    widget('text_input','Tickers del experimento').set_value('AAPL').run(timeout=60)
    widget('text_input','Inicio de entrenamiento').set_value('full').run(timeout=60)
    widget('text_input','Capas ocultas').set_value('4').run(timeout=60)
    widget('number_input','Iteraciones máximas').set_value(10).run(timeout=60)
    widget('button','Lanzar entrenamiento').click().run(timeout=60)
    assert not app.exception,[(item.message) for item in app.exception]
    tracked=dict(app.session_state['tracked_jobs'])
    assert len(tracked)==1,'UI must dispatch exactly one durable job'
    identity=next(iter(tracked))
    deadline=time.monotonic()+240
    while True:
        job=api.get(f'/jobs/{identity}')
        if job['status'] in {'completed','failed','cancelled'}:break
        assert time.monotonic()<deadline,'UI training job timed out'
        time.sleep(1)
    assert job['status']=='completed',job.get('error')
    assert job['payload_json']['dataset_version']==active
    assert job['payload_json']['tickers']==['AAPL']
    assert job['payload_json']['family']=='mlp'
    assert job['attempts']>=1 and job['started_at'] and job['completed_at']
    result=api.get(f'/experiments/{identity}')
    assert result['artifact_uri'].startswith('s3://') and result['mlflow_run_id']
    assert any(item['path'].endswith('_mlp_pipeline.joblib') for item in result['artifacts'])
    app.run(timeout=60)
    assert not app.exception,[(item.message) for item in app.exception]
    assert app.session_state['research_results']['experiment']['experiment_id']==identity
    report={'unmocked':True,'sections':SECTIONS,'ui_dispatched_job':identity,'dataset_version':active,
            'job_status':job['status'],'artifact_uri':result['artifact_uri'],'mlflow_run_id':result['mlflow_run_id']}
    Path(report_path).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':run()
