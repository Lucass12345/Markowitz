"""Observed serving state, distinct from synchronization claims."""
import os
from pathlib import Path
import requests
from fastapi import APIRouter
from sqlalchemy import select,text
from app.db import engine,optimization_run_provenance
from src.config import block_configuration
from src.data.load_data import active_data_dir,load_metadata,load_returns
from src.optimization.inputs_builder import build_model_inputs
from src.utils.serialization import to_builtin

router=APIRouter()


def probe(url):
    if not url:return {'status':'not_configured'}
    if not url.startswith(('https://','http://')):return {'status':'configured_unprobed'}
    try:
        response=requests.get(url,timeout=(1,2),allow_redirects=False)
        return {'status':'reachable' if response.ok else 'http_error','http_status':response.status_code}
    except requests.RequestException:
        return {'status':'unreachable'}


def serving_diagnostics():
    report={'services':{},'models':{},'tracking':None}
    try:
        directory=active_data_dir()
        returns=load_returns(data_dir=directory)
        metadata=load_metadata(data_dir=directory)
        report['dataset']={'version':returns.attrs['dataset_version'],'as_of':str(returns.index.max()),
                           'base_currency':'USD','n_assets':len(returns.columns),'n_dates':len(returns),
                           'asset_observations':returns.notna().sum().to_dict(),**block_configuration(metadata)}
        for name,mode in [('lstm','ai_returns_lstm'),('garch','ai_volatility')]:
            try:
                mu,_,_=build_model_inputs(mode,returns)
                diagnostic=mu.attrs['input_diagnostics']
                signals=diagnostic['signals']
                report['models'][name]={'status':'compatible','ai_assets':sum(s['source']=='ai' for s in signals),
                    'historical_assets':sum(s['source']=='historical' for s in signals),'signals':signals,
                    'excluded_assets':diagnostic['excluded_assets'],'prediction_policy':diagnostic['prediction_policy']}
            except (ValueError,FileNotFoundError,ConnectionError) as exc:
                report['models'][name]={'status':'unavailable_or_incompatible','reason':str(exc)}
    except (ValueError,OSError) as exc:
        report['dataset']={'status':'unavailable','reason':str(exc)}
    try:
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
            report['services']['database']={'status':'reachable'}
            row=conn.execute(select(optimization_run_provenance).order_by(optimization_run_provenance.c.run_id.desc()).limit(1)).mappings().first()
            report['tracking']=dict(row) if row else None
    except Exception:
        report['services'].setdefault('database',{'status':'unreachable'})
    mlflow=os.getenv('MLFLOW_TRACKING_URI')
    endpoint=os.getenv('MINIO_ENDPOINT')
    report['services']['mlflow']=probe(f'{mlflow.rstrip("/")}/health' if mlflow else None)
    scheme='https' if os.getenv('MINIO_SECURE','false').lower()=='true' else 'http'
    report['services']['minio']=probe(f'{scheme}://{endpoint}/minio/health/live' if endpoint else None)
    report['services']['github_actions']={'status':'not_probed','note':'Artifacts on GitHub do not imply activation in this runtime'}
    return to_builtin(report)


@router.get('')
def diagnostics():return serving_diagnostics()
