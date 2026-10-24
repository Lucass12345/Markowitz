"""Central presentation API. Algorithms run only through the durable worker queue."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from fastapi import APIRouter,Request,Query,HTTPException
from fastapi.responses import Response
from sqlalchemy import select,text
from src.platform import database as db
from src.platform.jobs import JobService
from src.platform.registry import RegistryService
from src.platform.operations import submit
from src.platform.artifacts import get_store,safe_relative
from src.platform.errors import NotFoundError
from src.utils.serialization import to_builtin
from api.schemas.platform import (RefreshRequest,TrainingRequest,OptimizationRequest,BacktestRequest,
                                 UniverseRequest,UniverseUpdate,InferenceRequest,JobRequest)

router=APIRouter()


def database(request):return getattr(request.app.state,'platform_database',db.engine)
def store(request):return getattr(request.app.state,'platform_store',None) or get_store()
def actor(request):return getattr(request.state,'actor','service')
def registry(request):return RegistryService(database(request),getattr(request.app.state,'platform_store',None))
def dataset_service(request):
    from src.platform.datasets import DatasetService
    return DatasetService(database=database(request),store=getattr(request.app.state,'platform_store',None))

def enqueue(request,kind,payload):
    return submit(kind,payload,actor(request),request.headers.get('Idempotency-Key'),database(request),getattr(request.app.state,'platform_store',None))


@router.get('/datasets')
def datasets(request:Request):return {'items':dataset_service(request).list_versions()}

@router.post('/datasets/import-legacy',status_code=202)
def import_legacy(request:Request):return enqueue(request,'dataset.import',{})

@router.post('/datasets/refresh',status_code=202)
def refresh(payload:RefreshRequest,request:Request):return enqueue(request,'dataset.refresh',payload.model_dump(exclude_none=True))

@router.post('/datasets/rollback',status_code=202)
def dataset_rollback(request:Request):
    with database(request).connect() as conn:
        identity=conn.execute(select(db.active_refs.c.version_id).where(db.active_refs.c.kind=='dataset',db.active_refs.c.key=='previous')).scalar()
    if not identity:raise NotFoundError('No previous dataset')
    return enqueue(request,'dataset.activate',{'dataset_version':identity})

@router.post('/datasets/{version}/activate',status_code=202)
def activate_dataset(version:str,request:Request):return enqueue(request,'dataset.activate',{'dataset_version':version})

@router.get('/datasets/{version}')
def dataset_detail(version:str,request:Request):
    from src.config import block_configuration
    record=dataset_service(request).resolve(version)
    with tempfile.TemporaryDirectory(prefix='dataset-view-') as temporary:
        snapshot=dataset_service(request).materialize(record['id'],temporary)
        metadata=snapshot.load_metadata()
        return to_builtin({**record,'assets':metadata.to_dict('records'),**block_configuration(metadata),
                           'quality':record['manifest_json'].get('quality',{})})

@router.get('/datasets/{version}/data')
def dataset_data(version:str,request:Request,kind:str=Query('returns',pattern='^(prices|returns)$'),limit:int=Query(1000,ge=1,le=10000)):
    with tempfile.TemporaryDirectory(prefix='dataset-view-') as temporary:
        snapshot=dataset_service(request).materialize(version,temporary)
        frame=snapshot.load_prices() if kind=='prices' else snapshot.load_returns()
        frame=frame.tail(limit).rename_axis('date').reset_index()
        return {'items':to_builtin(frame)}

@router.get('/universes/presets/original')
def original_universe(request:Request):
    from src.platform.universes import UniverseService
    return UniverseService(database=database(request)).original_preset()

@router.get('/universes')
def universes(request:Request):
    from src.platform.universes import UniverseService
    return {'items':UniverseService(database=database(request)).list()}

@router.post('/universes',status_code=201)
def create_universe(payload:UniverseRequest,request:Request):
    from src.platform.universes import UniverseService
    return UniverseService(database=database(request)).create(payload.name,payload.assets,payload.config,actor=actor(request))

@router.patch('/universes/{universe_id}')
def update_universe(universe_id:str,payload:UniverseUpdate,request:Request):
    from src.platform.universes import UniverseService
    return UniverseService(database=database(request)).update(universe_id,**payload.model_dump(exclude_none=True),actor=actor(request))

@router.get('/universes/{universe_id}')
def universe(universe_id:str,request:Request):
    from src.platform.universes import UniverseService
    return UniverseService(database=database(request)).get(universe_id)

@router.delete('/universes/{universe_id}')
def remove_universe(universe_id:str,request:Request):
    from src.platform.universes import UniverseService
    return UniverseService(database=database(request)).delete(universe_id,actor(request))

@router.post('/experiments',status_code=202)
def train(payload:TrainingRequest,request:Request):return enqueue(request,'model.train',payload.model_dump(exclude_none=True))

@router.get('/experiments')
def experiments(request:Request,dataset_version:str|None=None):return _experiment_list(request,'model.train',dataset_version)

@router.get('/experiments/{identity}')
def experiment(identity:str,request:Request):return _experiment_detail(request,identity)

@router.get('/models')
def models(request:Request,family:str|None=None,ticker:str|None=None,state:str|None=None,dataset_version:str|None=None):
    rows=registry(request).list_models(family,ticker,state)
    if dataset_version:
        rows=[r for r in rows if (r.get('inference_dataset_version') or r['dataset_version'])==dataset_version]
    return {'items':rows}

@router.get('/models/{identity}')
def model(identity:str,request:Request):
    result=registry(request).get(identity)
    manifest=store(request).read_json(result['model_artifact_uri'])
    result['artifacts']=[{'id':identity,'path':name,'filename':Path(name).name} for name in manifest['files']]
    return result

@router.post('/models/{identity}/activate',status_code=202)
def activate_model(identity:str,request:Request):return enqueue(request,'model.activate',{'model_id':identity})

@router.post('/models/{identity}/archive')
def archive_model(identity:str,request:Request):return registry(request).archive(identity,actor(request))

@router.post('/models/{identity}/infer',status_code=202)
def infer(identity:str,payload:InferenceRequest,request:Request):return enqueue(request,'model.infer',{'model_id':identity,**payload.model_dump(exclude_none=True)})

@router.post('/models/{family}/{ticker}/rollback',status_code=202)
def rollback_model(family:str,ticker:str,request:Request):
    with database(request).connect() as conn:
        identity=conn.execute(select(db.active_refs.c.version_id).where(db.active_refs.c.kind==f'model:{family}:previous',db.active_refs.c.key==ticker)).scalar()
    if not identity:raise NotFoundError('No previous model')
    return enqueue(request,'model.activate',{'model_id':identity})

@router.post('/optimizations/validate')
def validate_optimization(payload:OptimizationRequest,request:Request):
    from src.optimization.optimize_blocks import validate_optimizer_inputs
    import numpy as np
    import pandas as pd
    config=payload.model_dump(exclude_none=True)
    with tempfile.TemporaryDirectory(prefix='validate-') as temporary:
        snapshot=dataset_service(request).materialize(config.get('dataset_version'),temporary)
        meta=snapshot.load_metadata()
        tickers=list(meta.ticker)
        if config.get('universe_id'):
            from src.platform.universes import UniverseService
            universe=UniverseService(database=database(request)).get(config['universe_id'])
            assets=[a for a in universe['assets_json'] if a.get('enabled',True)]
            tickers=[a['ticker'] for a in assets]
            for a in assets:
                if a.get('block'):meta.loc[meta.ticker==a['ticker'],'block']=a['block']
        elif config.get('tickers') is not None:
            tickers=config['tickers']
            missing=set(tickers)-set(meta.ticker)
            if missing:raise ValueError(f'Universe tickers absent from dataset: {sorted(missing)}')
        validate_optimizer_inputs(pd.Series(np.zeros(len(tickers)),index=tickers),np.eye(len(tickers)),tickers,meta,config['targets'],config['max_weight'])
    return {'valid':True,'dataset_version':snapshot.version,'message':'Static capacity valid; worker checks observed risk and model compatibility'}

@router.post('/optimizations',status_code=202)
def optimize(payload:OptimizationRequest,request:Request):return enqueue(request,'optimization',payload.model_dump(exclude_none=True))

@router.get('/optimizations')
def optimizations(request:Request,dataset_version:str|None=None):return _experiment_list(request,'optimization',dataset_version)

@router.get('/optimizations/{identity}')
def optimization(identity:str,request:Request):return _experiment_detail(request,identity)

@router.post('/backtests',status_code=202)
def backtest(payload:BacktestRequest,request:Request):return enqueue(request,'backtest',payload.model_dump(exclude_none=True))

@router.get('/backtests')
def backtests(request:Request,dataset_version:str|None=None):return _experiment_list(request,'backtest',dataset_version)

@router.get('/backtests/{identity}')
def backtest_result(identity:str,request:Request):return _experiment_detail(request,identity)

@router.get('/jobs')
def jobs(request:Request):
    rows=JobService(database(request)).list_jobs()
    for row in rows:row.pop('technical_error',None)
    return {'items':rows}

@router.post('/jobs',status_code=202)
def submit_job(payload:JobRequest,request:Request):
    schemas={'dataset.refresh':RefreshRequest,'model.train':TrainingRequest,'optimization':OptimizationRequest,'backtest':BacktestRequest}
    data=schemas[payload.kind](**payload.payload).model_dump(exclude_none=True) if payload.kind in schemas else payload.payload
    if payload.kind=='model.infer':
        identity=data.get('model_id')
        if not isinstance(identity,str) or not identity:raise ValueError('model.infer requires model_id')
        data={'model_id':identity,**InferenceRequest(**{k:v for k,v in data.items() if k!='model_id'}).model_dump(exclude_none=True)}
    if payload.kind=='dataset.import' and data:raise ValueError('dataset.import accepts no payload')
    return enqueue(request,payload.kind,{**data,'timeout_seconds':payload.timeout_seconds,'max_attempts':payload.max_attempts})

@router.get('/jobs/{identity}')
def job(identity:str,request:Request):return JobService(database(request)).get(identity)

@router.get('/jobs/{identity}/logs')
def logs(identity:str,request:Request,after:int=Query(0,ge=0)):return {'items':JobService(database(request)).logs(identity,after)}

@router.post('/jobs/{identity}/cancel')
def cancel(identity:str,request:Request):return JobService(database(request)).cancel(identity,actor(request))

@router.get('/artifacts/{identity}')
def artifact(identity:str,request:Request,path:str='platform_result.json'):
    safe_relative(path)
    with database(request).connect() as conn:
        uri=conn.execute(select(db.experiments.c.artifact_uri).where(db.experiments.c.id==identity)).scalar()
        if not uri:uri=conn.execute(select(db.datasets.c.artifact_uri).where(db.datasets.c.id==identity)).scalar()
        if not uri:uri=conn.execute(select(db.models.c.artifact_uri).where(db.models.c.id==identity)).scalar()
    if not uri:raise NotFoundError('Artifact reference not found')
    canonical=store(request)
    manifest=canonical.read_json(uri)
    if path not in manifest['files']:raise NotFoundError('Artifact file not found')
    base=uri.removesuffix('/manifest.json')
    content=canonical.read_bytes(f'{base}/files/{path}')
    info=manifest['files'][path]
    if hashlib.sha256(content).hexdigest()!=info['sha256'] or len(content)!=info['size']:
        raise ValueError('Artifact integrity failed')
    return Response(content,media_type='application/octet-stream',headers={'Content-Disposition':f'attachment; filename="{Path(path).name}"'})

@router.get('/system')
def system(request:Request):
    from datetime import timedelta
    from src.platform.universes import UniverseService
    services={}
    connection=database(request)
    try:
        with connection.connect() as conn:conn.execute(text('SELECT 1'))
        services['postgresql']={'status':'reachable'}
    except Exception:services['postgresql']={'status':'unreachable'}
    try:
        canonical=store(request);canonical.client.bucket_exists(canonical.bucket)
        services['minio']={'status':'reachable','canonical':True}
    except Exception:services['minio']={'status':'unavailable','canonical':True}
    try:
        import redis
        redis.Redis.from_url(os.getenv('CELERY_BROKER_URL','redis://127.0.0.1:6379/0'),socket_connect_timeout=1,socket_timeout=1).ping()
        services['redis']={'status':'reachable'}
    except Exception:services['redis']={'status':'unavailable'}
    from api.routers.diagnostics import probe
    uri=os.getenv('MLFLOW_TRACKING_URI')
    services['mlflow']=probe(f'{uri.rstrip("/")}/health' if uri and uri.startswith(('http:','https:')) else uri)
    try:active=dataset_service(request).resolve()
    except ValueError:active=None
    with connection.connect() as conn:
        worker_rows=[to_builtin(dict(r)) for r in conn.execute(select(db.workers)).mappings()]
    from datetime import datetime, timezone
    recent=[]
    for worker in worker_rows:
        seen=datetime.fromisoformat(worker['last_seen'])
        if seen.tzinfo is None:seen=seen.replace(tzinfo=timezone.utc)
        if seen>db.utcnow()-timedelta(seconds=90):recent.append(worker)
    latest=JobService(connection).list_jobs(20)
    return {'active_dataset':active,'models':registry(request).list_models(state='active'),'services':services,
            'workers':worker_rows,'jobs':[{k:v for k,v in row.items() if k!='technical_error'} for row in latest],
            'alerts':(['No active canonical dataset: import or refresh first'] if not active else [])+(['No recent worker heartbeat'] if not recent else []),
            'presets':{'lstm':{'seq_len':40,'forecast_horizon':5,'target_aggregation':'mean','epochs':50},
                       'mlp':{'forecast_horizon':5,'target_aggregation':'mean'},'garch':{'forecast_horizon':5,'distribution':'t'}}}


def _experiment_list(request,kind,version=None):
    rows=registry(request).list_experiments(kind)
    return {'items':[r for r in rows if not version or r['dataset_version']==version]}

def _experiment_detail(request,identity):
    result=registry(request).experiment(identity)
    uri=result.get('artifact_uri')
    if uri:
        manifest=store(request).read_json(uri)
        result['artifacts']=[{'id':identity,'path':name,'filename':Path(name).name} for name in manifest['files']]
    return result
