"""Worker operation dispatch and snapshot pinning. HTTP only queues these calls."""
import json
from pathlib import Path
from src.platform import database as db
from src.platform.jobs import JobService,engine_for
from src.platform.registry import RegistryService
from src.platform.errors import NotFoundError
from src.utils.serialization import to_builtin
from src.optimization.provenance import code_version


def pin_request(kind,payload,database=None,store=None):
    from src.platform.datasets import DatasetService
    from src.platform.universes import UniverseService
    payload=dict(payload)
    database=engine_for(database)
    datasets=DatasetService(store=store,database=database)
    dataset=None
    if kind not in {'dataset.import','reconcile'}:
        try:
            dataset=datasets.resolve(payload.get('dataset_version'))
        except (ValueError,NotFoundError):
            if kind!='dataset.refresh':raise
        else:payload['dataset_version']=dataset['version']
    if payload.get('universe_id'):
        payload['universe_snapshot']=UniverseService(database=database).get(payload['universe_id'])
    if kind=='dataset.refresh' and not payload.get('tickers') and not payload.get('universe_id'):
        if dataset is not None:
            # Scheduled updates preserve the published universe, including
            # custom blocks; the historical preset is only for first bootstrap.
            assets=dataset['manifest_json'].get('assets') or []
            if not assets:raise ValueError('Canonical dataset has no registered universe metadata')
            payload['universe_snapshot']={'source_dataset_version':dataset['version'],'assets_json':[
                {'ticker':asset['ticker'],'block':asset['block'],'enabled':True} for asset in assets]}
        else:
            universe=UniverseService(database=database).original_preset()
            payload.update(universe_id=universe['id'],universe_snapshot=universe)
    registry=RegistryService(database=database,store=store)
    if kind=='optimization':
        from src.platform.optimization import _mode
        mode,estimator=_mode(payload)
        family='lstm' if mode=='ai_returns_lstm' else 'mlp' if mode=='ai_returns' else None
        vol=estimator=='gjr_garch' or mode=='ai_volatility'
        tickers=payload.get('tickers')
        if payload.get('universe_snapshot'):
            tickers=[a['ticker'] for a in payload['universe_snapshot']['assets_json'] if a.get('enabled',True)]
        for field,ids_field,model_family in [('pinned_return_models','return_model_ids',family),
                                             ('pinned_volatility_models','volatility_model_ids','garch' if vol else None)]:
            identities=payload.get(ids_field)
            records=[registry.get(identity) for identity in identities] if identities is not None else registry.resolve_active(model_family,tickers=tickers,dataset_version=payload.get('dataset_version')) if model_family else []
            payload[field]=records
    if kind in {'model.infer','model.activate'}:
        payload['pinned_model']=registry.get(payload['model_id'])
    return to_builtin(payload)


def submit(kind,payload,actor='service',key=None,database=None,store=None):
    pinned=pin_request(kind,payload,database,store)
    return JobService(database).submit(kind,pinned,actor,key,pinned={'code':code_version()})


def execute_operation(kind,config,context):
    from src.platform.datasets import DatasetService
    registry=RegistryService(database=context.database,store=context.store)
    if kind=='dataset.import':
        return {'result':DatasetService(store=context.store,database=context.database).import_legacy(context)}
    if kind=='dataset.activate':
        context.progress(95,'committing','Activating verified dataset reference')
        return {'result':DatasetService(store=context.store,database=context.database).activate(context.dataset_version,actor=context.actor)}
    if kind=='dataset.refresh':
        return {'result':DatasetService(store=context.store,database=context.database).refresh(config,context)}
    if kind=='model.activate':
        context.progress(90,'committing','Validating artifacts before model activation')
        return {'result':registry.activate(config['model_id'],context.actor)}
    # Resume a publication that uploaded a complete bundle before tracking/SQL failed.
    try: previous=registry.experiment(context.job_id)
    except NotFoundError:previous=None
    if previous and previous['status']=='completed':return previous['result_json']
    if previous and previous.get('artifact_uri'):
        root=context.store.materialize(previous['artifact_uri'],context.temp_dir/'resume')
        result=json.loads((root/'platform_result.json').read_text())['output']
        result['artifacts_dir']=str(root)
    elif kind=='model.train':
        from src.platform.models import ModelService
        result=ModelService(store=context.store,database=context.database).train(config['family'],config,context)
    elif kind=='model.infer':
        from src.platform.models import ModelService
        dataset=DatasetService(store=context.store,database=context.database)
        snapshot=dataset.materialize(context.dataset_version,context.temp_dir/'dataset')
        result=ModelService(store=context.store,database=context.database).infer(config['pinned_model'],snapshot,context)
    elif kind=='optimization':
        from src.platform.optimization import OptimizationService
        result=OptimizationService().run(config,context)
    elif kind=='backtest':
        from src.platform.backtesting import BacktestService
        result=BacktestService().run(config,context)
    elif kind=='reconcile':
        return {'result':{'expired_leases_recovered':JobService(context.database).recover_expired()}}
    else:raise ValueError('Unsupported worker operation')
    published=registry.publish(kind,config,result,context)
    return published
