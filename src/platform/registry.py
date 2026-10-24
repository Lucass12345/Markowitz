"""Model/experiment registry and immutable publication saga.

Objects and MLflow are verified before a SQL reference becomes visible. A retry
reuses the experiment/job identity; activation is a separate reversible mutation.
"""
import json
import os
from pathlib import Path
import tempfile
import uuid
from sqlalchemy import select,insert,update
from src.platform import database as db
from src.platform.jobs import engine_for
from src.platform.artifacts import safe_relative
from src.platform.errors import NotFoundError,ConflictError,TransientError
from src.optimization.provenance import json_bytes,code_version
from src.utils.serialization import to_builtin


def _row(record):
    result=dict(record)
    result['model_artifact_uri']=result['artifact_uri']
    result['model_run_id']=result['experiment_id']
    inference=result.get('inference_json') or {}
    if inference:
        result.update(inference_dataset_version=inference['dataset_version'],
                      inference_as_of=inference['as_of'],artifact_uri=inference['artifact_uri'],
                      prediction_file=inference['prediction_file'])
    return to_builtin(result)


def audit(conn,action,entity,actor,details=None):
    conn.execute(insert(db.audit).values(at=db.utcnow(),actor=actor,action=action,entity_id=entity,details_json=to_builtin(details or {})))


def set_ref(conn,kind,key,version_id,actor):
    result=conn.execute(update(db.active_refs).where(db.active_refs.c.kind==kind,db.active_refs.c.key==key)
                        .values(version_id=version_id,updated_at=db.utcnow(),actor=actor))
    if not result.rowcount:
        conn.execute(insert(db.active_refs).values(kind=kind,key=key,version_id=version_id,updated_at=db.utcnow(),actor=actor))


class RegistryService:
    def __init__(self,database=None,store=None):
        self.database=engine_for(database)
        self.store=store

    def get(self,model_id):
        with self.database.connect() as conn:
            record=conn.execute(select(db.models).where(db.models.c.id==model_id)).mappings().first()
        if not record:raise NotFoundError('Model not found')
        return _row(record)

    def list_models(self,family=None,ticker=None,state=None,dataset_version=None):
        query=select(db.models).order_by(db.models.c.created_at.desc())
        for name,value in [('family',family),('ticker',ticker),('state',state),('dataset_version',dataset_version)]:
            if value:query=query.where(db.models.c[name]==value)
        with self.database.connect() as conn:
            return [_row(row) for row in conn.execute(query.limit(2000)).mappings()]

    def resolve_active(self,family,tickers=None,dataset_version=None):
        with self.database.connect() as conn:
            refs=conn.execute(select(db.active_refs.c.version_id).where(db.active_refs.c.kind==f'model:{family}')).scalars().all()
        rows=[self.get(identity) for identity in refs]
        return [r for r in rows if (not tickers or r['ticker'] in tickers) and
                (not dataset_version or (r.get('inference_dataset_version') or r['dataset_version'])==dataset_version)]

    def activate(self,model_id,actor='service'):
        record=self.get(model_id)
        if not record.get('accepted') or record['state'] not in {'validated','active','archived'}:
            raise ConflictError('Only a model with accepted validation evidence may be activated')
        store=self.store
        if store is None:
            from src.platform.artifacts import get_store
            store=get_store()
        with tempfile.TemporaryDirectory(prefix='verify-model-') as tmp:
            store.materialize(record['model_artifact_uri'],tmp)
        kind=f'model:{record["family"]}'
        with self.database.begin() as conn:
            row=conn.execute(select(db.models).where(db.models.c.id==model_id).with_for_update()).mappings().one()
            if not row['accepted'] or row['state']=='rejected':raise ConflictError('Rejected model cannot become active')
            previous=conn.execute(select(db.active_refs.c.version_id).where(db.active_refs.c.kind==kind,
                                      db.active_refs.c.key==record['ticker']).with_for_update()).scalar()
            if previous!=model_id:
                if previous:
                    set_ref(conn,kind+':previous',record['ticker'],previous,actor)
                    conn.execute(update(db.models).where(db.models.c.id==previous).values(state='archived'))
                set_ref(conn,kind,record['ticker'],model_id,actor)
            conn.execute(update(db.models).where(db.models.c.id==model_id).values(state='active'))
            audit(conn,'model.activate',model_id,actor,{'previous':previous})
        return self.get(model_id)

    def archive(self,model_id,actor='service'):
        from sqlalchemy import delete
        record=self.get(model_id)
        with self.database.begin() as conn:
            conn.execute(update(db.models).where(db.models.c.id==model_id).values(state='archived'))
            conn.execute(delete(db.active_refs).where(db.active_refs.c.kind==f'model:{record["family"]}',
                         db.active_refs.c.key==record['ticker'],db.active_refs.c.version_id==model_id))
            audit(conn,'model.archive',model_id,actor)
        return self.get(model_id)

    def rollback(self,family,ticker,actor='service'):
        with self.database.connect() as conn:
            identity=conn.execute(select(db.active_refs.c.version_id).where(db.active_refs.c.kind==f'model:{family}:previous',
                                 db.active_refs.c.key==ticker)).scalar()
        if not identity:raise NotFoundError('No previous model version')
        return self.activate(identity,actor)

    def experiment(self,experiment_id):
        with self.database.connect() as conn:
            row=conn.execute(select(db.experiments).where(db.experiments.c.id==experiment_id)).mappings().first()
        if not row:raise NotFoundError('Experiment not found')
        return to_builtin(dict(row))

    def list_experiments(self,kind=None,limit=100):
        query=select(db.experiments).order_by(db.experiments.c.created_at.desc()).limit(limit)
        if kind:query=query.where(db.experiments.c.kind==kind)
        with self.database.connect() as conn:return [to_builtin(dict(row)) for row in conn.execute(query).mappings()]

    def publish(self,kind,config,result,context):
        # Completed experiment is the idempotent saga checkpoint.
        with self.database.connect() as conn:
            existing=conn.execute(select(db.experiments).where(db.experiments.c.id==context.job_id)).mappings().first()
        if existing and existing['status']=='completed':return existing['result_json']
        root=Path(result['artifacts_dir'])
        if not root.resolve().is_relative_to(Path(context.temp_dir).resolve()):
            raise ValueError('Worker artifacts must belong to its isolated temporary directory')
        context.check_cancelled()
        config=to_builtin(config)
        model_records=result.get('models',[])
        for record in model_records:
            safe_relative(record['relative_path'])
            if not (root/record['relative_path']).exists():raise ValueError('Required model weights/specification missing')
            if record.get('prediction_file'):
                safe_relative(record['prediction_file'])
                if not (root/record['prediction_file']).is_file():raise ValueError('Required model prediction artifact missing')
        sanitized={key:to_builtin(value) for key,value in result.items() if key not in {'artifacts_dir','predictions_path','model_dir','run_output_dir'}}
        (root/'platform_result.json').write_bytes(json_bytes({'job_id':context.job_id,'config':config,'code':code_version(),'output':sanitized}))
        with self.database.begin() as conn:
            if not existing:
                conn.execute(insert(db.experiments).values(id=context.job_id,job_id=context.job_id,kind=kind,
                    dataset_version=context.dataset_version,actor=context.actor,status='publishing',config_json=config,
                    code_json=code_version(),created_at=db.utcnow()))
            else:
                conn.execute(update(db.experiments).where(db.experiments.c.id==context.job_id).values(status='publishing'))
        context.progress(88,'uploading','Publishing complete immutable bundle to MinIO')
        manifest=context.store.upload_directory(root,f'experiments/{kind}/{context.job_id}')
        with tempfile.TemporaryDirectory(dir=context.temp_dir) as verification:
            context.store.materialize(manifest['uri'],verification)
        with self.database.begin() as conn:
            conn.execute(update(db.experiments).where(db.experiments.c.id==context.job_id).values(
                artifact_uri=manifest['uri'],content_hash=manifest['sha256'],sync_json={'minio':'verified','mlflow':'pending'}))
        mlflow_id=self._track(context,kind,config,result,root,manifest,existing)
        context.progress(98,'committing','Artifact integrity and required tracking verified; committing registry')
        published={**sanitized,'experiment_id':context.job_id,'artifact_uri':manifest['uri'],
                   'content_hash':manifest['sha256'],'mlflow_run_id':mlflow_id,'status':'completed'}
        with self.database.begin() as conn:
            for record in model_records:
                identity=str(uuid.uuid5(uuid.UUID(context.job_id),f'{record["family"]}:{record["ticker"]}'))
                values={key:to_builtin(record.get(key)) for key in ('family','ticker','horizon','features','preprocessing',
                    'validation_metrics','relative_path','prediction_file','training_cutoff','as_of')}
                values.update(id=identity,experiment_id=context.job_id,dataset_version=context.dataset_version,
                    state='validated' if record.get('accepted') else 'rejected',accepted=bool(record.get('accepted')),artifact_uri=manifest['uri'],
                    content_hash=manifest['sha256'],config_json=config,created_at=db.utcnow())
                exists=conn.execute(select(db.models.c.id).where(db.models.c.id==identity)).scalar()
                if not exists:conn.execute(insert(db.models).values(**values))
            if kind=='model.infer':
                identity=config['pinned_model']['id']
                inference={'dataset_version':context.dataset_version,'as_of':result['as_of'],
                    'artifact_uri':manifest['uri'],'prediction_file':result['prediction_file']}
                conn.execute(update(db.models).where(db.models.c.id==identity).values(inference_json=inference))
                audit(conn,'model.infer',identity,context.actor,{'experiment_id':context.job_id,**inference})
            conn.execute(update(db.experiments).where(db.experiments.c.id==context.job_id).values(status='completed',
                metrics_json=to_builtin(result.get('metrics',{})),result_json=to_builtin(published),mlflow_run_id=mlflow_id,
                sync_json={'minio':'verified','mlflow':'completed' if mlflow_id else 'explicitly_optional'},completed_at=db.utcnow()))
            audit(conn,f'{kind}.publish',context.job_id,context.actor,{'artifact_uri':manifest['uri']})
        return to_builtin(published)

    def _track(self,context,kind,config,result,root,manifest,existing):
        required=os.getenv('PLATFORM_MLFLOW_REQUIRED','true').lower()=='true'
        uri=os.getenv('MLFLOW_TRACKING_URI')
        if not uri:
            if required:raise TransientError('MLflow is required for publication but not configured')
            return None
        try:
            import mlflow
            client=mlflow.tracking.MlflowClient(tracking_uri=uri)
            name=os.getenv('MLFLOW_EXPERIMENT_NAME','markowitz-platform')
            experiment=client.get_experiment_by_name(name)
            if experiment is None:
                experiment_id=client.create_experiment(name)
            else:experiment_id=experiment.experiment_id
            run_id=existing.get('mlflow_run_id') if existing else None
            if not run_id:
                found=client.search_runs([experiment_id],filter_string=f"tags.platform_job_id = '{context.job_id}'",max_results=1)
                run_id=found[0].info.run_id if found else client.create_run(experiment_id,tags={
                    'platform_job_id':context.job_id,'dataset_version':context.dataset_version or '',
                    'kind':kind,'artifact_uri':manifest['uri'],'code_commit':code_version()['commit']}).info.run_id
            with self.database.begin() as conn:
                conn.execute(update(db.experiments).where(db.experiments.c.id==context.job_id).values(mlflow_run_id=run_id))
            for key,value in config.items():
                if value is not None:client.log_param(run_id,key,json.dumps(to_builtin(value),sort_keys=True)[:500])
            for key,value in _metric_leaves(to_builtin(result.get('metrics',{}))):client.log_metric(run_id,key,value)
            # MinIO is canonical. MLflow records the immutable URI plus reports.
            client.log_artifact(run_id,str(root/'platform_result.json'))
            client.set_terminated(run_id,status='FINISHED')
            return run_id
        except Exception as exc:
            if required:raise TransientError(f'MLflow publication incomplete: {type(exc).__name__}') from exc
            return None


def _metric_leaves(value,prefix=''):
    if isinstance(value,dict):
        for key,item in value.items():yield from _metric_leaves(item,f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(value,(int,float)) and not isinstance(value,bool):
        import re
        import hashlib
        safe=re.sub(r'[^a-zA-Z0-9_./ -]','_',prefix)
        if safe!=prefix:safe=f'{safe[:225]}_{hashlib.sha256(prefix.encode()).hexdigest()[:8]}'
        yield safe[:240],float(value)
