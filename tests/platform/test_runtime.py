import json
from datetime import timedelta
from pathlib import Path
import uuid
import pytest
from sqlalchemy import create_engine,select,update
from src.platform import database as db
from src.platform.artifacts import MemoryArtifactStore
from src.platform.jobs import JobService,JobContext
from src.platform.registry import RegistryService
from src.platform.errors import ConflictError,JobCancelled,TransientError

@pytest.fixture
def database(tmp_path):
    engine=create_engine(f'sqlite:///{tmp_path}/platform.db')
    db.initialize(engine)
    return engine


def test_canonical_artifacts_are_immutable_checked_and_path_confined(tmp_path):
    store=MemoryArtifactStore();source=tmp_path/'source';source.mkdir()
    (source/'weights.pt').write_bytes(b'actual-content')
    manifest=store.upload_directory(source,'models/job')
    assert store.upload_directory(source,'models/job')==manifest
    assert (store.materialize(manifest['uri'],tmp_path/'cache')/'weights.pt').read_bytes()==b'actual-content'
    key=next(k for k in store.objects if '/files/' in k)
    store.objects[key]=b'corrupt'
    with pytest.raises(ValueError,match='checksum'):store.materialize(manifest['uri'],tmp_path/'bad')
    with pytest.raises(ValueError,match='Unsafe'):store.upload_directory(source,'../outside')


def test_api_restart_preserves_pending_and_running_jobs(database):
    first=JobService(database).submit('optimization',{'dataset_version':'v1'})
    restarted=JobService(database)
    assert restarted.get(first['id'])['status']=='pending'
    claimed=restarted.claim(first['id'],'worker-a')
    assert JobService(database).get(first['id'])['status']=='running'
    assert restarted.claim(first['id'],'worker-b') is None
    restarted.finish(first['id'],'worker-a',{'ok':True})
    assert restarted.get(first['id'])['status']=='completed'


def test_idempotency_and_outbox_redelivery_do_not_duplicate_execution(database):
    service=JobService(database)
    first=service.submit('optimization',{'dataset_version':'v1'},idempotency_key='same')
    assert service.submit('optimization',{'dataset_version':'v1'},idempotency_key='same')['id']==first['id']
    with pytest.raises(ConflictError):service.submit('optimization',{'dataset_version':'v2'},idempotency_key='same')
    sent=[];service.dispatch_pending(lambda *args:sent.append(args))
    assert sent[0][0]==first['id']
    assert service.claim(first['id'],'worker')
    assert service.claim(first['id'],'duplicate') is None
    assert service.dispatch_pending(lambda *args:sent.append(args))==0


def test_transient_retry_permanent_failure_and_expired_worker_recovery(database):
    service=JobService(database)
    first=service.submit('optimization',{'attempt':1})['id'];service.claim(first,'w')
    assert service.finish(first,'w',error=TransientError('broker interrupted'))['status']=='retrying'
    with database.begin() as conn:
        conn.execute(update(db.jobs).where(db.jobs.c.id==first).values(next_attempt_at=db.utcnow()-timedelta(seconds=1)))
    assert service.claim(first,'w2')
    assert service.finish(first,'w2',error=ValueError('infeasible'))['status']=='failed'
    second=service.submit('optimization',{'attempt':2})['id'];service.claim(second,'dead')
    with database.begin() as conn:
        conn.execute(update(db.jobs).where(db.jobs.c.id==second).values(lease_until=db.utcnow()-timedelta(seconds=1)))
    assert service.recover_expired()==1
    assert service.claim(second,'recovered')


def test_safe_cancellation_and_publication_boundary(database,tmp_path):
    service=JobService(database)
    pending=service.submit('optimization',{'phase':'pending'})['id']
    assert service.cancel(pending)['status']=='cancelled'
    active=service.submit('optimization',{'phase':'running'})['id'];record=service.claim(active,'worker')
    ctx=JobContext(record,database,MemoryArtifactStore(),tmp_path/'job')
    ctx.progress(50,'training')
    service.cancel(active)
    with pytest.raises(JobCancelled):ctx.check_cancelled()
    service.finish(active,'worker',error=JobCancelled())
    other=service.submit('optimization',{'phase':'publish'})['id'];record=service.claim(other,'worker')
    ctx=JobContext(record,database,MemoryArtifactStore(),tmp_path/'other');ctx.progress(98,'committing')
    with pytest.raises(ConflictError):service.cancel(other)


def test_missing_required_tracking_never_completes_registry_or_activates(database,tmp_path,monkeypatch):
    store=MemoryArtifactStore();service=JobService(database)
    job=service.submit('optimization',{'dataset_version':'v1'})['id'];record=service.claim(job,'w')
    ctx=JobContext(record,database,store,tmp_path/'job');ctx.temp_dir.mkdir()
    directory=ctx.temp_dir/'result';directory.mkdir();(directory/'metric.json').write_text('{}')
    monkeypatch.delenv('MLFLOW_TRACKING_URI',raising=False);monkeypatch.setenv('PLATFORM_MLFLOW_REQUIRED','true')
    registry=RegistryService(database,store)
    with pytest.raises(TransientError):registry.publish('optimization',{'dataset_version':'v1'},
        {'artifacts_dir':str(directory),'metrics':{'sharpe':1.}},ctx)
    assert registry.experiment(job)['status']=='publishing'
    assert not registry.list_models()
    assert service.get(job)['status']=='running'
    monkeypatch.setenv('PLATFORM_MLFLOW_REQUIRED','false')
    result=registry.publish('optimization',{'dataset_version':'v1'}, {'artifacts_dir':str(directory),'metrics':{'sharpe':1.}},ctx)
    assert result['status']=='completed' and result['artifact_uri'].startswith('s3://')


def test_failed_automatic_request_can_be_resubmitted_but_explicit_key_is_stable(database):
    service=JobService(database)
    first=service.submit('optimization',{'case':'automatic'})['id']
    service.claim(first,'w');service.finish(first,'w',error=ValueError('bad configuration'))
    assert service.submit('optimization',{'case':'automatic'})['id']!=first
    explicit=service.submit('optimization',{'case':'explicit'},idempotency_key='stable')['id']
    service.cancel(explicit)
    assert service.submit('optimization',{'case':'explicit'},idempotency_key='stable')['id']==explicit


def test_concurrent_delivery_has_one_claim_owner(database):
    from concurrent.futures import ThreadPoolExecutor
    service=JobService(database)
    identity=service.submit('optimization',{'case':'concurrent'})['id']
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda owner:service.claim(identity,owner),['a','b','c','d']))
    assert sum(result is not None for result in results)==1
    assert service.get(identity)['attempts']==1


def test_inference_reference_and_completed_experiment_commit_together(database,tmp_path,monkeypatch):
    from sqlalchemy import insert
    import src.platform.registry as module
    monkeypatch.setenv('PLATFORM_MLFLOW_REQUIRED','false')
    monkeypatch.delenv('MLFLOW_TRACKING_URI',raising=False)
    store=MemoryArtifactStore();service=JobService(database)
    model_id=str(uuid.uuid4())
    with database.begin() as conn:
        conn.execute(insert(db.models).values(id=model_id,experiment_id=str(uuid.uuid4()),family='mlp',ticker='A',
            dataset_version='v1',horizon=5,state='validated',accepted=True,artifact_uri='s3://original/manifest.json',
            content_hash='0'*64,created_at=db.utcnow()))
    config={'dataset_version':'v2','pinned_model':{'id':model_id}}
    identity=service.submit('model.infer',config)['id'];record=service.claim(identity,'worker')
    ctx=JobContext(record,database,store,tmp_path/'inference');ctx.temp_dir.mkdir()
    directory=ctx.temp_dir/'output';directory.mkdir();(directory/'prediction.csv').write_text('mu_pred\n0.001\n')
    output={'artifacts_dir':str(directory),'as_of':'2025-01-31','prediction_file':'prediction.csv','metrics':{}}
    original=module.audit
    def fail_inference(conn,action,*args,**kwargs):
        if action=='model.infer':raise RuntimeError('interrupted transaction')
        return original(conn,action,*args,**kwargs)
    monkeypatch.setattr(module,'audit',fail_inference)
    registry=RegistryService(database,store)
    with pytest.raises(RuntimeError,match='interrupted'):registry.publish('model.infer',config,output,ctx)
    assert registry.experiment(identity)['status']=='publishing'
    assert registry.get(model_id)['inference_json'] is None
    monkeypatch.setattr(module,'audit',original)
    result=registry.publish('model.infer',config,output,ctx)
    assert registry.get(model_id)['inference_dataset_version']=='v2'
    assert registry.get(model_id)['artifact_uri']==result['artifact_uri']
    assert registry.experiment(identity)['status']=='completed'
    assert registry.publish('model.infer',config,output,ctx)==result
