from threading import Event
import time
import pytest
from sqlalchemy import create_engine, insert
from app.db import metadata,optimization_jobs
from api.services.jobs import JobManager,QueueFullError,utcnow
from src.optimization.errors import OptimizationInputError


def database(tmp_path):
    engine=create_engine(f'sqlite:///{tmp_path}/jobs.db',connect_args={'check_same_thread':False})
    metadata.create_all(engine)
    return engine


def finish(manager,job_id):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        row=manager.get(job_id)
        if row['status'] in {'completed','failed'}:return row
        time.sleep(.01)
    pytest.fail('Job did not finish')


def test_jobs_deduplicate_versioned_classic_config_and_persist_duration(tmp_path):
    calls=[]
    manager=JobManager(database(tmp_path),runner=lambda payload:calls.append(payload) or {'ok':True},key_builder=lambda payload:payload['version'])
    try:
        first=manager.submit({'version':'v1'})
        assert finish(manager,first['id'])['duration_seconds']>=0
        assert manager.submit({'version':'v1'})['id']==first['id']
        second=manager.submit({'version':'v2'});finish(manager,second['id'])
        assert len(calls)==2
    finally:manager.close()


def test_queue_bounded_errors_recoverable_and_restart_marks_interrupted(tmp_path):
    entered,release=Event(),Event()
    def slow(payload):
        entered.set();release.wait(5);raise OptimizationInputError('Infeasible block OTHER')
    db=database(tmp_path)
    manager=JobManager(db,runner=slow,max_workers=1,max_pending=0,key_builder=lambda payload:None)
    try:
        first=manager.submit({'model_type':'classic'});assert entered.wait(2)
        with pytest.raises(QueueFullError):manager.submit({})
        release.set()
        row=finish(manager,first['id'])
        assert row['error_status']==422 and 'OTHER' in row['error'] and row['technical_error']
    finally:release.set();manager.close()
    with db.begin() as conn:
        conn.execute(insert(optimization_jobs).values(id='interrupted',status='running',created_at=utcnow(),payload_json={}))
    restarted=JobManager(db,key_builder=lambda payload:None)
    try:assert restarted.get('interrupted')['status']=='failed'
    finally:restarted.close()


def test_job_http_contract_returns_202_and_polls_completion(tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import jobs
    manager=JobManager(database(tmp_path),runner=lambda payload:{'result':{'ok':True}},key_builder=lambda payload:None)
    monkeypatch.setattr(jobs,'get_manager',lambda:manager)
    app=FastAPI();app.include_router(jobs.router,prefix='/jobs')
    try:
        client=TestClient(app)
        response=client.post('/jobs',json={'model_type':'classic','targets':{'OTHER':1}})
        assert response.status_code==202
        job_id=response.json()['id'];finish(manager,job_id)
        status=client.get(f'/jobs/{job_id}')
        assert status.status_code==200 and status.json()['status']=='completed'
        assert 'technical_error' not in status.json()
        assert client.get('/jobs/missing').status_code==404
    finally:manager.close()
