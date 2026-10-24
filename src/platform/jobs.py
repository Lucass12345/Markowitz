"""Durable SQL jobs/outbox, leased claims and cooperative cancellation.

Redis transports IDs; SQL owns state and deduplicates delivery. API restart never
changes an existing job. An independent scheduler redelivers pending work and
recovers expired leases. Side effects are published by immutable domain bundles.
"""
from datetime import timedelta
import hashlib
import os
from pathlib import Path
import tempfile
import time
import traceback
import uuid
from sqlalchemy import select, insert, update, and_, or_
from sqlalchemy.exc import IntegrityError, OperationalError
from src.platform import database as db
from src.platform.errors import ConflictError, NotFoundError, JobCancelled, LeaseLost, TransientError
from src.optimization.provenance import json_bytes, code_version
from src.utils.serialization import to_builtin

TERMINAL = {'completed', 'failed', 'cancelled'}
KINDS = {'dataset.refresh','dataset.import','dataset.activate','model.train','model.infer','model.activate',
         'optimization','backtest','reconcile'}


def engine_for(database=None):
    return getattr(database, 'engine', database) if database is not None else db.engine


def event(conn, job_id, stage, message, level='info'):
    conn.execute(insert(db.job_events).values(job_id=job_id, at=db.utcnow(), stage=stage,
                                             message=str(message)[:8000], level=level))


class JobService:
    def __init__(self, database=None, publisher=None):
        self.database = engine_for(database)
        self.publisher = publisher

    def submit(self, kind, payload, actor='service', idempotency_key=None, pinned=None):
        if kind not in KINDS:
            raise ValueError('Unknown job kind')
        payload = to_builtin(dict(payload))
        context = to_builtin(pinned or {})
        request_hash = hashlib.sha256(json_bytes({'kind':kind,'payload':payload,'actor':actor,'context':context})).hexdigest()
        # Explicit keys are scoped to actor; default identity includes all pinned dependencies.
        key = hashlib.sha256(f'{actor}:{idempotency_key}'.encode()).hexdigest() if idempotency_key else request_hash
        queue = 'gpu' if payload.get('parameters',payload).get('device') == 'cuda' else 'cpu'
        timeout = int(payload.get('timeout_seconds', 7200 if kind in {'model.train','backtest'} else 900))
        if not 1 <= timeout <= 86400:
            raise ValueError('Job timeout must be between 1 and 86400 seconds')
        job_id = str(uuid.uuid4())
        try:
            with self.database.begin() as conn:
                existing = conn.execute(select(db.jobs).where(db.jobs.c.idempotency_key == key)).mappings().first()
                if existing:
                    if existing['request_hash'] != request_hash:
                        raise ConflictError('Idempotency key already belongs to another configuration')
                    if idempotency_key or existing['status'] not in {'failed','cancelled'}:
                        return {**to_builtin(dict(existing)), 'reused':True}
                    # A failed automatic cache entry must not prevent recovery.
                    # Explicit client keys continue to identify exactly one job.
                    conn.execute(update(db.jobs).where(db.jobs.c.id==existing['id'],
                        db.jobs.c.idempotency_key==key).values(idempotency_key=None))
                conn.execute(insert(db.jobs).values(id=job_id,kind=kind,status='pending',actor=actor,
                    payload_json=payload,context_json=context,request_hash=request_hash,idempotency_key=key,
                    progress=0,stage='queued',attempts=0,max_attempts=int(payload.get('max_attempts',3)),
                    timeout_seconds=timeout,cancel_requested=False,created_at=db.utcnow()))
                conn.execute(insert(db.outbox).values(job_id=job_id,queue=queue,deliveries=0))
                event(conn,job_id,'queued','Durably queued in SQL; awaiting worker')
        except IntegrityError:
            with self.database.connect() as conn:
                row = conn.execute(select(db.jobs).where(db.jobs.c.idempotency_key == key)).mappings().one()
            if row['request_hash'] != request_hash:
                raise ConflictError('Idempotency key conflict')
            return {**to_builtin(dict(row)),'reused':True}
        return {'id':job_id,'status':'pending','progress':0,'stage':'queued','reused':False}

    def get(self, job_id, technical=False):
        with self.database.connect() as conn:
            row = conn.execute(select(db.jobs).where(db.jobs.c.id == job_id)).mappings().first()
        if not row:
            raise NotFoundError('Job not found')
        result = to_builtin(dict(row))
        if not technical:
            result.pop('technical_error',None)
        return result

    def list_jobs(self, limit=100):
        with self.database.connect() as conn:
            return [to_builtin(dict(r)) for r in conn.execute(select(db.jobs).order_by(db.jobs.c.created_at.desc()).limit(limit)).mappings()]

    def logs(self, job_id, after=0):
        self.get(job_id)
        with self.database.connect() as conn:
            return [to_builtin(dict(r)) for r in conn.execute(select(db.job_events).where(
                db.job_events.c.job_id == job_id, db.job_events.c.id > after).order_by(db.job_events.c.id).limit(1000)).mappings()]

    def cancel(self, job_id, actor='service'):
        with self.database.begin() as conn:
            row = conn.execute(select(db.jobs).where(db.jobs.c.id == job_id).with_for_update()).mappings().first()
            if not row:
                raise NotFoundError('Job not found')
            if row['status'] in TERMINAL:
                return self.get(job_id)
            if row['stage'] == 'committing':
                raise ConflictError('Publication is committing atomically; cancellation is no longer safe')
            pending = row['status'] in {'pending','retrying'}
            conn.execute(update(db.jobs).where(db.jobs.c.id == job_id).values(cancel_requested=True,
                **({'status':'cancelled','stage':'cancelled','completed_at':db.utcnow()} if pending else {})))
            event(conn,job_id,'cancel_requested',f'Cancellation requested by {actor}')
        return self.get(job_id)

    def claim(self, job_id, owner=None):
        owner = owner or str(uuid.uuid4())
        now = db.utcnow()
        with self.database.connect() as conn:
            row = conn.execute(select(db.jobs).where(db.jobs.c.id == job_id)).mappings().first()
        if not row or row['cancel_requested'] or row['status'] not in {'pending','retrying'}:
            return None
        if row['next_attempt_at'] and _aware(row['next_attempt_at']) > now:
            return None
        with self.database.begin() as conn:
            changed = conn.execute(update(db.jobs).where(db.jobs.c.id == job_id,
                db.jobs.c.status.in_(['pending','retrying']),db.jobs.c.cancel_requested == False,
                db.jobs.c.attempts == row['attempts']).values(status='running',stage='starting',
                attempts=row['attempts']+1,lease_owner=owner,lease_until=now+timedelta(seconds=row['timeout_seconds']+60),
                started_at=now,completed_at=None,error=None,error_status=None))
            if changed.rowcount != 1:
                return None
            event(conn,job_id,'starting',f'Worker claimed attempt {row["attempts"]+1}')
        return self.get(job_id,technical=True)

    def retry(self, job_id, actor='service'):
        """Resume the same pinned publication after an operator fixes a failure."""
        with self.database.begin() as conn:
            row=conn.execute(select(db.jobs).where(db.jobs.c.id==job_id).with_for_update()).mappings().first()
            if not row:raise NotFoundError('Job not found')
            if row['status'] not in {'failed','cancelled'}:
                raise ConflictError('Only failed or cancelled jobs can be retried')
            if row['idempotency_key'] is None:
                raise ConflictError('This job was superseded by a resubmission; retry the newer job')
            conn.execute(update(db.jobs).where(db.jobs.c.id==job_id).values(status='pending',stage='queued',
                cancel_requested=False,progress=0,lease_owner=None,lease_until=None,next_attempt_at=None,
                completed_at=None,error=None,error_status=None,max_attempts=row['attempts']+3))
            conn.execute(update(db.outbox).where(db.outbox.c.job_id==job_id).values(last_dispatched_at=None))
            event(conn,job_id,'queued',f'Manual retry by {actor}; original pinned inputs retained')
        return self.get(job_id)

    def finish(self, job_id, owner, result=None, error=None, resources=None):
        now = db.utcnow()
        with self.database.begin() as conn:
            row = conn.execute(select(db.jobs).where(db.jobs.c.id == job_id).with_for_update()).mappings().one()
            if row['status'] != 'running' or row['lease_owner'] != owner:
                raise LeaseLost('Worker lease no longer owns this job')
            fields = {'lease_until':None,'resource_json':to_builtin(resources or {}),
                      'duration_seconds':(now-_aware(row['started_at'])).total_seconds()}
            if row['cancel_requested'] or isinstance(error,JobCancelled):
                fields.update(status='cancelled',stage='cancelled',completed_at=now)
            elif error is not None:
                transient = is_transient(error)
                retry = transient and row['attempts'] < row['max_attempts']
                fields.update(status='retrying' if retry else 'failed',stage='retry_wait' if retry else 'failed',
                    error=_safe_error(error),technical_error=_redact(''.join(traceback.format_exception(error)))[-24000:],
                    error_class='transient' if transient else 'permanent',error_status=getattr(error,'status_code',503 if transient else 422 if isinstance(error,ValueError) else 500),
                    next_attempt_at=now+timedelta(seconds=min(300,2**row['attempts'])) if retry else None,
                    completed_at=None if retry else now)
            else:
                fields.update(status='completed',stage='completed',progress=100,result_json=to_builtin(result),completed_at=now)
            conn.execute(update(db.jobs).where(db.jobs.c.id == job_id).values(**fields))
            event(conn,job_id,fields['stage'],fields.get('error') or fields['status'], 'error' if error else 'info')
        return self.get(job_id)

    def recover_expired(self):
        now = db.utcnow()
        with self.database.begin() as conn:
            rows = conn.execute(select(db.jobs).where(db.jobs.c.status == 'running',db.jobs.c.lease_until < now).with_for_update(skip_locked=True)).mappings().all()
            for row in rows:
                status = 'cancelled' if row['cancel_requested'] else 'retrying' if row['attempts'] < row['max_attempts'] else 'failed'
                conn.execute(update(db.jobs).where(db.jobs.c.id == row['id']).values(status=status,stage='lease_expired',
                    lease_owner=None,lease_until=None,next_attempt_at=now,
                    completed_at=None if status=='retrying' else now,error='Worker lease expired; recovery scheduled',error_status=503))
                event(conn,row['id'],'lease_expired',f'Worker lost; recovered as {status}','warning')
            return len(rows)

    def dispatch_pending(self, sender=None):
        if sender is None:
            from worker.celery_app import app
            sender = lambda job_id,queue,timeout: app.send_task('markowitz.execute_job',args=[job_id],task_id=job_id,
                queue=queue,soft_time_limit=timeout,time_limit=timeout+30,retry=False)
        now = db.utcnow()
        count = 0
        with self.database.begin() as conn:
            rows = conn.execute(select(db.jobs,db.outbox.c.queue).join(db.outbox,db.outbox.c.job_id==db.jobs.c.id).where(
                db.jobs.c.status.in_(['pending','retrying']),db.jobs.c.cancel_requested == False,
                or_(db.jobs.c.next_attempt_at == None, db.jobs.c.next_attempt_at <= now),
                or_(db.outbox.c.last_dispatched_at == None,db.outbox.c.last_dispatched_at < now-timedelta(seconds=30)))
                .with_for_update(skip_locked=True).limit(50)).mappings().all()
            for row in rows:
                sender(row['id'],row['queue'],row['timeout_seconds'])
                conn.execute(update(db.outbox).where(db.outbox.c.job_id==row['id']).values(
                    last_dispatched_at=now,deliveries=db.outbox.c.deliveries+1))
                count += 1
        return count


def _aware(value):
    from datetime import timezone
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _safe_error(exc):
    return _redact(str(exc))[:4000]


def _redact(message):
    for name in ('MINIO_SECRET_KEY','MINIO_ACCESS_KEY','MARKOWITZ_API_KEY','MARKOWITZ_READ_API_KEY',
                 'DATABASE_URL','CELERY_BROKER_URL','MLFLOW_TRACKING_PASSWORD','MLFLOW_TRACKING_TOKEN'):
        value = os.getenv(name)
        if value:
            message = message.replace(value,'[redacted]')
    import re
    return re.sub(r'(\w+://)[^\s/@:]+:[^\s/@]+@',r'\1[redacted]@',message)


def is_transient(exc):
    from minio.error import S3Error
    return isinstance(exc,(TransientError,ConnectionError,TimeoutError,OperationalError)) or (
        isinstance(exc,S3Error) and exc.code in {'SlowDown','ServiceUnavailable','InternalError','RequestTimeout'})


class JobContext:
    def __init__(self, record, database=None, store=None, directory=None):
        self.database = engine_for(database)
        self.job_id, self.actor = record['id'], record['actor']
        self.owner = record['lease_owner']
        self.dataset_version = record['payload_json'].get('dataset_version')
        self.record = record
        self._store = store
        self._directory = tempfile.TemporaryDirectory(prefix=f'markowitz-{self.job_id}-') if directory is None else None
        self.temp_dir = Path(directory or self._directory.name)
        self.deadline = time.monotonic()+record['timeout_seconds']

    @property
    def store(self):
        if self._store is None:
            from src.platform.artifacts import get_store
            self._store = get_store()
        return self._store

    def check_cancelled(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError('Configured job timeout exceeded')
        with self.database.connect() as conn:
            row = conn.execute(select(db.jobs.c.status,db.jobs.c.cancel_requested,db.jobs.c.lease_owner).where(db.jobs.c.id==self.job_id)).one()
        if row.cancel_requested:
            raise JobCancelled('Job cancellation acknowledged at a safe checkpoint')
        if row.status != 'running' or row.lease_owner != self.owner:
            raise LeaseLost('Worker lease was superseded')

    def progress(self, percent, stage, message=None):
        self.check_cancelled()
        if not 0 <= float(percent) <= 100:
            raise ValueError('Progress must be between 0 and 100')
        with self.database.begin() as conn:
            changed = conn.execute(update(db.jobs).where(db.jobs.c.id==self.job_id,db.jobs.c.status=='running',
                db.jobs.c.lease_owner==self.owner,db.jobs.c.cancel_requested==False).values(
                    progress=float(percent),stage=str(stage)[:80]))
            if changed.rowcount != 1:
                raise JobCancelled('Cancellation raced with progress checkpoint')
            if message:
                event(conn,self.job_id,stage,message)

    def close(self):
        if self._directory:
            self._directory.cleanup()
