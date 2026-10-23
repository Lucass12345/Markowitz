"""Bounded in-process execution for a single API worker; state survives restart."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import hashlib
from pathlib import Path
from threading import BoundedSemaphore, Lock
import time
import traceback
import uuid
from sqlalchemy import select, insert, update
from app.db import engine, init_db, optimization_jobs as jobs
from src.optimization.errors import OptimizationInputError
from src.optimization.provenance import code_version, json_bytes
from src.utils.serialization import to_builtin


def utcnow():return datetime.now(timezone.utc)


def context_key(payload):
    # Active DB model selection can change without changing CSVs: bypass model caches.
    if payload['model_type']!='classic' or payload.get('covariance_estimator')=='gjr_garch':
        return None
    from src.data.load_data import active_data_dir
    directory=active_data_dir()
    hasher=hashlib.sha256(json_bytes({'config':payload,'code':code_version(),'dataset_dir':str(directory)}))
    for name in ['returns.csv','assets_metadata.csv','manifest.json']:
        path=directory/name
        if path.exists():
            hasher.update(name.encode());hasher.update(path.read_bytes())
    return hasher.hexdigest()


class QueueFullError(RuntimeError):pass


class JobManager:
    def __init__(self,database=None,runner=None,max_workers=2,max_pending=4,key_builder=context_key):
        self.database=database if database is not None else engine
        if database is None:init_db()
        self.runner=runner
        self.key_builder=key_builder
        self.pool=ThreadPoolExecutor(max_workers=max_workers,thread_name_prefix='optimization')
        self.capacity=BoundedSemaphore(max_workers+max_pending)
        self.lock=Lock()
        with self.database.begin() as conn:
            conn.execute(update(jobs).where(jobs.c.status.in_(['pending','running'])).values(
                status='failed',completed_at=utcnow(),error='API restarted before job completed; resubmit to retry',error_status=503))

    def get(self,job_id):
        with self.database.begin() as conn:
            row=conn.execute(select(jobs).where(jobs.c.id==job_id)).mappings().first()
        return to_builtin(dict(row)) if row else None

    def submit(self,payload):
        # Serialize submission to deduplicate matching jobs in the same process.
        with self.lock:
            key=self.key_builder(payload)
            if key:
                with self.database.begin() as conn:
                    row=conn.execute(select(jobs).where(jobs.c.cache_key==key,
                        jobs.c.status.in_(['pending','running','completed']),jobs.c.created_at>utcnow()-timedelta(hours=1))
                        .order_by(jobs.c.created_at.desc()).limit(1)).mappings().first()
                if row:
                    return {**to_builtin(dict(row)),'reused':True}
            if not self.capacity.acquire(blocking=False):
                raise QueueFullError('Optimization worker capacity reached; retry later')
            job_id=str(uuid.uuid4())
            try:
                with self.database.begin() as conn:
                    conn.execute(insert(jobs).values(id=job_id,status='pending',created_at=utcnow(),payload_json=to_builtin(payload),cache_key=key))
                self.pool.submit(self._execute,job_id,payload,key)
            except Exception:
                self.capacity.release()
                raise
            return {'id':job_id,'status':'pending','reused':False}

    def _execute(self,job_id,payload,key):
        start=time.monotonic()
        try:
            with self.database.begin() as conn:
                conn.execute(update(jobs).where(jobs.c.id==job_id).values(status='running',started_at=utcnow()))
            if self.runner is None:
                from api.services.optimization_service import run_optimization
                response=run_optimization(payload)
            else:
                response=self.runner(payload)
            # Only reuse if all dependencies still match the original request.
            cache_key=key if key and self.key_builder(payload)==key else None
            changes={'status':'completed','result_json':to_builtin(response),'cache_key':cache_key}
        except Exception as exc:
            changes={'status':'failed','error':str(exc),'technical_error':traceback.format_exc(),
                     'error_status':422 if isinstance(exc,OptimizationInputError) else 500}
        finally:
            try:
                with self.database.begin() as conn:
                    conn.execute(update(jobs).where(jobs.c.id==job_id).values(**changes,completed_at=utcnow(),duration_seconds=time.monotonic()-start))
            finally:
                self.capacity.release()

    def close(self):
        self.pool.shutdown(wait=True)


_manager=None
_manager_lock=Lock()


def get_manager():
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager=JobManager()
    return _manager
