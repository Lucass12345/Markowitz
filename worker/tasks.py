"""Celery only hosts execution; all state and business logic live in domain services."""
import os
import resource
import socket
import time
from celery.exceptions import SoftTimeLimitExceeded
from worker.celery_app import app
from src.platform.jobs import JobService,JobContext
from src.platform.errors import LeaseLost


@app.task(name='markowitz.execute_job',acks_late=True,reject_on_worker_lost=True)
def execute_job(job_id):
    from src.platform.database import initialize
    from src.platform.operations import execute_operation
    initialize()
    service=JobService()
    record=service.claim(job_id,owner=f'{socket.gethostname()}:{os.getpid()}:{time.time_ns()}')
    if record is None:
        return {'duplicate_or_terminal':True}
    context=JobContext(record)
    started=time.monotonic()
    error=result=None
    try:
        from threadpoolctl import threadpool_limits
        threads=max(1,int(os.getenv('PLATFORM_CPU_THREADS','1')))
        threadpool_limits(limits=threads)
        if record['kind'] in {'model.train','backtest'}:
            import torch
            torch.set_num_threads(threads)
        result=execute_operation(record['kind'],record['payload_json'],context)
    except SoftTimeLimitExceeded:
        error=TimeoutError('Worker soft time limit exceeded')
    except Exception as exc:
        error=exc
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        resources={'wall_seconds':time.monotonic()-started,'max_rss_kb':usage.ru_maxrss,
                   'cpu_user_seconds':usage.ru_utime,'cpu_system_seconds':usage.ru_stime}
        try:
            service.finish(job_id,record['lease_owner'],result,error,resources)
        except LeaseLost:
            pass
        context.close()
    return {'job_id':job_id}


from celery.signals import worker_ready

@worker_ready.connect
def start_heartbeat(**kwargs):
    from threading import Thread
    from worker.scheduler import heartbeat
    from src.platform.database import initialize
    initialize()
    def pulse():
        while True:
            try:
                heartbeat('worker',capabilities={'queue':os.getenv('PLATFORM_WORKER_QUEUE','cpu'),
                    'cpu_threads':int(os.getenv('PLATFORM_CPU_THREADS','1'))})
            except Exception:
                pass
            time.sleep(20)
    Thread(target=pulse,daemon=True,name='worker-heartbeat').start()
