"""Independent SQL outbox relay and daily dataset scheduler; no model execution."""
import os
import socket
import time
from datetime import timedelta
from sqlalchemy import select,insert,update
from src.platform import database as db
from src.platform.jobs import JobService,_aware
from src.platform.operations import submit


def heartbeat(kind,identity=None,capabilities=None):
    identity=identity or f'{kind}:{socket.gethostname()}:{os.getpid()}'
    now=db.utcnow()
    with db.engine.begin() as conn:
        changed=conn.execute(update(db.workers).where(db.workers.c.id==identity).values(last_seen=now,
                            capabilities_json=capabilities or {},resources_json={'pid':os.getpid()}))
        if not changed.rowcount:
            conn.execute(insert(db.workers).values(id=identity,kind=kind,last_seen=now,
                capabilities_json=capabilities or {},resources_json={'pid':os.getpid()}))


def schedule_due(database=None):
    if os.getenv('YAHOO_REFRESH_ENABLED','true').lower() != 'true':
        return
    database=database if database is not None else db.engine
    now=db.utcnow()
    with database.begin() as conn:
        if not conn.execute(select(db.schedules.c.id).where(db.schedules.c.id=='yahoo-daily')).scalar():
            conn.execute(insert(db.schedules).values(id='yahoo-daily',enabled=True,
                interval_seconds=int(os.getenv('YAHOO_REFRESH_INTERVAL_SECONDS','86400')),
                next_run_at=now+timedelta(seconds=30),config_json={'activate':True,'refresh_mode':'incremental'},updated_at=now))
        rows=conn.execute(select(db.schedules).where(db.schedules.c.enabled==True,db.schedules.c.next_run_at<=now)
                          .with_for_update(skip_locked=True)).mappings().all()
        due=[dict(r) for r in rows]
    for row in due:
        # A stable schedule instant provides deduplication across scheduler restarts.
        key=f'schedule:{row["id"]}:{_aware(row["next_run_at"]).isoformat()}'
        submit('dataset.refresh',row['config_json'],actor='scheduler',key=key,database=database)
        with database.begin() as conn:
            conn.execute(update(db.schedules).where(db.schedules.c.id==row['id'],db.schedules.c.next_run_at==row['next_run_at'])
                .values(next_run_at=now+timedelta(seconds=row['interval_seconds']),updated_at=now))


def main():
    db.initialize()
    service=JobService()
    while True:
        heartbeat('scheduler')
        try:
            service.recover_expired()
            schedule_due()
            service.dispatch_pending()
        except Exception as exc:
            # No credentials/connection strings in process logs.
            print(f'{{"event":"scheduler_retry","error_type":"{type(exc).__name__}"}}',flush=True)
        time.sleep(5)

if __name__=='__main__':main()
