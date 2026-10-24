import socket
import sys
from datetime import timedelta
from sqlalchemy import select
from src.platform import database as db


def main():
    kind=sys.argv[1] if len(sys.argv)>1 else 'worker'
    hostname=socket.gethostname()
    with db.engine.connect() as conn:
        row=conn.execute(select(db.workers.c.id).where(db.workers.c.kind==kind,
            db.workers.c.id.like(f'{kind}:{hostname}:%'),db.workers.c.last_seen>db.utcnow()-timedelta(seconds=90))).first()
    raise SystemExit(0 if row else 1)

if __name__=='__main__':main()
