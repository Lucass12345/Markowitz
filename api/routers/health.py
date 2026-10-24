import os
from fastapi import APIRouter, Request
from starlette.responses import JSONResponse
from sqlalchemy import text
from app.db import engine

router=APIRouter()


@router.get('')
@router.get('/live')
def healthcheck():
    return {'status':'ok','service':'markowitz-platform-api'}


@router.get('/ready')
def readiness(request: Request):
    # Importing the first snapshot requires a ready API; data availability is
    # reported separately by the authenticated system diagnostics endpoint.
    checks={'database':False,'broker':False,'artifacts':False}
    try:
        with getattr(request.app.state,'platform_database',engine).connect() as conn:conn.execute(text('SELECT 1'))
        checks['database']=True
    except Exception:
        pass
    try:
        import redis
        redis.Redis.from_url(os.getenv('CELERY_BROKER_URL','redis://127.0.0.1:6379/0'),
                            socket_connect_timeout=1,socket_timeout=1).ping()
        checks['broker']=True
    except Exception:
        pass
    try:
        from api.routers.platform import store
        canonical=store(request)
        canonical.client.bucket_exists(canonical.bucket)
        checks['artifacts']=True
    except Exception:
        pass
    ready=all(checks.values())
    return JSONResponse({'status':'ready' if ready else 'not_ready','checks':checks},status_code=200 if ready else 503)
