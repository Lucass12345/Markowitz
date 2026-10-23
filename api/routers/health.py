from fastapi import APIRouter
from starlette.responses import JSONResponse
from sqlalchemy import text
from app.db import engine
from src.data.load_data import active_data_dir,assert_usd_dataset

router=APIRouter()


@router.get('')
@router.get('/live')
def healthcheck():
    return {'status':'ok','service':'markowitz-platform-api'}


@router.get('/ready')
def readiness():
    checks={'database':False,'dataset':False}
    try:
        with engine.connect() as conn:conn.execute(text('SELECT 1'))
        checks['database']=True
    except Exception:
        pass
    try:
        directory=active_data_dir()
        assert_usd_dataset(directory)
        checks['dataset']=all((directory/name).is_file() for name in ['returns.csv','assets_metadata.csv'])
    except Exception:
        pass
    ready=all(checks.values())
    return JSONResponse({'status':'ready' if ready else 'not_ready','checks':checks},status_code=200 if ready else 503)
