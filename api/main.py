from contextlib import asynccontextmanager
import json
import logging
import time
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.platform import router as platform_router
from api.routers.diagnostics import router as diagnostics_router
from api.routers.jobs import router as jobs_router
from api.routers.health import router as health_router
from api.routers.optimize import router as optimize_router
from api.routers.runs import router as runs_router
from api.security import security_config,authorize

logger=logging.getLogger('markowitz.requests')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler=logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
logger.propagate=False


@asynccontextmanager
async def lifespan(app):
    from src.platform.database import initialize
    initialize(getattr(app.state,'platform_database',None))
    yield


def create_app():
    production,admin,reader,origins=security_config()
    app=FastAPI(title='Markowitz Platform API',version='2.0.0',lifespan=lifespan,
                docs_url=None if production else '/docs',openapi_url=None if production else '/openapi.json')

    @app.middleware('http')
    async def audit_and_auth(request,call_next):
        request_id=uuid.uuid4().hex
        supplied=request.headers.get('X-API-Key','')
        request.state.actor = "reader" if reader and supplied==reader else "admin" if admin else "development"
        start=time.monotonic()
        response=authorize(request,admin,reader)
        if response is None:
            response=await call_next(request)
        response.headers['X-Request-ID']=request_id
        logger.info(json.dumps({'event':'http_request','request_id':request_id,'method':request.method,
                                'path':request.url.path,'status':response.status_code,
                                'duration_seconds':round(time.monotonic()-start,6)}))
        return response

    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,
                       allow_methods=['GET','POST','PATCH','DELETE','OPTIONS'],allow_headers=['Content-Type','X-API-Key','Idempotency-Key'])
    app.include_router(health_router,prefix='/health',tags=['health'])
    app.include_router(optimize_router,prefix='/optimize',tags=['optimize'])
    app.include_router(runs_router,prefix='/runs',tags=['runs'])
    app.include_router(jobs_router,prefix='/jobs',tags=['jobs'])
    app.include_router(platform_router,prefix='/api/v1',tags=['platform'])
    app.include_router(diagnostics_router,prefix='/diagnostics',tags=['diagnostics'])
    from fastapi.responses import JSONResponse
    from src.platform.errors import PlatformError, ServiceUnavailableError
    async def domain_error(request,exc):
        return JSONResponse({'detail':str(exc)},status_code=getattr(exc,'status_code',422))
    app.add_exception_handler(PlatformError,domain_error)
    app.add_exception_handler(ValueError,domain_error)
    app.add_exception_handler(ServiceUnavailableError,domain_error)
    return app


app=create_app()
