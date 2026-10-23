"""Administrative API credentials; opt-in locally, mandatory in production."""
import hmac
import os
from starlette.responses import JSONResponse


def security_config():
    production=os.getenv('APP_ENV','development')=='production'
    admin=os.getenv('MARKOWITZ_API_KEY','')
    reader=os.getenv('MARKOWITZ_READ_API_KEY','')
    origins=[value.strip() for value in os.getenv('CORS_ALLOWED_ORIGINS','http://localhost:8501,http://localhost:8502').split(',') if value.strip()]
    if production and (len(admin)<32 or admin.startswith('REPLACE_') or not origins or any('*' in origin for origin in origins)):
        raise RuntimeError('Production requires MARKOWITZ_API_KEY of at least 32 characters and explicit CORS_ALLOWED_ORIGINS')
    if reader and (len(reader)<32 or reader==admin):
        raise RuntimeError('Read-only key must be distinct and at least 32 characters')
    return production,admin,reader,origins


def authorize(request,admin,reader):
    if request.method=='OPTIONS' or request.url.path in {'/health','/health/live','/health/ready'}:
        return None
    if not admin:
        return None
    supplied=request.headers.get('X-API-Key','')
    if hmac.compare_digest(supplied.encode(),admin.encode()):
        return None
    if reader and hmac.compare_digest(supplied.encode(),reader.encode()):
        if request.method=='GET':
            return None
        return JSONResponse({'detail':'Read-only credential cannot modify resources'},status_code=403)
    return JSONResponse({'detail':'Valid X-API-Key required'},status_code=401)
