from fastapi import APIRouter, HTTPException
from api.schemas.optimize import OptimizeRequest
from api.services.jobs import get_manager, QueueFullError

router=APIRouter()


@router.post('',status_code=202)
def submit_job(payload:OptimizeRequest):
    try:
        return get_manager().submit(payload.model_dump())
    except QueueFullError as exc:
        raise HTTPException(status_code=503,detail=str(exc),headers={'Retry-After':'5'}) from exc


@router.get('/{job_id}')
def job_status(job_id:str):
    result=get_manager().get(job_id)
    if result is None:
        raise HTTPException(status_code=404,detail='Job not found')
    # Tracebacks remain in the database for operator diagnosis.
    result.pop('technical_error',None)
    if result.get('error_status') == 500:
        result['error'] = 'Optimization failed; consult persisted technical error using the job ID'
    return result
