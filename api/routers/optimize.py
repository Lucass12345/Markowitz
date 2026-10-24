from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, Request

from api.schemas.optimize import OptimizeRequest, OptimizeResponse
from src.platform.compat import optimization_request as run_optimization
from src.optimization.errors import OptimizationInputError

router = APIRouter()


@router.post("", response_model=OptimizeResponse)
def optimize_portfolio(payload: OptimizeRequest, response_status: Response) -> OptimizeResponse:
    try:
        response = run_optimization(payload.model_dump(), save_to_db=True, track_to_mlflow=True)
        if response["status"] in {"pending", "running", "retrying"}:
            response_status.status_code = 202
        return OptimizeResponse(**response)
    except OptimizationInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=getattr(exc,'status_code',422),detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503,detail='Required platform service unavailable') from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Optimization failed; consult server logs") from exc

@router.get("/metadata")
def optimization_metadata(request: Request):
    from api.routers.platform import dataset_detail
    return dataset_detail('active',request)
