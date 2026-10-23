from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.optimize import OptimizeRequest, OptimizeResponse
from api.services.optimization_service import run_optimization
from src.optimization.errors import OptimizationInputError

router = APIRouter()


@router.post("", response_model=OptimizeResponse)
def optimize_portfolio(payload: OptimizeRequest) -> OptimizeResponse:
    try:
        response = run_optimization(payload.model_dump(), save_to_db=True, track_to_mlflow=True)
        return OptimizeResponse(**response)
    except OptimizationInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Optimization failed; consult server logs") from exc

@router.get("/metadata")
def optimization_metadata():
    from src.config import block_configuration
    from src.data.load_data import active_data_dir, load_metadata
    try:
        directory = active_data_dir()
        return {"dataset_version": directory.name, **block_configuration(load_metadata(data_dir=directory))}
    except OptimizationInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
