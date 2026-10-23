from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.repository import get_run_by_id, get_run_history
from api.schemas.runs import RunDetailResponse

from src.utils.serialization import to_builtin

router = APIRouter()


@router.get("")
def list_runs(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    df = get_run_history(limit=limit)
    if df.empty:
        return []

    rows = df.to_dict(orient="records")
    for row in rows:
        if "created_at" in row and row["created_at"] is not None:
            row["created_at"] = str(row["created_at"])
    return to_builtin(rows)


@router.get("/{run_id}", response_model=RunDetailResponse)
def get_run_detail(run_id: int) -> RunDetailResponse:
    run = get_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run id={run_id} not found")

    if "created_at" in run and run["created_at"] is not None:
        run["created_at"] = str(run["created_at"])

    return RunDetailResponse(run=to_builtin(run))