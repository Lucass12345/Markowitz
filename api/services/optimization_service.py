from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.repository import save_run
from app.tracking import track_run_mlflow
from src.optimization.optimizer_service import optimize_portfolio_mode


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return [_to_builtin(row) for row in value.to_dict(orient="records")]

    if isinstance(value, pd.Series):
        return _to_builtin(value.to_dict())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]

    return value


def run_optimization(payload: dict, save_to_db: bool = True, track_to_mlflow: bool = True) -> dict:
    """
    Executes the portfolio optimization using the existing repo backend.

    Expected payload keys:
        - model_type
        - targets
        - max_weight
        - rf_annual_simple
        - n_frontier_points
        - run_label (optional)
    """
    result = optimize_portfolio_mode(
        model_type=payload["model_type"],
        targets=payload["targets"],
        max_weight=payload["max_weight"],
        rf_annual_simple=payload["rf_annual_simple"],
        n_frontier_points=payload["n_frontier_points"],
        **{name: payload[name] for name in ("min_history", "min_overlap", "eigenvalue_floor", "max_psd_adjustment") if name in payload},
    )

    run_id = None
    mlflow_run_id = None
    mlflow_error = None

    if save_to_db:
        config = {
            "mode": payload["model_type"],
            "targets": payload["targets"],
            "max_weight": payload["max_weight"],
            "rf_annual_simple": payload["rf_annual_simple"],
            "n_frontier_points": payload["n_frontier_points"],
        }

        run_id = save_run(
            run_label=payload.get("run_label"),
            config=config,
            result=result,
        )

        if track_to_mlflow:
            mlflow_run_id, mlflow_error = track_run_mlflow(
                run_db_id=run_id,
                run_label=payload.get("run_label"),
                config=config,
                result=result,
            )

    serialized_result = _to_builtin(result)

    return {
        "run_id": run_id,
        "status": "success",
        "message": "Optimization executed successfully",
        "result": serialized_result,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_error": mlflow_error,
    }