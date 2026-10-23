from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.repository import save_run, record_run_tracking
from app.tracking import track_run_details
from src.optimization.optimizer_service import optimize_portfolio_mode


from src.utils.serialization import to_builtin as _to_builtin


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
        **{name: payload[name] for name in ("min_history", "min_overlap", "eigenvalue_floor", "max_psd_adjustment", "prediction_fallback", "max_prediction_age_days", "covariance_estimator", "lstm_direction_min", "lstm_amplitude_min") if name in payload},
    )

    run_id = None
    mlflow_run_id = None
    mlflow_error = None

    if save_to_db:
        config = {**payload, **result["inputs"], "mode": payload["model_type"]}

        run_id = save_run(
            run_label=payload.get("run_label"),
            config=config,
            result=result,
        )

        if track_to_mlflow:
            tracking = track_run_details(
                run_db_id=run_id,
                run_label=payload.get("run_label"),
                config=config,
                result=result,
            )
            record_run_tracking(run_id, tracking)
            result["tracking"] = tracking
            mlflow_run_id, mlflow_error = tracking["mlflow"]["run_id"], tracking["mlflow"].get("error")

    serialized_result = _to_builtin(result)

    return {
        "run_id": run_id,
        "status": "success",
        "message": "Optimization executed successfully",
        "result": serialized_result,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_error": mlflow_error,
    }