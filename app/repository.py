from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import desc, insert, select, update

from app.db import engine, init_db, optimization_runs, optimization_run_provenance


from src.utils.serialization import to_builtin as _to_builtin


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return _to_builtin(df.to_dict(orient="records"))


def build_comparison_payload(result: dict) -> list[dict]:
    rows = []
    for key, label in [
        ("minvar", "Min Variance"),
        ("maxsharpe", "Max Sharpe"),
        ("blocks", "Max Sharpe + Blocks"),
    ]:
        stats = result[key]["stats"] if key != "blocks" else result["blocks"]["stats"]
        rows.append(
            {
                "portfolio": label,
                "return_annual": float(stats["return_annual"]),
                "return_log_annual": float(stats.get("return_log_annual", stats["return_annual"])),
                "sharpe_daily": stats.get("sharpe_daily"),
                "metrics_version": stats.get("metrics_version", 1),
                "vol_annual": float(stats["vol_annual"]),
                "sharpe_annual": float(stats["sharpe_annual"]),
            }
        )
    return rows


def save_run(run_label: str | None, config: dict, result: dict) -> int:
    init_db()

    payload = {
        "run_label": run_label or None,
        "model_type": str(config["mode"]),
        "max_weight": float(config["max_weight"]),
        "rf_annual_simple": float(config["rf_annual_simple"]),
        "n_frontier_points": int(config["n_frontier_points"]),
        "targets_json": _to_builtin(config["targets"]),
        "metrics_json": _to_builtin(result["blocks"]["stats"]),
        "comparison_json": build_comparison_payload(result),
        "block_exposure_json": _df_to_records(result["blocks"]["exposure_by_block"]),
        "weights_json": _df_to_records(result["blocks"]["weights_df"]),
        "frontier_json": _df_to_records(result["frontier"]),
        "solver_message": str(result["blocks"].get("solver_message", "")),
    }

    with engine.begin() as conn:
        stmt = insert(optimization_runs).values(**payload)
        inserted = conn.execute(stmt)

        try:
            inserted_id = inserted.inserted_primary_key[0]
        except Exception:
            row = conn.execute(
                select(optimization_runs.c.id).order_by(desc(optimization_runs.c.id)).limit(1)
            ).first()
            inserted_id = int(row[0])

        if result.get("provenance"):
            conn.execute(insert(optimization_run_provenance).values(run_id=int(inserted_id),
                provenance_json=_to_builtin(result["provenance"]), tracking_json={}))
    return int(inserted_id)


def get_run_history(limit: int = 100) -> pd.DataFrame:
    init_db()

    stmt = (
        select(
            optimization_runs.c.id,
            optimization_runs.c.created_at,
            optimization_runs.c.run_label,
            optimization_runs.c.model_type,
            optimization_runs.c.max_weight,
            optimization_runs.c.rf_annual_simple,
            optimization_runs.c.n_frontier_points,
            optimization_runs.c.metrics_json,
        )
        .order_by(desc(optimization_runs.c.created_at), desc(optimization_runs.c.id))
        .limit(limit)
    )

    with engine.begin() as conn:
        rows = conn.execute(stmt).mappings().all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["return_annual"] = df["metrics_json"].apply(lambda x: x.get("return_annual"))
    df["vol_annual"] = df["metrics_json"].apply(lambda x: x.get("vol_annual"))
    df["sharpe_annual"] = df["metrics_json"].apply(lambda x: x.get("sharpe_annual"))
    df = df.drop(columns=["metrics_json"])
    return df


def get_run_by_id(run_id: int) -> dict | None:
    init_db()

    stmt = select(optimization_runs).where(optimization_runs.c.id == int(run_id))

    with engine.begin() as conn:
        row = conn.execute(stmt).mappings().first()

    if not row:
        return None

    result = dict(row)
    with engine.begin() as conn:
        details = conn.execute(select(optimization_run_provenance).where(optimization_run_provenance.c.run_id == run_id)).mappings().first()
    result["provenance"] = dict(details) if details else None
    return result


def record_run_tracking(run_id, tracking):
    with engine.begin() as conn:
        conn.execute(update(optimization_run_provenance).where(optimization_run_provenance.c.run_id == run_id)
                     .values(tracking_json=_to_builtin(tracking)))
