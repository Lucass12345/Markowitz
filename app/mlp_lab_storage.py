from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    JSON,
    func,
    insert,
    select,
    update,
)

from app.db import engine, init_db

ROOT = Path(__file__).resolve().parents[1]
meta = MetaData()

mlp_prediction_runs = Table(
    "mlp_prediction_runs",
    meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("run_id", String(64), unique=True, nullable=False),
    Column("model_family", String(50), nullable=False, server_default="mlp"),
    Column("start_date", String(32), nullable=True),
    Column("forecast_horizon", Integer, nullable=False),
    Column("target_aggregation", String(16), nullable=False),
    Column("lags", Integer, nullable=False),
    Column("short_window", Integer, nullable=False),
    Column("long_window", Integer, nullable=False),
    Column("hidden_layers_json", JSON, nullable=True),
    Column("activation", String(32), nullable=True),
    Column("solver", String(32), nullable=True),
    Column("scaler", String(32), nullable=True),
    Column("alpha", Float, nullable=True),
    Column("learning_rate", String(32), nullable=True),
    Column("learning_rate_init", Float, nullable=True),
    Column("max_iter", Integer, nullable=True),
    Column("batch_size", String(32), nullable=True),
    Column("early_stopping", Boolean, nullable=True),
    Column("validation_fraction", Float, nullable=True),
    Column("n_iter_no_change", Integer, nullable=True),
    Column("clip_pred", Float, nullable=True),
    Column("winsorize_quantile", Float, nullable=True),
    Column("include_abs_features", Boolean, nullable=True),
    Column("include_square_features", Boolean, nullable=True),
    Column("include_sign_features", Boolean, nullable=True),
    Column("total_tickers_in_universe", Integer, nullable=True),
    Column("tickers_with_predictions", Integer, nullable=True),
    Column("mean_mu_pred", Float, nullable=True),
    Column("mean_rmse_test", Float, nullable=True),
    Column("mean_dir_acc_test", Float, nullable=True),
    Column("mean_corr_test", Float, nullable=True),
    Column("mean_score", Float, nullable=True),
    Column("selection_policy_json", JSON, nullable=True),
    Column("is_active", Boolean, nullable=False, server_default="0"),
    Column("run_output_dir", Text, nullable=True),
    Column("run_model_dir", Text, nullable=True),
)

mlp_prediction_run_tickers = Table(
    "mlp_prediction_run_tickers",
    meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("prediction_run_id", Integer, ForeignKey("mlp_prediction_runs.id", ondelete="CASCADE"), nullable=False),
    Column("ticker", String(64), nullable=False),
    Column("mu_pred", Float, nullable=True),
    Column("score", Float, nullable=True),
    Column("rmse_train", Float, nullable=True),
    Column("rmse_val", Float, nullable=True),
    Column("rmse_test", Float, nullable=True),
    Column("mae_train", Float, nullable=True),
    Column("mae_val", Float, nullable=True),
    Column("mae_test", Float, nullable=True),
    Column("dir_acc_train", Float, nullable=True),
    Column("dir_acc_val", Float, nullable=True),
    Column("dir_acc_test", Float, nullable=True),
    Column("corr_train", Float, nullable=True),
    Column("corr_val", Float, nullable=True),
    Column("corr_test", Float, nullable=True),
    Column("r2_train", Float, nullable=True),
    Column("r2_val", Float, nullable=True),
    Column("r2_test", Float, nullable=True),
    Column("positive_precision_test", Float, nullable=True),
    Column("pred_positive_rate_test", Float, nullable=True),
)


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np
        import pandas as pd
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    return str(value)


def ensure_mlp_lab_tables() -> None:
    init_db()
    meta.create_all(engine)


def persist_mlp_lab_run(*, run_id: str, summary: dict, predictions_df: pd.DataFrame, metrics_df: pd.DataFrame, run_output_dir: str, run_model_dir: str) -> int | None:
    ensure_mlp_lab_tables()
    merged = metrics_df.merge(predictions_df, on="ticker", how="outer", suffixes=("", "_prediction"))
    run_payload = {
        "run_id": run_id,
        "model_family": "mlp",
        "start_date": summary.get("start_date"),
        "forecast_horizon": summary.get("forecast_horizon"),
        "target_aggregation": summary.get("target_aggregation"),
        "lags": summary.get("lags"),
        "short_window": summary.get("short_window"),
        "long_window": summary.get("long_window"),
        "hidden_layers_json": _jsonable(summary.get("hidden_layers")),
        "activation": summary.get("activation"),
        "solver": summary.get("solver"),
        "scaler": summary.get("scaler"),
        "alpha": summary.get("alpha"),
        "learning_rate": summary.get("learning_rate"),
        "learning_rate_init": summary.get("learning_rate_init"),
        "max_iter": summary.get("max_iter"),
        "batch_size": str(summary.get("batch_size")),
        "early_stopping": summary.get("early_stopping"),
        "validation_fraction": summary.get("validation_fraction"),
        "n_iter_no_change": summary.get("n_iter_no_change"),
        "clip_pred": summary.get("clip_pred"),
        "winsorize_quantile": summary.get("winsorize_quantile"),
        "include_abs_features": summary.get("include_abs_features"),
        "include_square_features": summary.get("include_square_features"),
        "include_sign_features": summary.get("include_sign_features"),
        "total_tickers_in_universe": summary.get("total_tickers_in_universe"),
        "tickers_with_predictions": summary.get("tickers_with_predictions"),
        "mean_mu_pred": summary.get("mean_mu_pred"),
        "mean_rmse_test": summary.get("mean_rmse_test"),
        "mean_dir_acc_test": summary.get("mean_dir_acc_test"),
        "mean_corr_test": summary.get("mean_corr_test"),
        "mean_score": summary.get("mean_score"),
        "selection_policy_json": _jsonable(summary.get("selection_policy", {"policy_type": "all_assets"})),
        "is_active": False,
        "run_output_dir": run_output_dir,
        "run_model_dir": run_model_dir,
    }

    cols = [
        "ticker", "mu_pred", "score",
        "rmse_train", "rmse_val", "rmse_test",
        "mae_train", "mae_val", "mae_test",
        "dir_acc_train", "dir_acc_val", "dir_acc_test",
        "corr_train", "corr_val", "corr_test",
        "r2_train", "r2_val", "r2_test",
        "positive_precision_test", "pred_positive_rate_test",
    ]
    available_cols = [c for c in cols if c in merged.columns]

    with engine.begin() as conn:
        inserted = conn.execute(insert(mlp_prediction_runs).values(**run_payload))
        run_pk = inserted.inserted_primary_key[0]
        ticker_rows = []
        for _, row in merged[available_cols].iterrows():
            payload = {c: _jsonable(row.get(c)) for c in available_cols}
            payload["prediction_run_id"] = run_pk
            ticker_rows.append(payload)
        if ticker_rows:
            conn.execute(insert(mlp_prediction_run_tickers), ticker_rows)
    return run_pk


def get_mlp_prediction_runs_df() -> pd.DataFrame:
    ensure_mlp_lab_tables()
    with engine.begin() as conn:
        rows = conn.execute(
            select(
                mlp_prediction_runs.c.id,
                mlp_prediction_runs.c.created_at,
                mlp_prediction_runs.c.run_id,
                mlp_prediction_runs.c.forecast_horizon,
                mlp_prediction_runs.c.target_aggregation,
                mlp_prediction_runs.c.start_date,
                mlp_prediction_runs.c.activation,
                mlp_prediction_runs.c.hidden_layers_json,
                mlp_prediction_runs.c.tickers_with_predictions,
                mlp_prediction_runs.c.mean_mu_pred,
                mlp_prediction_runs.c.mean_rmse_test,
                mlp_prediction_runs.c.mean_dir_acc_test,
                mlp_prediction_runs.c.mean_corr_test,
                mlp_prediction_runs.c.mean_score,
                mlp_prediction_runs.c.is_active,
                mlp_prediction_runs.c.selection_policy_json,
            ).order_by(mlp_prediction_runs.c.created_at.desc())
        ).mappings().all()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["hidden_layers_json"] = df["hidden_layers_json"].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
        df["selection_policy_json"] = df["selection_policy_json"].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    return df


def set_active_mlp_prediction_run(run_id: str, selection_policy: dict | None = None) -> None:
    ensure_mlp_lab_tables()
    policy = selection_policy or {"policy_type": "all_assets"}
    with engine.begin() as conn:
        conn.execute(update(mlp_prediction_runs).values(is_active=False))
        conn.execute(
            update(mlp_prediction_runs)
            .where(mlp_prediction_runs.c.run_id == run_id)
            .values(is_active=True, selection_policy_json=_jsonable(policy))
        )


def get_active_mlp_prediction_run() -> dict | None:
    ensure_mlp_lab_tables()
    with engine.begin() as conn:
        row = conn.execute(select(mlp_prediction_runs).where(mlp_prediction_runs.c.is_active == True).limit(1)).mappings().first()
    return dict(row) if row else None


def get_mlp_run_tickers_df(run_id: str) -> pd.DataFrame:
    ensure_mlp_lab_tables()
    with engine.begin() as conn:
        rows = conn.execute(
            select(
                mlp_prediction_run_tickers.c.ticker,
                mlp_prediction_run_tickers.c.mu_pred,
                mlp_prediction_run_tickers.c.score,
                mlp_prediction_run_tickers.c.rmse_train,
                mlp_prediction_run_tickers.c.rmse_val,
                mlp_prediction_run_tickers.c.rmse_test,
                mlp_prediction_run_tickers.c.mae_train,
                mlp_prediction_run_tickers.c.mae_val,
                mlp_prediction_run_tickers.c.mae_test,
                mlp_prediction_run_tickers.c.dir_acc_train,
                mlp_prediction_run_tickers.c.dir_acc_val,
                mlp_prediction_run_tickers.c.dir_acc_test,
                mlp_prediction_run_tickers.c.corr_train,
                mlp_prediction_run_tickers.c.corr_val,
                mlp_prediction_run_tickers.c.corr_test,
                mlp_prediction_run_tickers.c.r2_train,
                mlp_prediction_run_tickers.c.r2_val,
                mlp_prediction_run_tickers.c.r2_test,
                mlp_prediction_run_tickers.c.positive_precision_test,
                mlp_prediction_run_tickers.c.pred_positive_rate_test,
            )
            .select_from(
                mlp_prediction_run_tickers.join(
                    mlp_prediction_runs,
                    mlp_prediction_run_tickers.c.prediction_run_id == mlp_prediction_runs.c.id,
                )
            )
            .where(mlp_prediction_runs.c.run_id == run_id)
        ).mappings().all()
    return pd.DataFrame(rows)


def _restore_validation_evidence(df: pd.DataFrame, active: dict) -> pd.DataFrame:
    out = df.copy()
    directory = active.get("run_output_dir") or active.get("local_run_output_dir")
    if not directory:
        return out
    for filename in ("metrics.csv", "predictions.csv"):
        path = Path(directory) / filename
        if not path.exists():
            continue
        stored = pd.read_csv(path)
        columns = [name for name in ("dir_acc_val", "corr_val", "rmse_val", "positive_precision_val") if name in stored]
        if not columns:
            continue
        stored = stored[["ticker"] + columns].rename(columns={name: f"{name}_artifact" for name in columns})
        out = out.merge(stored, on="ticker", how="left", validate="one_to_one")
        for name in columns:
            artifact = out.pop(f"{name}_artifact")
            out[name] = out[name].fillna(artifact) if name in out else artifact
    return out


def _apply_selection_policy(df: pd.DataFrame, selection_policy: dict | None) -> pd.DataFrame:
    from src.models.train_returns_mlp_lab import build_composite_score
    out = df.copy()
    if out.empty:
        out["use_for_optimizer"] = pd.Series(dtype=bool)
        return out
    policy = selection_policy or {"policy_type": "all_assets"}
    policy_type = str(policy.get("policy_type", "all_assets")).strip().lower()
    # Scores persisted by older runs used test metrics; always recompute from val.
    out["score"] = build_composite_score(out)
    evidence = out["score"].notna()
    def numeric(name):
        return pd.to_numeric(out[name], errors="coerce") if name in out else pd.Series(float("nan"), index=out.index)
    direction, corr, rmse, score = numeric("dir_acc_val"), numeric("corr_val"), numeric("rmse_val"), out["score"]
    if policy_type == "all_assets":
        mask = evidence.copy()
    elif policy_type == "dir_acc_min":
        mask = direction >= float(policy.get("dir_acc_min", .52))
    elif policy_type == "corr_and_rmse":
        mask = (corr >= float(policy.get("corr_min", 0.))) & (rmse <= float(policy.get("rmse_max", rmse.max())))
    elif policy_type == "composite_min_score":
        mask = score >= float(policy.get("min_score", .60))
    elif policy_type == "composite_top_n":
        top_n = int(policy.get("top_n", max(1, len(out) // 3)))
        mask = score.rank(method="first", ascending=False) <= top_n
    elif policy_type == "hybrid_quality":
        mask = ((direction >= float(policy.get("dir_acc_min", .52)))
                & (corr >= float(policy.get("corr_min", 0.)))
                & (score >= float(policy.get("min_score", .50))))
    else:
        raise ValueError(f"Unknown MLP selection policy: {policy_type}")
    out["use_for_optimizer"] = (mask & evidence).fillna(False).astype(bool)
    out["selection_split"] = "validation"
    out["selection_reason"] = "validation_policy_rejected"
    out.loc[~evidence, "selection_reason"] = "missing_validation_evidence"
    out.loc[out.use_for_optimizer, "selection_reason"] = "validation_policy_accepted"
    return out


def load_active_mlp_predictions_df() -> pd.DataFrame:
    active = get_active_mlp_prediction_run()
    if active is None:
        return pd.DataFrame()
    df = get_mlp_run_tickers_df(active["run_id"])
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = _restore_validation_evidence(df, active)
    return _apply_selection_policy(df, active.get("selection_policy_json"))
