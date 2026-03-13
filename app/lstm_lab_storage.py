
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    select,
    update,
    insert,
    func,
    JSON,
)

from sqlalchemy.exc import SQLAlchemyError

from app.db import engine, init_db

try:
    import mlflow
except Exception:
    mlflow = None

try:
    from minio import Minio
except Exception:
    Minio = None


ROOT = Path(__file__).resolve().parents[1]
metadata = MetaData()

JsonType = JSON


prediction_runs = Table(
    "prediction_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("run_id", String(64), unique=True, nullable=False),
    Column("model_family", String(50), nullable=False, default="lstm"),
    Column("target_aggregation", String(20), nullable=False),
    Column("forecast_horizon", Integer, nullable=False),
    Column("seq_len", Integer, nullable=False),
    Column("start_date", String(32), nullable=True),
    Column("selected_features_json", JsonType, nullable=True),
    Column("epochs", Integer, nullable=True),
    Column("batch_size", Integer, nullable=True),
    Column("lr", Float, nullable=True),
    Column("hidden_size", Integer, nullable=True),
    Column("num_layers", Integer, nullable=True),
    Column("dropout", Float, nullable=True),
    Column("total_tickers_in_universe", Integer, nullable=True),
    Column("tickers_with_predictions", Integer, nullable=True),
    Column("mean_mu_pred", Float, nullable=True),
    Column("min_mu_pred", Float, nullable=True),
    Column("max_mu_pred", Float, nullable=True),
    Column("mean_rmse_test", Float, nullable=True),
    Column("mean_dir_acc_test", Float, nullable=True),
    Column("local_run_output_dir", Text, nullable=True),
    Column("local_model_dir", Text, nullable=True),
    Column("minio_bucket", String(200), nullable=True),
    Column("minio_prefix", Text, nullable=True),
    Column("mlflow_run_id", String(100), nullable=True),
    Column("is_active", Boolean, nullable=False, default=False),
)

prediction_run_tickers = Table(
    "prediction_run_tickers",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("prediction_run_id", Integer, ForeignKey("prediction_runs.id", ondelete="CASCADE"), nullable=False),
    Column("ticker", String(50), nullable=False),
    Column("mu_pred", Float, nullable=True),
    Column("mae_train", Float, nullable=True),
    Column("mae_val", Float, nullable=True),
    Column("mae_test", Float, nullable=True),
    Column("rmse_train", Float, nullable=True),
    Column("rmse_val", Float, nullable=True),
    Column("rmse_test", Float, nullable=True),
    Column("dir_acc_train", Float, nullable=True),
    Column("dir_acc_val", Float, nullable=True),
    Column("dir_acc_test", Float, nullable=True),
    Column("corr_train", Float, nullable=True),
    Column("corr_val", Float, nullable=True),
    Column("corr_test", Float, nullable=True),
    Column("best_val_loss", Float, nullable=True),
    Column("error", Text, nullable=True),
)


def ensure_lstm_lab_tables() -> None:
    init_db()
    metadata.create_all(engine)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)


def _maybe_log_to_mlflow(summary: dict, run_output_dir: str) -> str | None:
    if mlflow is None:
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "markowitz-streamlit")

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name=f"lstm_lab_{summary['run_id']}") as run:
            mlflow.log_params({
                "model_family": "lstm",
                "target_aggregation": summary.get("target_aggregation"),
                "forecast_horizon": summary.get("forecast_horizon"),
                "seq_len": summary.get("seq_len"),
                "start_date": summary.get("start_date"),
                "epochs": summary.get("epochs"),
                "batch_size": summary.get("batch_size"),
                "lr": summary.get("lr"),
                "hidden_size": summary.get("hidden_size"),
                "num_layers": summary.get("num_layers"),
                "dropout": summary.get("dropout"),
                "tickers_with_predictions": summary.get("tickers_with_predictions"),
            })

            if summary.get("mean_mu_pred") is not None:
                mlflow.log_metric("mean_mu_pred", float(summary["mean_mu_pred"]))
            if summary.get("min_mu_pred") is not None:
                mlflow.log_metric("min_mu_pred", float(summary["min_mu_pred"]))
            if summary.get("max_mu_pred") is not None:
                mlflow.log_metric("max_mu_pred", float(summary["max_mu_pred"]))
            if summary.get("mean_rmse_test") is not None:
                mlflow.log_metric("mean_rmse_test", float(summary["mean_rmse_test"]))
            if summary.get("mean_dir_acc_test") is not None:
                mlflow.log_metric("mean_dir_acc_test", float(summary["mean_dir_acc_test"]))

            try:
                mlflow.log_artifacts(run_output_dir, artifact_path="lstm_lab_outputs")
            except Exception:
                pass

            return run.info.run_id
    except Exception:
        return None


def _maybe_upload_dir_to_minio(run_id: str, run_output_dir: str, run_model_dir: str) -> tuple[str | None, str | None]:
    if Minio is None:
        return None, None

    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    if not endpoint or not access_key or not secret_key:
        return None, None

    bucket = os.getenv("LSTM_LAB_MINIO_BUCKET", "markowitz-lstm-lab")
    prefix = f"prediction-runs/{run_id}"

    try:
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        found = client.bucket_exists(bucket)
        if not found:
            client.make_bucket(bucket)

        for base_dir, top_name in [(run_output_dir, "outputs"), (run_model_dir, "models")]:
            base_dir_path = Path(base_dir)
            if not base_dir_path.exists():
                continue

            for file_path in base_dir_path.rglob("*"):
                if file_path.is_file():
                    object_name = f"{prefix}/{top_name}/{file_path.relative_to(base_dir_path).as_posix()}"
                    client.fput_object(bucket, object_name, str(file_path))

        return bucket, prefix
    except Exception:
        return None, None


def persist_lstm_lab_run(
    *,
    run_id: str,
    summary: dict,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    run_output_dir: str,
    run_model_dir: str,
) -> int | None:
    ensure_lstm_lab_tables()

    metrics_tmp = metrics_df.copy()
    pred_tmp = predictions_df.copy()

    mean_rmse_test = None
    mean_dir_acc_test = None

    if "rmse_test" in metrics_tmp.columns:
        s = pd.to_numeric(metrics_tmp["rmse_test"], errors="coerce")
        if s.notna().any():
            mean_rmse_test = float(s.mean())

    if "dir_acc_test" in metrics_tmp.columns:
        s = pd.to_numeric(metrics_tmp["dir_acc_test"], errors="coerce")
        if s.notna().any():
            mean_dir_acc_test = float(s.mean())

    summary = {
        **summary,
        "mean_rmse_test": mean_rmse_test,
        "mean_dir_acc_test": mean_dir_acc_test,
    }

    mlflow_run_id = _maybe_log_to_mlflow(summary, run_output_dir)
    minio_bucket, minio_prefix = _maybe_upload_dir_to_minio(run_id, run_output_dir, run_model_dir)

    run_payload = {
        "run_id": run_id,
        "model_family": "lstm",
        "target_aggregation": summary.get("target_aggregation"),
        "forecast_horizon": summary.get("forecast_horizon"),
        "seq_len": summary.get("seq_len"),
        "start_date": summary.get("start_date"),
        "selected_features_json": _jsonable(summary.get("selected_features")),
        "epochs": summary.get("epochs"),
        "batch_size": summary.get("batch_size"),
        "lr": summary.get("lr"),
        "hidden_size": summary.get("hidden_size"),
        "num_layers": summary.get("num_layers"),
        "dropout": summary.get("dropout"),
        "total_tickers_in_universe": summary.get("total_tickers_in_universe"),
        "tickers_with_predictions": summary.get("tickers_with_predictions"),
        "mean_mu_pred": summary.get("mean_mu_pred"),
        "min_mu_pred": summary.get("min_mu_pred"),
        "max_mu_pred": summary.get("max_mu_pred"),
        "mean_rmse_test": summary.get("mean_rmse_test"),
        "mean_dir_acc_test": summary.get("mean_dir_acc_test"),
        "local_run_output_dir": run_output_dir,
        "local_model_dir": run_model_dir,
        "minio_bucket": minio_bucket,
        "minio_prefix": minio_prefix,
        "mlflow_run_id": mlflow_run_id,
        "is_active": False,
    }

    try:
        with engine.begin() as conn:
            inserted = conn.execute(insert(prediction_runs).values(**run_payload))
            run_pk = int(inserted.inserted_primary_key[0])

            if not metrics_tmp.empty:
                metrics_clean = metrics_tmp.copy()
                metrics_clean = metrics_clean.where(pd.notnull(metrics_clean), None)

                pred_map = {}
                if not pred_tmp.empty and {"ticker", "mu_pred"}.issubset(pred_tmp.columns):
                    pred_map = dict(zip(pred_tmp["ticker"], pred_tmp["mu_pred"]))

                rows = []
                for _, row in metrics_clean.iterrows():
                    rows.append({
                        "prediction_run_id": run_pk,
                        "ticker": row.get("ticker"),
                        "mu_pred": pred_map.get(row.get("ticker"), row.get("mu_pred")),
                        "mae_train": row.get("mae_train"),
                        "mae_val": row.get("mae_val"),
                        "mae_test": row.get("mae_test"),
                        "rmse_train": row.get("rmse_train"),
                        "rmse_val": row.get("rmse_val"),
                        "rmse_test": row.get("rmse_test"),
                        "dir_acc_train": row.get("dir_acc_train"),
                        "dir_acc_val": row.get("dir_acc_val"),
                        "dir_acc_test": row.get("dir_acc_test"),
                        "corr_train": row.get("corr_train"),
                        "corr_val": row.get("corr_val"),
                        "corr_test": row.get("corr_test"),
                        "best_val_loss": row.get("best_val_loss"),
                        "error": row.get("error"),
                    })
                if rows:
                    conn.execute(insert(prediction_run_tickers), rows)

        return run_pk
    except SQLAlchemyError:
        return None


def set_active_prediction_run(run_id: str) -> None:
    ensure_lstm_lab_tables()
    with engine.begin() as conn:
        conn.execute(update(prediction_runs).values(is_active=False))
        conn.execute(update(prediction_runs).where(prediction_runs.c.run_id == run_id).values(is_active=True))


def get_prediction_runs_df() -> pd.DataFrame:
    ensure_lstm_lab_tables()
    with engine.begin() as conn:
        rows = conn.execute(
            select(
                prediction_runs.c.id,
                prediction_runs.c.created_at,
                prediction_runs.c.run_id,
                prediction_runs.c.target_aggregation,
                prediction_runs.c.forecast_horizon,
                prediction_runs.c.seq_len,
                prediction_runs.c.start_date,
                prediction_runs.c.tickers_with_predictions,
                prediction_runs.c.mean_mu_pred,
                prediction_runs.c.mean_rmse_test,
                prediction_runs.c.mean_dir_acc_test,
                prediction_runs.c.is_active,
                prediction_runs.c.mlflow_run_id,
                prediction_runs.c.minio_bucket,
                prediction_runs.c.minio_prefix,
            ).order_by(prediction_runs.c.created_at.desc())
        ).mappings().all()
    return pd.DataFrame(rows)


def get_active_prediction_run() -> dict | None:
    ensure_lstm_lab_tables()
    with engine.begin() as conn:
        row = conn.execute(
            select(prediction_runs).where(prediction_runs.c.is_active == True).limit(1)
        ).mappings().first()
    return dict(row) if row else None


def load_active_predictions_df() -> pd.DataFrame:
    active = get_active_prediction_run()
    if not active:
        return pd.DataFrame(columns=["ticker", "mu_pred"])

    run_id = active["run_id"]

    ensure_lstm_lab_tables()
    with engine.begin() as conn:
        rows = conn.execute(
            select(
                prediction_run_tickers.c.ticker,
                prediction_run_tickers.c.mu_pred,
                prediction_run_tickers.c.rmse_test,
                prediction_run_tickers.c.dir_acc_test,
                prediction_run_tickers.c.corr_test,
            ).select_from(
                prediction_run_tickers.join(
                    prediction_runs,
                    prediction_run_tickers.c.prediction_run_id == prediction_runs.c.id,
                )
            ).where(prediction_runs.c.run_id == run_id)
        ).mappings().all()

    return pd.DataFrame(rows)
