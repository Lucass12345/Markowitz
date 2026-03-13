from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()

files = {}

files["src/models/lstm_features.py"] = dedent("""
from __future__ import annotations

import pandas as pd

AVAILABLE_FEATURES = [
    "return",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_5",
    "rolling_std_5",
    "rolling_mean_20",
    "rolling_std_20",
    "momentum_5",
    "momentum_20",
]


def build_lstm_feature_frame(series: pd.Series) -> pd.DataFrame:
    s = series.dropna().astype(float).copy()

    df = pd.DataFrame(index=s.index)
    df["return"] = s
    df["lag_1"] = s.shift(1)
    df["lag_2"] = s.shift(2)
    df["lag_3"] = s.shift(3)
    df["rolling_mean_5"] = s.rolling(5).mean()
    df["rolling_std_5"] = s.rolling(5).std()
    df["rolling_mean_20"] = s.rolling(20).mean()
    df["rolling_std_20"] = s.rolling(20).std()
    df["momentum_5"] = s - s.shift(5)
    df["momentum_20"] = s - s.shift(20)

    return df


def select_feature_columns(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    cols = [c for c in selected_features if c in df.columns]
    if not cols:
        raise ValueError("No valid LSTM features selected.")
    return df[cols].dropna()
""")

files["src/models/sequence_utils.py"] = dedent("""
import numpy as np
import pandas as pd


def build_lstm_sequences(
    series: pd.Series,
    seq_len: int = 20,
    forecast_horizon: int = 5
):
    s = series.dropna().astype(float).values

    X, y = [], []

    for i in range(seq_len, len(s) - forecast_horizon + 1):
        x_window = s[i - seq_len:i]
        y_future = s[i:i + forecast_horizon].mean()

        X.append(x_window.reshape(-1, 1))
        y.append(y_future)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def build_lstm_sequences_multifeature(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    seq_len: int = 20,
    forecast_horizon: int = 5,
    target_aggregation: str = "mean",
):
    X, y = [], []

    df = feature_df.copy()
    target = target_series.copy().astype(float)

    aligned = df.copy()
    aligned["target"] = target
    aligned = aligned.dropna()

    values = aligned.drop(columns=["target"]).values.astype(np.float32)
    t = aligned["target"].values.astype(np.float32)

    for i in range(seq_len, len(aligned) - forecast_horizon + 1):
        x_window = values[i - seq_len:i]
        future_slice = t[i:i + forecast_horizon]

        if target_aggregation == "sum":
            y_future = future_slice.sum()
        else:
            y_future = future_slice.mean()

        X.append(x_window)
        y.append(y_future)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y
""")

files["app/lstm_lab_storage.py"] = dedent("""
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
)
from sqlalchemy.dialects.postgresql import JSONB
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

try:
    JsonType = JSONB
except Exception:
    JsonType = Text


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
""")

files["src/optimization/lstm_lab_loader.py"] = dedent("""
from __future__ import annotations

import pandas as pd

from app.lstm_lab_storage import load_active_predictions_df


def load_active_lstm_predictions_df() -> pd.DataFrame:
    return load_active_predictions_df()
""")

files["src/models/train_returns_lstm_lab.py"] = dedent("""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_returns
from src.models.return_model_lstm import ReturnLSTM
from src.models.lstm_features import build_lstm_feature_frame, select_feature_columns
from src.models.sequence_utils import build_lstm_sequences_multifeature
from app.lstm_lab_storage import persist_lstm_lab_run

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "lstm_lab" / "runs"
MODEL_ROOT = ROOT / "models_artifacts" / "returns_lstm" / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def corrcoef_safe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def temporal_split(X, y, dates, train_frac=0.70, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return (
        X[:train_end], y[:train_end], dates[:train_end],
        X[train_end:val_end], y[train_end:val_end], dates[train_end:val_end],
        X[val_end:], y[val_end:], dates[val_end:],
    )


def scale_3d_train(X_train, X_val, X_test):
    feat_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    feat_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

    X_train_s = (X_train - feat_mean) / feat_std
    X_val_s = (X_val - feat_mean) / feat_std
    X_test_s = (X_test - feat_mean) / feat_std

    return X_train_s, X_val_s, X_test_s, feat_mean, feat_std


def scale_y_train(y_train, y_val, y_test):
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-8:
        y_std = 1.0

    y_train_s = (y_train - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    return y_train_s, y_val_s, y_test_s, y_mean, y_std


def inverse_y(y_scaled, y_mean, y_std):
    return y_scaled * y_std + y_mean


def run_lstm_lab(
    selected_features: list[str],
    seq_len: int = 40,
    forecast_horizon: int = 5,
    target_aggregation: str = "mean",
    start_date: str | None = "2022-01-01",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
):
    returns = load_returns()

    if start_date and start_date != "full":
        returns = returns.loc[start_date:]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = OUTPUT_ROOT / run_id
    run_model_dir = MODEL_ROOT / run_id
    per_ticker_dir = run_output_dir / "per_ticker"
    curves_dir = run_output_dir / "loss_curves"

    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)
    per_ticker_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    pred_rows = []

    total_tickers = len(list(returns.columns))

    for ticker in returns.columns:
        try:
            series = returns[ticker].dropna().astype(float)

            if len(series) < max(seq_len + forecast_horizon + 30, 120):
                continue

            q_low = float(series.quantile(0.01))
            q_high = float(series.quantile(0.99))
            series = series.clip(lower=q_low, upper=q_high)

            feat_df = build_lstm_feature_frame(series)
            feat_df = select_feature_columns(feat_df, selected_features)

            common_idx = feat_df.index.intersection(series.index)
            feat_df = feat_df.loc[common_idx]
            target_series = series.loc[common_idx]

            aligned = feat_df.copy()
            aligned["target"] = target_series
            aligned = aligned.dropna()

            if len(aligned) < max(seq_len + forecast_horizon + 30, 120):
                continue

            X, y = build_lstm_sequences_multifeature(
                feature_df=aligned.drop(columns=["target"]),
                target_series=aligned["target"],
                seq_len=seq_len,
                forecast_horizon=forecast_horizon,
                target_aggregation=target_aggregation,
            )

            target_dates = aligned.index[seq_len: len(aligned) - forecast_horizon + 1]

            if len(X) < 150:
                continue

            (
                X_train, y_train, d_train,
                X_val, y_val, d_val,
                X_test, y_test, d_test,
            ) = temporal_split(X, y, target_dates)

            if min(len(X_train), len(X_val), len(X_test)) == 0:
                continue

            X_train_s, X_val_s, X_test_s, x_mean, x_std = scale_3d_train(X_train, X_val, X_test)
            y_train_s, y_val_s, y_test_s, y_mean, y_std = scale_y_train(y_train, y_val, y_test)

            X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
            y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
            y_val_t = torch.tensor(y_val_s, dtype=torch.float32)
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

            train_loader = DataLoader(
                TensorDataset(X_train_t, y_train_t),
                batch_size=batch_size,
                shuffle=False,
            )

            model = ReturnLSTM(
                input_size=X_train.shape[2],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                dense_size=32,
            ).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.SmoothL1Loss()

            best_val = np.inf
            best_state = None
            train_losses = []
            val_losses = []

            for _ in range(epochs):
                model.train()
                batch_losses = []

                for xb, yb in train_loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)

                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    batch_losses.append(loss.item())

                train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
                train_losses.append(train_loss)

                model.eval()
                with torch.no_grad():
                    val_pred_scaled = model(X_val_t.to(DEVICE)).cpu().numpy()
                    val_loss = float(criterion(torch.tensor(val_pred_scaled), y_val_t).item())

                val_losses.append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            with torch.no_grad():
                pred_train_scaled = model(X_train_t.to(DEVICE)).cpu().numpy()
                pred_val_scaled = model(X_val_t.to(DEVICE)).cpu().numpy()
                pred_test_scaled = model(X_test_t.to(DEVICE)).cpu().numpy()
                last_pred_scaled = model(torch.tensor(X_test_s[-1:], dtype=torch.float32).to(DEVICE)).cpu().numpy()[0]

            pred_train = inverse_y(pred_train_scaled, y_mean, y_std)
            pred_val = inverse_y(pred_val_scaled, y_mean, y_std)
            pred_test = inverse_y(pred_test_scaled, y_mean, y_std)
            last_pred = float(inverse_y(last_pred_scaled, y_mean, y_std))

            hist_mean = float(series.mean())
            hist_std = float(series.std())
            lower = hist_mean - 2.5 * hist_std
            upper = hist_mean + 2.5 * hist_std
            last_pred = float(np.clip(last_pred, lower, upper))
            last_pred = float(np.clip(last_pred, -0.02, 0.02))

            train_metrics = {
                "mae_train": mae(y_train, pred_train),
                "rmse_train": rmse(y_train, pred_train),
                "dir_acc_train": directional_accuracy(y_train, pred_train),
                "corr_train": corrcoef_safe(y_train, pred_train),
            }
            val_metrics = {
                "mae_val": mae(y_val, pred_val),
                "rmse_val": rmse(y_val, pred_val),
                "dir_acc_val": directional_accuracy(y_val, pred_val),
                "corr_val": corrcoef_safe(y_val, pred_val),
            }
            test_metrics = {
                "mae_test": mae(y_test, pred_test),
                "rmse_test": rmse(y_test, pred_test),
                "dir_acc_test": directional_accuracy(y_test, pred_test),
                "corr_test": corrcoef_safe(y_test, pred_test),
            }

            pred_rows.append({
                "ticker": ticker,
                "mu_pred": last_pred,
            })

            summary_rows.append({
                "ticker": ticker,
                "best_val_loss": best_val,
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "mu_pred": last_pred,
                **train_metrics,
                **val_metrics,
                **test_metrics,
            })

            pred_detail = pd.concat([
                pd.DataFrame({
                    "date": pd.to_datetime(d_train),
                    "split": "train",
                    "y_true": y_train,
                    "y_pred": pred_train.reshape(-1),
                }),
                pd.DataFrame({
                    "date": pd.to_datetime(d_val),
                    "split": "val",
                    "y_true": y_val,
                    "y_pred": pred_val.reshape(-1),
                }),
                pd.DataFrame({
                    "date": pd.to_datetime(d_test),
                    "split": "test",
                    "y_true": y_test,
                    "y_pred": pred_test.reshape(-1),
                }),
            ], ignore_index=True)

            pred_detail.to_csv(per_ticker_dir / f"{ticker}_predictions.csv", index=False)

            pd.DataFrame({
                "epoch": np.arange(1, len(train_losses) + 1),
                "train_loss": train_losses,
                "val_loss": val_losses,
            }).to_csv(curves_dir / f"{ticker}_losses.csv", index=False)

            torch.save(model.state_dict(), run_model_dir / f"{ticker}_lstm.pt")

            with open(run_model_dir / f"{ticker}_meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "ticker": ticker,
                    "selected_features": selected_features,
                    "seq_len": seq_len,
                    "forecast_horizon": forecast_horizon,
                    "target_aggregation": target_aggregation,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "lr": lr,
                    "start_date": start_date,
                    "x_mean": x_mean.tolist(),
                    "x_std": x_std.tolist(),
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "metrics": {
                        **train_metrics,
                        **val_metrics,
                        **test_metrics,
                        "best_val_loss": best_val,
                        "mu_pred": last_pred,
                    },
                }, f, indent=2)

        except Exception as e:
            summary_rows.append({
                "ticker": ticker,
                "error": repr(e),
            })

    predictions_df = pd.DataFrame(pred_rows).sort_values("ticker").reset_index(drop=True) if pred_rows else pd.DataFrame(columns=["ticker", "mu_pred"])
    metrics_df = pd.DataFrame(summary_rows).sort_values("ticker").reset_index(drop=True) if summary_rows else pd.DataFrame()

    predictions_df.to_csv(run_output_dir / "predictions.csv", index=False)
    metrics_df.to_csv(run_output_dir / "metrics.csv", index=False)

    summary = {
        "run_id": run_id,
        "total_tickers_in_universe": total_tickers,
        "tickers_with_predictions": int(len(predictions_df)),
        "start_date": start_date,
        "selected_features": selected_features,
        "seq_len": seq_len,
        "forecast_horizon": forecast_horizon,
        "target_aggregation": target_aggregation,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "mean_mu_pred": float(predictions_df["mu_pred"].mean()) if not predictions_df.empty else None,
        "min_mu_pred": float(predictions_df["mu_pred"].min()) if not predictions_df.empty else None,
        "max_mu_pred": float(predictions_df["mu_pred"].max()) if not predictions_df.empty else None,
        "mean_rmse_test": float(pd.to_numeric(metrics_df["rmse_test"], errors="coerce").mean()) if "rmse_test" in metrics_df.columns else None,
        "mean_dir_acc_test": float(pd.to_numeric(metrics_df["dir_acc_test"], errors="coerce").mean()) if "dir_acc_test" in metrics_df.columns else None,
    }

    with open(run_output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(run_output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    latest_ptr = ROOT / "outputs" / "lstm_lab" / "latest_run.json"
    latest_ptr.parent.mkdir(parents=True, exist_ok=True)
    latest_ptr.write_text(json.dumps({
        "run_id": run_id,
        "run_output_dir": str(run_output_dir),
        "run_model_dir": str(run_model_dir),
    }, indent=2), encoding="utf-8")

    db_run_pk = persist_lstm_lab_run(
        run_id=run_id,
        summary=summary,
        predictions_df=predictions_df,
        metrics_df=metrics_df,
        run_output_dir=str(run_output_dir),
        run_model_dir=str(run_model_dir),
    )

    return {
        "run_id": run_id,
        "db_run_pk": db_run_pk,
        "predictions_path": str(run_output_dir / "predictions.csv"),
        "metrics_path": str(run_output_dir / "metrics.csv"),
        "per_ticker_dir": str(per_ticker_dir),
        "curves_dir": str(curves_dir),
        "model_dir": str(run_model_dir),
        "summary": summary,
        "predictions": predictions_df,
        "metrics": metrics_df,
    }
""")

files["app/pages/03_LSTM_Lab.py"] = dedent("""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_returns
try:
    from src.data.load_data import load_prices
except Exception:
    load_prices = None

from src.models.train_returns_lstm_lab import run_lstm_lab
from app.lstm_lab_storage import get_prediction_runs_df, set_active_prediction_run


st.set_page_config(page_title="LSTM Lab", page_icon="🧠", layout="wide")
st.title("LSTM Lab")
st.caption("Full-universe configurable LSTM training, prediction, ranking, filtering and ticker-level inspection.")

RETURNS = load_returns()
ALL_TICKERS = list(RETURNS.columns)


def get_prices():
    global load_prices
    if load_prices is not None:
        try:
            p = load_prices()
            if isinstance(p, pd.DataFrame) and not p.empty:
                return p
        except Exception:
            pass

    synthetic = (1 + RETURNS.fillna(0)).cumprod() * 100
    return synthetic


PRICES = get_prices()


def load_latest_saved_result():
    latest_path = ROOT / "outputs" / "lstm_lab" / "latest_run.json"
    if not latest_path.exists():
        return None

    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        run_output_dir = Path(latest["run_output_dir"])
        predictions_path = run_output_dir / "predictions.csv"
        metrics_path = run_output_dir / "metrics.csv"
        summary_path = run_output_dir / "summary.json"
        config_path = run_output_dir / "config.json"

        if not predictions_path.exists() or not metrics_path.exists():
            return None

        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

        return {
            "run_id": latest["run_id"],
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "per_ticker_dir": str(run_output_dir / "per_ticker"),
            "curves_dir": str(run_output_dir / "loss_curves"),
            "predictions": pd.read_csv(predictions_path),
            "metrics": pd.read_csv(metrics_path),
            "summary": {**config, **summary},
        }
    except Exception:
        return None


def build_selected_features():
    selected = []

    with st.sidebar:
        st.header("Configuration")
        st.subheader("Feature selection")

        with st.expander("Base features", expanded=True):
            f_return = st.checkbox("return", value=True)
            f_lag1 = st.checkbox("lag_1", value=True)
            f_lag2 = st.checkbox("lag_2", value=True)
            f_lag3 = st.checkbox("lag_3", value=False)

        with st.expander("Rolling features", expanded=True):
            f_rm5 = st.checkbox("rolling_mean_5", value=True)
            f_rs5 = st.checkbox("rolling_std_5", value=True)
            f_rm20 = st.checkbox("rolling_mean_20", value=True)
            f_rs20 = st.checkbox("rolling_std_20", value=True)

        with st.expander("Momentum features", expanded=False):
            f_m5 = st.checkbox("momentum_5", value=False)
            f_m20 = st.checkbox("momentum_20", value=False)

        if f_return: selected.append("return")
        if f_lag1: selected.append("lag_1")
        if f_lag2: selected.append("lag_2")
        if f_lag3: selected.append("lag_3")
        if f_rm5: selected.append("rolling_mean_5")
        if f_rs5: selected.append("rolling_std_5")
        if f_rm20: selected.append("rolling_mean_20")
        if f_rs20: selected.append("rolling_std_20")
        if f_m5: selected.append("momentum_5")
        if f_m20: selected.append("momentum_20")

        start_date = st.selectbox(
            "Training data start date",
            options=["full", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
            index=3,
        )

        seq_len = st.slider("Lookback window (trading days)", 20, 60, 40)
        forecast_horizon = st.slider("Forecast horizon (trading days)", 1, 10, 5)
        target_aggregation = st.radio("Target aggregation", options=["mean", "sum"], horizontal=True)

        hidden_size = st.slider("Hidden size", 16, 128, 64, step=16)
        num_layers = st.slider("Num layers", 1, 4, 2)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
        epochs = st.slider("Epochs", 10, 150, 50, step=10)
        batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
        lr = st.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)

    return {
        "selected_features": selected,
        "start_date": start_date,
        "seq_len": seq_len,
        "forecast_horizon": forecast_horizon,
        "target_aggregation": target_aggregation,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }


cfg = build_selected_features()

st.info(
    f\"\"\"\\n**Universe:** all available tickers in returns dataset (`{len(ALL_TICKERS)}` assets)  
**Target:** `{cfg["target_aggregation"]}` of next `{cfg["forecast_horizon"]}` trading-day returns  
**Input window:** last `{cfg["seq_len"]}` trading days  
**Training data start:** `{cfg["start_date"]}`  
**Selected features:** {", ".join(cfg["selected_features"]) if cfg["selected_features"] else "None"}
\"\"\"
)

run_button = st.button("Train and generate full-universe predictions", use_container_width=True)

if run_button:
    if not cfg["selected_features"]:
        st.error("Select at least one feature.")
        st.stop()

    with st.spinner("Training configurable LSTM models for all available tickers..."):
        result = run_lstm_lab(
            selected_features=cfg["selected_features"],
            seq_len=cfg["seq_len"],
            forecast_horizon=cfg["forecast_horizon"],
            target_aggregation=cfg["target_aggregation"],
            start_date=cfg["start_date"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    st.session_state["lstm_lab_result"] = result

if "lstm_lab_result" not in st.session_state:
    latest = load_latest_saved_result()
    if latest is not None:
        st.session_state["lstm_lab_result"] = latest

result = st.session_state.get("lstm_lab_result")

runs_df = get_prediction_runs_df()

with st.expander("Stored LSTM prediction runs (Postgres)", expanded=False):
    if runs_df.empty:
        st.info("No prediction runs stored in Postgres yet.")
    else:
        st.dataframe(runs_df, use_container_width=True, hide_index=True)
        active_candidates = runs_df["run_id"].tolist()
        selected_active = st.selectbox("Set active run for optimizer", active_candidates)
        if st.button("Set selected run as active"):
            set_active_prediction_run(selected_active)
            st.success(f"Active prediction run set to: {selected_active}")
            st.rerun()

if result is None:
    st.warning("No LSTM Lab run available yet. Launch one from this page.")
    st.stop()

pred_df = result["predictions"].copy()
metrics_df = result["metrics"].copy()
summary = result.get("summary", {})

if pred_df.empty or metrics_df.empty:
    st.warning("Run completed but no predictions/metrics were generated.")
    st.stop()

merged = metrics_df.merge(pred_df, on=["ticker", "mu_pred"], how="outer")

st.success(f"LSTM run available: {result['run_id']}")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Universe size", int(summary.get("total_tickers_in_universe", len(ALL_TICKERS))))
with c2:
    st.metric("Tickers predicted", int(summary.get("tickers_with_predictions", len(pred_df))))
with c3:
    st.metric("Mean RMSE test", f"{merged['rmse_test'].mean():.6f}" if "rmse_test" in merged.columns else "-")
with c4:
    st.metric("Mean dir. acc. test", f"{merged['dir_acc_test'].mean():.2%}" if "dir_acc_test" in merged.columns else "-")

c5, c6, c7, c8 = st.columns(4)
with c5:
    st.metric("Mean mu_pred", f"{pred_df['mu_pred'].mean():.6f}")
with c6:
    st.metric("Median mu_pred", f"{pred_df['mu_pred'].median():.6f}")
with c7:
    st.metric("Min mu_pred", f"{pred_df['mu_pred'].min():.6f}")
with c8:
    st.metric("Max mu_pred", f"{pred_df['mu_pred'].max():.6f}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Rankings", "Filter & Table", "Ticker Detail"])

with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Predicted return distribution")
        fig_hist = px.histogram(pred_df, x="mu_pred", nbins=30)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("RMSE test distribution")
        if "rmse_test" in merged.columns:
            fig_rmse = px.histogram(merged.dropna(subset=["rmse_test"]), x="rmse_test", nbins=30)
            st.plotly_chart(fig_rmse, use_container_width=True)

    st.subheader("Artifacts")
    st.code(
        f"Run ID: {result['run_id']}\\n"
        f"Predictions: {result['predictions_path']}\\n"
        f"Metrics: {result['metrics_path']}\\n"
        f"Per ticker predictions: {result['per_ticker_dir']}\\n"
        f"Loss curves: {result['curves_dir']}"
    )

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Top 5 best model quality")
        quality_df = merged.dropna(subset=["rmse_test"]).sort_values(["rmse_test", "mae_test"], ascending=[True, True]).head(5)
        st.dataframe(
            quality_df[["ticker", "rmse_test", "mae_test", "dir_acc_test", "corr_test", "mu_pred"]],
            use_container_width=True,
            hide_index=True,
        )

        if not quality_df.empty:
            fig_quality = px.bar(quality_df, x="ticker", y="rmse_test", title="Top 5 lowest RMSE test")
            st.plotly_chart(fig_quality, use_container_width=True)

    with c2:
        st.subheader("Top 5 highest predicted returns")
        opp_df = merged.sort_values("mu_pred", ascending=False).head(5)
        st.dataframe(
            opp_df[["ticker", "mu_pred", "rmse_test", "dir_acc_test", "corr_test"]],
            use_container_width=True,
            hide_index=True,
        )

        if not opp_df.empty:
            fig_opp = px.bar(opp_df, x="ticker", y="mu_pred", title="Top 5 highest mu_pred")
            st.plotly_chart(fig_opp, use_container_width=True)

with tab3:
    st.subheader("Global filtered table")

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        search_ticker = st.text_input("Search ticker", "")
    with colf2:
        min_dir_acc = st.slider("Min directional accuracy test", 0.0, 1.0, 0.0, 0.01)
    with colf3:
        max_rmse = st.number_input(
            "Max RMSE test",
            min_value=0.0,
            value=float(np.nanmax(merged["rmse_test"])) if "rmse_test" in merged.columns and merged["rmse_test"].notna().any() else 1.0
        )

    filtered = merged.copy()

    if search_ticker.strip():
        filtered = filtered[filtered["ticker"].astype(str).str.contains(search_ticker.strip(), case=False, na=False)]

    if "dir_acc_test" in filtered.columns:
        filtered = filtered[(filtered["dir_acc_test"].isna()) | (filtered["dir_acc_test"] >= min_dir_acc)]

    if "rmse_test" in filtered.columns:
        filtered = filtered[(filtered["rmse_test"].isna()) | (filtered["rmse_test"] <= max_rmse)]

    filtered = filtered.sort_values(["mu_pred", "dir_acc_test"], ascending=[False, False])
    st.dataframe(filtered, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Ticker detail")

    ticker_options = list(merged["ticker"].dropna().astype(str).unique())
    selected_ticker = st.selectbox("Select ticker", options=ticker_options)

    ticker_row = merged[merged["ticker"] == selected_ticker].iloc[0]
    detail_cols = st.columns(5)
    with detail_cols[0]:
        st.metric("mu_pred", f"{float(ticker_row['mu_pred']):.6f}")
    with detail_cols[1]:
        st.metric("RMSE test", f"{float(ticker_row['rmse_test']):.6f}" if pd.notna(ticker_row.get("rmse_test")) else "-")
    with detail_cols[2]:
        st.metric("DirAcc test", f"{float(ticker_row['dir_acc_test']):.2%}" if pd.notna(ticker_row.get("dir_acc_test")) else "-")
    with detail_cols[3]:
        st.metric("Corr test", f"{float(ticker_row['corr_test']):.4f}" if pd.notna(ticker_row.get("corr_test")) else "-")
    with detail_cols[4]:
        st.metric("Best val loss", f"{float(ticker_row['best_val_loss']):.6f}" if pd.notna(ticker_row.get("best_val_loss")) else "-")

    sub1, sub2 = st.columns(2)

    with sub1:
        st.markdown("**Metrics by split**")
        split_metrics = pd.DataFrame([
            {"split": "train", "mae": ticker_row.get("mae_train"), "rmse": ticker_row.get("rmse_train"), "dir_acc": ticker_row.get("dir_acc_train"), "corr": ticker_row.get("corr_train")},
            {"split": "val", "mae": ticker_row.get("mae_val"), "rmse": ticker_row.get("rmse_val"), "dir_acc": ticker_row.get("dir_acc_val"), "corr": ticker_row.get("corr_val")},
            {"split": "test", "mae": ticker_row.get("mae_test"), "rmse": ticker_row.get("rmse_test"), "dir_acc": ticker_row.get("dir_acc_test"), "corr": ticker_row.get("corr_test")},
        ])
        st.dataframe(split_metrics, use_container_width=True, hide_index=True)

    with sub2:
        prices_t = PRICES[selected_ticker].dropna().copy()
        last_5 = prices_t.tail(5)

        if not last_5.empty:
            last_date = pd.to_datetime(last_5.index[-1])
            future_date = pd.bdate_range(last_date, periods=int(summary.get("forecast_horizon", 5)) + 1)[-1]
            last_price = float(last_5.iloc[-1])
            mu_pred = float(ticker_row["mu_pred"])
            horizon = int(summary.get("forecast_horizon", 5))
            projected_price = float(last_price * ((1 + mu_pred) ** horizon))

            chart_df = pd.DataFrame({
                "date": list(pd.to_datetime(last_5.index)) + [future_date],
                "price": list(last_5.values) + [projected_price],
                "type": ["historical"] * len(last_5) + ["projected"],
            })

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=chart_df[chart_df["type"] == "historical"]["date"],
                y=chart_df[chart_df["type"] == "historical"]["price"],
                mode="lines+markers",
                name="Historical price"
            ))
            fig_price.add_trace(go.Scatter(
                x=[last_5.index[-1], future_date],
                y=[last_price, projected_price],
                mode="lines+markers",
                name="Projection"
            ))
            fig_price.update_layout(title=f"{selected_ticker}: last 5 prices + projected price")
            st.plotly_chart(fig_price, use_container_width=True)

            st.caption(f"Projected price in {horizon} trading days: {projected_price:.4f}")

    pred_file = Path(result["per_ticker_dir"]) / f"{selected_ticker}_predictions.csv"
    if pred_file.exists():
        pred_detail = pd.read_csv(pred_file)
        pred_detail["date"] = pd.to_datetime(pred_detail["date"])

        st.markdown("**Actual vs predicted returns by split**")
        split_choice = st.multiselect("Splits to show", options=["train", "val", "test"], default=["val", "test"])

        plot_detail = pred_detail[pred_detail["split"].isin(split_choice)].copy()

        if not plot_detail.empty:
            fig_det = go.Figure()
            for split in split_choice:
                sub = plot_detail[plot_detail["split"] == split]
                if not sub.empty:
                    fig_det.add_trace(go.Scatter(x=sub["date"], y=sub["y_true"], mode="lines", name=f"y_true_{split}"))
                    fig_det.add_trace(go.Scatter(x=sub["date"], y=sub["y_pred"], mode="lines", name=f"y_pred_{split}"))
            fig_det.update_layout(title=f"{selected_ticker}: actual vs predicted returns")
            st.plotly_chart(fig_det, use_container_width=True)

            st.dataframe(plot_detail.tail(30), use_container_width=True, hide_index=True)

    loss_file = Path(result["curves_dir"]) / f"{selected_ticker}_losses.csv"
    if loss_file.exists():
        losses = pd.read_csv(loss_file)
        st.markdown("**Loss curves**")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=losses["epoch"], y=losses["train_loss"], mode="lines", name="train_loss"))
        fig_loss.add_trace(go.Scatter(x=losses["epoch"], y=losses["val_loss"], mode="lines", name="val_loss"))
        fig_loss.update_layout(title=f"{selected_ticker}: train vs val loss")
        st.plotly_chart(fig_loss, use_container_width=True)
""")

for rel_path, content in files.items():
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    print(f"WROTE: {rel_path}")

print("\\nDone. Now run:")
print(r"C:\\Users\\lucas\\anaconda3\\envs\\IIA_cuda_env\\python.exe -c \"from app.lstm_lab_storage import ensure_lstm_lab_tables; ensure_lstm_lab_tables(); print('DB OK')\"")
print("docker compose restart streamlit fastapi")