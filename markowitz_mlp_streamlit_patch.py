from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil


def locate_repo_root() -> Path:
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "app").exists() and (candidate / "src").exists() and (candidate / "Data").exists():
            return candidate
    raise RuntimeError("No pude encontrar la raíz del repo. Ejecutá este script desde la carpeta raíz de Markowitz.")


def backup_file(path: Path, tag: str = "mlp_lab_fix") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{tag}_{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


MLP_LAB_STORAGE = r'''from __future__ import annotations

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
    merged = metrics_df.merge(predictions_df, on="ticker", how="outer")
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


def _apply_selection_policy(df: pd.DataFrame, selection_policy: dict | None) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["use_for_optimizer"] = []
        return out

    policy = selection_policy or {"policy_type": "all_assets"}
    policy_type = str(policy.get("policy_type", "all_assets")).strip().lower()
    dir_acc = pd.to_numeric(out.get("dir_acc_test"), errors="coerce")
    corr = pd.to_numeric(out.get("corr_test"), errors="coerce")
    rmse = pd.to_numeric(out.get("rmse_test"), errors="coerce")
    score = pd.to_numeric(out.get("score"), errors="coerce")

    if policy_type == "all_assets":
        mask = pd.Series(True, index=out.index)
    elif policy_type == "dir_acc_min":
        mask = dir_acc >= float(policy.get("dir_acc_min", 0.52))
    elif policy_type == "corr_and_rmse":
        mask = (corr >= float(policy.get("corr_min", 0.0))) & (rmse <= float(policy.get("rmse_max", rmse.max())))
    elif policy_type == "composite_min_score":
        mask = score >= float(policy.get("min_score", 0.60))
    elif policy_type == "composite_top_n":
        top_n = int(policy.get("top_n", max(1, len(out) // 3)))
        order = score.rank(method="first", ascending=False)
        mask = order <= top_n
    elif policy_type == "hybrid_quality":
        mask = (
            (dir_acc >= float(policy.get("dir_acc_min", 0.52)))
            & (corr >= float(policy.get("corr_min", 0.0)))
            & (score >= float(policy.get("min_score", 0.50)))
        )
    else:
        mask = pd.Series(True, index=out.index)

    out["use_for_optimizer"] = mask.fillna(False).astype(bool)
    return out


def load_active_mlp_predictions_df() -> pd.DataFrame:
    active = get_active_mlp_prediction_run()
    if active is None:
        return pd.DataFrame()
    df = get_mlp_run_tickers_df(active["run_id"])
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return _apply_selection_policy(df, active.get("selection_policy_json"))
'''

MLP_LAB_LOADER = r'''from __future__ import annotations

import pandas as pd
from app.mlp_lab_storage import load_active_mlp_predictions_df


def load_active_mlp_predictions() -> pd.DataFrame:
    return load_active_mlp_predictions_df()
'''

TRAIN_RETURNS_MLP_LAB = r'''from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.data.load_data import load_returns
from app.mlp_lab_storage import persist_mlp_lab_run

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "mlp_lab" / "runs"
MODEL_ROOT = ROOT / "models_artifacts" / "returns_mlp_lab" / "runs"

warnings.filterwarnings("ignore", category=ConvergenceWarning)


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


def r2_safe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def positive_precision(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_pred > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(y_true[mask] > 0))


def pred_positive_rate(y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_pred) == 0:
        return np.nan
    return float(np.mean(y_pred > 0))


def _parse_hidden_layers(hidden_layers):
    if isinstance(hidden_layers, tuple):
        return hidden_layers
    if isinstance(hidden_layers, list):
        return tuple(int(v) for v in hidden_layers if int(v) > 0)
    raw = str(hidden_layers).strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    parsed = tuple(int(p) for p in parts if int(p) > 0)
    if not parsed:
        raise ValueError("hidden_layers debe tener al menos una capa, por ejemplo 256,128,64")
    return parsed


def build_scaler(name: str):
    name = str(name).strip().lower()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler(feature_range=(-1, 1))
    return "passthrough"


def build_mlp_feature_frame(series: pd.Series, lags: int, short_window: int, long_window: int, include_abs_features: bool, include_square_features: bool, include_sign_features: bool) -> pd.DataFrame:
    s = series.dropna().astype(float).sort_index()
    lagged = s.shift(1)
    df = pd.DataFrame(index=s.index)
    df["ret"] = s
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = s.shift(lag)
    df[f"roll_mean_{short_window}"] = lagged.rolling(short_window).mean()
    df[f"roll_mean_{long_window}"] = lagged.rolling(long_window).mean()
    df[f"roll_std_{short_window}"] = lagged.rolling(short_window).std()
    df[f"roll_std_{long_window}"] = lagged.rolling(long_window).std()
    df[f"roll_min_{short_window}"] = lagged.rolling(short_window).min()
    df[f"roll_min_{long_window}"] = lagged.rolling(long_window).min()
    df[f"roll_max_{short_window}"] = lagged.rolling(short_window).max()
    df[f"roll_max_{long_window}"] = lagged.rolling(long_window).max()
    df[f"momentum_{short_window}"] = lagged.rolling(short_window).sum()
    df[f"momentum_{long_window}"] = lagged.rolling(long_window).sum()
    df[f"ewm_{short_window}"] = lagged.ewm(span=short_window, adjust=False).mean()
    df[f"ewm_{long_window}"] = lagged.ewm(span=long_window, adjust=False).mean()
    denom_s = df[f"roll_std_{short_window}"].replace(0, np.nan)
    denom_l = df[f"roll_std_{long_window}"].replace(0, np.nan)
    df[f"zscore_{short_window}"] = (lagged - df[f"roll_mean_{short_window}"]) / denom_s
    df[f"zscore_{long_window}"] = (lagged - df[f"roll_mean_{long_window}"]) / denom_l
    if include_abs_features:
        abs_lagged = lagged.abs()
        df["abs_ret_1"] = abs_lagged
        df[f"roll_abs_mean_{short_window}"] = abs_lagged.rolling(short_window).mean()
        df[f"roll_abs_mean_{long_window}"] = abs_lagged.rolling(long_window).mean()
    if include_square_features:
        sq_lagged = lagged ** 2
        df["sq_ret_1"] = sq_lagged
        df[f"roll_sq_mean_{short_window}"] = sq_lagged.rolling(short_window).mean()
        df[f"roll_sq_mean_{long_window}"] = sq_lagged.rolling(long_window).mean()
    if include_sign_features:
        df["sign_1"] = np.sign(lagged)
        df[f"sign_mean_{short_window}"] = np.sign(lagged).rolling(short_window).mean()
        df[f"sign_mean_{long_window}"] = np.sign(lagged).rolling(long_window).mean()
    return df


def build_target(series: pd.Series, forecast_horizon: int, target_aggregation: str) -> pd.Series:
    future_steps = [series.shift(-i) for i in range(1, forecast_horizon + 1)]
    future_df = pd.concat(future_steps, axis=1)
    if target_aggregation == "sum":
        return future_df.sum(axis=1)
    return future_df.mean(axis=1)


def temporal_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def metric_bundle(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
        "corr": corrcoef_safe(y_true, y_pred),
        "r2": r2_safe(y_true, y_pred),
    }


def _rank_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    if higher_is_better:
        return s.rank(pct=True, ascending=True).fillna(0.5)
    return (1.0 - s.rank(pct=True, ascending=True)).fillna(0.5)


def build_composite_score(df: pd.DataFrame) -> pd.Series:
    mu_rank = _rank_score(df["mu_pred"], higher_is_better=True)
    dir_rank = _rank_score(df["dir_acc_test"], higher_is_better=True)
    corr_rank = _rank_score(df["corr_test"].fillna(0.0), higher_is_better=True)
    rmse_rank = _rank_score(df["rmse_test"], higher_is_better=False)
    pos_rank = _rank_score(df["positive_precision_test"].fillna(0.0), higher_is_better=True)
    return (0.35 * mu_rank + 0.25 * dir_rank + 0.15 * corr_rank + 0.15 * rmse_rank + 0.10 * pos_rank).astype(float)


def run_mlp_lab(*, start_date: str = "2022-01-01", forecast_horizon: int = 5, target_aggregation: str = "mean", lags: int = 20, short_window: int = 5, long_window: int = 20, hidden_layers = "256,128,64", activation: str = "relu", solver: str = "adam", scaler: str = "standard", alpha: float = 1e-4, learning_rate: str = "adaptive", learning_rate_init: float = 1e-3, max_iter: int = 500, batch_size = "auto", early_stopping: bool = True, validation_fraction: float = 0.15, n_iter_no_change: int = 20, clip_pred: float = 0.02, winsorize_quantile: float = 0.01, include_abs_features: bool = True, include_square_features: bool = True, include_sign_features: bool = True, random_state: int = 42):
    returns = load_returns()
    if start_date and start_date != "full":
        returns = returns.loc[start_date:]
    hidden_layers_tuple = _parse_hidden_layers(hidden_layers)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = OUTPUT_ROOT / run_id
    run_model_dir = MODEL_ROOT / run_id
    per_ticker_dir = run_output_dir / "per_ticker"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)
    per_ticker_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = []
    metrics_rows = []
    total_tickers = len(list(returns.columns))

    for ticker in returns.columns:
        try:
            series = returns[ticker].dropna().astype(float).copy()
            if len(series) < max(long_window + forecast_horizon + 30, 180):
                continue
            if winsorize_quantile and winsorize_quantile > 0:
                q_low = float(series.quantile(winsorize_quantile))
                q_high = float(series.quantile(1.0 - winsorize_quantile))
                series = series.clip(lower=q_low, upper=q_high)
            feat_df = build_mlp_feature_frame(series, lags, short_window, long_window, include_abs_features, include_square_features, include_sign_features)
            feat_df["target"] = build_target(series, forecast_horizon, target_aggregation)
            aligned = feat_df.dropna().copy()
            if len(aligned) < 150:
                continue
            train_df, val_df, test_df = temporal_split(aligned)
            if min(len(train_df), len(val_df), len(test_df)) == 0:
                continue

            feature_cols = [c for c in aligned.columns if c not in ["ret", "target"]]
            X_train = train_df[feature_cols]
            y_train = train_df["target"]
            X_val = val_df[feature_cols]
            y_val = val_df["target"]
            X_test = test_df[feature_cols]
            y_test = test_df["target"]

            model = Pipeline([
                ("scaler", build_scaler(scaler)),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=hidden_layers_tuple,
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=early_stopping,
                    validation_fraction=validation_fraction,
                    n_iter_no_change=n_iter_no_change,
                )),
            ])
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)
            pred_latest = float(model.predict(aligned[feature_cols].tail(1))[0])
            pred_latest = max(min(pred_latest, clip_pred), -clip_pred)

            detail_df = pd.concat([
                pd.DataFrame({"date": train_df.index, "split": "train", "y_true": y_train.values, "y_pred": pred_train}),
                pd.DataFrame({"date": val_df.index, "split": "val", "y_true": y_val.values, "y_pred": pred_val}),
                pd.DataFrame({"date": test_df.index, "split": "test", "y_true": y_test.values, "y_pred": pred_test}),
            ], ignore_index=True)
            detail_df.to_csv(per_ticker_dir / f"{ticker}_predictions.csv", index=False)
            joblib.dump(model, run_model_dir / f"{ticker}_mlp_pipeline.joblib")

            pred_rows.append({"ticker": ticker, "mu_pred": pred_latest})
            metrics_train = metric_bundle(y_train, pred_train)
            metrics_val = metric_bundle(y_val, pred_val)
            metrics_test = metric_bundle(y_test, pred_test)
            metrics_rows.append({
                "ticker": ticker,
                "rmse_train": metrics_train["rmse"], "rmse_val": metrics_val["rmse"], "rmse_test": metrics_test["rmse"],
                "mae_train": metrics_train["mae"], "mae_val": metrics_val["mae"], "mae_test": metrics_test["mae"],
                "dir_acc_train": metrics_train["dir_acc"], "dir_acc_val": metrics_val["dir_acc"], "dir_acc_test": metrics_test["dir_acc"],
                "corr_train": metrics_train["corr"], "corr_val": metrics_val["corr"], "corr_test": metrics_test["corr"],
                "r2_train": metrics_train["r2"], "r2_val": metrics_val["r2"], "r2_test": metrics_test["r2"],
                "positive_precision_test": positive_precision(y_test, pred_test),
                "pred_positive_rate_test": pred_positive_rate(pred_test),
            })
        except Exception:
            continue

    predictions_df = pd.DataFrame(pred_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    if predictions_df.empty or metrics_df.empty:
        raise RuntimeError("MLP Lab no pudo generar predicciones válidas para ningún ticker.")

    merged = metrics_df.merge(predictions_df, on="ticker", how="inner")
    merged["score"] = build_composite_score(merged)
    predictions_df = merged[["ticker", "mu_pred", "score", "dir_acc_test", "corr_test", "rmse_test", "positive_precision_test"]].copy()
    metrics_df = merged[[
        "ticker", "rmse_train", "rmse_val", "rmse_test", "mae_train", "mae_val", "mae_test",
        "dir_acc_train", "dir_acc_val", "dir_acc_test", "corr_train", "corr_val", "corr_test",
        "r2_train", "r2_val", "r2_test", "positive_precision_test", "pred_positive_rate_test", "score", "mu_pred",
    ]].copy()

    predictions_path = run_output_dir / "predictions.csv"
    metrics_path = run_output_dir / "metrics.csv"
    predictions_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    config = {
        "start_date": start_date,
        "forecast_horizon": forecast_horizon,
        "target_aggregation": target_aggregation,
        "lags": lags,
        "short_window": short_window,
        "long_window": long_window,
        "hidden_layers": list(hidden_layers_tuple),
        "activation": activation,
        "solver": solver,
        "scaler": scaler,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "batch_size": str(batch_size),
        "early_stopping": early_stopping,
        "validation_fraction": validation_fraction,
        "n_iter_no_change": n_iter_no_change,
        "clip_pred": clip_pred,
        "winsorize_quantile": winsorize_quantile,
        "include_abs_features": include_abs_features,
        "include_square_features": include_square_features,
        "include_sign_features": include_sign_features,
        "selection_policy": {"policy_type": "all_assets"},
    }
    summary = {
        **config,
        "run_id": run_id,
        "total_tickers_in_universe": total_tickers,
        "tickers_with_predictions": int(len(predictions_df)),
        "mean_mu_pred": float(predictions_df["mu_pred"].mean()),
        "min_mu_pred": float(predictions_df["mu_pred"].min()),
        "max_mu_pred": float(predictions_df["mu_pred"].max()),
        "mean_rmse_test": float(metrics_df["rmse_test"].mean()),
        "mean_dir_acc_test": float(metrics_df["dir_acc_test"].mean()),
        "mean_corr_test": float(metrics_df["corr_test"].fillna(0.0).mean()),
        "mean_score": float(metrics_df["score"].mean()),
    }

    (run_output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest_ptr = ROOT / "outputs" / "mlp_lab" / "latest_run.json"
    latest_ptr.parent.mkdir(parents=True, exist_ok=True)
    latest_ptr.write_text(json.dumps({"run_id": run_id, "run_output_dir": str(run_output_dir), "run_model_dir": str(run_model_dir)}, indent=2), encoding="utf-8")

    persist_mlp_lab_run(run_id=run_id, summary=summary, predictions_df=predictions_df, metrics_df=metrics_df, run_output_dir=str(run_output_dir), run_model_dir=str(run_model_dir))

    return {
        "run_id": run_id,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "per_ticker_dir": str(per_ticker_dir),
        "run_output_dir": str(run_output_dir),
        "run_model_dir": str(run_model_dir),
        "predictions": predictions_df,
        "metrics": metrics_df,
        "summary": summary,
    }
'''

MLP_LAB_PAGE = r'''from __future__ import annotations

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

from src.models.train_returns_mlp_lab import run_mlp_lab
from app.mlp_lab_storage import get_mlp_prediction_runs_df, set_active_mlp_prediction_run


def get_prices(returns: pd.DataFrame) -> pd.DataFrame:
    global load_prices
    if load_prices is not None:
        try:
            p = load_prices()
            if isinstance(p, pd.DataFrame) and not p.empty:
                return p
        except Exception:
            pass
    return (1 + returns.fillna(0)).cumprod() * 100


def load_latest_saved_result():
    latest_path = ROOT / "outputs" / "mlp_lab" / "latest_run.json"
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
            "run_output_dir": str(run_output_dir),
            "predictions": pd.read_csv(predictions_path),
            "metrics": pd.read_csv(metrics_path),
            "summary": {**config, **summary},
        }
    except Exception:
        return None


def build_config() -> dict:
    with st.sidebar:
        st.header("MLP Lab configuration")
        start_date = st.selectbox("Training data start date", ["full", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"], index=3)
        forecast_horizon = st.slider("Forecast horizon (trading days)", 1, 10, 5)
        target_aggregation = st.radio("Target aggregation", ["mean", "sum"], horizontal=True)

        st.subheader("Feature engineering")
        lags = st.slider("Number of lags", 5, 60, 20, step=5)
        short_window = st.slider("Short rolling window", 3, 20, 5)
        long_window = st.slider("Long rolling window", 10, 60, 20)
        include_abs_features = st.checkbox("Include absolute-return features", value=True)
        include_square_features = st.checkbox("Include squared-return features", value=True)
        include_sign_features = st.checkbox("Include sign features", value=True)
        winsorize_quantile = st.slider("Winsorize quantile", 0.0, 0.10, 0.01, 0.01)
        clip_pred = st.slider("Clip final mu_pred", 0.005, 0.05, 0.02, 0.005)

        st.subheader("MLP architecture")
        hidden_layers = st.text_input("Hidden layers (comma separated)", value="256,128,64")
        activation = st.selectbox("Activation", ["relu", "tanh", "logistic", "identity"], index=0)
        solver = st.selectbox("Solver", ["adam", "sgd", "lbfgs"], index=0)
        scaler = st.selectbox("Scaler", ["standard", "robust", "minmax", "none"], index=0)
        alpha = st.selectbox("L2 alpha", [1e-5, 1e-4, 1e-3, 1e-2], index=1)
        learning_rate = st.selectbox("Learning-rate schedule", ["constant", "adaptive", "invscaling"], index=1)
        learning_rate_init = st.selectbox("Learning-rate init", [1e-4, 5e-4, 1e-3, 5e-3], index=2)
        max_iter = st.slider("Max iterations", 100, 2000, 500, step=100)
        batch_size = st.selectbox("Batch size", ["auto", 16, 32, 64, 128], index=0)
        early_stopping = st.checkbox("Early stopping", value=True)
        validation_fraction = st.slider("Internal validation fraction", 0.05, 0.30, 0.15, 0.05)
        n_iter_no_change = st.slider("n_iter_no_change", 5, 50, 20, step=5)

    return {
        "start_date": start_date,
        "forecast_horizon": forecast_horizon,
        "target_aggregation": target_aggregation,
        "lags": lags,
        "short_window": short_window,
        "long_window": long_window,
        "include_abs_features": include_abs_features,
        "include_square_features": include_square_features,
        "include_sign_features": include_sign_features,
        "winsorize_quantile": winsorize_quantile,
        "clip_pred": clip_pred,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "solver": solver,
        "scaler": scaler,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "batch_size": batch_size,
        "early_stopping": early_stopping,
        "validation_fraction": validation_fraction,
        "n_iter_no_change": n_iter_no_change,
    }


def render_overview(pred_df: pd.DataFrame, merged: pd.DataFrame, result: dict):
    left, right = st.columns(2)
    with left:
        st.subheader("Predicted return distribution")
        st.plotly_chart(px.histogram(pred_df, x="mu_pred", nbins=30), use_container_width=True)
    with right:
        st.subheader("Composite quality score distribution")
        st.plotly_chart(px.histogram(merged, x="score", nbins=30), use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("RMSE test distribution")
        st.plotly_chart(px.histogram(merged.dropna(subset=["rmse_test"]), x="rmse_test", nbins=30), use_container_width=True)
    with right2:
        st.subheader("Directional accuracy test distribution")
        st.plotly_chart(px.histogram(merged.dropna(subset=["dir_acc_test"]), x="dir_acc_test", nbins=30), use_container_width=True)

    st.code(
        f"Run ID: {result['run_id']}\n"
        f"Predictions: {result['predictions_path']}\n"
        f"Metrics: {result['metrics_path']}\n"
        f"Per-ticker detail: {result['per_ticker_dir']}"
    )


def render_rankings(merged: pd.DataFrame):
    left, right = st.columns(2)
    with left:
        st.subheader("Top 10 composite score")
        top_score = merged.sort_values("score", ascending=False).head(10)
        st.dataframe(top_score[["ticker", "score", "mu_pred", "dir_acc_test", "corr_test", "rmse_test"]], use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(top_score, x="ticker", y="score", title="Top 10 by composite score"), use_container_width=True)
    with right:
        st.subheader("Top 10 predicted returns")
        top_mu = merged.sort_values("mu_pred", ascending=False).head(10)
        st.dataframe(top_mu[["ticker", "mu_pred", "score", "dir_acc_test", "corr_test", "rmse_test"]], use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(top_mu, x="ticker", y="mu_pred", title="Top 10 by mu_pred"), use_container_width=True)


def render_filter_table(merged: pd.DataFrame):
    st.subheader("Global filter table")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        search_ticker = st.text_input("Search ticker", "")
    with c2:
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.01)
    with c3:
        min_dir_acc = st.slider("Min dir. acc. test", 0.0, 1.0, 0.0, 0.01)
    with c4:
        max_rmse = st.number_input("Max RMSE test", min_value=0.0, value=float(np.nanmax(merged["rmse_test"])) if merged["rmse_test"].notna().any() else 1.0)

    filtered = merged.copy()
    if search_ticker.strip():
        filtered = filtered[filtered["ticker"].astype(str).str.contains(search_ticker.strip(), case=False, na=False)]
    filtered = filtered[(filtered["score"].isna()) | (filtered["score"] >= min_score)]
    filtered = filtered[(filtered["dir_acc_test"].isna()) | (filtered["dir_acc_test"] >= min_dir_acc)]
    filtered = filtered[(filtered["rmse_test"].isna()) | (filtered["rmse_test"] <= max_rmse)]
    filtered = filtered.sort_values(["score", "mu_pred"], ascending=[False, False])
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_ticker_detail(merged: pd.DataFrame, result: dict, summary: dict, prices: pd.DataFrame):
    st.subheader("Ticker detail")
    ticker_options = list(merged["ticker"].dropna().astype(str).unique())
    if not ticker_options:
        st.info("No ticker details available.")
        return
    selected_ticker = st.selectbox("Select ticker", options=ticker_options)
    row = merged[merged["ticker"] == selected_ticker].iloc[0]

    metric_cols = st.columns(6)
    metric_cols[0].metric("mu_pred", f"{float(row['mu_pred']):.6f}")
    metric_cols[1].metric("Score", f"{float(row['score']):.3f}")
    metric_cols[2].metric("RMSE test", f"{float(row['rmse_test']):.6f}" if pd.notna(row.get("rmse_test")) else "-")
    metric_cols[3].metric("DirAcc test", f"{float(row['dir_acc_test']):.2%}" if pd.notna(row.get("dir_acc_test")) else "-")
    metric_cols[4].metric("Corr test", f"{float(row['corr_test']):.4f}" if pd.notna(row.get("corr_test")) else "-")
    metric_cols[5].metric("R2 test", f"{float(row['r2_test']):.4f}" if pd.notna(row.get("r2_test")) else "-")

    left, right = st.columns(2)
    with left:
        split_metrics = pd.DataFrame([
            {"split": "train", "mae": row.get("mae_train"), "rmse": row.get("rmse_train"), "dir_acc": row.get("dir_acc_train"), "corr": row.get("corr_train"), "r2": row.get("r2_train")},
            {"split": "val", "mae": row.get("mae_val"), "rmse": row.get("rmse_val"), "dir_acc": row.get("dir_acc_val"), "corr": row.get("corr_val"), "r2": row.get("r2_val")},
            {"split": "test", "mae": row.get("mae_test"), "rmse": row.get("rmse_test"), "dir_acc": row.get("dir_acc_test"), "corr": row.get("corr_test"), "r2": row.get("r2_test")},
        ])
        st.dataframe(split_metrics, use_container_width=True, hide_index=True)
    with right:
        if selected_ticker in prices.columns:
            prices_t = prices[selected_ticker].dropna().copy()
            last_5 = prices_t.tail(5)
            if not last_5.empty:
                last_date = pd.to_datetime(last_5.index[-1])
                horizon = int(summary.get("forecast_horizon", 5))
                future_date = pd.bdate_range(last_date, periods=horizon + 1)[-1]
                last_price = float(last_5.iloc[-1])
                mu_pred = float(row["mu_pred"])
                projected_price = float(last_price * ((1 + mu_pred) ** horizon))
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=last_5.index, y=last_5.values, mode="lines+markers", name="Historical price"))
                fig_price.add_trace(go.Scatter(x=[last_5.index[-1], future_date], y=[last_price, projected_price], mode="lines+markers", name="Projection"))
                fig_price.update_layout(title=f"{selected_ticker}: last 5 prices + projected price")
                st.plotly_chart(fig_price, use_container_width=True)
                st.caption(f"Projected price in {horizon} trading days: {projected_price:.4f}")

    pred_file = Path(result["per_ticker_dir"]) / f"{selected_ticker}_predictions.csv"
    if pred_file.exists():
        pred_detail = pd.read_csv(pred_file)
        pred_detail["date"] = pd.to_datetime(pred_detail["date"])
        st.markdown("**Actual vs predicted returns by split**")
        split_choice = st.multiselect("Splits to show", ["train", "val", "test"], default=["val", "test"])
        plot_df = pred_detail[pred_detail["split"].isin(split_choice)].copy()
        if not plot_df.empty:
            fig = go.Figure()
            for split in split_choice:
                sub = plot_df[plot_df["split"] == split]
                if not sub.empty:
                    fig.add_trace(go.Scatter(x=sub["date"], y=sub["y_true"], mode="lines", name=f"y_true_{split}"))
                    fig.add_trace(go.Scatter(x=sub["date"], y=sub["y_pred"], mode="lines", name=f"y_pred_{split}"))
            fig.update_layout(title=f"{selected_ticker}: actual vs predicted returns")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(plot_df.tail(40), use_container_width=True, hide_index=True)


def build_selection_policy_ui(runs_df: pd.DataFrame) -> dict:
    policy_type = st.selectbox(
        "Optimizer selection policy",
        ["all_assets", "dir_acc_min", "corr_and_rmse", "composite_min_score", "composite_top_n", "hybrid_quality"],
        help=(
            "all_assets: usa mu_pred para todos.\n"
            "dir_acc_min: solo activos con directional accuracy mínima.\n"
            "corr_and_rmse: requiere correlación mínima y RMSE máxima.\n"
            "composite_min_score: filtra por score compuesto.\n"
            "composite_top_n: usa solo los top N por score.\n"
            "hybrid_quality: combina dir_acc, corr y score."
        ),
    )
    if policy_type == "all_assets":
        return {"policy_type": policy_type}
    if policy_type == "dir_acc_min":
        return {"policy_type": policy_type, "dir_acc_min": st.slider("dir_acc_min", 0.0, 1.0, 0.52, 0.01)}
    if policy_type == "corr_and_rmse":
        rmse_default = float(runs_df["mean_rmse_test"].dropna().median()) if not runs_df.empty and runs_df["mean_rmse_test"].notna().any() else 0.02
        return {"policy_type": policy_type, "corr_min": st.slider("corr_min", -1.0, 1.0, 0.0, 0.01), "rmse_max": st.number_input("rmse_max", min_value=0.0, value=rmse_default)}
    if policy_type == "composite_min_score":
        return {"policy_type": policy_type, "min_score": st.slider("min_score", 0.0, 1.0, 0.60, 0.01)}
    if policy_type == "composite_top_n":
        return {"policy_type": policy_type, "top_n": int(st.number_input("top_n", min_value=1, value=10, step=1))}
    return {
        "policy_type": policy_type,
        "dir_acc_min": st.slider("hybrid dir_acc_min", 0.0, 1.0, 0.52, 0.01),
        "corr_min": st.slider("hybrid corr_min", -1.0, 1.0, 0.0, 0.01),
        "min_score": st.slider("hybrid min_score", 0.0, 1.0, 0.55, 0.01),
    }


def main():
    st.set_page_config(page_title="MLP Lab", page_icon="🧠", layout="wide")
    st.title("MLP Lab")
    st.caption(
        "Esta página usa directamente Data/returns.csv, guarda historial de runs, permite elegir un run activo para Markowitz y muestra detalle por activo. "
        "En ai_returns, Markowitz consumirá el run activo y aplicará la selection policy elegida."
    )

    returns = load_returns()
    prices = get_prices(returns)
    cfg = build_config()

    st.info(
        f"""**Universe:** `{len(returns.columns)}` assets  
**Target:** `{cfg['target_aggregation']}` of next `{cfg['forecast_horizon']}` trading-day returns  
**Feature set:** lags=`{cfg['lags']}`, short_window=`{cfg['short_window']}`, long_window=`{cfg['long_window']}`  
**Network:** hidden_layers=`{cfg['hidden_layers']}`, activation=`{cfg['activation']}`, solver=`{cfg['solver']}`, scaler=`{cfg['scaler']}`"""
    )

    if st.button("Train and generate full-universe MLP predictions", use_container_width=True):
        with st.spinner("Training configurable MLP models for all available tickers..."):
            result = run_mlp_lab(**cfg)
        st.session_state["mlp_lab_result"] = result

    if "mlp_lab_result" not in st.session_state:
        latest = load_latest_saved_result()
        if latest is not None:
            st.session_state["mlp_lab_result"] = latest

    result = st.session_state.get("mlp_lab_result")
    runs_df = get_mlp_prediction_runs_df()

    with st.expander("Stored MLP prediction runs", expanded=False):
        if runs_df.empty:
            st.info("No MLP runs stored yet.")
        else:
            st.dataframe(runs_df, use_container_width=True, hide_index=True)
            selected_run = st.selectbox("Set active run for optimizer", runs_df["run_id"].tolist())
            selection_policy = build_selection_policy_ui(runs_df)
            if st.button("Set selected MLP run as active"):
                set_active_mlp_prediction_run(selected_run, selection_policy)
                st.success(f"Active MLP run set to: {selected_run}")
                st.rerun()

    if result is None:
        st.warning("No MLP Lab run available yet. Launch one from this page.")
        return

    pred_df = result["predictions"].copy()
    metrics_df = result["metrics"].copy()
    summary = result.get("summary", {})
    merged = metrics_df.copy()

    st.success(f"MLP run available: {result['run_id']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe size", int(summary.get("total_tickers_in_universe", len(returns.columns))))
    c2.metric("Tickers predicted", int(summary.get("tickers_with_predictions", len(pred_df))))
    c3.metric("Mean RMSE test", f"{merged['rmse_test'].mean():.6f}")
    c4.metric("Mean dir. acc. test", f"{merged['dir_acc_test'].mean():.2%}")
    c5.metric("Mean score", f"{merged['score'].mean():.3f}")

    t1, t2, t3, t4 = st.tabs(["Overview", "Rankings", "Filter & Table", "Ticker Detail"])
    with t1:
        render_overview(pred_df, merged, result)
    with t2:
        render_rankings(merged)
    with t3:
        render_filter_table(merged)
    with t4:
        render_ticker_detail(merged, result, summary, prices)


if __name__ == "__main__":
    main()
'''

INPUTS_BUILDER = r'''from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.optimization.lstm_lab_loader import load_active_lstm_predictions_df
from src.optimization.mlp_lab_loader import load_active_mlp_predictions

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "outputs" / "predictions"


def _read_prediction_csv(filename: str) -> pd.DataFrame:
    path = PRED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Generalo primero desde el pipeline correspondiente.")
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{filename} debe contener la columna 'ticker'.")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


def load_predictions(prediction_file: str = "predictions.csv") -> pd.DataFrame:
    return _read_prediction_csv(prediction_file)


def load_volatility_predictions(prediction_file: str = "volatility_predictions.csv") -> pd.DataFrame:
    preds = _read_prediction_csv(prediction_file)
    required_cols = {"ticker", "vol_pred"}
    missing = required_cols - set(preds.columns)
    if missing:
        raise ValueError(f"{prediction_file} no tiene columnas requeridas: {sorted(missing)}")
    preds["vol_pred"] = preds["vol_pred"].astype(float)
    preds = preds.dropna(subset=["vol_pred"]).copy()
    preds = preds.drop_duplicates(subset=["ticker"], keep="last").copy()
    return preds


def _build_ai_vol_cov(returns_subset: pd.DataFrame, pred_vol: pd.Series) -> pd.DataFrame:
    hist_cov = returns_subset.cov()
    hist_vol = returns_subset.std().replace(0, np.nan)
    corr = hist_cov.div(hist_vol, axis=0).div(hist_vol, axis=1)
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d = pd.Series(pred_vol, index=returns_subset.columns, dtype=float)
    cov_ai = corr.mul(d, axis=0).mul(d, axis=1)
    return (cov_ai + cov_ai.T) / 2.0


def _build_ai_returns_from_active_mlp(returns: pd.DataFrame):
    preds = load_active_mlp_predictions().copy()
    if preds.empty:
        raise ValueError("No hay un run activo de MLP Lab para usar en ai_returns.")
    preds["ticker"] = preds["ticker"].astype(str).str.strip()
    preds = preds.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")
    if "mu_pred" not in preds.columns:
        raise ValueError("Active MLP Lab predictions must contain column 'mu_pred'.")
    common = [t for t in returns.columns if t in preds.index]
    if not common:
        raise ValueError("No hay tickers en común entre returns.csv y el run activo de MLP Lab.")
    returns_sub = returns[common].copy()
    preds = preds.loc[common].copy()
    hist_mu = returns_sub.mean()
    mu = hist_mu.copy()
    use_mask = preds["use_for_optimizer"].astype(bool) if "use_for_optimizer" in preds.columns else pd.Series(True, index=preds.index)
    mu.loc[use_mask.index[use_mask]] = preds.loc[use_mask, "mu_pred"].astype(float)
    cov = returns_sub.cov()
    return mu.loc[common], cov.loc[common, common], common


def build_model_inputs(model_type: str, returns: pd.DataFrame):
    model_type = str(model_type).strip().lower()
    returns = returns.copy()
    returns.columns = returns.columns.astype(str).str.strip()

    if model_type == "classic":
        mu = returns.mean()
        cov = returns.cov()
        return mu, cov, list(mu.index)

    if model_type == "ai_returns":
        try:
            return _build_ai_returns_from_active_mlp(returns)
        except Exception:
            preds = load_predictions("predictions.csv").set_index("ticker")
            if "mu_pred" not in preds.columns:
                raise ValueError("predictions.csv debe contener columna 'mu_pred'")
            common = [t for t in returns.columns if t in preds.index]
            if not common:
                raise ValueError("No hay tickers en común entre returns.csv y predictions.csv")
            returns_sub = returns[common].copy()
            mu = preds.loc[common, "mu_pred"].astype(float)
            cov = returns_sub.cov()
            return mu, cov, common

    if model_type == "ai_returns_lstm":
        preds = load_active_lstm_predictions_df().copy()
        preds["ticker"] = preds["ticker"].astype(str).str.strip()
        preds = preds.set_index("ticker")
        if "mu_pred" not in preds.columns:
            raise ValueError("Active LSTM Lab predictions must contain column 'mu_pred'")
        if "dir_acc_test" not in preds.columns:
            raise ValueError("Active LSTM Lab predictions must contain column 'dir_acc_test'")
        common = [t for t in returns.columns if t in preds.index]
        if not common:
            raise ValueError("No active LSTM Lab predictions available for optimizer universe.")
        returns_sub = returns[common].copy()
        preds = preds.loc[common].copy()
        hist_mu = returns_sub.mean()
        mu = hist_mu.copy()
        good_mask = preds["dir_acc_test"].astype(float) > 0.52
        mu.loc[good_mask.index[good_mask]] = preds.loc[good_mask, "mu_pred"].astype(float)
        cov = returns_sub.cov()
        return mu.loc[common], cov.loc[common, common], common

    if model_type == "ai_volatility":
        preds = load_volatility_predictions("volatility_predictions.csv").set_index("ticker")
        common = [t for t in returns.columns if t in preds.index]
        if not common:
            raise ValueError("No hay tickers en común entre returns.csv y volatility_predictions.csv")
        returns_sub = returns[common].copy()
        mu = returns_sub.mean()
        pred_vol = preds.loc[common, "vol_pred"].astype(float)
        cov = _build_ai_vol_cov(returns_sub, pred_vol)
        return mu.loc[common], cov.loc[common, common], common

    raise ValueError(f"model_type inválido: {model_type}. Usá 'classic', 'ai_returns', 'ai_returns_lstm' o 'ai_volatility'.")
'''

RUN_SCRIPT = r'''from __future__ import annotations

from src.models.train_returns_mlp_lab import run_mlp_lab

if __name__ == "__main__":
    result = run_mlp_lab(
        start_date="2022-01-01",
        forecast_horizon=5,
        target_aggregation="mean",
        lags=20,
        short_window=5,
        long_window=20,
        hidden_layers="256,128,64",
        activation="relu",
        solver="adam",
        scaler="standard",
        alpha=1e-4,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        batch_size="auto",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        clip_pred=0.02,
        winsorize_quantile=0.01,
        include_abs_features=True,
        include_square_features=True,
        include_sign_features=True,
    )
    print("RUN_ID:", result["run_id"])
    print("PREDICTIONS:", result["predictions_path"])
    print("METRICS:", result["metrics_path"])
'''

PATCH_TARGETS = {
    Path("app/mlp_lab_storage.py"): MLP_LAB_STORAGE,
    Path("src/optimization/mlp_lab_loader.py"): MLP_LAB_LOADER,
    Path("src/models/train_returns_mlp_lab.py"): TRAIN_RETURNS_MLP_LAB,
    Path("app/pages/02_MLP_Lab.py"): MLP_LAB_PAGE,
    Path("src/optimization/inputs_builder.py"): INPUTS_BUILDER,
    Path("scripts/run_train_returns_mlp_lab.py"): RUN_SCRIPT,
}


def patch_streamlit_app(root: Path) -> list[str]:
    notes = []
    path = root / "app" / "streamlit_app.py"
    if not path.exists():
        notes.append("No encontré app/streamlit_app.py, salteo patch de texto UI.")
        return notes
    original = path.read_text(encoding="utf-8")
    updated = original.replace('"ai_returns": "AI Returns (MLP)",', '"ai_returns": "AI Returns (Active MLP Lab)",')
    updated = updated.replace(
        '"ai_returns": "Predicted expected returns from MLP + historical covariance.",',
        '"ai_returns": "Active MLP Lab run + historical covariance. Assets that fail the active selection policy fall back to historical mean return.",',
    )
    updated = updated.replace('"ai_volatility": "AI Volatility (GJR-GJR-GARCH)",', '"ai_volatility": "AI Volatility (GJR-GARCH)",')
    if updated != original:
        backup_file(path, tag="mlp_lab_fix")
        path.write_text(updated, encoding="utf-8", newline="\n")
        notes.append("app/streamlit_app.py actualizado")
    else:
        notes.append("app/streamlit_app.py ya estaba actualizado o no necesitó cambios")
    return notes


def main() -> None:
    root = locate_repo_root()
    print(f"Repo root detectado: {root}")
    inputs_builder = root / "src" / "optimization" / "inputs_builder.py"
    if inputs_builder.exists():
        backup = backup_file(inputs_builder, tag="mlp_lab_fix")
        print(f"BACKUP: {backup}")
    for rel_path, content in PATCH_TARGETS.items():
        full_path = root / rel_path
        write_text(full_path, content)
        print(f"WROTE: {rel_path.as_posix()}")
    for note in patch_streamlit_app(root):
        print(note)
    print("\nDONE ✅")
    print("\nSiguientes pasos:")
    print("1) docker compose restart streamlit fastapi")
    print("2) Abrí la nueva página 'MLP Lab' en Streamlit")
    print("3) Entrená un run")
    print("4) Elegí la selection policy y seteá el run como activo")
    print("5) En la app principal usá el modo ai_returns")


if __name__ == "__main__":
    main()
