from __future__ import annotations

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

from src.data.load_data import load_returns, active_data_dir

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
    dataset_dir = active_data_dir()
    returns = load_returns(data_dir=dataset_dir)
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

            pred_rows.append({"ticker": ticker, "mu_pred": pred_latest,
                              "dataset_version": dataset_dir.name, "base_currency": "USD"})
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
    predictions_df = merged[["ticker", "mu_pred", "score", "dir_acc_test", "corr_test", "rmse_test", "positive_precision_test", "dataset_version", "base_currency"]].copy()
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
        "dataset": str(dataset_dir),
        "dataset_version": dataset_dir.name,
        "base_currency": "USD",
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

    from app.mlp_lab_storage import persist_mlp_lab_run
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
