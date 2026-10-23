from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_returns, active_data_dir
from src.models.return_model_lstm import ReturnLSTM
from src.models.lstm_features import build_lstm_feature_frame, select_feature_columns
from src.models.sequence_utils import build_lstm_sequences_multifeature
from src.models.lstm_search import fit_candidates, default_candidates, refit_deployment

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


def temporal_split(X, y, dates, train_frac=0.70, val_frac=0.15, forecast_horizon=5):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return (
        X[:train_end-forecast_horizon+1], y[:train_end-forecast_horizon+1], dates[:train_end-forecast_horizon+1],
        X[train_end:val_end-forecast_horizon+1], y[train_end:val_end-forecast_horizon+1], dates[train_end:val_end-forecast_horizon+1],
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
    search: bool = True,
    feature_search: bool = False,
    refit_latest: bool = True,
    candidates: list | None = None,
    patience: int = 8,
    persist_db: bool = True,
    tickers: list | None = None,
    publish_latest: bool = True,
    returns_df: pd.DataFrame | None = None,
    dataset_version: str | None = None,
):
    if forecast_horizon != 5:
        raise ValueError("El horizonte de producción es de cinco ruedas bursátiles")
    dataset_dir = active_data_dir()
    dataset_version = dataset_version or dataset_dir.name
    returns = load_returns(data_dir=dataset_dir) if returns_df is None else returns_df.copy()
    if tickers is not None:
        returns = returns[tickers]

    if start_date and start_date != "full":
        returns = returns.loc[start_date:]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
                summary_rows.append({"ticker": ticker, "error": "insufficient_history", "n_observations": len(series)})
                continue

            feat_df = build_lstm_feature_frame(series).dropna()
            feat_df = select_feature_columns(feat_df, selected_features)

            common_idx = feat_df.index.intersection(series.index)
            feat_df = feat_df.loc[common_idx]
            target_series = series.loc[common_idx]

            aligned = feat_df.copy()
            aligned["target"] = target_series
            aligned = aligned.dropna()

            if len(aligned) < max(seq_len + forecast_horizon + 30, 120):
                summary_rows.append({"ticker": ticker, "error": "insufficient_feature_history", "n_observations": len(aligned)})
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
                summary_rows.append({"ticker": ticker, "error": "insufficient_sequences", "n_sequences": len(X)})
                continue

            (
                X_train, y_train, d_train,
                X_val, y_val, d_val,
                X_test, y_test, d_test,
            ) = temporal_split(X, y, target_dates, forecast_horizon=forecast_horizon)

            if min(len(X_train), len(X_val), len(X_test)) == 0:
                continue

            X_train_s, X_val_s, X_test_s, x_mean, x_std = scale_3d_train(X_train, X_val, X_test)
            y_train_s, y_val_s, y_test_s, y_mean, y_std = scale_y_train(y_train, y_val, y_test)

            X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
            y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
            y_val_t = torch.tensor(y_val_s, dtype=torch.float32)
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

            configs = candidates if candidates is not None else (default_candidates() if search else [
                dict(hidden_sizes=[hidden_size] * num_layers, activation="relu", lr=lr)])
            if feature_search and "return" in selected_features and len(selected_features) > 1:
                configs = [dict(c, feature_indices=indices, feature_set=name)
                           for c in configs for name, indices in [("selected_features", list(range(len(selected_features)))),
                                                                  ("returns_only", [selected_features.index("return")])]]
            fitted = fit_candidates(X_train_s, y_train_s, X_val_s, y_val_s,
                                    configs, epochs=epochs, batch_size=batch_size,
                                    dropout=dropout, patience=patience, device=DEVICE)
            model = fitted["model"]
            best_val = fitted["best_val"]
            train_losses, val_losses = fitted["train_losses"], fitted["val_losses"]
            pd.DataFrame(fitted["candidates"]).to_csv(run_output_dir / f"{ticker}_candidates.csv", index=False)
            # Inference window includes the latest observed return, unlike labeled windows.
            latest_X = (feat_df.iloc[-seq_len:].values.astype(np.float32) - x_mean) / x_std

            model.eval()
            with torch.no_grad():
                pred_train_scaled = model(X_train_t.to(DEVICE)).cpu().numpy()
                pred_val_scaled = model(X_val_t.to(DEVICE)).cpu().numpy()
                pred_test_scaled = model(X_test_t.to(DEVICE)).cpu().numpy()
                last_pred_scaled = model(torch.tensor(latest_X[None], dtype=torch.float32).to(DEVICE)).cpu().numpy()[0]

            pred_train = inverse_y(pred_train_scaled, y_mean, y_std)
            pred_val = inverse_y(pred_val_scaled, y_mean, y_std)
            pred_test = inverse_y(pred_test_scaled, y_mean, y_std)
            last_pred = float(inverse_y(last_pred_scaled, y_mean, y_std))

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

            evaluation_scalers = {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                                  "y_mean": y_mean, "y_std": y_std}
            refit_epochs = int(np.argmin(val_losses)) + 1
            if refit_latest:
                # Test metrics above describe the untouched evaluation checkpoint.
                torch.save(model.state_dict(), run_model_dir / f"{ticker}_evaluation.pt")
                model, x_mean, x_std, y_mean, y_std = refit_deployment(
                    X, y, fitted["config"], epochs=refit_epochs, batch_size=batch_size,
                    dropout=dropout, device=DEVICE)
                latest_X = (feat_df.iloc[-seq_len:].values.astype(np.float32) - x_mean) / x_std
                with torch.no_grad():
                    last_pred = float(model(torch.tensor(latest_X[None], dtype=torch.float32).to(DEVICE)).cpu().item() * y_std + y_mean)

            pred_rows.append({
                "ticker": ticker,
                "mu_pred": last_pred / forecast_horizon if target_aggregation == "sum" else last_pred,
                "forecast_log_return_5d": last_pred if target_aggregation == "sum" else last_pred * forecast_horizon,
                "forecast_simple_return_5d": float(np.expm1(last_pred if target_aggregation == "sum" else last_pred * forecast_horizon)),
                "as_of": str(feat_df.index[-1].date()),
                "forecast_horizon": forecast_horizon,
                "dataset_version": dataset_version,
                "base_currency": "USD",
                "use_for_optimizer": val_metrics["rmse_val"] < rmse(y_val, np.full_like(y_val, y_train.mean())),
            })

            summary_rows.append({
                "ticker": ticker,
                "best_val_loss": best_val,
                "selected_config": json.dumps(fitted["config"]),
                "baseline_rmse_test": rmse(y_test, np.full_like(y_test, y_train.mean())),
                "baseline_rmse_val": rmse(y_val, np.full_like(y_val, y_train.mean())),
                "zero_rmse_test": rmse(y_test, np.zeros_like(y_test)),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "mu_pred": last_pred / forecast_horizon if target_aggregation == "sum" else last_pred,
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
                    "selected_config": fitted["config"],
                    "refit_latest": refit_latest,
                    "refit_epochs": refit_epochs if refit_latest else None,
                    "evaluation_metrics_before_refit": True,
                    "evaluation_scalers": evaluation_scalers,
                    "as_of": str(feat_df.index[-1].date()),
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
        "dataset": str(dataset_dir),
        "total_tickers_in_universe": total_tickers,
        "tickers_with_predictions": int(len(predictions_df)),
        "start_date": start_date,
        "selected_features": selected_features,
        "seq_len": seq_len,
        "forecast_horizon": forecast_horizon,
        "target_aggregation": target_aggregation,
        "feature_search": feature_search,
        "search": search,
        "selection_metric": "validation_mse",
        "test_used_for_selection": False,
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
    if publish_latest and not predictions_df.empty:
        latest_ptr.write_text(json.dumps({
            "run_id": run_id,
            "run_output_dir": str(run_output_dir),
            "run_model_dir": str(run_model_dir),
        }, indent=2), encoding="utf-8")

    db_run_pk = None
    if persist_db:
        from app.lstm_lab_storage import persist_lstm_lab_run
        db_run_pk = persist_lstm_lab_run(
            run_id=run_id, summary=summary, predictions_df=predictions_df,
            metrics_df=metrics_df, run_output_dir=str(run_output_dir),
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
