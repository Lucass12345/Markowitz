"""Real model training in isolated worker directories and verified-bundle inference.

This module never activates models, writes legacy pointers, or publishes partial
artifacts. Evaluation uses chronological train/validation/test partitions; only
validation evidence participates in acceptance. Publication belongs to the worker.
"""
from __future__ import annotations

from collections.abc import Mapping
import inspect
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), indent=2, allow_nan=False), encoding="utf-8")


def _config(config):
    if hasattr(config, "model_dump"):
        config = config.model_dump(exclude_none=True)
    if not isinstance(config, Mapping):
        raise ValueError("Model configuration must be a mapping")
    result = dict(config)
    parameters = result.pop("parameters", {}) or {}
    if not isinstance(parameters, Mapping):
        raise ValueError("Model parameters must be a mapping")
    return {**parameters, **result}


def _path(root, relative):
    path = Path(str(relative))
    if path.is_absolute() or ".." in path.parts or "\\" in str(relative):
        raise ValueError("Unsafe model artifact path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Model artifact escapes its verified bundle")
    return resolved


def _check(context):
    callback = getattr(context, "check_cancelled", None)
    if callback:
        callback()


def _progress(context, percent, stage, message=None):
    _check(context)
    callback = getattr(context, "progress", None)
    if callback:
        callback(float(percent), stage, message)


def _device(requested):
    import torch
    requested = str(requested or "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this worker")
    return requested


def _frame(returns, version, config):
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("Training requires a nonempty returns frame")
    if not version or returns.attrs.get("dataset_version", str(version)) != str(version):
        raise ValueError("Training dataset version does not match the pinned snapshot")
    if returns.attrs.get("base_currency") != "USD":
        raise ValueError("Training requires verified USD daily log returns")
    if (not isinstance(returns.index, pd.DatetimeIndex) or returns.index.has_duplicates
            or not returns.index.is_monotonic_increasing or returns.columns.has_duplicates):
        raise ValueError("Training requires unique assets and sorted unique dates")
    tickers = config.get("tickers") or list(returns.columns)
    if len(tickers) != len(set(tickers)) or set(tickers) - set(returns.columns):
        raise ValueError("Training tickers must be unique members of the snapshot")
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker or Path(ticker).name != ticker or ticker in {".", ".."} or "\\" in ticker:
            raise ValueError("Invalid ticker artifact name")
    frame = returns[tickers].copy()
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Training returns contain infinite values")
    frame.attrs.update(dataset_version=str(version), base_currency="USD")
    return frame


def _scores(y, pred, train_mean, baseline=None):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    if not len(y):
        return {"n": 0}
    std = float(np.std(y))
    result = dict(n=len(y), rmse=float(np.sqrt(np.mean((y-pred)**2))),
                  mae=float(np.mean(abs(y-pred))), bias=float(np.mean(pred-y)),
                  amplitude_ratio=float(np.std(pred)/std) if std > 0 else None,
                  dir_acc=float(np.mean(np.sign(y) == np.sign(pred))),
                  corr=float(np.corrcoef(y, pred)[0, 1]) if std > 0 and np.std(pred) > 0 and len(y) > 1 else None,
                  baseline_mean_rmse=float(np.sqrt(np.mean((y-train_mean)**2))),
                  baseline_zero_rmse=float(np.sqrt(np.mean(y**2))))
    if baseline is not None:
        b = np.asarray(baseline, float)[valid]
        result.update(baseline_ewma_rmse=float(np.sqrt(np.mean((y-b)**2)))
                      if np.isfinite(b).all() else None)
        if np.isfinite(b).all() and (pred > 0).all() and (b > 0).all():
            result.update(qlike=float(np.mean(np.log(pred)+y/pred)),
                          baseline_ewma_qlike=float(np.mean(np.log(b)+y/b)))
    return _clean(result)


def _split_metrics(details, divisor=1):
    details = details.copy()
    details[["y_true", "y_pred"]] /= divisor
    train = details.loc[details.split.eq("train"), "y_true"]
    train_mean = float(train.mean())
    return {name: _scores(part.y_true, part.y_pred, train_mean,
                          part.ewma if "ewma" in part else None)
            for name, part in details.groupby("split", sort=False)}


def _accept(family, metrics, config, extra):
    val = metrics.get("val", {})
    def finite(name):
        value = val.get(name)
        return isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))

    if not val.get("n") or not finite("rmse"):
        return False, "missing_validation_evidence"
    if (family == "lstm" or config.get("require_baseline_improvement", False)) and not finite("baseline_mean_rmse"):
        return False, "missing_validation_evidence"
    if family == "lstm":
        if val["rmse"] >= val["baseline_mean_rmse"]:
            return False, "validation_mean_baseline_not_beaten"
        if not finite("dir_acc") or val["dir_acc"] <= float(config.get("lstm_direction_min", .52)):
            return False, "validation_direction_below_threshold"
        if not finite("amplitude_ratio") or val["amplitude_ratio"] <= float(config.get("lstm_amplitude_min", .15)):
            return False, "validation_amplitude_below_threshold"
    elif family == "mlp":
        if not extra.get("validation_composite_available", False):
            return False, "missing_validation_evidence"
    elif not extra.get("converged") or not extra.get("covariance_stationary"):
        return False, "garch_convergence_or_stationarity_failed"
    if config.get("require_baseline_improvement", False) and val["rmse"] >= val["baseline_mean_rmse"]:
        return False, "validation_mean_baseline_not_beaten"
    return True, "validation_policy_accepted"


class ModelService:
    def __init__(self, dataset_service=None, store=None, database=None):
        self.dataset_service = dataset_service
        self.store = store
        self.database = database

    def train(self, family, config, context):
        config = _config(config)
        from src.platform.datasets import DatasetService
        datasets = self.dataset_service or DatasetService(
            store=self.store or getattr(context, "store", None),
            database=self.database or getattr(context, "database", None))
        requested = config.get("dataset_version") or getattr(context, "dataset_version", None)
        # Resolve once, then materialize the resolved immutable identity.
        record = datasets.resolve(requested)
        snapshot = datasets.materialize(record["id"], Path(context.temp_dir) / "dataset")
        result = self.train_on_frame(family, snapshot.load_returns(), snapshot.version, config, context)
        result["dataset_id"] = snapshot.id
        return result

    def train_on_frame(self, family, returns_df, dataset_version, config, context):
        family = str(family).lower()
        if family not in {"lstm", "mlp", "garch"}:
            raise ValueError("family must be lstm, mlp, or garch")
        config = _config(config)
        horizon = int(config.get("forecast_horizon", config.get("horizon", 5)))
        if horizon != 5:
            raise ValueError("Production models predict the next five trading days")
        if config.get("target_aggregation", "mean") not in {"mean", "sum"}:
            raise ValueError("target_aggregation must be mean or sum")
        frame = _frame(returns_df, dataset_version, config)
        config.update(forecast_horizon=horizon, seed=int(config.get("seed", 42)))
        job_id = str(context.job_id)
        if not job_id or Path(job_id).name != job_id or job_id in {".", ".."} or "\\" in job_id:
            raise ValueError("Invalid training job identity")
        temporary = Path(context.temp_dir)
        temporary.mkdir(parents=True, exist_ok=True)
        _progress(context, 0, "model_training", family)
        staging = temporary / f".models-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            if family == "garch":
                records = self._train_garch(frame, config, context, staging)
            else:
                records = self._train_returns(family, frame, config, context, staging)
            if not records:
                raise ValueError("No asset produced a complete fitted model; inspect available history/configuration")
            predictions = []
            for record in records:
                _check(context)
                prediction = self._predict_local(record, staging, frame, config.get("device", "auto"))
                prediction.update(self._validation_columns(record))
                predictions.append(prediction)
                record.update(as_of=prediction["as_of"], training_cutoff=prediction["as_of"], cutoff=prediction["as_of"])
            prediction_file = f"predictions/{family}.csv"
            (staging / "predictions").mkdir(exist_ok=True)
            pd.DataFrame(predictions).to_csv(staging / prediction_file, index=False)
            for record in records:
                record["prediction_file"] = prediction_file
            metrics = {record["ticker"]: record["metrics"] for record in records}
            _write(staging / "models.json", records)
            _write(staging / "metrics.json", metrics)
            _write(staging / "config.json", {**config, "family": family, "dataset_version": dataset_version,
                                              "job_id": job_id, "selection_split": "validation",
                                              "test_used_for_selection": False})
            _check(context)
            finished = temporary / f"model-bundle-{uuid.uuid4().hex}"
            staging.rename(finished)
            _progress(context, 100, "model_training_complete", family)
            return _clean(dict(dataset_version=dataset_version, family=family, job_id=job_id,
                               artifacts_dir=str(finished), predictions_path=str(finished / prediction_file),
                               predictions=predictions, models=records, metrics=metrics))
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _validation_columns(record):
        val = record["validation_metrics"]
        result = {f"{key}_val": value for key, value in val.items()}
        if record["family"] == "lstm":
            result["amp_ratio"] = val.get("amplitude_ratio")
        result.update(use_for_optimizer=record["accepted"], selection_reason=record["selection_reason"],
                      selection_split="validation", validation_metrics=json.dumps(_clean(val)))
        return result

    def _train_returns(self, family, frame, config, context, root):
        if family == "lstm":
            import torch
            from src.models.train_returns_lstm_lab import run_lstm_lab as trainer
        else:
            from src.models.train_returns_mlp_lab import run_mlp_lab as trainer
        allowed = inspect.signature(trainer).parameters
        kwargs = {key: value for key, value in config.items() if key in allowed}
        kwargs.update(returns_df=frame, dataset_version=frame.attrs["dataset_version"],
                      output_root=root / "evaluation", model_root=root / "weights", run_id=str(context.job_id),
                      persist_db=False, publish_latest=False,
                      progress=lambda pct, stage, message=None: _progress(context, pct * .9, stage, message),
                      check_cancelled=lambda: _check(context))
        if family == "lstm":
            from src.models.lstm_features import AVAILABLE_FEATURES
            kwargs.setdefault("selected_features", AVAILABLE_FEATURES)
            unknown = set(kwargs["selected_features"]) - set(AVAILABLE_FEATURES)
            if unknown:
                raise ValueError(f"Unknown LSTM features: {sorted(unknown)}")
            kwargs.update(device=_device(config.get("device")), manage_threads=False, seed=config["seed"])
            if not kwargs.get("search", True) and "candidates" not in kwargs:
                kwargs["candidates"] = [dict(hidden_sizes=config.get("hidden_sizes", [int(config.get("hidden_size", 64))] * int(config.get("num_layers", 2))),
                    activation=config.get("activation", "relu"), lr=config.get("lr", 1e-3),
                    betas=config.get("betas", [.9, .999]), eps=config.get("eps", 1e-8),
                    weight_decay=config.get("weight_decay", 0))]
            # Trainers keep their legacy seed behavior; isolate its process RNG state.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if kwargs["device"] == "cuda" else []):
                result = trainer(**kwargs)
            model_dir = Path(result["model_dir"])
        else:
            kwargs["random_state"] = config["seed"]
            result = trainer(**kwargs)
            model_dir = Path(result["run_model_dir"])
        records = []
        for row in result["predictions"].to_dict("records"):
            ticker = row["ticker"]
            details = pd.read_csv(Path(result["per_ticker_dir"]) / f"{ticker}_predictions.csv")
            divisor = config["forecast_horizon"] if config.get("target_aggregation", "mean") == "sum" else 1
            metrics = _split_metrics(details, divisor)
            if family == "lstm":
                checkpoint = model_dir / f"{ticker}_lstm.pt"
                metadata = model_dir / f"{ticker}_meta.json"
                preprocessing = json.loads(metadata.read_text())
                features = preprocessing["selected_features"]
                extra = {}
            else:
                checkpoint = model_dir / f"{ticker}_mlp_pipeline.joblib"
                metadata = model_dir / f"{ticker}_preprocessing.json"
                preprocessing = json.loads(metadata.read_text())
                preprocessing["start_date"] = config.get("start_date", "2022-01-01")
                _write(metadata, preprocessing)
                features = preprocessing["feature_columns"]
                extra = dict(validation_composite_available=bool(np.isfinite(row.get("score", np.nan))))
                metrics["val"]["positive_precision"] = row.get("positive_precision_val")
            accepted, reason = _accept(family, metrics, config, extra)
            records.append(dict(ticker=ticker, family=family, horizon=config["forecast_horizon"],
                forecast_horizon=config["forecast_horizon"], target_aggregation=config.get("target_aggregation", "mean"),
                dataset_version=frame.attrs["dataset_version"], base_currency="USD",
                model_run_id=str(context.job_id), job_id=str(context.job_id), seed=config["seed"],
                relative_path=checkpoint.relative_to(root).as_posix(), metadata_path=metadata.relative_to(root).as_posix(),
                features=features, preprocessing=preprocessing, metrics=metrics, validation_metrics=metrics.get("val", {}),
                accepted=accepted, use_for_optimizer=accepted, state="validated", selection_reason=reason,
                prediction_units="daily_log_return", selection_split="validation"))
        return records

    def _train_garch(self, frame, config, context, root):
        from src.models.vol_model_garch import diagnose_garch, fit_gjr
        validation_days, test_days = int(config.get("validation_days", 20)), int(config.get("test_days", 20))
        minimum = max(252, int(config.get("min_obs", 252)))
        if min(validation_days, test_days) < 1:
            raise ValueError("GARCH validation_days and test_days must be positive")
        fit_config = {key: config.get(key, default) for key, default in (("dist", "t"), ("mean", "Zero"), ("maxiter", 300))}
        records = []
        for number, ticker in enumerate(frame):
            _progress(context, 90 * number / len(frame.columns), "garch_ticker", ticker)
            series = frame[ticker].dropna().astype(float)
            if len(series) < minimum + validation_days + test_days:
                raise ValueError(f"{ticker}: GARCH requires {minimum + validation_days + test_days} observations")
            train = series.iloc[:-(validation_days+test_days)]
            training_fit = fit_gjr(train, **fit_config)
            val = diagnose_garch(series.iloc[:-test_days], horizon=5, test_size=validation_days,
                                 include_predictions=True, check_cancelled=lambda: _check(context), **fit_config)
            test = diagnose_garch(series, horizon=5, test_size=test_days,
                                  include_predictions=True, check_cancelled=lambda: _check(context), **fit_config)
            if val["backtest"]["n"] != validation_days or test["backtest"]["n"] != test_days:
                raise ValueError(f"{ticker}: GARCH rolling evaluation has failed origins")
            train_detail = pd.DataFrame(dict(date=train.index, split="train", y_true=train.pow(2).values,
                y_pred=np.asarray(training_fit.conditional_volatility)**2 / 10000,
                ewma=train.pow(2).ewm(alpha=.06, adjust=False).mean().values))
            details = pd.concat([train_detail, pd.DataFrame(val["backtest"]["predictions"]).assign(split="val"),
                                 pd.DataFrame(test["backtest"]["predictions"]).assign(split="test")], ignore_index=True)
            metrics = _split_metrics(details)
            metrics["target_units"] = "squared_daily_log_return"
            final_fit = fit_gjr(series, **fit_config)
            params = dict(parameter_names=list(final_fit.params.index), parameters=final_fit.params.tolist(),
                          **fit_config, forecast_horizon=5, as_of=str(series.index[-1].date()))
            checkpoint = root / "weights" / f"{ticker}_garch.json"
            _write(checkpoint, params)
            _write(root / "evaluation" / f"{ticker}_diagnostics.json", {"validation": val, "test": test})
            detail_path = root / "evaluation" / f"{ticker}_predictions.csv"
            details.to_csv(detail_path, index=False)
            accepted, reason = _accept("garch", metrics, config, val)
            records.append(dict(ticker=ticker, family="garch", horizon=5, forecast_horizon=5,
                dataset_version=frame.attrs["dataset_version"], base_currency="USD", model_run_id=str(context.job_id),
                job_id=str(context.job_id), seed=config["seed"], relative_path=checkpoint.relative_to(root).as_posix(),
                metadata_path=checkpoint.relative_to(root).as_posix(), features=["daily_log_return"], preprocessing={"scale": 100, **fit_config},
                metrics=metrics, validation_metrics=metrics["val"], accepted=accepted, use_for_optimizer=accepted,
                state="validated", selection_reason=reason, prediction_units="daily_log_volatility", selection_split="validation"))
        return records

    @staticmethod
    def _predict_local(record, root, frame, device="auto"):
        ticker, family = record["ticker"], record["family"]
        checkpoint = _path(root, record["relative_path"])
        metadata = json.loads(_path(root, record["metadata_path"]).read_text())
        series = frame[ticker].dropna().astype(float)
        start = metadata.get("start_date")
        if start and start != "full":
            series = series.loc[start:]
        if series.empty:
            raise ValueError(f"No inference history for {ticker}")
        result = dict(ticker=ticker, family=family, dataset_version=frame.attrs["dataset_version"],
            base_currency="USD", forecast_horizon=int(record["horizon"]),
            prediction_units=record["prediction_units"], model_run_id=record["model_run_id"])
        if family == "lstm":
            import torch
            from src.models.lstm_features import build_lstm_feature_frame, select_feature_columns
            from src.models.return_model_lstm import ReturnLSTM
            features = select_feature_columns(build_lstm_feature_frame(series), metadata["selected_features"])
            length = int(metadata["seq_len"])
            if len(features) < length:
                raise ValueError(f"Insufficient LSTM inference window for {ticker}")
            config = metadata["selected_config"]
            target_device = _device(device)
            # Initial parameter draws must not perturb subsequent worker jobs.
            with torch.random.fork_rng(devices=[]):
                model = ReturnLSTM(input_size=features.shape[1], hidden_sizes=config["hidden_sizes"],
                    activation=config["activation"], dropout=metadata["dropout"], feature_indices=config.get("feature_indices"))
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.to(target_device).eval()
            values = (features.iloc[-length:].to_numpy(np.float32)-np.asarray(metadata["x_mean"]))/np.asarray(metadata["x_std"])
            with torch.no_grad():
                raw = float(model(torch.tensor(values[None], dtype=torch.float32, device=target_device)).item())
            raw = raw * metadata["y_std"] + metadata["y_mean"]
            as_of = features.index[-1]
        elif family == "mlp":
            import joblib
            from src.models.train_returns_mlp_lab import build_mlp_feature_frame
            series = series.clip(lower=metadata["winsor_lower"], upper=metadata["winsor_upper"])
            features = build_mlp_feature_frame(series, **{key: metadata[key] for key in (
                "lags", "short_window", "long_window", "include_abs_features", "include_square_features", "include_sign_features")})
            features = features[metadata["feature_columns"]].dropna()
            if features.empty:
                raise ValueError(f"Insufficient MLP inference features for {ticker}")
            raw = float(joblib.load(checkpoint).predict(features.tail(1))[0])
            as_of = features.index[-1]
        elif family == "garch":
            from arch import arch_model
            if len(series) < 252:
                raise ValueError(f"Insufficient GARCH inference history for {ticker}")
            fixed = arch_model(series*100, mean=metadata["mean"], vol="GARCH", p=1, o=1, q=1,
                               dist=metadata["dist"], rescale=False).fix(np.asarray(metadata["parameters"]))
            variance = fixed.forecast(horizon=5).variance.iloc[-1].to_numpy() / 10000
            if not np.isfinite(variance).all() or (variance <= 0).any():
                raise ValueError("Invalid GARCH inference variance")
            result.update(vol_pred=float(np.sqrt(variance[0])), variance_pred=float(variance[0]),
                          variance_sum_5d=float(variance.sum()), vol_5d=float(np.sqrt(variance.sum())))
            as_of = series.index[-1]
        else:
            raise ValueError("Unsupported model family in verified bundle")
        if family != "garch":
            mu = raw / record["horizon"] if metadata["target_aggregation"] == "sum" else raw
            if family == "mlp":
                mu = float(np.clip(mu, -metadata["clip_pred"], metadata["clip_pred"]))
            if not np.isfinite(mu):
                raise ValueError("Nonfinite model prediction")
            result.update(mu_pred=mu, forecast_log_return_5d=mu*record["horizon"],
                          forecast_simple_return_5d=float(np.expm1(mu*record["horizon"])))
        result["as_of"] = str(as_of.date())
        return _clean(result)

    def infer(self, model_record, snapshot, context):
        record = dict(model_record)
        uri = record.get("model_artifact_uri") or record.get("artifact_uri")
        if not uri:
            raise ValueError("Inference requires a verified registry model artifact URI")
        _progress(context, 0, "model_inference", record.get("ticker"))
        store = self.store or getattr(context, "store", None)
        if store is None:
            from src.platform.artifacts import get_store
            store = get_store()
        root = Path(store.materialize(uri, Path(context.temp_dir) / f"model-{uuid.uuid4().hex}"))
        # Bundle metadata is authoritative; SQL identity must refer to that model.
        records = json.loads((root / "models.json").read_text())
        matches = [item for item in records if item["ticker"] == record.get("ticker")
                   and item["family"] == record.get("family")
                   and item["model_run_id"] == record.get("model_run_id", record.get("job_id"))]
        if len(matches) != 1:
            raise ValueError("Registry model identity does not match its verified bundle")
        trained = matches[0]
        frame = _frame(snapshot.load_returns(), snapshot.version, {"tickers": [trained["ticker"]]})
        observed = frame[trained["ticker"]].dropna().index.max()
        if pd.Timestamp(trained["training_cutoff"]) > observed:
            raise ValueError("Model training cutoff is later than the inference history")
        prediction = self._predict_local(trained, root, frame, record.get("device", "auto"))
        prediction.update(self._validation_columns(trained))
        output = Path(context.temp_dir) / f"inference-bundle-{uuid.uuid4().hex}"
        output.mkdir(parents=True)
        relative = f"predictions/{trained['family']}.csv"
        (output / "predictions").mkdir()
        pd.DataFrame([prediction]).to_csv(output / relative, index=False)
        _write(output / "inference.json", {"model_run_id": trained["model_run_id"], "job_id": str(context.job_id),
            "model_artifact_uri": uri, "dataset_version": snapshot.version, "as_of": prediction["as_of"]})
        _progress(context, 100, "model_inference_complete", trained["ticker"])
        return _clean(dict(prediction, predictions=[prediction], prediction_file=relative,
            predictions_path=str(output / relative), artifacts_dir=str(output), dataset_version=snapshot.version,
            horizon=trained["horizon"], model_run_id=trained["model_run_id"], job_id=str(context.job_id)))
