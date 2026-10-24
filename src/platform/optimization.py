"""Shared portfolio domain for serving and past-only walk-forward decisions.

The frame-level entry point does not read active datasets, model pointers or local
prediction files. It receives the exact USD dataset and forecasts selected by its
caller. The worker service loads only versioned registry artifacts and writes to
its disposable job directory; publication belongs to the platform publisher.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS
from src.markowitz import rf_simple_annual_to_log_daily
from src.optimization.covariance import RiskConfig
from src.optimization.errors import OptimizationInputError
from src.optimization.frontier_engine import run_constrained_frontier, run_frontier
from src.optimization.inputs_builder import build_model_inputs
from src.optimization.optimize_blocks import run_blocks_optimizer, validate_optimizer_inputs
from src.portfolio_risk import estimated_risk
from src.utils.serialization import to_builtin


def _configuration(config):
    if hasattr(config, "model_dump"):
        return config.model_dump(exclude_none=True)
    if not isinstance(config, Mapping):
        raise OptimizationInputError("Optimization configuration must be a mapping")
    return dict(config)


def _mode(config):
    requested = str(config.get("model_type", "classic")).lower()
    aliases = {"lstm": "ai_returns_lstm", "mlp": "ai_returns", "garch": "ai_volatility",
               "gjr_garch": "ai_volatility", "combined": "ai_returns_lstm", "ai_combined": "ai_returns_lstm"}
    mode = aliases.get(requested, requested)
    if mode not in {"classic", "ai_returns", "ai_returns_lstm", "ai_volatility"}:
        raise OptimizationInputError(f"Invalid model_type: {requested}")
    return_model = config.get("return_model")
    if return_model is not None:
        names = {"historical": "classic", "classic": "classic", "lstm": "ai_returns_lstm", "mlp": "ai_returns"}
        if str(return_model).lower() not in names:
            raise OptimizationInputError(f"Invalid return model: {return_model}")
        mode = names[str(return_model).lower()]
    estimator = config.get("covariance_estimator")
    risk_model = config.get("volatility_model")
    if risk_model is not None:
        if str(risk_model).lower() in {"garch", "gjr_garch"}:
            estimator = "gjr_garch"
        elif str(risk_model).lower() in {"historical", "classic"}:
            estimator = estimator or "historical"
        else:
            raise OptimizationInputError(f"Invalid volatility model: {risk_model}")
    if requested in {"combined", "ai_combined"}:
        estimator = estimator or "gjr_garch"
    return mode, estimator


def _forecast_frame(value):
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value or [])


def risk_configuration(config):
    """Resolve identical risk eligibility controls for serving and research."""
    values = {**dict(config.get("risk_config") or {}),
              **{key: config[key] for key in ("min_history", "min_overlap", "eigenvalue_floor", "max_psd_adjustment") if key in config}}
    return RiskConfig(**values)


def _cutoff_frame(returns, config):
    frame = returns.copy()
    version = frame.attrs.get("dataset_version") or config.get("dataset_version")
    if not version:
        raise OptimizationInputError("An explicit dataset_version is required")
    if config.get("dataset_version") and str(config["dataset_version"]) != str(version):
        raise OptimizationInputError("Requested dataset_version differs from the supplied data")
    if frame.attrs.get("base_currency", config.get("base_currency")) != "USD":
        raise OptimizationInputError("Optimization requires verified USD daily log returns")
    frame.attrs.update(dataset_version=str(version), base_currency="USD")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.isna().any():
        raise OptimizationInputError("Portfolio data requires valid observation dates")
    if config.get("as_of"):
        cutoff = pd.Timestamp(config["as_of"])
        if pd.isna(cutoff):
            raise OptimizationInputError("Invalid as_of date")
        if frame.index.tz is None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        elif frame.index.tz is not None:
            cutoff = cutoff.tz_localize(frame.index.tz) if cutoff.tzinfo is None else cutoff.tz_convert(frame.index.tz)
        frame = frame.loc[frame.index <= cutoff].copy()
    tickers = config.get("tickers")
    if tickers is not None:
        tickers = [str(ticker).strip() for ticker in tickers]
        if not tickers or len(tickers) != len(set(tickers)):
            raise OptimizationInputError("Universe tickers must be nonempty and unique")
        missing = set(tickers) - set(frame.columns)
        if missing:
            raise OptimizationInputError(f"Universe tickers absent from dataset: {sorted(missing)}")
        frame = frame.loc[:, tickers]
    if frame.empty:
        raise OptimizationInputError("No portfolio observations at the requested cutoff")
    return frame


def _matching_signal_dates(return_predictions, volatility_predictions):
    if return_predictions.empty or volatility_predictions.empty:
        return
    if "ticker" not in return_predictions or "ticker" not in volatility_predictions:
        return  # The central prediction contract produces the precise schema error.
    left = return_predictions.set_index("ticker")
    right = volatility_predictions.set_index("ticker")
    if left.index.has_duplicates or right.index.has_duplicates:
        return
    for ticker in left.index.intersection(right.index):
        first = pd.to_datetime(left.loc[ticker].get("as_of"), utc=True, errors="coerce")
        last = pd.to_datetime(right.loc[ticker].get("as_of", right.loc[ticker].get("train_end")), utc=True, errors="coerce")
        if pd.notna(first) and pd.notna(last) and first.normalize() != last.normalize():
            raise OptimizationInputError(f"Return and risk signals have different as_of dates for {ticker}")


def optimize_frame(returns, metadata, config, return_predictions=None, volatility_predictions=None):
    """Apply the existing financial engine to one explicitly pinned dataset frame.

    ``None`` predictions mean missing signals, using the configured historical
    fallback. They never trigger reads from a current production model pointer.
    Set ``calculate_frontiers=False`` for historical decisions; this only omits
    plotting frontiers and does not change portfolio selection or constraints.
    """
    config = _configuration(config)
    data = _cutoff_frame(returns, config)
    mode, estimator = _mode(config)
    risk_config = risk_configuration(config)
    periods = config.get("periods", TRADING_DAYS)
    if not isinstance(periods, int) or isinstance(periods, bool) or periods <= 0:
        raise OptimizationInputError("periods must be a positive integer")
    rf_annual = float(config.get("rf_annual_simple", 0.))
    if not np.isfinite(rf_annual) or rf_annual <= -1:
        raise OptimizationInputError("Risk-free rate must be finite and greater than -1")
    rf_daily = rf_simple_annual_to_log_daily(rf_annual, periods)
    return_predictions = _forecast_frame(return_predictions)
    volatility_predictions = _forecast_frame(volatility_predictions)
    for family, frame in (("returns", return_predictions if mode in {"ai_returns", "ai_returns_lstm"} else pd.DataFrame()),
                          ("volatility", volatility_predictions if estimator == "gjr_garch" or (mode == "ai_volatility" and estimator is None) else pd.DataFrame())):
        if not frame.empty and "prediction_units" not in frame:
            raise OptimizationInputError(f"Explicit {family} forecasts require prediction_units")
    if mode in {"ai_returns", "ai_returns_lstm"} and estimator == "gjr_garch":
        _matching_signal_dates(return_predictions, volatility_predictions)
    mu, covariance, tickers = build_model_inputs(
        mode, data, risk_config,
        prediction_fallback=config.get("prediction_fallback", "historical"),
        max_prediction_age_days=config.get("max_prediction_age_days", 7),
        covariance_estimator=estimator,
        lstm_direction_min=config.get("lstm_direction_min", .52),
        lstm_amplitude_min=config.get("lstm_amplitude_min", .15),
        return_predictions=return_predictions,
        volatility_predictions=volatility_predictions,
    )
    targets = config.get("targets")
    if not targets:
        raise OptimizationInputError("Explicit block targets are required")
    max_weight = float(config.get("max_weight", .15))
    blocks = validate_optimizer_inputs(mu, covariance, tickers, metadata, targets, max_weight)
    covariance = covariance.loc[tickers, tickers]
    portfolio = run_blocks_optimizer(mu, covariance, tickers, metadata, targets, max_weight, rf_daily, periods)
    frontiers = config.get("calculate_frontiers", True)
    if frontiers:
        n_points = config.get("n_frontier_points", 80)
        if not isinstance(n_points, int) or n_points < 1:
            raise OptimizationInputError("n_frontier_points must be a positive integer")
        classical = run_frontier(mu, covariance, rf_daily, periods, n_points)
        constrained = run_constrained_frontier(mu, covariance, blocks, targets, max_weight, periods, n_points)
    else:
        classical = {"frontier": pd.DataFrame(), "minvar": None, "maxsharpe": None}
        constrained = {"points": pd.DataFrame(), "diagnostics": {"omitted_for_decision": True}}
    return {
        "model_type": mode,
        "inputs": {**config, "model_type": mode, "dataset_version": data.attrs["dataset_version"],
                   "base_currency": "USD", "as_of": data.index.max().isoformat(), "targets": dict(targets),
                   "max_weight": max_weight, "periods": periods, "rf_annual_simple": rf_annual, "rf_log_daily": rf_daily,
                   "risk_config": asdict(risk_config), "covariance_estimator": mu.attrs["input_diagnostics"]["estimator"]},
        "diagnostics": {"risk": mu.attrs["input_diagnostics"]},
        "frontier": classical["frontier"], "minvar": classical["minvar"], "maxsharpe": classical["maxsharpe"],
        "constrained_frontier": constrained, "blocks": portfolio,
        "portfolio_risk": estimated_risk(portfolio["weights_series"], covariance, data, periods, risk_config.min_overlap),
        "estimates": {"mu": mu, "covariance": covariance, "tickers": tickers},
    }


def _enabled(value):
    return str(value).strip().lower() not in {"false", "0", "no", "none"}


class OptimizationService:
    """Worker adapter: immutable inputs in, disposable artifacts out."""

    def __init__(self, datasets=None, registry=None, universes=None):
        self.datasets = datasets
        self.registry = registry
        self.universes = universes

    def _services(self, context):
        if self.datasets is None:
            from src.platform.datasets import DatasetService
            self.datasets = DatasetService(store=context.store, database=context.database)
        if self.registry is None:
            from src.platform.registry import RegistryService
            self.registry = RegistryService(store=context.store, database=context.database)

    def _universe(self, config, metadata, context):
        universe = config.get("universe_snapshot")
        if universe is None and not config.get("universe_id"):
            return config, metadata
        if universe is None:
            if self.universes is None:
                from src.platform.universes import UniverseService
                self.universes = UniverseService(database=context.database)
            universe = self.universes.get(config["universe_id"])
        assets = universe.get("assets_json", universe.get("assets", [])) if isinstance(universe, Mapping) else universe
        enabled = [asset for asset in assets if _enabled(asset.get("enabled", asset.get("active", True)))]
        if not enabled:
            raise OptimizationInputError("The selected universe has no enabled assets")
        config = {**config, "tickers": [str(asset["ticker"]).strip() for asset in enabled], "universe_snapshot": universe}
        metadata = metadata.copy().set_index("ticker")
        for asset in enabled:
            ticker = str(asset["ticker"]).strip()
            if ticker not in metadata.index:
                raise OptimizationInputError(f"Universe ticker {ticker} is absent from dataset metadata")
            if asset.get("block") is not None:
                metadata.loc[ticker, "block"] = asset["block"]
        return config, metadata.reset_index()

    def _records(self, family, ids, tickers, version, pinned_records=None):
        records = (list(pinned_records) if pinned_records is not None else
                   [self.registry.get(model_id) for model_id in ids] if ids is not None else
                   self.registry.resolve_active(family, tickers=tickers, dataset_version=version))
        seen = set()
        for record in records:
            if record is None:
                raise OptimizationInputError("Selected model does not exist")
            record_family = str(record.get("family", "")).lower()
            record_family = "garch" if record_family in {"garch", "gjr_garch"} else record_family
            if record_family != family or str(record.get("state", "")).lower() not in {"active", "validated"}:
                raise OptimizationInputError(f"Selected model {record.get('id')} is not a validated {family} model")
            if str(record.get("inference_dataset_version", record.get("dataset_version"))) != str(version):
                raise OptimizationInputError(f"Model {record.get('id')} is incompatible with dataset {version}")
            if record["ticker"] not in tickers or record["ticker"] in seen:
                raise OptimizationInputError("Selected models must cover unique tickers in the requested universe")
            seen.add(record["ticker"])
        return records

    def _predictions(self, records, family, data, context, work):
        rows = []
        materialized = {}
        cutoff = pd.to_datetime(data.index.max(), utc=True)
        for record in records:
            context.check_cancelled()
            trained_until = record.get("training_cutoff") or record.get("preprocessing", {}).get("training_cutoff")
            if trained_until and pd.to_datetime(trained_until, utc=True) > cutoff:
                raise OptimizationInputError(f"Model {record['id']} was trained after the requested decision cutoff")
            if family in {"lstm", "mlp"} and int(record.get("horizon", 5)) != 5:
                raise OptimizationInputError(f"Model {record['id']} must predict a five-session daily-average return")
            uri = record.get("artifact_uri")
            if not uri:
                raise OptimizationInputError(f"Model {record['id']} has no verified artifact reference")
            if uri not in materialized:
                destination = work / "model_cache" / hashlib.sha256(uri.encode()).hexdigest()
                materialized[uri] = Path(context.store.materialize(uri, destination))
            bundle = materialized[uri]
            relative = record.get("prediction_file") or "predictions.csv"
            prediction_file = bundle / relative
            if not prediction_file.resolve().is_relative_to(bundle.resolve()):
                raise OptimizationInputError("Model prediction path escapes its artifact bundle")
            predictions = pd.read_csv(prediction_file)
            required = {"ticker", "dataset_version", "base_currency", "as_of", "prediction_units",
                        "vol_pred" if family == "garch" else "mu_pred"}
            if not required.issubset(predictions):
                raise OptimizationInputError(f"Model artifact is missing forecast contract fields: {sorted(required - set(predictions))}")
            selected = predictions.loc[predictions.ticker.astype(str).eq(str(record["ticker"]))].copy()
            if len(selected) != 1:
                raise OptimizationInputError(f"Model artifact requires one forecast for {record['ticker']}")
            selected["registry_model_id"] = str(record["id"])
            selected["artifact_uri"] = uri
            rows.append(selected)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def run(self, config, context):
        config = _configuration(config)
        self._services(context)
        pinned = getattr(context, "dataset_version", None) or config.get("dataset_version")
        record = self.datasets.resolve(pinned)
        version = record["version"]
        context.check_cancelled()
        context.progress(10, "dataset", "Loading the pinned optimization dataset")
        work = Path(context.temp_dir) / "optimization"
        work.mkdir(parents=True, exist_ok=True)
        snapshot = self.datasets.materialize(version, work / "dataset")
        for explicit in (config.get("dataset_version"), getattr(context, "dataset_version", None)):
            if explicit and str(explicit) not in {str(snapshot.id), str(snapshot.version)}:
                raise OptimizationInputError("Optimization job and request reference different dataset versions")
        data, metadata = snapshot.load_returns(), snapshot.load_metadata()
        config = {**config, "dataset_version": version}
        config, metadata = self._universe(config, metadata, context)
        data = _cutoff_frame(data, config)
        mode, estimator = _mode(config)
        return_family = {"ai_returns_lstm": "lstm", "ai_returns": "mlp"}.get(mode)
        risk_family = "garch" if estimator == "gjr_garch" or (mode == "ai_volatility" and estimator is None) else None
        return_records = self._records(return_family, config.get("return_model_ids"), list(data), version, config.get("pinned_return_models")) if return_family else []
        risk_records = self._records(risk_family, config.get("volatility_model_ids"), list(data), version, config.get("pinned_volatility_models")) if risk_family else []
        return_predictions = self._predictions(return_records, return_family, data, context, work)
        volatility_predictions = self._predictions(risk_records, risk_family, data, context, work)
        context.progress(30, "optimization", "Validating signals and portfolio constraints")
        context.check_cancelled()
        result = optimize_frame(data, metadata, config, return_predictions, volatility_predictions)
        result["model_references"] = {"returns": return_records, "volatility": risk_records}
        context.check_cancelled()
        artifacts = work / "artifacts"
        artifacts.mkdir(exist_ok=True)
        data.to_csv(artifacts / "returns.csv", index_label="date")
        metadata.to_csv(artifacts / "assets_metadata.csv", index=False)
        result["estimates"]["mu"].to_csv(artifacts / "mu.csv")
        result["estimates"]["covariance"].to_csv(artifacts / "covariance.csv")
        result["blocks"]["weights_df"].to_csv(artifacts / "weights.csv", index=False)
        result["blocks"]["exposure_by_block"].to_csv(artifacts / "block_exposures.csv", index=False)
        result["frontier"].to_csv(artifacts / "frontier_classical.csv", index=False)
        result["constrained_frontier"]["points"].to_csv(artifacts / "frontier_constrained.csv", index=False)
        (artifacts / "result.json").write_text(json.dumps(to_builtin(result), sort_keys=True, indent=2, allow_nan=False))
        context.progress(85, "artifacts", "Optimization artifacts ready for immutable publication")
        return {"dataset_version": version, "dataset_id": snapshot.id, "metrics": result["blocks"]["stats"],
                "artifacts_dir": str(artifacts), "result": to_builtin(result),
                "model_ids": [record["id"] for record in return_records + risk_records]}
