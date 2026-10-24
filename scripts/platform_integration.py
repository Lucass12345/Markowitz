#!/usr/bin/env python3
"""Exercise the real API, workers, PostgreSQL, Redis, MinIO and MLflow.

Run inside the integration API container. ``before-restart`` writes durable
evidence; restart API and worker externally, then run ``after-restart`` with the
same state path. No provider requests, mocks, or writes to historical CSVs occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TICKERS = ["AAPL", "TLT"]
PARAMETERS = {
    "lstm": {"start_date": "2024-01-01", "device": "cpu", "seed": 42,
             "seq_len": 8, "epochs": 2, "hidden_size": 4, "num_layers": 1,
             "selected_features": ["return", "lag_1"], "search": False,
             "dropout": 0, "batch_size": 64, "target_aggregation": "sum"},
    "mlp": {"start_date": "2024-01-01", "device": "cpu", "seed": 42,
            "hidden_layers": [5], "max_iter": 20, "early_stopping": False,
            "lags": 3, "short_window": 3, "long_window": 8,
            "target_aggregation": "sum"},
    "garch": {"seed": 42, "validation_days": 3, "test_days": 3},
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def wait_for(callback, description, timeout=180, interval=1):
    from redis.exceptions import RedisError
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            value = callback()
            if value:
                return value
        except (requests.RequestException, OSError, ConnectionError, OperationalError, RedisError) as exc:
            last_error = type(exc).__name__
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for {description}" + (f" ({last_error})" if last_error else ""))


def rows(value, *keys):
    if isinstance(value, list):
        return value
    for key in (*keys, "items", "results"):
        if isinstance(value.get(key), list):
            return value[key]
    raise AssertionError(f"Expected a list response with one of {keys}")


class Harness:
    def __init__(self, base_url=None, timeout=1200):
        self.base_url = (base_url or os.getenv("PLATFORM_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout
        self.http = requests.Session()
        if os.getenv("MARKOWITZ_API_KEY"):
            self.http.headers["X-API-Key"] = os.environ["MARKOWITZ_API_KEY"]
        self.engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
        require(self.engine.dialect.name == "postgresql", "Integration requires real PostgreSQL")
        from src.platform.artifacts import ArtifactStore
        self.store = ArtifactStore()
        from mlflow.tracking import MlflowClient
        self.mlflow = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        import redis
        self.redis = redis.Redis.from_url(os.environ["CELERY_BROKER_URL"])

    def request(self, method, path, payload=None, *, key=None, expected=(200,)):
        headers = {"Idempotency-Key": key} if key else {}
        response = self.http.request(method, self.base_url + path, json=payload, headers=headers, timeout=30)
        require(response.status_code in expected,
                f"{method} {path}: expected {expected}, got {response.status_code}: {response.text[:1200]}")
        return response.json() if response.content else {}

    def sql(self, query, **parameters):
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(text(query), parameters).mappings()]

    def ready(self):
        wait_for(lambda: self.http.get(self.base_url + "/health/live", timeout=5).status_code == 200,
                 "API liveness")
        wait_for(lambda: self.redis.ping(), "Redis")
        wait_for(lambda: self.sql("SELECT 1"), "PostgreSQL")
        endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
        scheme = "https" if os.environ.get("MINIO_SECURE", "false").lower() == "true" else "http"
        wait_for(lambda: requests.get(f"{scheme}://{endpoint}/minio/health/live", timeout=5).ok, "MinIO")
        wait_for(lambda: requests.get(os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/health", timeout=5).ok,
                 "MLflow")
        print("Real API, PostgreSQL, Redis, MinIO and MLflow are reachable", flush=True)

    def submit(self, path, payload, key):
        submitted = self.request("POST", "/api/v1" + path, payload, key=key, expected=(202,))
        identity = submitted.get("id", submitted.get("job_id"))
        require(identity, f"{path} did not return a durable job ID")
        return str(identity)

    def completed(self, identity, expected_status="completed"):
        previous = None

        def poll():
            nonlocal previous
            job = self.request("GET", "/api/v1/jobs/" + identity)
            status = job["status"]
            stage = (status, job.get("stage"))
            if stage != previous:
                print(f"Job {identity}: {status}, {job.get('stage')}", flush=True)
                previous = stage
            require(status not in ({"completed", "failed", "cancelled"} - {expected_status}),
                    f"Job {identity} {status}: {job.get('error')} (stage={job.get('stage')})")
            return job if status == expected_status else None

        return wait_for(poll, f"job {identity}", timeout=self.timeout, interval=1)

    @staticmethod
    def result(job):
        result = job.get("result_json", job.get("result"))
        require(isinstance(result, dict), f"Job {job.get('id')} has no persisted result")
        while set(result) == {"result"} and isinstance(result["result"], dict):
            result = result["result"]
        return result

    def verify_experiment(self, identity, expected_files=()):
        records = self.sql("SELECT * FROM platform_experiments WHERE job_id = :identity", identity=identity)
        require(len(records) == 1, f"Expected one experiment for job {identity}")
        record = records[0]
        require(record["status"] == "completed", "Experiment publication is incomplete")
        require(record["artifact_uri"].startswith("s3://"), "Experiment has no canonical object-store URI")
        require(record["mlflow_run_id"], "Completed experiment must have required MLflow run")
        run = self.mlflow.get_run(record["mlflow_run_id"])
        require(run.info.status == "FINISHED", "MLflow run is not finished")
        require(run.data.tags.get("platform_job_id") == identity, "MLflow job identity differs from SQL")
        require(run.data.tags.get("dataset_version") == record["dataset_version"], "MLflow dataset differs from SQL")
        require(any(item.path == "platform_result.json" for item in self.mlflow.list_artifacts(run.info.run_id)),
                "MLflow publication report is missing")
        manifest = self.store.read_json(record["artifact_uri"])
        require(manifest["sha256"] == record["content_hash"], "SQL and MinIO bundle hashes differ")
        with tempfile.TemporaryDirectory(prefix="integration-verify-") as temporary:
            root = Path(self.store.materialize(record["artifact_uri"], temporary))
            for relative in expected_files:
                require((root / relative).is_file() and (root / relative).stat().st_size > 0,
                        f"Published bundle is missing {relative}")
            payload = json.loads((root / "platform_result.json").read_text())
            require(payload["job_id"] == identity, "Transported publication identity differs")
        return {"id": identity, "artifact_uri": record["artifact_uri"],
                "content_hash": record["content_hash"], "mlflow_run_id": record["mlflow_run_id"],
                "minio_files": sorted(manifest["files"])}

    def train(self, family, version, run_id):
        payload = {"family": family, "dataset_version": version, "tickers": TICKERS,
                   "parameters": PARAMETERS[family], "timeout_seconds": self.timeout}
        key = f"integration:{run_id}:train:{family}"
        identity = self.submit("/experiments", payload, key)
        job = self.completed(identity)
        result = self.result(job)
        require(result.get("dataset_version") == version, "Training used another dataset")
        models = rows(self.request("GET", f"/api/v1/models?family={family}&dataset_version={version}"), "models")
        models = [model for model in models if model["experiment_id"] == identity]
        require({model["ticker"] for model in models} == set(TICKERS), f"{family} did not fit both requested assets")
        expected = ["models.json", "metrics.json", "config.json", f"predictions/{family}.csv"]
        expected.extend(model["relative_path"] for model in models)
        evidence = self.verify_experiment(identity, expected)
        for model in models:
            require(model["state"] in {"validated", "rejected"}, "Training must not activate models")
            require(model["features"] and model["preprocessing"], "Model preprocessing contract is absent")
            require(model["validation_metrics"].get("n", 0) > 0, "Model has no out-of-sample validation evidence")
        return {"id": identity, "path": "/experiments", "payload": payload, "key": key,
                "models": models, "evidence": evidence, "attempts": job["attempts"]}

    def infer_and_activate(self, model, version, run_id):
        path = f"/models/{model['id']}/activate"
        activation = self.submit(path, {}, f"integration:{run_id}:activate:{model['id']}")
        if model["state"] == "rejected":
            rejected = self.completed(activation, expected_status="failed")
            require(rejected.get("error_status") == 409, "Rejected model activation must report a validation conflict")
        else:
            self.completed(activation)
        payload = {"dataset_version": version}
        inference = self.submit(f"/models/{model['id']}/infer", payload,
                                f"integration:{run_id}:infer:{model['id']}")
        result = self.result(self.completed(inference))
        predictions = result.get("predictions") or [result]
        require(len(predictions) == 1 and predictions[0]["ticker"] == model["ticker"], "Inference identity changed")
        field = "vol_pred" if model["family"] == "garch" else "mu_pred"
        require(math.isfinite(float(predictions[0][field])), "Reloaded model emitted a nonfinite prediction")
        import pandas as pd
        with tempfile.TemporaryDirectory(prefix="integration-training-prediction-") as temporary:
            root = Path(self.store.materialize(model["artifact_uri"], temporary))
            original = pd.read_csv(root / f"predictions/{model['family']}.csv").set_index("ticker").loc[model["ticker"]]
            require(math.isclose(float(predictions[0][field]), float(original[field]), rel_tol=1e-7, abs_tol=1e-10),
                    "Reloaded model differs from the prediction stored during training")
        evidence = self.verify_experiment(inference, [f"predictions/{model['family']}.csv", "inference.json"])
        return {"model_id": model["id"], "family": model["family"], "ticker": model["ticker"],
                "job_id": inference, "activated": model["state"] != "rejected",
                "prediction": predictions[0], "evidence": evidence}

    def verify_portfolio(self, evidence):
        import pandas as pd
        with tempfile.TemporaryDirectory(prefix="integration-portfolio-") as temporary:
            root = Path(self.store.materialize(evidence["artifact_uri"], temporary))
            weights = pd.read_csv(root / "weights.csv").set_index("ticker")["weight"]
            require(set(weights.index) == set(TICKERS), "Optimizer changed the chosen universe")
            require(all(math.isfinite(float(value)) and 0 <= value <= 1 for value in weights), "Portfolio weights are invalid")
            require(math.isclose(float(weights.sum()), 1., abs_tol=1e-7), "Portfolio weights do not sum to one")
            require(all(math.isclose(float(value), .5, abs_tol=1e-7) for value in weights), "Portfolio violated explicit block targets")

    def verify_backtest(self, evidence, strategies):
        import pandas as pd
        with tempfile.TemporaryDirectory(prefix="integration-backtest-") as temporary:
            root = Path(self.store.materialize(evidence["artifact_uri"], temporary))
            curves = pd.read_csv(root / "wealth_curves.csv", index_col=0)
            require(set(curves.columns) == set(strategies) and len(curves) == 6, "Strategies have different OOS calendars")
            require(all(math.isfinite(float(value)) and value > 0 for value in curves.to_numpy().ravel()), "Backtest wealth is invalid")
            for strategy in strategies:
                decisions = json.loads((root / f"{strategy}_decisions.json").read_text())
                require(decisions, f"{strategy} has no rebalance evidence")
                for index, decision in enumerate(decisions):
                    require(pd.Timestamp(decision["train_end"]) <= pd.Timestamp(decision["decision_date"]) < pd.Timestamp(decision["holding_start"]),
                            "Backtest training/decision/holding chronology is invalid")
                    weights = [float(value) for value in decision["weights"].values()]
                    require(all(math.isfinite(value) and value >= 0 for value in weights) and math.isclose(sum(weights), 1., abs_tol=1e-7),
                            "Backtest produced invalid holdings")
                    if strategy == "buy_and_hold" and index:
                        require(decision["turnover"] == 0, "Buy-and-hold unexpectedly rebalanced")
                    else:
                        require(all(math.isclose(value, .5, abs_tol=1e-7) for value in weights),
                                "Backtest rebalance violated the shared block targets")
                    for training in decision.get("diagnostics", {}).get("training", []):
                        require(pd.Timestamp(training["train_cutoff"]) <= pd.Timestamp(decision["decision_date"]), "Fold trained on future data")
                        model_root = root / training["artifacts"]
                        for model in training["models"]:
                            require((model_root / model["relative_path"]).is_file(), "Backtest fold model weights were not published")

    def before_restart(self):
        self.ready()
        run_id = uuid.uuid4().hex
        fixture = Path(__file__).resolve().parents[1] / "Data"
        originals = {name: hashlib.sha256((fixture / name).read_bytes()).hexdigest()
                     for name in ("prices.csv", "returns.csv", "assets_metadata.csv")}
        identity = self.submit("/datasets/import-legacy", {}, f"integration:{run_id}:import")
        imported = self.result(self.completed(identity))
        datasets = rows(self.request("GET", "/api/v1/datasets"), "datasets", "versions")
        active = [item for item in datasets if item.get("active")]
        require(len(active) == 1, "Legacy import must establish one active SQL dataset")
        dataset = active[0]
        require(dataset["base_currency"] == "USD" if "base_currency" in dataset else
                dataset.get("manifest", dataset.get("manifest_json", {}))["base_currency"] == "USD",
                "Imported dataset currency is not USD")
        require(imported["version"] == dataset["version"], "Import result and active SQL version differ")
        from src.platform.datasets import DatasetService
        datasets_service = DatasetService(store=self.store, database=self.engine)
        with tempfile.TemporaryDirectory(prefix="integration-dataset-") as temporary:
            snapshot = datasets_service.materialize(dataset["version"], temporary)
            require(set(TICKERS).issubset(snapshot.load_returns().columns), "Historical fixture lacks integration tickers")
        self.request("GET", "/health/ready")
        training = {family: self.train(family, dataset["version"], run_id) for family in PARAMETERS}
        inference = []
        for trained in training.values():
            for model in trained["models"]:
                evidence = self.infer_and_activate(model, dataset["version"], run_id)
                if evidence:
                    inference.append(evidence)
        mlp_ids = [item["model_id"] for item in inference if item["family"] == "mlp" and item["activated"]]
        require(len(mlp_ids) == len(TICKERS), "Both real MLP fits must validate for the AI optimizer integration case")
        optimization = {"dataset_version": dataset["version"], "tickers": TICKERS, "model_type": "classic",
                        "targets": {"US": .5, "FUNDS_BONDS": .5}, "max_weight": 1.,
                        "min_history": 252, "min_overlap": 126, "n_frontier_points": 10}
        portfolios = []
        for mode in ("classic", "ai_returns"):
            payload = {**optimization, "model_type": mode}
            if mode == "ai_returns":
                payload.update(return_model_ids=mlp_ids, prediction_fallback="required")
            job_id = self.submit("/optimizations", payload, f"integration:{run_id}:optimize:{mode}")
            self.completed(job_id)
            evidence = self.verify_experiment(job_id, ["weights.csv", "result.json", "returns.csv"])
            self.verify_portfolio(evidence)
            portfolios.append(evidence)
        fold_models = {family: {**parameters, "start_date": "full"} for family, parameters in PARAMETERS.items()}
        universe = self.request("POST", "/api/v1/universes", {"name": f"Integration {run_id}",
            "assets": [{"ticker": "AAPL", "block": "US", "enabled": True},
                       {"ticker": "TLT", "block": "FUNDS_BONDS", "enabled": True}],
            "config": {"targets": optimization["targets"], "max_weight": 1.}}, expected=(201,))
        payload = {"dataset_version": dataset["version"], "universe_id": universe["id"],
                   "strategies": ["classic", "lstm", "mlp", "gjr_garch", "combined", "equal_weight", "buy_and_hold"],
                   "train_window": 340, "min_train": 300, "oos_sessions": 6, "rebalance_every": 5,
                   "optimization": optimization, "model_parameters": fold_models,
                   "timeout_seconds": self.timeout}
        backtest_id = self.submit("/backtests", payload, f"integration:{run_id}:backtest")
        self.completed(backtest_id)
        backtest = self.verify_experiment(backtest_id, ["report.json", "wealth_curves.csv", "metrics.csv"])
        self.verify_backtest(backtest, payload["strategies"])
        for name, digest in originals.items():
            require(hashlib.sha256((fixture / name).read_bytes()).hexdigest() == digest,
                    "Integration changed an immutable historical source CSV")
        return {"run_id": run_id, "dataset": dataset, "import_job_id": identity,
                "training": training, "inference": inference, "optimization": portfolios,
                "backtest": backtest, "historical_fixture_hashes": originals,
                "before_restart_verified": True, "after_restart_verified": False}

    def after_restart(self, state):
        self.ready()
        version = state["dataset"]["version"]
        require(self.sql("SELECT version_id FROM platform_active_refs WHERE kind='dataset' AND key='active'")[0]["version_id"]
                == state["dataset"]["id"], "Dataset activation was lost after restart")
        for trained in state["training"].values():
            job = self.completed(trained["id"])
            require(job["attempts"] == trained["attempts"], "A completed training reran during restart")
            identity = self.submit(trained["path"], trained["payload"], trained["key"])
            require(identity == trained["id"], "Idempotent API replay created another job")
            evidence = self.verify_experiment(identity, [model["relative_path"] for model in trained["models"]])
            require(evidence["content_hash"] == trained["evidence"]["content_hash"], "Restart changed model artifacts")
            require(evidence["mlflow_run_id"] == trained["evidence"]["mlflow_run_id"], "Restart duplicated MLflow runs")
        for previous in state["inference"]:
            identity = self.submit(f"/models/{previous['model_id']}/infer", {"dataset_version": version},
                                   f"integration:{state['run_id']}:reload:{previous['model_id']}")
            predictions = self.result(self.completed(identity))
            prediction = (predictions.get("predictions") or [predictions])[0]
            field = "vol_pred" if previous["family"] == "garch" else "mu_pred"
            require(math.isclose(float(prediction[field]), float(previous["prediction"][field]), rel_tol=1e-7, abs_tol=1e-10),
                    "The same model and dataset changed prediction after process restart")
        # A real duplicate Celery delivery must not execute a completed SQL job.
        from worker.celery_app import app
        duplicate = state["training"]["mlp"]
        app.send_task("markowitz.execute_job", args=[duplicate["id"]], task_id=duplicate["id"], queue="cpu")

        def duplicate_consumed():
            if self.redis.llen("cpu"):
                return False
            inspector = app.control.inspect(timeout=2)
            active = inspector.active() or {}
            reserved = inspector.reserved() or {}
            return bool(active) and not any(task.get("id") == duplicate["id"]
                for tasks in [*active.values(), *reserved.values()] for task in tasks)

        wait_for(duplicate_consumed, "real duplicate task delivery", timeout=60)
        require(self.completed(duplicate["id"])["attempts"] == duplicate["attempts"], "Duplicate delivery retrained the model")
        require(len(self.sql("SELECT id FROM platform_experiments WHERE job_id=:identity", identity=duplicate["id"])) == 1,
                "Duplicate delivery created another experiment")
        state["after_restart_verified"] = True
        state["duplicate_delivery_verified"] = True
        return state


def run(phase="all", state_path="/integration/state.json", base_url=None, timeout=1200):
    state_path = Path(state_path)
    harness = Harness(base_url=base_url, timeout=timeout)
    try:
        state = harness.before_restart() if phase in {"all", "before-restart"} else json.loads(state_path.read_text())
        if phase in {"all", "after-restart"}:
            state = harness.after_restart(state)
        state["process_restart_expected"] = phase != "all"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, default=str, allow_nan=False))
        report = {"run_id": state["run_id"], "dataset_version": state["dataset"]["version"],
                  "before_restart_verified": state["before_restart_verified"],
                  "after_restart_verified": state["after_restart_verified"],
                  "duplicate_delivery_verified": state.get("duplicate_delivery_verified", False),
                  "external_process_restart_expected": phase != "all",
                  "models": {family: [{"ticker": model["ticker"], "state": model["state"]}
                                       for model in value["models"]] for family, value in state["training"].items()},
                  "model_reload_evidence": state["inference"],
                  "experiments": [value["evidence"] for value in state["training"].values()] + state["optimization"] + [state["backtest"]]}
        (state_path.parent / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        harness.engine.dispose()
        harness.http.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "before-restart", "after-restart"), default="all")
    parser.add_argument("--state", default="/integration/state.json")
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds per real job")
    arguments = parser.parse_args()
    run(arguments.phase, arguments.state, arguments.api_url, arguments.timeout)
