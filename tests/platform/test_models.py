"""Small real fits exercise worker artifacts, temporal evaluation and inference."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from src.platform.artifacts import MemoryArtifactStore
from src.platform.models import ModelService, _accept, _scores, _split_metrics


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def returns():
    random = np.random.default_rng(42)
    frame = pd.DataFrame({"AAA": random.normal(.0005, .009, 340)},
                         index=pd.bdate_range("2023-01-02", periods=340))
    frame.attrs.update(dataset_version="tiny-v1", base_currency="USD")
    return frame


def context(directory, identity="tiny-fit"):
    return SimpleNamespace(job_id=identity, temp_dir=Path(directory), actor="test")


def configuration(family):
    common = dict(start_date="full", device="cpu", seed=42)
    if family == "lstm":
        return dict(common, seq_len=8, epochs=2, hidden_size=4, num_layers=1,
                    selected_features=["return", "lag_1"], search=False,
                    dropout=0, batch_size=64, target_aggregation="sum")
    if family == "mlp":
        return dict(common, hidden_layers=[5], max_iter=20, early_stopping=False,
                    lags=3, short_window=3, long_window=8, target_aggregation="sum")
    return dict(common, validation_days=3, test_days=3)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory, returns):
    service = ModelService()
    return {family: service.train_on_frame(family, returns, "tiny-v1", configuration(family),
                                         context(tmp_path_factory.mktemp(family), f"tiny-{family}"))
            for family in ("lstm", "mlp", "garch")}


@pytest.mark.parametrize("family", ["lstm", "mlp", "garch"])
def test_real_training_saves_complete_isolated_bundle(family, fitted, returns):
    result = fitted[family]
    root = Path(result["artifacts_dir"])
    record, prediction = result["models"][0], result["predictions"][0]
    assert root.is_dir()
    assert (root / record["relative_path"]).is_file()
    assert (root / record["metadata_path"]).is_file()
    assert (root / record["prediction_file"]).is_file()
    assert record["horizon"] == 5
    assert record["dataset_version"] == "tiny-v1"
    assert record["training_cutoff"] == str(returns.index[-1].date())
    assert record["cutoff"] == record["as_of"] == prediction["as_of"]
    assert prediction["base_currency"] == record["base_currency"] == "USD"
    assert prediction["use_for_optimizer"] == record["accepted"]
    assert record["validation_metrics"] == record["metrics"]["val"]
    assert all(record["metrics"][split]["n"] > 0 for split in ("train", "val", "test"))
    assert json.loads((root / "config.json").read_text())["test_used_for_selection"] is False
    assert not list(root.rglob("latest_run.json"))
    # These manifests must be valid JSON, including rejected or missing metrics.
    json.loads((root / "models.json").read_text(), parse_constant=lambda value: pytest.fail(value))
    if family == "lstm":
        assert record["preprocessing"]["selected_config"]["hidden_sizes"] == [4]
        assert prediction["amp_ratio"] == record["validation_metrics"]["amplitude_ratio"]


@pytest.mark.parametrize("family", ["lstm", "mlp", "garch"])
def test_same_seed_reproduces_real_fit(family, fitted, returns, tmp_path):
    import torch
    torch.manual_seed(814)
    state = torch.random.get_rng_state().clone()
    again = ModelService().train_on_frame(family, returns, "tiny-v1", configuration(family),
                                         context(tmp_path, f"repeat-{family}"))
    original = fitted[family]
    key = "vol_pred" if family == "garch" else "mu_pred"
    assert again["predictions"][0][key] == pytest.approx(original["predictions"][0][key], rel=1e-12)
    assert again["models"][0]["metrics"] == original["models"][0]["metrics"]
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("family", ["lstm", "mlp", "garch"])
def test_inference_loads_original_verified_weights(family, fitted, returns, tmp_path):
    result = fitted[family]
    store = MemoryArtifactStore()
    original = store.upload_directory(result["artifacts_dir"], f"models/{family}")
    # The registry's current artifact may be a later prediction-only bundle.
    predictions = tmp_path / "only-predictions"
    predictions.mkdir()
    (predictions / "prediction.json").write_text("{}")
    later = store.upload_directory(predictions, f"inference/{family}")
    record = dict(result["models"][0], model_artifact_uri=original["uri"], artifact_uri=later["uri"])
    snapshot = SimpleNamespace(version="tiny-v1", load_returns=lambda: returns.copy())
    output = ModelService(store=store).infer(record, snapshot, context(tmp_path / "worker", "infer-job"))
    key = "vol_pred" if family == "garch" else "mu_pred"
    assert output[key] == pytest.approx(result["predictions"][0][key], rel=1e-12)
    assert output["as_of"] == str(returns.index[-1].date())
    assert output["model_run_id"] == result["models"][0]["model_run_id"]
    assert output["job_id"] == "infer-job"
    assert (Path(output["artifacts_dir"]) / output["prediction_file"]).is_file()
    assert set(Path(output["artifacts_dir"]).rglob("*.pt")) == set()
    manifest = json.loads((Path(output["artifacts_dir"]) / "inference.json").read_text())
    assert manifest["model_artifact_uri"] == original["uri"]


@pytest.mark.parametrize("family", ["lstm", "mlp"])
def test_sum_target_exports_daily_units(family, fitted):
    result = fitted[family]
    prediction, record = result["predictions"][0], result["models"][0]
    assert prediction["prediction_units"] == "daily_log_return"
    assert prediction["forecast_log_return_5d"] == pytest.approx(5 * prediction["mu_pred"])
    assert prediction["forecast_simple_return_5d"] == pytest.approx(np.expm1(5 * prediction["mu_pred"]))
    root = Path(result["artifacts_dir"])
    details = pd.read_csv(next((root / "evaluation").rglob("AAA_predictions.csv")))
    expected = _split_metrics(details, divisor=5)
    assert record["validation_metrics"]["rmse"] == pytest.approx(expected["val"]["rmse"])
    assert record["validation_metrics"]["baseline_mean_rmse"] == pytest.approx(expected["val"]["baseline_mean_rmse"])


def test_garch_parameters_are_reloadable_and_variance_units_are_daily(fitted):
    result = fitted["garch"]
    record, prediction = result["models"][0], result["predictions"][0]
    parameters = json.loads((Path(result["artifacts_dir"]) / record["relative_path"]).read_text())
    assert len(parameters["parameters"]) == len(parameters["parameter_names"])
    assert {"omega", "alpha[1]", "gamma[1]", "beta[1]"} <= set(parameters["parameter_names"])
    assert prediction["prediction_units"] == "daily_log_volatility"
    assert prediction["variance_pred"] == pytest.approx(prediction["vol_pred"] ** 2)
    assert prediction["variance_sum_5d"] == pytest.approx(prediction["vol_5d"] ** 2)
    assert prediction["variance_sum_5d"] > prediction["variance_pred"] > 0
    assert record["metrics"]["target_units"] == "squared_daily_log_return"
    assert record["validation_metrics"]["baseline_ewma_rmse"] >= 0


def test_acceptance_never_uses_test_and_lstm_thresholds_are_strict():
    val = dict(n=20, rmse=.1, baseline_mean_rmse=.2, dir_acc=.6, amplitude_ratio=.2)
    metrics = dict(val=val, test=dict(rmse=1000, dir_acc=0))
    assert _accept("lstm", metrics, {}, {})[0]
    metrics["test"] = dict(rmse=0, dir_acc=1, amplitude_ratio=10)
    assert _accept("lstm", metrics, {}, {})[0]
    for boundary in (dict(dir_acc=.52), dict(amplitude_ratio=.15)):
        assert not _accept("lstm", dict(val=dict(val, **boundary)), {}, {})[0]
    for missing in (None, float("nan"), float("inf")):
        assert not _accept("lstm", dict(val=dict(val, baseline_mean_rmse=missing)), {}, {})[0]
        assert not _accept("lstm", dict(val=dict(val, dir_acc=missing)), {}, {})[0]
    assert not _accept("mlp", metrics, {}, {"validation_composite_available": False})[0]
    assert not _accept("garch", metrics, {}, {"converged": True, "covariance_stationary": False})[0]


def test_missing_and_constant_metrics_remain_missing():
    assert _scores([np.nan], [1], 0) == {"n": 0}
    scores = _scores([1, 1], [1, 1], np.nan, [np.nan, 1])
    assert scores["amplitude_ratio"] is scores["corr"] is None
    assert scores["baseline_mean_rmse"] is scores["baseline_ewma_rmse"] is None
    assert "qlike" not in scores


def test_targets_and_partitions_purge_overlapping_futures():
    from src.models.sequence_utils import build_lstm_sequences_multifeature
    from src.models.train_returns_lstm_lab import temporal_split as lstm_split
    from src.models.train_returns_mlp_lab import build_target, temporal_split as mlp_split
    series = pd.Series(np.arange(100, dtype=float), index=pd.bdate_range("2024-01-01", periods=100))
    features = series.to_frame("return")
    _, mean = build_lstm_sequences_multifeature(features, series, seq_len=3, forecast_horizon=5)
    _, summed = build_lstm_sequences_multifeature(features, series, seq_len=3, forecast_horizon=5, target_aggregation="sum")
    assert mean[0] == np.mean([3, 4, 5, 6, 7])
    assert np.allclose(summed, 5 * mean)
    assert build_target(series, 5, "sum").iloc[0] == sum([1, 2, 3, 4, 5])
    assert build_target(series, 5, "mean").tail(5).isna().all()
    train, val, test = mlp_split(features, forecast_horizon=5)
    assert series.index.get_loc(train.index[-1]) + 5 <= series.index.get_loc(val.index[0])
    assert series.index.get_loc(val.index[-1]) + 5 <= series.index.get_loc(test.index[0])
    splits = lstm_split(np.zeros((100, 3, 1)), series.to_numpy(), np.arange(100), forecast_horizon=5)
    assert splits[2][-1] + 5 <= splits[5][0]
    assert splits[5][-1] + 5 <= splits[8][0]


def test_train_pins_resolved_snapshot_and_does_not_resolve_local_pointer(returns, tmp_path, monkeypatch):
    seen = []
    snapshot = SimpleNamespace(id="immutable-dataset-id", version="tiny-v1", load_returns=lambda: returns)
    datasets = SimpleNamespace(
        resolve=lambda version: (seen.append(("resolve", version)) or {"id": snapshot.id}),
        materialize=lambda identity, destination: (seen.append(("materialize", identity, destination)) or snapshot))
    service = ModelService(dataset_service=datasets)
    monkeypatch.setattr(service, "train_on_frame", lambda family, frame, version, cfg, ctx: {"dataset_version": version})
    result = service.train("mlp", {"dataset_version": "requested-version"}, context(tmp_path))
    assert seen == [("resolve", "requested-version"), ("materialize", snapshot.id, tmp_path / "dataset")]
    assert result == {"dataset_version": "tiny-v1", "dataset_id": snapshot.id}


def test_inference_rejects_registry_identity_and_future_training_cutoff(fitted, returns, tmp_path):
    result = fitted["mlp"]
    store = MemoryArtifactStore()
    uri = store.upload_directory(result["artifacts_dir"], "models/identity")["uri"]
    record = dict(result["models"][0], model_artifact_uri=uri)
    snapshot = SimpleNamespace(version="tiny-v1", load_returns=lambda: returns.copy())
    with pytest.raises(ValueError, match="identity"):
        ModelService(store=store).infer(dict(record, model_run_id="another-training"), snapshot, context(tmp_path))
    earlier = returns.iloc[:-1].copy()
    earlier.attrs.update(returns.attrs)
    snapshot.load_returns = lambda: earlier
    with pytest.raises(ValueError, match="cutoff"):
        ModelService(store=store).infer(record, snapshot, context(tmp_path))


@pytest.mark.parametrize("family", ["lstm", "mlp", "garch"])
def test_fitted_weights_predict_new_snapshot_without_retraining(family, fitted, returns, tmp_path, monkeypatch):
    from src.models import vol_model_garch
    result = fitted[family]
    store = MemoryArtifactStore()
    uri = store.upload_directory(result["artifacts_dir"], f"models/frozen-{family}")["uri"]
    record = dict(result["models"][0], model_artifact_uri=uri)
    newer = pd.concat([returns, pd.DataFrame({"AAA": [.008, -.004]},
                      index=pd.bdate_range(returns.index[-1] + pd.offsets.BDay(), periods=2))])
    newer.attrs.update(dataset_version="tiny-v2", base_currency="USD")
    snapshot = SimpleNamespace(version="tiny-v2", load_returns=lambda: newer)
    def cannot_fit(*args, **kwargs):
        pytest.fail("Inference attempted to refit a model")
    monkeypatch.setattr(ModelService, "train_on_frame", cannot_fit)
    monkeypatch.setattr(vol_model_garch, "fit_gjr", cannot_fit)
    output = ModelService(store=store).infer(record, snapshot, context(tmp_path, "new-snapshot"))
    assert output["dataset_version"] == "tiny-v2"
    assert output["as_of"] == str(newer.index[-1].date())
    assert output["model_run_id"] == record["model_run_id"]


def test_cancelled_training_does_not_leave_partial_model_bundle(returns, tmp_path):
    class Cancelled(Exception):
        pass
    ctx = context(tmp_path)
    def cancelled():
        raise Cancelled()
    ctx.check_cancelled = cancelled
    with pytest.raises(Cancelled):
        ModelService().train_on_frame("lstm", returns, "tiny-v1", configuration("lstm"), ctx)
    assert not list(tmp_path.glob("model-bundle-*"))
    assert not list(tmp_path.glob(".models-*"))


@pytest.mark.parametrize("change", [{"base_currency": "ARS"}, {"dataset_version": "wrong"}])
def test_training_rejects_unverified_units_or_version(returns, tmp_path, change):
    invalid = returns.copy()
    invalid.attrs.update(change)
    with pytest.raises(ValueError):
        ModelService().train_on_frame("mlp", invalid, "tiny-v1", configuration("mlp"), context(tmp_path))
