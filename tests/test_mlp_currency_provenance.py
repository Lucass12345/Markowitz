"""Keep MLP predictions tied to the USD dataset they actually consumed."""
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.data import load_data
from src.models import predict
from src.optimization.mlp_lab_loader import load_active_mlp_predictions


@pytest.fixture
def usd_dataset(tmp_path, monkeypatch):
    data = tmp_path / "Data"
    snapshot = data / "dynamic" / "usd-v1"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(json.dumps({"base_currency": "USD"}))
    (data / "active_dataset.json").write_text(json.dumps({"version": snapshot.name}))
    returns = pd.DataFrame({"NEW": np.random.default_rng(42).normal(0, .01, 270)},
                           index=pd.bdate_range("2023-01-01", periods=270))
    returns.to_csv(snapshot / "returns.csv")
    monkeypatch.setattr(load_data, "DATA_DIR", data)
    return data, snapshot


def fake_storage(monkeypatch, run_directory, predictions):
    storage = ModuleType("app.mlp_lab_storage")
    storage.get_active_mlp_prediction_run = lambda: {"run_id": "selected", "run_output_dir": str(run_directory)}
    storage.get_mlp_run_tickers_df = lambda run_id: predictions.copy() if run_id == "selected" else None
    storage._apply_selection_policy = lambda frame, policy: frame.assign(use_for_optimizer=True)
    monkeypatch.setitem(sys.modules, "app.mlp_lab_storage", storage)
    return storage


@pytest.mark.parametrize("provenance_file", ["predictions.csv", "config.json"])
def test_active_mlp_loads_local_usd_provenance_without_schema_change(usd_dataset, tmp_path, monkeypatch, provenance_file):
    _, snapshot = usd_dataset
    run = tmp_path / "run"
    run.mkdir()
    info = {"dataset_version": snapshot.name, "base_currency": "USD"}
    if provenance_file.endswith("csv"):
        pd.DataFrame({"ticker": ["NEW"], **{key: [value] for key, value in info.items()}}).to_csv(run / provenance_file, index=False)
    else:
        (run / provenance_file).write_text(json.dumps(info))
    fake_storage(monkeypatch, run, pd.DataFrame({"ticker": ["NEW"], "mu_pred": [.001]}))
    result = load_active_mlp_predictions()
    assert result.iloc[0].dataset_version == snapshot.name
    assert result.iloc[0].base_currency == "USD"
    assert result.iloc[0].use_for_optimizer


@pytest.mark.parametrize("info", [
    {},
    {"dataset_version": "old-mixed", "base_currency": "USD"},
    {"dataset_version": "usd-v1", "base_currency": "ARS"},
])
def test_old_or_unverifiable_mlp_predictions_rejected(usd_dataset, tmp_path, monkeypatch, info):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(info))
    fake_storage(monkeypatch, run, pd.DataFrame({"ticker": ["NEW"], "mu_pred": [.001]}))
    with pytest.raises(ValueError):
        load_active_mlp_predictions()


def test_legacy_merge_preserves_one_usd_dataset_provenance(usd_dataset, tmp_path, monkeypatch):
    _, snapshot = usd_dataset
    monkeypatch.setattr(predict, "OUTPUT_DIR", tmp_path)
    provenance = {"ticker": ["NEW"], "dataset_version": [snapshot.name], "base_currency": ["USD"]}
    pd.DataFrame({**provenance, "mu_pred": [.001]}).to_csv(tmp_path / "return_predictions.csv", index=False)
    pd.DataFrame({**provenance, "vol_pred": [.01]}).to_csv(tmp_path / "volatility_predictions.csv", index=False)
    result = predict.build_prediction_table()
    assert list(result.columns).count("dataset_version") == 1
    assert list(result.columns).count("base_currency") == 1
    assert not any(name.endswith(("_x", "_y")) for name in result.columns)
    assert result.iloc[0].base_currency == "USD"


@pytest.mark.parametrize("wrong_field, wrong_value", [("dataset_version", "old-mixed"), ("base_currency", "ARS")])
def test_legacy_merge_rejects_mismatched_inputs_before_overwriting(usd_dataset, tmp_path, monkeypatch, wrong_field, wrong_value):
    _, snapshot = usd_dataset
    monkeypatch.setattr(predict, "OUTPUT_DIR", tmp_path)
    ret = pd.DataFrame({"ticker": ["NEW"], "mu_pred": [.001], "dataset_version": [snapshot.name], "base_currency": ["USD"]})
    vol = ret.drop(columns="mu_pred").assign(vol_pred=.01)
    vol[wrong_field] = wrong_value
    ret.to_csv(tmp_path / "return_predictions.csv", index=False)
    vol.to_csv(tmp_path / "volatility_predictions.csv", index=False)
    previous = tmp_path / "predictions.csv"
    previous.write_text("previous valid predictions")
    with pytest.raises(ValueError):
        predict.build_prediction_table()
    assert previous.read_text() == "previous valid predictions"


def test_classic_mlp_training_keeps_the_snapshot_read_at_start(usd_dataset, tmp_path, monkeypatch):
    from src.models import train_returns as trainer
    data, snapshot = usd_dataset
    seen = []
    original_load = trainer.load_returns
    def capture_load(*, data_dir):
        seen.append(data_dir)
        return original_load(data_dir=data_dir)
    class Model:
        def fit(self, X, y):
            (data / "active_dataset.json").write_text(json.dumps({"version": "changed-during-training"}))
        def evaluate(self, X, y):
            return {"mse": .001, "rmse": .01}
        def predict(self, X):
            return np.zeros(len(X))
        def save(self, path):
            pass
    monkeypatch.setattr(trainer, "load_returns", capture_load)
    monkeypatch.setattr(trainer, "ReturnMLPModel", Model)
    monkeypatch.setattr(trainer, "OUTPUT_DIR", tmp_path / "outputs")
    result = trainer.train_all_return_models()
    assert seen == [snapshot]
    assert result.iloc[0].dataset_version == snapshot.name
    assert result.iloc[0].base_currency == "USD"


def test_mlp_lab_persists_pinned_usd_provenance(usd_dataset, tmp_path, monkeypatch):
    from src.models import train_returns_mlp_lab as lab
    data, snapshot = usd_dataset
    persisted = []
    class Model:
        def __init__(self, steps):
            pass
        def fit(self, X, y):
            (data / "active_dataset.json").write_text(json.dumps({"version": "changed-during-training"}))
        def predict(self, X):
            return np.zeros(len(X))
    storage = ModuleType("app.mlp_lab_storage")
    storage.persist_mlp_lab_run = lambda **values: persisted.append(values)
    monkeypatch.setitem(sys.modules, "app.mlp_lab_storage", storage)
    monkeypatch.setattr(lab, "Pipeline", Model)
    monkeypatch.setattr(lab.joblib, "dump", lambda *args: None)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "OUTPUT_ROOT", tmp_path / "lab")
    monkeypatch.setattr(lab, "MODEL_ROOT", tmp_path / "models")
    result = lab.run_mlp_lab(start_date="full")
    assert result["predictions"].iloc[0].dataset_version == snapshot.name
    assert result["predictions"].iloc[0].base_currency == "USD"
    config = json.loads((Path(result["run_output_dir"]) / "config.json").read_text())
    assert config["dataset_version"] == snapshot.name and config["base_currency"] == "USD"
    assert persisted[0]["summary"]["dataset_version"] == snapshot.name
