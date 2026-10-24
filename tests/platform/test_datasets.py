"""Canonical dataset guarantees, using only explicit in-memory object storage."""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
import sys
from threading import Barrier

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select

from src.config import DEFAULT_CONFIG
from src.data.yahoo_dataset import classify_asset, refresh_dataset
from src.platform.artifacts import MemoryArtifactStore
from src.platform.datasets import DatasetService, validate_dataset
from src.platform.errors import ConflictError
from src.platform.universes import UniverseService


@pytest.fixture
def service(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'platform.sqlite'}")
    result = DatasetService(store=MemoryArtifactStore(), database=engine)
    yield result
    engine.dispose()


def source(tmp_path, version="v1", multiplier=1., updated_at=None):
    directory = tmp_path / version
    directory.mkdir()
    dates = pd.date_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"AAA": np.arange(100., 110.) * multiplier}, index=dates)
    prices.to_csv(directory / "prices.csv", index_label="date")
    prices.to_csv(directory / "prices_native.csv", index_label="date")
    np.log(prices / prices.shift()).dropna(how="all").to_csv(directory / "returns.csv", index_label="date")
    pd.DataFrame([{"ticker": "AAA", "block": "US", "currency": "USD", "base_currency": "USD"}]).to_csv(
        directory / "assets_metadata.csv", index=False)
    manifest = {"version": version, "base_currency": "USD", "return_type": "log", "start": "2024-01-01",
                "updated_at": updated_at or datetime.now(timezone.utc).isoformat()}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest


def publish(service, tmp_path, version="v1", activate=False, **kwargs):
    directory, manifest = source(tmp_path, version, **kwargs)
    return service.publish(directory, manifest, actor="tester", activate=activate)


def test_publication_activation_rollback_preserve_pinned_snapshot(service, tmp_path):
    first = publish(service, tmp_path, activate=True)
    snapshot = service.materialize(first["version"], tmp_path / "materialized")
    second = publish(service, tmp_path, "v2", activate=True, multiplier=2.)
    assert service.resolve()["id"] == second["id"]
    assert snapshot.load_prices().iloc[0, 0] == 100.
    assert snapshot.load_returns().attrs["dataset_version"] == "v1"
    assert snapshot.load_returns().attrs["base_currency"] == "USD"
    assert service.rollback(actor="reviewer")["id"] == first["id"]
    assert service.rollback()["id"] == second["id"]
    assert first["artifact_uri"].startswith("s3://test-artifacts/")
    assert first["as_of"] == "2024-01-10"
    assert not (tmp_path / "active_dataset.json").exists()
    with service.engine.connect() as connection:
        actions = connection.execute(select(service.db.audit.c.action)).scalars().all()
    assert actions.count("dataset.publish") == 2
    assert "dataset.rollback" in actions


def test_publication_without_activation_and_idempotent_immutable_version(service, tmp_path):
    directory, manifest = source(tmp_path)
    record = service.publish(directory, manifest)
    assert service.publish(directory, manifest)["id"] == record["id"]
    with pytest.raises(ValueError, match="No active"):
        service.resolve()
    prices = pd.read_csv(directory / "prices.csv", index_col=0)
    prices *= 2
    prices.to_csv(directory / "prices.csv")
    with pytest.raises(ValueError, match="Immutable"):
        service.publish(directory, manifest, activate=True)
    assert len(service.list_versions()) == 1


@pytest.mark.parametrize("change", ["missing_return", "wrong_return", "non_usd", "duplicate_ticker"])
def test_invalid_dataset_cannot_publish(service, tmp_path, change):
    directory, manifest = source(tmp_path)
    if change in {"missing_return", "wrong_return"}:
        returns = pd.read_csv(directory / "returns.csv", index_col=0)
        returns.iloc[2, 0] = np.nan if change == "missing_return" else .99
        returns.to_csv(directory / "returns.csv")
    elif change == "non_usd":
        manifest["base_currency"] = "ARS"
        (directory / "manifest.json").write_text(json.dumps(manifest))
    else:
        metadata = pd.read_csv(directory / "assets_metadata.csv")
        pd.concat([metadata, metadata]).to_csv(directory / "assets_metadata.csv", index=False)
    with pytest.raises(ValueError):
        service.publish(directory, manifest, activate=True)
    assert service.list_versions() == []
    assert service.store.objects == {}


def test_failed_activation_rolls_back_dataset_insert_and_both_references(service, tmp_path, monkeypatch):
    first = publish(service, tmp_path, activate=True)
    original = service._set_ref

    def reject_active(connection, key, version_id, actor):
        if key == "active":
            raise RuntimeError("injected database failure")
        return original(connection, key, version_id, actor)

    monkeypatch.setattr(service, "_set_ref", reject_active)
    with pytest.raises(RuntimeError, match="database failure"):
        publish(service, tmp_path, "v2", activate=True)
    assert service.resolve()["id"] == first["id"]
    assert len(service.list_versions()) == 1
    with service.engine.connect() as connection:
        refs = connection.execute(select(service.db.active_refs.c.key)).scalars().all()
    assert refs == ["active"]


def test_transport_corruption_cannot_activate_or_rollback(service, tmp_path):
    first = publish(service, tmp_path, activate=True)
    second = publish(service, tmp_path, "v2", activate=True)
    corrupted = next(key for key in service.store.objects if "/v1/" in key and key.endswith("/returns.csv"))
    service.store.objects[corrupted] = b"corrupted"
    with pytest.raises(ValueError, match="checksum"):
        service.activate(first["id"])
    with pytest.raises(ValueError, match="checksum"):
        service.rollback()
    assert service.resolve()["id"] == second["id"]


def test_failed_upload_verification_never_registers_dataset(service, tmp_path, monkeypatch):
    first = publish(service, tmp_path, activate=True)
    original = service.store.upload_directory

    def corrupt_after_upload(directory, prefix):
        result = original(directory, prefix)
        key = next(key for key in service.store.objects if "/v2/" in key and key.endswith("/prices.csv"))
        service.store.objects[key] = b"broken upload"
        return result

    monkeypatch.setattr(service.store, "upload_directory", corrupt_after_upload)
    with pytest.raises(ValueError, match="checksum"):
        publish(service, tmp_path, "v2", activate=True)
    assert service.resolve()["id"] == first["id"]
    assert len(service.list_versions()) == 1


class Quotes:
    def __init__(self, values=None, dividends=None, on_request=None):
        self.values = values if values is not None else pd.Series(np.arange(100., 112.), index=pd.date_range("2024-01-01", periods=12))
        self.dividends = dividends or {}
        self.requests = []
        self.on_request = on_request

    def __call__(self, ticker):
        if ticker != "AAA":
            raise AssertionError(f"Unexpected network-equivalent request: {ticker}")
        owner = self

        class Asset:
            info = {"currency": "USD", "country": "United States", "quoteType": "EQUITY", "longName": "Test"}

            def history(self, **kwargs):
                owner.requests.append(kwargs)
                if owner.on_request:
                    owner.on_request()
                frame = owner.values.rename("Close").to_frame()
                frame["Dividends"] = [owner.dividends.get(str(date.date()), 0.) for date in frame.index]
                frame = frame.loc[frame.index >= pd.Timestamp(kwargs["start"])]
                if kwargs.get("end"):
                    frame = frame.loc[frame.index < pd.Timestamp(kwargs["end"])]
                return frame

        return Asset()


def yahoo_snapshot(tmp_path, quotes, *, previous=None, **kwargs):
    manifest = refresh_dataset(["AAA"], start="2024-01-01", data_dir=tmp_path, provider=quotes,
                               previous_dir=previous, activate=False, **kwargs)
    return tmp_path / "dynamic" / manifest["version"], manifest


def test_incremental_requests_overlap_and_retains_history(tmp_path):
    initial = Quotes().values.iloc[:10]
    previous, _ = yahoo_snapshot(tmp_path, Quotes(initial))
    quotes = Quotes()
    directory, manifest = yahoo_snapshot(tmp_path, quotes, previous=previous, refresh_mode="incremental", overlap_days=3)
    assert [item["start"] for item in quotes.requests] == ["2024-01-07"]
    assert manifest["refresh_details"]["AAA"]["mode"] == "incremental"
    assert len(pd.read_csv(directory / "prices.csv")) == 12
    assert validate_dataset(directory)["observations"] == 11


def test_adjusted_overlap_revision_downloads_entire_asset_again(tmp_path):
    previous, _ = yahoo_snapshot(tmp_path, Quotes(Quotes().values.iloc[:10]))
    revised = Quotes().values.copy()
    revised.iloc[:10] *= .5
    quotes = Quotes(revised)
    directory, manifest = yahoo_snapshot(tmp_path, quotes, previous=previous, refresh_mode="incremental", overlap_days=3)
    assert [item["start"] for item in quotes.requests] == ["2024-01-07", "2024-01-01"]
    assert manifest["refresh_details"]["AAA"]["revision_detected"]
    assert pd.read_csv(directory / "prices.csv")["AAA"].iloc[0] == 50.
    validate_dataset(directory)


def test_retracted_corporate_action_is_replaced_by_full_refresh(tmp_path):
    previous, _ = yahoo_snapshot(tmp_path, Quotes(Quotes().values.iloc[:10], dividends={"2024-01-08": 1.}))
    quotes = Quotes()
    directory, manifest = yahoo_snapshot(tmp_path, quotes, previous=previous, refresh_mode="incremental", overlap_days=3)
    assert manifest["refresh_details"]["AAA"]["corporate_actions_revised"]
    assert pd.read_csv(directory / "corporate_actions.csv").empty
    assert len(quotes.requests) == 2


def test_history_extension_performs_full_refresh(tmp_path):
    previous, _ = yahoo_snapshot(tmp_path, Quotes(Quotes().values.iloc[4:10]))
    old_manifest = json.loads((previous / "manifest.json").read_text())
    old_manifest["start"] = "2024-01-05"
    (previous / "manifest.json").write_text(json.dumps(old_manifest))
    directory, manifest = yahoo_snapshot(tmp_path, Quotes(), previous=previous, refresh_mode="incremental", overlap_days=3)
    assert manifest["refresh_details"]["AAA"]["mode"] == "full_history_extension"
    assert len(pd.read_csv(directory / "prices.csv")) == 12


class Context:
    def __init__(self, service, tmp_path, version=None):
        self.store, self.database = service.store, service.engine
        self.dataset_version, self.temp_dir = version, tmp_path / "job"
        self.actor, self.job_id = "worker-user", "test-job"
        self.events = []
        self.cancelled = False

    def check_cancelled(self):
        if self.cancelled:
            raise RuntimeError("Job cancelled")

    def progress(self, value, stage, message=None):
        assert 0 <= value <= 100
        self.events.append((value, stage, message))


def test_refresh_uses_frozen_universe_and_parent_with_injected_database(service, tmp_path, monkeypatch):
    first = publish(service, tmp_path, activate=True)
    publish(service, tmp_path, "v2", activate=True, multiplier=2.)
    universes = UniverseService(database=service.engine)
    universe = universes.create("editable", [{"ticker": "AAA", "block": "CUSTOM"}])
    frozen = universe.copy()
    universes.update(universe["id"], assets=[{"ticker": "BBB", "block": "OTHER"}])
    quotes = Quotes()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=quotes))
    context = Context(service, tmp_path, first["version"])
    record = service.refresh({"universe_id": universe["id"], "universe_snapshot": frozen,
                              "start": "2024-01-01", "revision_overlap_days": 3}, context)
    assert record["manifest"]["parent_version"] == "v1"
    assert record["manifest"]["assets"][0]["block"] == "CUSTOM"
    assert record["manifest"]["universe_snapshot"]["assets"][0]["ticker"] == "AAA"
    assert quotes.requests[0]["start"] == "2024-01-07"
    assert context.events[-1][0] == 100
    assert record["actor"] == "worker-user"
    assert service.resolve()["version"] == "v2"


def test_refresh_universe_id_resolves_on_injected_database(service, tmp_path, monkeypatch):
    universe = UniverseService(database=service.engine).create("sql-only", [{"ticker": "AAA", "block": "CUSTOM"}])
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=Quotes()))
    record = service.refresh({"universe_id": universe["id"], "start": "2024-01-01"})
    assert record["manifest"]["assets"][0]["block"] == "CUSTOM"


def test_refresh_rejects_conflicting_job_parent_before_downloading(service, tmp_path):
    first = publish(service, tmp_path, activate=True)
    second = publish(service, tmp_path, "v2")
    with pytest.raises(ValueError, match="different parent"):
        service.refresh({"tickers": ["AAA"], "dataset_version": second["version"]}, Context(service, tmp_path, first["version"]))
    assert len(service.list_versions()) == 2


def test_periodic_full_refresh_revisits_old_prices(service, tmp_path, monkeypatch):
    first = publish(service, tmp_path, activate=True, updated_at="2024-01-12T00:00:00+00:00")
    quotes = Quotes()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=quotes))
    record = service.refresh({"tickers": ["AAA"], "start": "2024-01-01"}, Context(service, tmp_path, first["version"]))
    assert quotes.requests[0]["start"] == "2024-01-01"
    assert record["manifest"]["refresh_mode"] == "full"
    assert record["manifest"]["last_full_refresh_at"] == record["manifest"]["updated_at"]


def test_cancellation_during_download_leaves_active_and_registry_unchanged(service, tmp_path, monkeypatch):
    first = publish(service, tmp_path, activate=True)
    context = Context(service, tmp_path, first["version"])
    quotes = Quotes(on_request=lambda: setattr(context, "cancelled", True))
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=quotes))
    with pytest.raises(RuntimeError, match="Job cancelled"):
        service.refresh({"tickers": ["AAA"], "start": "2024-01-01", "activate": True}, context)
    assert service.resolve()["id"] == first["id"]
    assert len(service.list_versions()) == 1


def test_refresh_recovers_committed_publication_without_reactivating_it(service, tmp_path, monkeypatch):
    parent = publish(service, tmp_path, activate=True)
    quotes = Quotes()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=quotes))
    context = Context(service, tmp_path, parent["version"])
    config = {"tickers": ["AAA"], "start": "2024-01-01", "activate": True}
    published = service.refresh(config, context)
    assert published["version"] == f"refresh-{context.job_id}"
    assert published["manifest"]["job_id"] == context.job_id
    newer = publish(service, tmp_path, "newer", activate=True)
    download_count = len(quotes.requests)
    recovered = service.refresh(config, context)
    assert recovered["id"] == published["id"]
    assert len(quotes.requests) == download_count
    assert service.resolve()["id"] == newer["id"]
    assert len(service.list_versions()) == 3
    assert sum(stage == "committing" for _, stage, _ in context.events) == 1


def test_editable_universes_preserve_original_preset_and_gold(service):
    universes = UniverseService(database=service.engine)
    preset = universes.original_preset()
    assert {asset["block"] for asset in preset["assets"]} == set(DEFAULT_CONFIG.blocks_order)
    assert next(asset for asset in preset["assets"] if asset["ticker"] == "GLD")["block"] == "GOLD"
    custom = universes.create("gold allocation", ["GLD", {"ticker": "AAA", "block": "CUSTOM", "enabled": False}])
    assert universes.tickers_and_overrides(custom["id"]) == (["GLD"], {"GLD": "GOLD"})
    changed = universes.update(custom["id"], assets=[{"ticker": "GLD", "block": "MY_GOLD"}])
    assert changed["assets"][0]["block"] == "MY_GOLD"
    assert universes.original_preset()["id"] == preset["id"]
    assert DEFAULT_CONFIG.targets["GOLD"] == .1
    universes.delete(custom["id"])
    with pytest.raises(ValueError, match="not found"):
        universes.get(custom["id"])


def test_duplicate_universe_names_report_conflict_and_preserve_previous_values(service):
    universes = UniverseService(database=service.engine)
    first = universes.create("existing", ["GLD"])
    other = universes.create("other", ["AAA"])
    with pytest.raises(ConflictError) as created:
        universes.create("existing", ["AAA"])
    assert created.value.status_code == 409
    with pytest.raises(ConflictError):
        universes.update(other["id"], name="existing")
    assert universes.get(other["id"])["name"] == "other"
    assert universes.get(first["id"])["assets"][0]["ticker"] == "GLD"


def test_original_universe_concurrent_creation_resolves_one_preset(service, monkeypatch):
    universes = UniverseService(database=service.engine)
    create = universes.create
    barrier = Barrier(2)

    def simultaneous_create(*args, **kwargs):
        barrier.wait(timeout=5)
        return create(*args, **kwargs)

    monkeypatch.setattr(universes, "create", simultaneous_create)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: universes.original_preset(), range(2)))
    assert results[0]["id"] == results[1]["id"]
    assert len(universes.list()) == 1


def test_gld_yahoo_identity_corrects_generic_commodity_category():
    row = classify_asset("GLD", {"quoteType": "ETF", "longName": "SPDR Gold Shares", "category": "Commodities Focused", "currency": "USD"})
    assert row["block"] == "GOLD"
    assert row["classification_basis"] == "verified_SPDR_Gold_Shares"


def test_legacy_import_uses_sql_and_leaves_original_files_untouched(service, tmp_path, monkeypatch):
    from src.data import load_data
    directory, _ = source(tmp_path, "legacy-data")
    original = {name: path.read_bytes() for name in ("prices.csv", "returns.csv", "assets_metadata.csv")
                for path in [directory / name]}
    monkeypatch.setattr(load_data, "DATA_DIR", directory)
    context = Context(service, tmp_path)
    record = service.import_legacy(context)
    assert service.resolve()["id"] == record["id"]
    assert record["manifest"]["legacy_import"]
    assert service.import_legacy(context)["id"] == record["id"]
    assert len(service.list_versions()) == 1
    assert any(event[1] == "committing" for event in context.events)
    for name, content in original.items():
        assert (directory / name).read_bytes() == content
