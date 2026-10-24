"""SQL-registered, immutable USD datasets transported through ArtifactStore.

Local paths are temporary materializations, never active-version pointers. The
legacy filesystem resolver is consulted only by the explicit migration operation.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib
import json
import tempfile
import uuid

import numpy as np
import pandas as pd
from sqlalchemy import and_, insert, or_, select, text, update

DATA_FILES = ("manifest.json", "returns.csv", "assets_metadata.csv", "prices.csv",
              "prices_native.csv", "fx_rates.csv", "corporate_actions.csv")


def _now():
    return datetime.now(timezone.utc)


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False, default=str))


def _content_hash(directory):
    directory = Path(directory)
    inventory = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                 for name in DATA_FILES if (directory / name).is_file()}
    return hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _database(value=None):
    module = importlib.import_module("src.platform.database")
    if value is None:
        module.initialize()
        return module.get_engine(), module
    if hasattr(value, "begin") and hasattr(value, "dialect"):
        module.initialize(value)
        return value, module
    engine = value.get_engine() if hasattr(value, "get_engine") else value.engine
    value.initialize()
    return engine, value


def _row(value):
    result = dict(value)
    manifest = result.get("manifest_json") or {}
    if isinstance(manifest, str):
        manifest = json.loads(manifest)
    result["manifest_json"] = manifest
    result["manifest"] = manifest
    for field in ("generated_at", "as_of"):
        if result.get(field) is not None and hasattr(result[field], "isoformat"):
            result[field] = result[field].isoformat()
    result["assets"] = manifest.get("assets", [])
    result["blocks"] = manifest.get("blocks", [])
    result["quality"] = manifest.get("quality", {})
    return result


def _progress(context, fraction, stage, message=None):
    if context is None:
        return
    check = getattr(context, "check_cancelled", None)
    if check:
        check()
    progress = getattr(context, "progress", None)
    if progress:
        progress(fraction * 100., stage, message)


def validate_dataset(directory, manifest=None):
    directory = Path(directory)
    manifest = manifest or json.loads((directory / "manifest.json").read_text())
    if manifest.get("base_currency") != "USD" or manifest.get("return_type") != "log":
        raise ValueError("Datasets require verified USD daily log returns")
    returns = pd.read_csv(directory / "returns.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv(directory / "prices.csv", index_col=0, parse_dates=True)
    metadata = pd.read_csv(directory / "assets_metadata.csv")
    for frame, name in ((returns, "returns"), (prices, "prices")):
        if frame.empty or frame.columns.has_duplicates or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Dataset {name} requires nonempty unique sorted dates and assets")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.isna().any():
            raise ValueError(f"Dataset {name} has invalid dates")
        values = frame.to_numpy(dtype=float)
        if np.isinf(values).any() or (name == "prices" and (values[np.isfinite(values)] <= 0).any()):
            raise ValueError(f"Dataset {name} has invalid observed values")
    if set(prices.columns) != set(returns.columns):
        raise ValueError("Dataset price and return assets differ")
    if "ticker" not in metadata or "block" not in metadata:
        raise ValueError("Dataset metadata requires ticker and block")
    if metadata.ticker.isna().any() or metadata.ticker.duplicated().any() or set(metadata.ticker) != set(returns.columns):
        raise ValueError("Dataset metadata must cover each return ticker exactly once")
    if metadata.block.isna().any() or metadata.block.astype(str).str.strip().eq("").any():
        raise ValueError("Dataset assets require explicit blocks")
    if "base_currency" in metadata and not metadata.base_currency.eq("USD").all():
        raise ValueError("Asset metadata contains non-USD base currencies")
    counts = returns.notna().sum()
    if (counts < 1).any():
        raise ValueError("Every dataset asset requires observed returns")
    tolerance = 2e-6 if manifest.get("legacy_import") else 1e-9
    for ticker in returns:
        source = prices[ticker].dropna().astype(float)
        expected = np.log(source / source.shift(1)).dropna()
        observed = returns[ticker].dropna()
        if not observed.index.equals(expected.index) or not np.allclose(observed, expected, rtol=1e-6, atol=tolerance):
            raise ValueError(f"USD return/price identity failed for {ticker}")
    as_of = returns.index[returns.notna().any(axis=1)].max()
    blocks = sorted(metadata.block.unique().astype(str).tolist())
    return {"asset_count": len(returns.columns), "observations": len(returns),
            "asset_observations": {ticker: int(counts[ticker]) for ticker in returns},
            "as_of": as_of.date().isoformat(), "blocks": blocks,
            "base_currency": "USD", "return_type": "daily_log", "price_identity_valid": True}


@dataclass(frozen=True)
class DatasetSnapshot:
    id: str
    version: str
    path: Path
    manifest: dict

    def load_returns(self):
        frame = pd.read_csv(self.path / "returns.csv", index_col=0, parse_dates=True).sort_index()
        frame.attrs.update(dataset_id=self.id, dataset_version=self.version, base_currency="USD",
                           dataset_dir=str(self.path), as_of=self.manifest.get("as_of"))
        return frame

    def load_metadata(self):
        return pd.read_csv(self.path / "assets_metadata.csv")

    def load_prices(self):
        return pd.read_csv(self.path / "prices.csv", index_col=0, parse_dates=True).sort_index()


class DatasetService:
    def __init__(self, store=None, database=None):
        self.engine, self.db = _database(database)
        if store is None:
            store = importlib.import_module("src.platform.artifacts").get_store()
        self.store = store

    @contextmanager
    def _transaction(self):
        # Serialize pointer changes even when the initial reference does not yet
        # exist. SQLite's write reservation also works across worker processes.
        with self.engine.connect() as connection:
            if self.engine.dialect.name == "sqlite":
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
                if self.engine.dialect.name == "postgresql":
                    connection.execute(text("SELECT pg_advisory_xact_lock(711202602)"))
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _audit(self, connection, action, entity_id, actor, details=None):
        if hasattr(self.db, "audit"):
            connection.execute(insert(self.db.audit).values(at=_now(), actor=actor,
                action=action, entity_id=entity_id, details_json=_json(details or {})))

    def list_versions(self):
        with self.engine.connect() as connection:
            records = connection.execute(select(self.db.datasets).order_by(self.db.datasets.c.generated_at.desc())).mappings().all()
            active = connection.execute(select(self.db.active_refs.c.version_id).where(and_(
                self.db.active_refs.c.kind == "dataset", self.db.active_refs.c.key == "active"))).scalar_one_or_none()
        return [{**_row(record), "active": record["id"] == active} for record in records]

    def resolve(self, version=None):
        with self.engine.connect() as connection:
            if version is None:
                version = connection.execute(select(self.db.active_refs.c.version_id).where(and_(
                    self.db.active_refs.c.kind == "dataset", self.db.active_refs.c.key == "active"))).scalar_one_or_none()
                if version is None:
                    raise ValueError("No active SQL-registered dataset; import or refresh a dataset first")
            record = connection.execute(select(self.db.datasets).where(or_(
                self.db.datasets.c.id == str(version), self.db.datasets.c.version == str(version)))).mappings().first()
        if record is None or record["status"] != "published":
            raise ValueError(f"Dataset version not published: {version}")
        return _row(record)

    def materialize(self, version, destination):
        record = self.resolve(version)
        return self._materialize_record(record, destination)

    def _materialize_record(self, record, destination):
        path = Path(self.store.materialize(record["artifact_uri"], Path(destination)))
        if _content_hash(path) != record["content_hash"]:
            raise ValueError("Dataset content hash differs from SQL registry")
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest != record["manifest_json"]:
            raise ValueError("Dataset manifest differs from SQL registry")
        validate_dataset(path, manifest)
        return DatasetSnapshot(record["id"], record["version"], path, manifest)

    def publish(self, local_dataset_dir, manifest, actor="system", activate=False):
        import shutil
        source = Path(local_dataset_dir)
        with tempfile.TemporaryDirectory(prefix="markowitz-dataset-publish-") as temporary:
            staging = Path(temporary) / "dataset"
            staging.mkdir()
            for name in DATA_FILES:
                if (source / name).is_symlink():
                    raise ValueError("Dataset artifacts cannot contain symbolic links")
                if (source / name).is_file():
                    shutil.copyfile(source / name, staging / name)
            return self._publish_staged(staging, manifest, actor=actor, activate=activate)

    def _publish_staged(self, local_dataset_dir, manifest, actor="system", activate=False):
        directory = Path(local_dataset_dir)
        version = manifest.get("version")
        if not isinstance(version, str) or not version or Path(version).name != version or version in {".", ".."}:
            raise ValueError("Invalid immutable dataset version")
        disk_manifest = json.loads((directory / "manifest.json").read_text())
        if disk_manifest != manifest:
            raise ValueError("Supplied manifest differs from dataset artifact")
        quality = validate_dataset(directory, manifest)
        # Enrich only this unregistered staging directory; published versions are never changed.
        assets = pd.read_csv(directory / "assets_metadata.csv").replace({np.nan: None}).to_dict(orient="records")
        manifest = _json({**manifest, "as_of": quality["as_of"], "quality": quality,
                          "assets": assets, "blocks": quality["blocks"]})
        (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False))
        digest = _content_hash(directory)
        with self.engine.connect() as connection:
            existing = connection.execute(select(self.db.datasets).where(self.db.datasets.c.version == version)).mappings().first()
        if existing:
            if existing["content_hash"] != digest:
                raise ValueError(f"Immutable dataset version already exists with other content: {version}")
            uri = existing["artifact_uri"]
        else:
            bundle = self.store.upload_directory(directory, f"datasets/{version}/{digest}")
            uri = bundle["uri"]
        # Neither a retry nor a new upload can change references until the
        # canonical transported content has passed the same integrity checks.
        with tempfile.TemporaryDirectory(prefix="markowitz-dataset-verify-") as temporary:
            restored = Path(self.store.materialize(uri, Path(temporary) / "dataset"))
            if _content_hash(restored) != digest:
                raise ValueError("Uploaded dataset checksum verification failed")
            validate_dataset(restored, manifest)
        with self._transaction() as connection:
            existing = connection.execute(select(self.db.datasets).where(self.db.datasets.c.version == version)).mappings().first()
            if existing:
                if existing["content_hash"] != digest or existing["status"] != "published":
                    raise ValueError(f"Immutable dataset version already exists with other content: {version}")
                record = dict(existing)
            else:
                record = {"id": str(uuid.uuid4()), "version": version, "content_hash": digest,
                          "manifest_json": manifest, "artifact_uri": uri, "status": "published",
                          "generated_at": _now(), "as_of": quality["as_of"], "actor": actor}
                connection.execute(insert(self.db.datasets).values(**record))
                self._audit(connection, "dataset.publish", record["id"], actor,
                            {"version": version, "content_hash": digest})
            if activate:
                self._activate_ref(connection, record["id"], actor)
        return _row(record)

    def _set_ref(self, connection, key, version_id, actor):
        condition = and_(self.db.active_refs.c.kind == "dataset", self.db.active_refs.c.key == key)
        value = {"version_id": version_id, "updated_at": _now(), "actor": actor}
        if connection.execute(select(self.db.active_refs.c.version_id).where(condition)).first():
            connection.execute(update(self.db.active_refs).where(condition).values(**value))
        else:
            connection.execute(insert(self.db.active_refs).values(kind="dataset", key=key, **value))

    def activate(self, version, actor="system"):
        target = self.resolve(version)
        # Verify remote integrity immediately before changing the authoritative reference.
        with tempfile.TemporaryDirectory(prefix="markowitz-dataset-activate-") as temporary:
            self.materialize(target["id"], Path(temporary) / "dataset")
        with self._transaction() as connection:
            self._activate_ref(connection, target["id"], actor)
        return self.resolve(target["id"])

    def _activate_ref(self, connection, target_id, actor, action="dataset.activate"):
        condition = and_(self.db.active_refs.c.kind == "dataset", self.db.active_refs.c.key == "active")
        current = connection.execute(select(self.db.active_refs.c.version_id).where(condition).with_for_update()).scalar_one_or_none()
        if current and current != target_id:
            self._set_ref(connection, "previous", current, actor)
        self._set_ref(connection, "active", target_id, actor)
        self._audit(connection, action, target_id, actor, {"previous_id": current})

    def rollback(self, actor="system"):
        with self._transaction() as connection:
            previous = connection.execute(select(self.db.active_refs.c.version_id).where(and_(
                self.db.active_refs.c.kind == "dataset", self.db.active_refs.c.key == "previous"))).scalar_one_or_none()
            if previous is None:
                raise ValueError("No previous dataset activation to roll back")
            # Selection and reference exchange share one lock. Another
            # activation cannot move 'previous' between these two operations.
            record = connection.execute(select(self.db.datasets).where(
                self.db.datasets.c.id == previous)).mappings().one()
            if record["status"] != "published":
                raise ValueError("Previous dataset is not published")
            with tempfile.TemporaryDirectory(prefix="markowitz-dataset-rollback-") as temporary:
                self._materialize_record(_row(record), Path(temporary) / "dataset")
            self._activate_ref(connection, previous, actor, "dataset.rollback")
        return self.resolve(previous)

    def refresh(self, config, context=None):
        from src.data.yahoo_dataset import refresh_dataset
        configuration = dict(config)
        tickers = configuration.get("tickers")
        overrides = dict(configuration.get("block_overrides") or {})
        universe = configuration.get("universe_snapshot")
        if universe is not None or configuration.get("universe_id"):
            from src.platform.universes import UniverseService
            if universe is None:
                universe = UniverseService(database=self.engine).get(configuration["universe_id"])
            enabled = [asset for asset in universe.get("assets_json", universe.get("assets", [])) if asset.get("enabled", asset.get("active", True))]
            tickers = tickers or [asset["ticker"] for asset in enabled]
            if set(tickers) - {asset["ticker"] for asset in enabled}:
                raise ValueError("Refresh tickers must be enabled in the frozen universe")
            overrides = {asset["ticker"]: asset["block"] for asset in enabled if asset.get("block") and asset["ticker"] in tickers} | overrides
        if not tickers:
            raise ValueError("Dataset refresh requires tickers or a configured universe")
        mode = configuration.get("refresh_mode", "full" if configuration.get("full_refresh", False) else "incremental")
        if mode not in {"full", "incremental"}:
            raise ValueError("Invalid dataset refresh mode")
        actor = configuration.get("actor", getattr(context, "actor", "system"))
        _progress(context, .05, "dataset.resolve", "Resolving dataset and universe snapshot")
        job_id = getattr(context, "job_id", None)
        job_version = None
        if job_id is not None:
            if not isinstance(job_id, str) or not job_id or len(job_id) > 120 or Path(job_id).name != job_id or "\\" in job_id:
                raise ValueError("Invalid dataset refresh job identity")
            job_version = f"refresh-{job_id}"
            try:
                recovered = self.resolve(job_version)
            except ValueError:
                pass
            else:
                if recovered["manifest"].get("job_id") != job_id:
                    raise ValueError("Dataset version belongs to another publication identity")
                # Publication and its activation already committed together.
                # Completing a recovered job must not undo a later activation.
                _progress(context, 1., "dataset.complete", "Recovered the job's previously published dataset")
                return recovered
        # A job's captured parent is authoritative. Explicit aliases may refer
        # to its ID or version, but cannot silently replace that snapshot.
        versions = [value for value in (getattr(context, "dataset_version", None),
                    configuration.get("dataset_version"), configuration.get("parent_version")) if value]
        parent = None
        for version in versions:
            selected = self.resolve(version)
            if parent and selected["id"] != parent["id"]:
                raise ValueError("Refresh job and request reference different parent datasets")
            parent = selected
        if parent is None and mode == "incremental":
            try:
                parent = self.resolve()
            except ValueError as exc:
                if "No active SQL-registered dataset" not in str(exc):
                    raise
        work = getattr(context, "temp_dir", None)
        if work is not None:
            Path(work).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="markowitz-dataset-refresh-", dir=work) as temporary:
            root = Path(temporary)
            previous = None
            if parent:
                previous = self.materialize(parent["id"], root / "previous")
                if mode == "incremental":
                    full_at = parent["manifest_json"].get("last_full_refresh_at") or parent["manifest_json"].get("updated_at")
                    interval = int(configuration.get("full_refresh_every_days", 30))
                    if interval < 1:
                        raise ValueError("Periodic full refresh interval must be positive")
                    full_date = pd.to_datetime(full_at, utc=True) if full_at else None
                    if full_date is None or (_now() - full_date.to_pydatetime()).days >= interval:
                        mode = "full"
            _progress(context, .15, "dataset.download", "Downloading adjusted prices and currency history")
            manifest = refresh_dataset(tickers, start=configuration.get("start", "2015-01-01"), end=configuration.get("end"),
                                       data_dir=root, activate=False, previous_dir=previous.path if previous else None,
                                       refresh_mode=mode, overlap_days=int(configuration.get("revision_overlap_days", 30)),
                                       block_overrides=overrides, max_fx_age_days=int(configuration.get("max_fx_age_days", 7)),
                                       check_cancelled=getattr(context, "check_cancelled", None),
                                       on_progress=lambda fraction, stage, message=None: _progress(context, .15 + .6 * fraction, stage, message))
            directory = root / "dynamic" / manifest["version"]
            if job_version is not None:
                manifest.update(version=job_version, job_id=job_id)
            manifest["parent_version"] = parent["version"] if parent else None
            if universe is not None:
                manifest["universe_snapshot"] = _json(universe)
            if mode == "full" or all(detail["mode"].startswith("full") for detail in manifest["refresh_details"].values()):
                manifest["last_full_refresh_at"] = manifest["updated_at"]
            elif previous:
                manifest["last_full_refresh_at"] = previous.manifest.get("last_full_refresh_at", previous.manifest.get("updated_at"))
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
            _progress(context, .8, "dataset.publish", "Validating USD snapshot and publishing immutable artifacts")
            _progress(context, .95, "committing", "Committing the verified dataset and its requested activation")
            record = self.publish(directory, manifest, actor=actor, activate=bool(configuration.get("activate", False)))
            _progress(context, 1., "dataset.complete", "Dataset snapshot published")
            return record

    def import_legacy(self, context=None):
        from src.data.load_data import active_data_dir, DATA_DIR
        source = active_data_dir()
        source_manifest = json.loads((source / "manifest.json").read_text()) if (source / "manifest.json").exists() else {}
        if source != DATA_DIR and (source_manifest.get("base_currency") != "USD" or source_manifest.get("return_type") != "log"):
            raise ValueError("Legacy dynamic snapshot does not verify USD log returns")
        fingerprint = hashlib.sha256(b"".join((source / name).read_bytes() for name in ("returns.csv", "prices.csv", "assets_metadata.csv"))).hexdigest()
        version = f"legacy-{fingerprint[:20]}"
        try:
            return self.resolve(version)
        except ValueError:
            pass
        _progress(context, .1, "dataset.import", "Importing verified legacy USD data")
        with tempfile.TemporaryDirectory(prefix="markowitz-import-legacy-") as temporary:
            directory = Path(temporary)
            import shutil
            for name in DATA_FILES:
                if name != "manifest.json" and (source / name).exists():
                    shutil.copyfile(source / name, directory / name)
            metadata = pd.read_csv(directory / "assets_metadata.csv")
            metadata["base_currency"] = "USD"
            # Repair only the copied descriptive label, never historical prices/returns.
            if "name" in metadata:
                gold = metadata.ticker.eq("GLD") & metadata["name"].fillna("").str.contains("SPDR", case=False) & metadata["name"].fillna("").str.contains("Gold", case=False)
                metadata.loc[gold, "block"] = "GOLD"
            metadata.to_csv(directory / "assets_metadata.csv", index=False)
            manifest = {**source_manifest, "version": version, "legacy_import": True, "source_version": source.name,
                        "base_currency": "USD", "return_type": "log", "updated_at": _now().isoformat()}
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
            _progress(context, .95, "committing", "Committing the legacy dataset and its activation")
            record = self.publish(directory, manifest, actor=getattr(context, "actor", "system"), activate=True)
        _progress(context, 1., "dataset.complete", "Legacy USD data registered and activated in SQL")
        return record
