"""Named universes and editable allocation blocks stored in SQL."""
from datetime import datetime, timezone
import re
import uuid

from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError
from src.config import DEFAULT_CONFIG
from src.platform.datasets import _database, _json
from src.platform.errors import ConflictError


def _public(record):
    result = dict(record)
    result["assets"] = result.get("assets_json", [])
    result["config"] = result.get("config_json", {})
    for name in ("created_at", "updated_at"):
        if result.get(name) is not None and hasattr(result[name], "isoformat"):
            result[name] = result[name].isoformat()
    return result


def _assets(values):
    if not isinstance(values, list) or not values:
        raise ValueError("A universe requires a nonempty asset list")
    result, seen = [], set()
    for value in values:
        if not isinstance(value, (str, dict)):
            raise ValueError("Universe assets must be tickers or objects")
        item = {"ticker": value} if isinstance(value, str) else dict(value)
        ticker = str(item.get("ticker", "")).strip().upper()
        if not re.fullmatch(r"[A-Z0-9^=._-]+", ticker) or ticker in seen:
            raise ValueError(f"Invalid or duplicate universe ticker: {ticker}")
        block = item.get("block")
        if block is not None and (not isinstance(block, str) or not block.strip()):
            raise ValueError(f"Invalid block override for {ticker}")
        # This is the original verified symbol preset. Users may override GOLD.
        if block is None and ticker == "GLD":
            block = "GOLD"
        enabled = item.get("enabled", item.get("active", True))
        if not isinstance(enabled, bool):
            raise ValueError(f"enabled must be boolean for {ticker}")
        result.append({**item, "ticker": ticker, "enabled": enabled, "block": block.strip() if block else None})
        seen.add(ticker)
    if not any(item["enabled"] for item in result):
        raise ValueError("A universe requires at least one enabled asset")
    return _json(result)


class UniverseService:
    def __init__(self, database=None):
        self.engine, self.db = _database(database)

    def list(self):
        with self.engine.connect() as connection:
            rows = connection.execute(select(self.db.universes).order_by(self.db.universes.c.name)).mappings().all()
        return [_public(row) for row in rows]

    list_universes = list

    def get(self, universe_id):
        with self.engine.connect() as connection:
            row = connection.execute(select(self.db.universes).where(self.db.universes.c.id == str(universe_id))).mappings().first()
        if row is None:
            raise ValueError(f"Universe not found: {universe_id}")
        return _public(row)

    resolve = get

    def _audit(self, connection, action, universe_id, actor):
        if hasattr(self.db, "audit"):
            connection.execute(insert(self.db.audit).values(at=datetime.now(timezone.utc), actor=actor,
                action=action, entity_id=str(universe_id), details_json={}))

    def create(self, name, assets, config=None, actor="system"):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 160:
            raise ValueError("Universe name is required")
        if config is not None and not isinstance(config, dict):
            raise ValueError("Universe config must be an object")
        now = datetime.now(timezone.utc)
        row = {"id": str(uuid.uuid4()), "name": name.strip(), "assets_json": _assets(assets),
               "config_json": _json(config or {}), "created_at": now, "updated_at": now}
        try:
            with self.engine.begin() as connection:
                connection.execute(insert(self.db.universes).values(**row))
                self._audit(connection, "universe.create", row["id"], actor)
        except IntegrityError as exc:
            raise ConflictError("A universe with this name already exists") from exc
        return self.get(row["id"])

    def update(self, universe_id, *, name=None, assets=None, config=None, actor="system"):
        current = self.get(universe_id)
        if name is not None and (not isinstance(name, str) or not name.strip() or len(name.strip()) > 160):
            raise ValueError("Universe name is required")
        if config is not None and not isinstance(config, dict):
            raise ValueError("Universe config must be an object")
        values = {"name": name.strip() if name is not None else current["name"],
                  "assets_json": _assets(assets) if assets is not None else current["assets_json"],
                  "config_json": _json(config) if config is not None else current["config_json"],
                  "updated_at": datetime.now(timezone.utc)}
        try:
            with self.engine.begin() as connection:
                connection.execute(update(self.db.universes).where(self.db.universes.c.id == str(universe_id)).values(**values))
                self._audit(connection, "universe.update", universe_id, actor)
        except IntegrityError as exc:
            raise ConflictError("A universe with this name already exists") from exc
        return self.get(universe_id)

    def delete(self, universe_id, actor="system"):
        self.get(universe_id)
        with self.engine.begin() as connection:
            connection.execute(delete(self.db.universes).where(self.db.universes.c.id == str(universe_id)))
            self._audit(connection, "universe.delete", universe_id, actor)
        return {"id": str(universe_id), "deleted": True}

    def original_preset(self):
        """Persist the historical allocation without treating it as global defaults."""
        from src.data.load_data import DATA_DIR
        import pandas as pd
        name = "Original Markowitz (AR/US/funds/gold)"
        for universe in self.list():
            if universe["config_json"].get("preset") == "original":
                return universe
        metadata = pd.read_csv(DATA_DIR / "assets_metadata.csv")
        assets = [{"ticker": row.ticker, "block": "GOLD" if row.ticker == "GLD" else row.block,
                   "enabled": True} for row in metadata.itertuples()]
        try:
            return self.create(name, assets, {"preset": "original", "targets": DEFAULT_CONFIG.targets,
                                             "max_weight": DEFAULT_CONFIG.max_weight_blocks})
        except ConflictError:
            for universe in self.list():
                if universe["config_json"].get("preset") == "original":
                    return universe
            raise

    def tickers_and_overrides(self, universe_id):
        assets = [asset for asset in self.get(universe_id)["assets_json"] if asset.get("enabled", True)]
        return ([asset["ticker"] for asset in assets],
                {asset["ticker"]: asset["block"] for asset in assets if asset.get("block")})
