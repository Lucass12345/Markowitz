from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
)

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SQLITE_URL = f"sqlite:///{(STORAGE_DIR / 'markowitz_app.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)

metadata = MetaData()

optimization_runs = Table(
    "optimization_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("run_label", String(255), nullable=True),
    Column("model_type", String(100), nullable=False),
    Column("max_weight", Float, nullable=False),
    Column("rf_annual_simple", Float, nullable=False),
    Column("n_frontier_points", Integer, nullable=False),
    Column("targets_json", JSON, nullable=False),
    Column("metrics_json", JSON, nullable=False),
    Column("comparison_json", JSON, nullable=False),
    Column("block_exposure_json", JSON, nullable=False),
    Column("weights_json", JSON, nullable=False),
    Column("frontier_json", JSON, nullable=False),
    Column("solver_message", Text, nullable=True),
)


optimization_run_provenance = Table(
    "optimization_run_provenance", metadata,
    Column("run_id", Integer, primary_key=True),
    Column("provenance_json", JSON, nullable=False),
    Column("tracking_json", JSON, nullable=False),
)


def init_db() -> None:
    metadata.create_all(engine)
