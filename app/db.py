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
    pool_timeout=5,
    connect_args={"connect_timeout": 5} if DATABASE_URL.startswith("postgresql") else {},
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


optimization_jobs = Table(
    "optimization_jobs", metadata,
    Column("id", String(36), primary_key=True),
    Column("status", String(20), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    Column("duration_seconds", Float),
    Column("cache_key", String(64)),
    Column("payload_json", JSON, nullable=False),
    Column("result_json", JSON),
    Column("error", Text),
    Column("technical_error", Text),
    Column("error_status", Integer),
)


def init_db() -> None:
    metadata.create_all(engine)
