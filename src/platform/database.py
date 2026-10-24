"""Additive platform schema. Existing experiment/history tables remain untouched."""
from datetime import datetime, timezone
from sqlalchemy import (Table, Column, String, Text, Integer, Float, Boolean, DateTime,
                        JSON, MetaData, UniqueConstraint, Index, text)
from app.db import engine, metadata as legacy_metadata

metadata = MetaData()

def utcnow():
    return datetime.now(timezone.utc)


datasets = Table('platform_datasets', metadata,
    Column('id', String(36), primary_key=True), Column('version', String(160), unique=True, nullable=False),
    Column('content_hash', String(64), unique=True, nullable=False), Column('manifest_json', JSON, nullable=False),
    Column('artifact_uri', Text, nullable=False), Column('status', String(32), nullable=False),
    Column('generated_at', DateTime(timezone=True), nullable=False), Column('as_of', String(64)), Column('actor', String(160)))
active_refs = Table('platform_active_refs', metadata,
    Column('kind', String(80), primary_key=True), Column('key', String(160), primary_key=True),
    Column('version_id', String(36), nullable=False), Column('updated_at', DateTime(timezone=True), nullable=False),
    Column('actor', String(160)))
universes = Table('platform_universes', metadata,
    Column('id', String(36), primary_key=True), Column('name', String(160), unique=True, nullable=False),
    Column('assets_json', JSON, nullable=False), Column('config_json', JSON, nullable=False),
    Column('created_at', DateTime(timezone=True), nullable=False), Column('updated_at', DateTime(timezone=True), nullable=False))
experiments = Table('platform_experiments', metadata,
    Column('id', String(36), primary_key=True), Column('job_id', String(36), unique=True, nullable=False),
    Column('kind', String(40), nullable=False), Column('dataset_version', String(160)), Column('actor', String(160)),
    Column('status', String(32), nullable=False), Column('config_json', JSON, nullable=False),
    Column('metrics_json', JSON), Column('result_json', JSON), Column('artifact_uri', Text), Column('content_hash', String(64)),
    Column('mlflow_run_id', String(160)), Column('code_json', JSON), Column('sync_json', JSON),
    Column('created_at', DateTime(timezone=True), nullable=False), Column('completed_at', DateTime(timezone=True)))
models = Table('platform_models', metadata,
    Column('id', String(36), primary_key=True), Column('experiment_id', String(36), nullable=False),
    Column('family', String(24), nullable=False), Column('ticker', String(64), nullable=False),
    Column('dataset_version', String(160), nullable=False), Column('horizon', Integer, nullable=False),
    Column('features', JSON), Column('preprocessing', JSON), Column('validation_metrics', JSON),
    Column('state', String(24), nullable=False), Column('accepted', Boolean, nullable=False, default=False), Column('artifact_uri', Text, nullable=False),
    Column('relative_path', Text), Column('prediction_file', Text), Column('content_hash', String(64), nullable=False),
    Column('training_cutoff', String(64)), Column('as_of', String(64)), Column('config_json', JSON),
    Column('inference_json', JSON), Column('created_at', DateTime(timezone=True), nullable=False),
    UniqueConstraint('experiment_id', 'family', 'ticker'))
jobs = Table('platform_jobs', metadata,
    Column('id', String(36), primary_key=True), Column('kind', String(40), nullable=False),
    Column('status', String(24), nullable=False), Column('actor', String(160), nullable=False),
    Column('payload_json', JSON, nullable=False), Column('context_json', JSON, nullable=False),
    Column('request_hash', String(64), nullable=False), Column('idempotency_key', String(200), unique=True),
    Column('progress', Float, nullable=False, default=0), Column('stage', String(80), nullable=False),
    Column('attempts', Integer, nullable=False, default=0), Column('max_attempts', Integer, nullable=False, default=3),
    Column('timeout_seconds', Integer, nullable=False), Column('cancel_requested', Boolean, nullable=False, default=False),
    Column('lease_owner', String(80)), Column('lease_until', DateTime(timezone=True)), Column('next_attempt_at', DateTime(timezone=True)),
    Column('created_at', DateTime(timezone=True), nullable=False), Column('started_at', DateTime(timezone=True)),
    Column('completed_at', DateTime(timezone=True)), Column('duration_seconds', Float),
    Column('result_json', JSON), Column('error', Text), Column('technical_error', Text), Column('error_status', Integer),
    Column('error_class', String(40)), Column('resource_json', JSON))
Index('platform_jobs_dispatch', jobs.c.status, jobs.c.next_attempt_at)
outbox = Table('platform_job_outbox', metadata,
    Column('job_id', String(36), primary_key=True), Column('queue', String(24), nullable=False),
    Column('last_dispatched_at', DateTime(timezone=True)), Column('deliveries', Integer, nullable=False, default=0))
job_events = Table('platform_job_events', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True), Column('job_id', String(36), nullable=False, index=True),
    Column('at', DateTime(timezone=True), nullable=False), Column('level', String(16), nullable=False),
    Column('stage', String(80)), Column('message', Text, nullable=False))
workers = Table('platform_workers', metadata,
    Column('id', String(160), primary_key=True), Column('kind', String(24), nullable=False),
    Column('last_seen', DateTime(timezone=True), nullable=False), Column('capabilities_json', JSON), Column('resources_json', JSON))
schedules = Table('platform_schedules', metadata,
    Column('id', String(80), primary_key=True), Column('enabled', Boolean, nullable=False),
    Column('interval_seconds', Integer, nullable=False), Column('next_run_at', DateTime(timezone=True), nullable=False),
    Column('config_json', JSON, nullable=False), Column('updated_at', DateTime(timezone=True), nullable=False))
audit = Table('platform_audit', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True), Column('at', DateTime(timezone=True), nullable=False),
    Column('actor', String(160)), Column('action', String(80), nullable=False), Column('entity_id', String(160)), Column('details_json', JSON))


def get_engine():
    return engine


def initialize(database=None):
    target=database if database is not None else engine
    with target.begin() as connection:
        if target.dialect.name=='postgresql':
            # API, scheduler and workers may bootstrap concurrently. checkfirst
            # alone does not serialize PostgreSQL catalog creation.
            connection.execute(text('SELECT pg_advisory_xact_lock(734920481)'))
        if database is None:
            legacy_metadata.create_all(connection)
        metadata.create_all(connection)
