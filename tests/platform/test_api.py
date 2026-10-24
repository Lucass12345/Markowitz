"""HTTP contracts against real domain services and isolated SQL/object adapters.

These are API integration tests, not validation of a deployed Redis/MinIO stack.
The separately runnable platform harness exercises those external services.
"""
from datetime import datetime, timezone
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, func

from api.main import create_app
from src.platform import database as db
from src.platform.artifacts import MemoryArtifactStore
from src.platform.datasets import DatasetService
from src.platform.jobs import JobService


@pytest.fixture
def platform(tmp_path, monkeypatch):
    monkeypatch.setenv('APP_ENV', 'development')
    monkeypatch.delenv('MARKOWITZ_API_KEY', raising=False)
    monkeypatch.delenv('MARKOWITZ_READ_API_KEY', raising=False)
    engine = create_engine(f'sqlite:///{tmp_path}/api.sqlite')
    db.initialize(engine)
    storage = MemoryArtifactStore()
    app = create_app()
    app.state.platform_database = engine
    app.state.platform_store = storage
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, engine=engine, store=storage, app=app, root=tmp_path)
    engine.dispose()


def publish(platform, version='v1', activate=True):
    directory = platform.root / version
    directory.mkdir()
    dates = pd.bdate_range('2024-01-01', periods=12)
    prices = pd.DataFrame({'AAA': np.arange(100., 112.), 'BBB': np.arange(200., 212.)}, index=dates)
    prices.to_csv(directory / 'prices.csv', index_label='date')
    np.log(prices / prices.shift()).dropna(how='all').to_csv(directory / 'returns.csv', index_label='date')
    pd.DataFrame({'ticker': ['AAA', 'BBB'], 'block': ['OTHER', 'OTHER'],
                  'base_currency': ['USD', 'USD'], 'currency': ['USD', 'USD']}).to_csv(directory / 'assets_metadata.csv', index=False)
    manifest = {'version': version, 'base_currency': 'USD', 'return_type': 'log',
                'updated_at': datetime.now(timezone.utc).isoformat()}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return DatasetService(database=platform.engine, store=platform.store).publish(directory, manifest, activate=activate)


def test_universe_crud_preserves_asset_metadata_and_gold_override(platform):
    client = platform.client
    response = client.post('/api/v1/universes', json={'name': 'Research', 'assets': [
        {'ticker': 'gld', 'sector': 'Precious metals'}, {'ticker': 'AAA', 'block': 'CUSTOM', 'enabled': False}]})
    assert response.status_code == 201, response.text
    universe = response.json()
    assert universe['assets_json'][0]['block'] == 'GOLD'
    assert universe['assets_json'][0]['sector'] == 'Precious metals'
    identity = universe['id']
    assert client.patch(f'/api/v1/universes/{identity}', json={'name': 'Revised'}).status_code == 200
    assert client.get(f'/api/v1/universes/{identity}').json()['name'] == 'Revised'
    assert len(client.get('/api/v1/universes').json()['items']) == 1
    assert client.delete(f'/api/v1/universes/{identity}').json()['deleted'] is True
    assert client.get('/api/v1/universes').json()['items'] == []


def test_duplicate_universe_and_invalid_assets_have_clear_http_errors(platform):
    payload = {'name': 'Unique', 'assets': [{'ticker': 'AAA'}]}
    assert platform.client.post('/api/v1/universes', json=payload).status_code == 201
    duplicate = platform.client.post('/api/v1/universes', json=payload)
    assert duplicate.status_code == 409 and 'already exists' in duplicate.json()['detail']
    bad = platform.client.post('/api/v1/universes', json={**payload, 'name': 'Invalid', 'assets': [{'ticker': 'AAA'}, {'ticker': 'AAA'}]})
    assert bad.status_code == 422 and 'duplicate' in bad.json()['detail']


def test_import_is_queued_idempotently_without_inline_dataset_work(platform, monkeypatch):
    monkeypatch.setattr(DatasetService, 'import_legacy', lambda *a, **k: pytest.fail('Dataset import ran in HTTP process'))
    first = platform.client.post('/api/v1/datasets/import-legacy', headers={'Idempotency-Key': 'import-once'})
    second = platform.client.post('/api/v1/datasets/import-legacy', headers={'Idempotency-Key': 'import-once'})
    assert first.status_code == second.status_code == 202
    assert second.json()['id'] == first.json()['id']
    assert second.json()['reused'] is True
    assert JobService(platform.engine).get(first.json()['id'])['status'] == 'pending'
    assert platform.client.get('/api/v1/datasets').json()['items'] == []
    with platform.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(db.outbox)).scalar() == 1


def test_training_pins_dataset_at_request_time_and_never_trains_inline(platform, monkeypatch):
    from src.platform.models import ModelService
    monkeypatch.setattr(ModelService, 'train', lambda *a, **k: pytest.fail('Training ran in HTTP process'))
    first = publish(platform)
    response = platform.client.post('/api/v1/experiments', json={'family': 'lstm', 'tickers': ['AAA'], 'parameters': {'epochs': 2}})
    assert response.status_code == 202, response.text
    identity = response.json()['id']
    second = publish(platform, 'v2')
    assert second['version'] != first['version']
    job = platform.client.get(f'/api/v1/jobs/{identity}').json()
    assert job['payload_json']['dataset_version'] == 'v1'
    assert job['status'] == 'pending' and job['payload_json']['parameters']['epochs'] == 2
    assert platform.client.get('/api/v1/experiments').json()['items'] == []


def test_optimizer_pins_universe_and_rejects_infeasible_static_capacity(platform):
    publish(platform)
    universe = platform.client.post('/api/v1/universes', json={'name': 'Both', 'assets': [{'ticker': 'AAA'}, {'ticker': 'BBB'}]}).json()
    configuration = {'model_type': 'classic', 'targets': {'OTHER': 1.}, 'max_weight': .6, 'universe_id': universe['id']}
    valid = platform.client.post('/api/v1/optimizations/validate', json=configuration)
    assert valid.status_code == 200, valid.text
    infeasible = platform.client.post('/api/v1/optimizations/validate', json={**configuration, 'max_weight': .4})
    assert infeasible.status_code == 422 and 'capacity' in infeasible.json()['detail']
    queued = platform.client.post('/api/v1/optimizations', json=configuration)
    assert queued.status_code == 202
    platform.client.patch(f"/api/v1/universes/{universe['id']}", json={'assets': [{'ticker': 'AAA'}]})
    job = JobService(platform.engine).get(queued.json()['id'])
    assert [asset['ticker'] for asset in job['payload_json']['universe_snapshot']['assets_json']] == ['AAA', 'BBB']


def test_job_logs_cancellation_and_missing_resource_contract(platform):
    response = platform.client.post('/api/v1/datasets/import-legacy')
    identity = response.json()['id']
    logs = platform.client.get(f'/api/v1/jobs/{identity}/logs').json()['items']
    assert len(logs) == 1 and logs[0]['stage'] == 'queued'
    canceled = platform.client.post(f'/api/v1/jobs/{identity}/cancel')
    assert canceled.status_code == 200 and canceled.json()['status'] == 'cancelled'
    assert platform.client.get(f'/api/v1/jobs/{identity}').json()['cancel_requested'] is True
    assert platform.client.get('/api/v1/jobs/not-found').status_code == 404
    assert platform.client.get('/api/v1/models/not-found').status_code == 404
    assert platform.client.get('/api/v1/artifacts/not-found').status_code == 404


def test_canonical_dataset_read_and_artifact_download_verify_content(platform):
    record = publish(platform)
    detail = platform.client.get('/api/v1/datasets/v1')
    assert detail.status_code == 200 and detail.json()['version'] == 'v1'
    assert {a['ticker'] for a in detail.json()['assets']} == {'AAA', 'BBB'}
    data = platform.client.get('/api/v1/datasets/v1/data', params={'kind': 'prices', 'limit': 2})
    assert data.status_code == 200 and len(data.json()['items']) == 2
    artifact_url = f"/api/v1/artifacts/{record['id']}"
    downloaded = platform.client.get(artifact_url, params={'path': 'prices.csv'})
    assert downloaded.status_code == 200 and b'AAA,BBB' in downloaded.content
    assert platform.client.get(artifact_url, params={'path': '../prices.csv'}).status_code == 422
    assert platform.client.get(artifact_url, params={'path': 'missing.csv'}).status_code == 404
    key = next(key for key in platform.store.objects if key.endswith('/files/prices.csv'))
    platform.store.objects[key] = b'changed'
    corrupted = platform.client.get(artifact_url, params={'path': 'prices.csv'})
    assert corrupted.status_code == 422 and 'integrity' in corrupted.json()['detail']


@pytest.mark.parametrize('route,payload', [
    ('/experiments', {'family': 'lstm', 'parameters': {'epochs': 1001}}),
    ('/experiments', {'family': 'lstm', 'parameters': {'forecast_horizon': 1}}),
    ('/optimizations', {'model_type': 'classic', 'targets': {'OTHER': .9}}),
    ('/jobs', {'kind': 'model.infer', 'payload': {}}),
    ('/jobs', {'kind': 'dataset.import', 'payload': {'path': '/etc/passwd'}}),
])
def test_invalid_requests_return_422_without_creating_jobs(platform, route, payload):
    response = platform.client.post('/api/v1' + route, json=payload)
    assert response.status_code == 422, response.text
    assert platform.client.get('/api/v1/jobs').json()['items'] == []


def test_read_only_service_key_cannot_modify_platform_resources(platform, monkeypatch):
    monkeypatch.setenv('MARKOWITZ_API_KEY', 'a' * 40)
    monkeypatch.setenv('MARKOWITZ_READ_API_KEY', 'r' * 40)
    app = create_app()
    app.state.platform_database = platform.engine
    app.state.platform_store = platform.store
    with TestClient(app) as client:
        assert client.get('/api/v1/jobs').status_code == 401
        reader = {'X-API-Key': 'r' * 40}
        assert client.get('/api/v1/jobs', headers=reader).status_code == 200
        assert client.post('/api/v1/datasets/import-legacy', headers=reader).status_code == 403
        assert client.patch('/api/v1/universes/anything', json={}, headers=reader).status_code == 403
        response = client.post('/api/v1/datasets/import-legacy', headers={'X-API-Key': 'a' * 40})
        assert response.status_code == 202 and response.headers['X-Request-ID']
        assert JobService(platform.engine).get(response.json()['id'])['actor'] == 'admin'


def test_readiness_supports_initial_bootstrap_without_active_dataset(platform, monkeypatch):
    import redis
    monkeypatch.setattr(redis.Redis, 'from_url', lambda *a, **k: SimpleNamespace(ping=lambda: True))
    platform.store.client = SimpleNamespace(bucket_exists=lambda bucket: False)
    response = platform.client.get('/health/ready')
    assert response.status_code == 200, response.text
    assert response.json()['checks'] == {'database': True, 'broker': True, 'artifacts': True}
    assert platform.client.get('/api/v1/datasets').json()['items'] == []
