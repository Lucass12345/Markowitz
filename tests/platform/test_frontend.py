"""Boundary and interaction checks for the API-only research terminal."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.api_client import APIError, PlatformAPI

ROOT = Path(__file__).resolve().parents[2]


def test_ui_has_no_business_persistence_or_training_imports():
    paths = [ROOT / 'app/Optimizador_Markowitz.py', *sorted((ROOT / 'app/pages').glob('*.py')),
             *sorted((ROOT / 'app/ui').glob('*.py'))]
    forbidden = ('src', 'sqlalchemy', 'minio', 'mlflow', 'torch', 'tensorflow', 'sklearn',
                 'app.db', 'app.repository', 'app.tracking', 'app.lstm_lab_storage', 'app.mlp_lab_storage', 'app.garch_lab_storage')
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            for module in imports:
                assert not any(module == name or module.startswith(name + '.') for name in forbidden), (path, module)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {'open', 'eval', 'exec', '__import__'}, path


def _reply(value):
    return SimpleNamespace(ok=True, status_code=200, json=lambda: value, headers={})


def _fake_api(monkeypatch):
    calls = []
    def get(url=None, **kwargs):
        path = url.split('/api/v1', 1)[1]
        if path == '/system':
            return _reply({'active_dataset': {'version': 'v1'}, 'services': {}, 'workers': []})
        if path == '/datasets':
            return _reply({'items': [{'version': 'v1', 'as_of': '2025-01-31'}, {'version': 'v2'}]})
        if path.startswith('/datasets/'):
            return _reply({'default_targets': {'OTHER': 1.}, 'blocks': {'OTHER': 20}})
        if path.startswith('/jobs/'):
            return _reply({'status': 'running', 'progress': 25, 'stage': 'worker training'})
        return _reply({'items': []})
    def post(url=None, **kwargs):
        calls.append((url, kwargs['json']))
        return _reply({'id': 'dispatch-job', 'status': 'queued'})
    monkeypatch.setattr(requests, 'get', get)
    monkeypatch.setattr(requests, 'post', post)
    return calls


@pytest.mark.parametrize('family', ['lstm', 'mlp', 'garch'])
def test_laboratory_dispatches_pinned_family_without_running_training(monkeypatch, family):
    calls = _fake_api(monkeypatch)
    app = AppTest.from_file(str(ROOT / 'app/Optimizador_Markowitz.py')).run(timeout=15)
    app.sidebar.selectbox[0].select('Model Laboratory').run(timeout=15)
    next(item for item in app.selectbox if item.label == 'Familia de modelo').select(family).run(timeout=15)
    next(item for item in app.selectbox if item.label == 'Dataset versionado').select('v2').run(timeout=15)
    next(item for item in app.button if item.label == 'Lanzar entrenamiento').click().run(timeout=15)
    assert not app.exception
    assert len(calls) == 1
    assert calls[0][0].endswith('/api/v1/experiments')
    assert calls[0][1]['family'] == family
    assert calls[0][1]['dataset_version'] == 'v2'
    assert calls[0][1]['parameters']['forecast_horizon'] == 5


def test_backtesting_dispatches_shared_constraints_and_costs(monkeypatch):
    calls = _fake_api(monkeypatch)
    app = AppTest.from_file(str(ROOT / 'app/Optimizador_Markowitz.py')).run(timeout=15)
    app.sidebar.selectbox[0].select('Backtesting').run(timeout=15)
    next(item for item in app.number_input if item.label == 'Comisión (bps por nocional)').set_value(8.).run(timeout=15)
    next(item for item in app.button if item.label == 'Ejecutar backtest').click().run(timeout=15)
    assert not app.exception
    payload = calls[0][1]
    assert payload['dataset_version'] == payload['optimization']['dataset_version'] == 'v1'
    assert payload['optimization']['targets'] == {'OTHER': 1.}
    assert payload['commission_bps'] == 8.
    assert payload['strategies'] == ['classic', 'equal_weight', 'buy_and_hold']


def test_transport_failure_is_explicit_and_never_runs_a_fallback(monkeypatch):
    def offline(**kwargs):
        raise requests.ConnectionError('offline')
    monkeypatch.setattr(requests, 'post', offline)
    with pytest.raises(APIError, match='no se ejecutó desde la interfaz'):
        PlatformAPI('http://backend').post('/experiments', {'family': 'lstm'})
    with pytest.raises(ValueError, match='relative'):
        PlatformAPI('http://backend').get('//external.example/secrets')


def test_transport_does_not_forward_credentials_through_redirects(monkeypatch):
    monkeypatch.setenv('MARKOWITZ_API_KEY', 'server-only-key')
    sent = []
    def get(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(ok=True, status_code=302)
    monkeypatch.setattr(requests, 'get', get)
    with pytest.raises(APIError, match='redirección'):
        PlatformAPI('http://backend').get('/system')
    assert len(sent) == 1
    assert sent[0]['allow_redirects'] is False
    assert sent[0]['headers'] == {'X-API-Key': 'server-only-key'}


def test_transient_poll_failure_preserves_job_and_retries(monkeypatch):
    _fake_api(monkeypatch)
    original_get = requests.get
    polls = []
    def get(url=None, **kwargs):
        if '/jobs/' in url:
            polls.append(url)
            if len(polls) == 1:
                raise requests.ConnectionError('temporary transport failure')
            return _reply({'status': 'completed', 'result_json': {'result': {}, 'run_id': 99}})
        return original_get(url=url, **kwargs)
    monkeypatch.setattr(requests, 'get', get)
    app = AppTest.from_file(str(ROOT / 'app/Optimizador_Markowitz.py')).run(timeout=15)
    app.sidebar.selectbox[0].select('Portfolio Optimizer').run(timeout=15)
    next(item for item in app.button if item.label == 'Correr optimización').click().run(timeout=15)
    assert not app.exception
    assert any('No se pudo conectar' in item.value for item in app.warning)
    assert 'dispatch-job' in app.session_state['tracked_jobs']
    app.run(timeout=15)
    assert not app.exception
    assert len(polls) == 2
    assert app.session_state['last_run_id'] == 99


def test_selected_universe_exposes_custom_allocation_blocks(monkeypatch):
    calls = _fake_api(monkeypatch)
    original_get = requests.get
    def get(url=None, **kwargs):
        if url.endswith('/universes'):
            return _reply({'items': [{'id': 'custom', 'name': 'Custom blocks', 'assets': [
                {'ticker': 'AAA', 'block': 'CUSTOM', 'enabled': True},
                {'ticker': 'BBB', 'block': 'DISABLED', 'enabled': False}]}]})
        return original_get(url=url, **kwargs)
    monkeypatch.setattr(requests, 'get', get)
    app = AppTest.from_file(str(ROOT / 'app/Optimizador_Markowitz.py')).run(timeout=15)
    app.sidebar.selectbox[0].select('Portfolio Optimizer').run(timeout=15)
    next(item for item in app.selectbox if item.label == 'Universo guardado').select('custom').run(timeout=15)
    assert any(item.label == 'CUSTOM' for item in app.number_input)
    assert not any(item.label == 'DISABLED' for item in app.number_input)
    next(item for item in app.button if item.label == 'Correr optimización').click().run(timeout=15)
    assert not app.exception
    assert calls[0][1]['targets'] == {'CUSTOM': 1.}
    assert calls[0][1]['universe_id'] == 'custom'
