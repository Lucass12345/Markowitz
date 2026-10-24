"""Exercise submission, job results and failure recovery using the real Streamlit app."""
from pathlib import Path
from types import SimpleNamespace

import requests
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def reply(payload, status=200):
    return SimpleNamespace(ok=status < 400, status_code=status, json=lambda: payload,
                           text=str(payload), headers={})


def backend(monkeypatch, job):
    calls = []
    def get(url=None, **kwargs):
        assert '/api/v1/' in url
        path = url.split('/api/v1', 1)[1]
        if path == '/system':
            return reply({'active_dataset': {'version': 'fixture'}, 'models': [], 'services': {'mlflow': {'status': 'not_configured'}}})
        if path == '/datasets':
            return reply({'items': [{'id': 'dataset-1', 'version': 'fixture', 'as_of': '2025-01-31'}]})
        if path == '/datasets/fixture':
            return reply({'version': 'fixture', 'blocks': {'OTHER': 10}, 'default_targets': {'OTHER': 1.}, 'as_of': '2025-01-31'})
        if path.startswith('/jobs/'):
            return reply(job)
        return reply({'items': []})
    def post(url=None, **kwargs):
        calls.append((url, kwargs.get('json')))
        return reply({'id': 'job-123', 'status': 'queued'}, 202)
    monkeypatch.setattr(requests, 'get', get)
    monkeypatch.setattr(requests, 'post', post)
    return calls


def optimizer():
    app = AppTest.from_file(str(ROOT / 'app/Optimizador_Markowitz.py')).run(timeout=15)
    app.sidebar.selectbox[0].select('Portfolio Optimizer').run(timeout=15)
    assert not app.exception
    return app


def test_optimizer_submits_and_displays_completed_job(monkeypatch):
    stats = {'return_annual': .1, 'vol_annual': .2, 'sharpe_annual': .5}
    result = {'frontier': [{'return_annual': .1, 'vol_annual': .2}],
        **{name: {'stats': stats} for name in ['minvar', 'maxsharpe']},
        'blocks': {'stats': stats, 'weights_df': [{'ticker': 'A', 'weight': 1.}],
                   'exposure_by_block': [{'block': 'OTHER', 'target': 1., 'achieved': 1., 'diff': 0.}]}}
    calls = backend(monkeypatch, {'status': 'completed', 'progress': 100,
                                 'result_json': {'result': result, 'run_id': 12}})
    app = optimizer()
    next(item for item in app.button if item.label == 'Correr optimización').click().run(timeout=15)
    assert not app.exception
    assert calls[0][0].endswith('/api/v1/optimizations')
    assert calls[0][1]['targets'] == {'OTHER': 1.}
    assert calls[0][1]['dataset_version'] == 'fixture'
    assert calls[0][1]['as_of'] == '2025-01-31'
    assert app.session_state['last_run_id'] == 12
    assert any('sin registro confirmado' in value.value for value in app.info)


def test_diagnostic_page_reports_unconfirmed_sync(monkeypatch):
    backend(monkeypatch, {})
    app = AppTest.from_file(str(ROOT / 'app/pages/06_Diagnostico.py')).run(timeout=15)
    assert not app.exception
    assert len(app.json) == 3
    assert 'no_recorded_sync' in app.json[2].value
    assert not any('sincronizado' in item.value.lower() for item in app.success)


def test_failed_job_reenables_submission_and_preserves_error(monkeypatch):
    calls = backend(monkeypatch, {'status': 'failed', 'error': 'Infeasible OTHER capacity'})
    app = optimizer()
    next(item for item in app.button if item.label == 'Correr optimización').click().run(timeout=15)
    assert not app.exception
    assert not next(item for item in app.button if item.label == 'Correr optimización').disabled
    assert any('Infeasible OTHER' in message.value for message in app.error)
    next(item for item in app.button if item.label == 'Correr optimización').click().run(timeout=15)
    assert not app.exception
    assert len(calls) == 2
