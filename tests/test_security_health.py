from fastapi.testclient import TestClient
import pytest
from api.main import create_app


def production(monkeypatch):
    monkeypatch.setenv('APP_ENV','production')
    monkeypatch.setenv('MARKOWITZ_API_KEY','a'*40)
    monkeypatch.setenv('MARKOWITZ_READ_API_KEY','r'*40)
    monkeypatch.setenv('CORS_ALLOWED_ORIGINS','https://portfolio.example.com')
    return TestClient(create_app())


def test_production_rejects_missing_secret_and_wildcard(monkeypatch):
    monkeypatch.setenv('APP_ENV','production');monkeypatch.delenv('MARKOWITZ_API_KEY',raising=False)
    with pytest.raises(RuntimeError,match='Production requires'):create_app()
    monkeypatch.setenv('MARKOWITZ_API_KEY','a'*40);monkeypatch.setenv('CORS_ALLOWED_ORIGINS','*')
    with pytest.raises(RuntimeError):create_app()


def test_auth_roles_cors_and_request_ids(monkeypatch):
    client=production(monkeypatch)
    assert client.get('/health/live').status_code==200
    assert client.get('/runs').status_code==401
    payload={'model_type':'classic','targets':{'OTHER':1}}
    assert client.post('/optimize',json=payload,headers={'X-API-Key':'r'*40}).status_code==403
    # Authorized request reaches schema validation.
    response=client.post('/optimize',json={},headers={'X-API-Key':'a'*40})
    assert response.status_code==422 and response.headers['X-Request-ID']
    headers={'Origin':'https://evil.example','Access-Control-Request-Method':'POST'}
    assert client.options('/jobs',headers=headers).status_code==400
    headers['Origin']='https://portfolio.example.com';headers['Access-Control-Request-Headers']='X-API-Key'
    assert client.options('/jobs',headers=headers).status_code==200


def test_readiness_distinct_from_liveness(monkeypatch):
    from api.routers import health
    class Offline:
        def connect(self):raise ConnectionError('sensitive database detail')
    monkeypatch.setattr(health,'engine',Offline())
    client=production(monkeypatch)
    assert client.get('/health/live').status_code==200
    response=client.get('/health/ready')
    assert response.status_code==503 and response.json()['checks']['database'] is False
    assert 'sensitive' not in response.text


def test_client_key_is_not_forwarded_to_other_services(monkeypatch):
    from app.api_client import api_headers_for
    monkeypatch.setenv('MARKOWITZ_API_KEY','private')
    assert api_headers_for('http://fastapi:8000/runs',['http://fastapi:8000'])=={'X-API-Key':'private'}
    assert api_headers_for('http://mlflow:5000',['http://fastapi:8000'])=={}
