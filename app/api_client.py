"""Server-side API transport. The browser never receives service credentials."""
from __future__ import annotations

import os
from urllib.parse import quote, urlsplit
import requests


def api_headers():
    value = os.getenv('MARKOWITZ_API_KEY') or os.getenv('MARKOWITZ_READ_API_KEY')
    return {'X-API-Key': value} if value else {}


def api_headers_for(url, trusted_bases):
    destination = urlsplit(url)
    if any((destination.scheme, destination.netloc) == (urlsplit(base).scheme, urlsplit(base).netloc) for base in trusted_bases):
        return api_headers()
    return {}


class APIError(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class PlatformAPI:
    """Only FastAPI owns business operations, persistence and artifact retrieval."""
    def __init__(self, base_url=None, timeout=20):
        self.base_url = (base_url or os.getenv('API_BASE_URL', 'http://127.0.0.1:8000')).rstrip('/')
        self.timeout = timeout

    def request(self, method, path, **kwargs):
        if not path.startswith('/') or path.startswith('//'):
            raise ValueError('API paths must be relative to the configured backend')
        try:
            response = getattr(requests, method.lower())(
                url=f'{self.base_url}/api/v1{path}', headers=api_headers(), timeout=self.timeout,
                allow_redirects=False, **kwargs)
        except requests.RequestException as exc:
            raise APIError('No se pudo conectar con FastAPI. El trabajo no se ejecutó desde la interfaz.') from exc
        if 300 <= response.status_code < 400:
            raise APIError('FastAPI devolvió una redirección inesperada.', response.status_code)
        if not response.ok:
            try:
                detail = response.json().get('detail', response.text)
            except (ValueError, AttributeError):
                detail = getattr(response, 'text', 'Respuesta inválida')
            raise APIError(f'API {response.status_code}: {detail}', response.status_code)
        return response

    def get(self, path, **params):
        return self._json(self.request('get', path, params=params or None))

    def post(self, path, payload=None):
        return self._json(self.request('post', path, json=payload or {}))

    def patch(self, path, payload):
        return self._json(self.request('patch', path, json=payload))

    def delete(self, path):
        return self._json(self.request('delete', path))

    @staticmethod
    def _json(response):
        try:
            return response.json()
        except ValueError as exc:
            raise APIError('FastAPI devolvió una respuesta JSON inválida.') from exc

    def download(self, artifact_id, path=None):
        response = self.request('get', f'/artifacts/{quote(str(artifact_id), safe="")}',
                                params={'path': path} if path else None)
        return response.content, response.headers.get('Content-Type', 'application/octet-stream')
