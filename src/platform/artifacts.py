"""Immutable object bundles. Local directories are verified disposable materializations."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
import uuid
from minio import Minio
from minio.error import S3Error
import urllib3
from src.optimization.provenance import json_bytes
from src.platform.errors import ServiceUnavailableError, TransientError


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise ValueError(f'Unsafe artifact path: {name}')
    return name


class ArtifactStore:
    def __init__(self, client=None, bucket=None):
        self.bucket = bucket or os.getenv('PLATFORM_ARTIFACT_BUCKET', 'markowitz-artifacts')
        self.client = client or self._client()

    @staticmethod
    def _client():
        endpoint = os.getenv('MINIO_ENDPOINT', '127.0.0.1:9000')
        access, secret = os.getenv('MINIO_ACCESS_KEY'), os.getenv('MINIO_SECRET_KEY')
        if not access or not secret:
            raise ServiceUnavailableError('MinIO service credentials are required for canonical artifacts')
        if os.getenv('APP_ENV') == 'production' and (access == 'minioadmin' or secret == 'minioadmin'):
            raise ServiceUnavailableError('Production requires distinct MinIO service credentials')
        return Minio(endpoint, access_key=access, secret_key=secret,
                     secure=os.getenv('MINIO_SECURE', 'false').lower() == 'true',
                     http_client=urllib3.PoolManager(timeout=urllib3.Timeout(connect=3, read=30), retries=False))

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            try:
                self.client.make_bucket(self.bucket)
            except S3Error as exc:
                if exc.code not in {'BucketAlreadyOwnedByYou', 'BucketAlreadyExists'}:
                    raise

    def _parse(self, uri):
        parsed = urlparse(uri)
        if parsed.scheme != 's3' or parsed.netloc != self.bucket:
            raise ValueError('Artifact URI must belong to the configured canonical bucket')
        key = parsed.path.lstrip('/')
        safe_relative(key)
        return key

    def _read(self, key):
        response = self.client.get_object(self.bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def read_bytes(self, uri):
        try:
            return self._read(self._parse(uri))
        except (urllib3.exceptions.HTTPError, OSError) as exc:
            raise TransientError('Artifact store is temporarily unavailable') from exc

    def read_json(self, uri):
        return json.loads(self.read_bytes(uri))

    def _put_immutable(self, key, content, content_type='application/octet-stream'):
        digest = hashlib.sha256(content).hexdigest()
        try:
            existing = self.client.stat_object(self.bucket, key)
        except S3Error as exc:
            if exc.code not in {'NoSuchKey', 'NoSuchObject', 'NoSuchBucket'}:
                raise
        else:
            if existing.size != len(content) or hashlib.sha256(self._read(key)).hexdigest() != digest:
                raise ValueError('Refusing to overwrite a different immutable artifact')
            return
        self.client.put_object(self.bucket, key, io.BytesIO(content), len(content), content_type=content_type,
                               metadata={'sha256': digest})

    def upload_directory(self, directory, prefix):
        directory = Path(directory)
        safe_relative(prefix)
        files = {}
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symbolic links are not allowed in model/dataset bundles')
            if path.is_file():
                name = safe_relative(path.relative_to(directory).as_posix())
                content = path.read_bytes()
                files[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'size': len(content)}
        if not files:
            raise ValueError('Cannot publish an empty artifact bundle')
        digest = hashlib.sha256(json_bytes(files)).hexdigest()
        base = f'{prefix}/{digest}'
        manifest = {'schema_version': 1, 'sha256': digest, 'files': files}
        try:
            self._ensure_bucket()
            for name in files:
                self._put_immutable(f'{base}/files/{name}', (directory / name).read_bytes())
            # The manifest is the commit marker and is always written last.
            self._put_immutable(f'{base}/manifest.json', json_bytes(manifest), 'application/json')
        except (urllib3.exceptions.HTTPError, OSError) as exc:
            raise TransientError('Artifact publication failed; no active reference changed') from exc
        return {**manifest, 'uri': f's3://{self.bucket}/{base}/manifest.json'}

    def materialize(self, uri, destination):
        key = self._parse(uri)
        if not key.endswith('/manifest.json'):
            raise ValueError('A bundle manifest URI is required')
        manifest = self.read_json(uri)
        files = manifest.get('files', {})
        if not files or hashlib.sha256(json_bytes(files)).hexdigest() != manifest.get('sha256'):
            raise ValueError('Artifact manifest integrity failed')
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        # Reject pre-existing symlinks: extraction must never escape its cache.
        if destination.is_symlink() or any(p.is_symlink() for p in destination.rglob('*')):
            raise ValueError('Unsafe materialization destination')
        base = key.removesuffix('/manifest.json')
        for name, info in files.items():
            safe_relative(name)
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Artifact path escapes its materialization')
            content = self.read_bytes(f's3://{self.bucket}/{base}/files/{name}')
            if len(content) != info['size'] or hashlib.sha256(content).hexdigest() != info['sha256']:
                raise ValueError(f'Artifact checksum mismatch: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}')
            temporary.write_bytes(content)
            temporary.replace(target)
        return destination


def get_store():
    return ArtifactStore()


class MemoryArtifactStore(ArtifactStore):
    """Explicit test adapter; never selected by serving configuration."""
    def __init__(self):
        self.bucket = 'test-artifacts'
        self.objects = {}
    def _ensure_bucket(self):
        pass
    def _put_immutable(self, key, content, content_type='application/octet-stream'):
        if key in self.objects and self.objects[key] != content:
            raise ValueError('Refusing immutable overwrite')
        self.objects[key] = content
    def _read(self, key):
        return self.objects[key]
