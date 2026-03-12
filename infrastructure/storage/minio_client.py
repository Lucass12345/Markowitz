from __future__ import annotations

import io
import os

from minio import Minio


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"


def get_minio_client() -> Minio:
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def ensure_bucket_exists(bucket_name: str) -> None:
    client = get_minio_client()
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def upload_bytes(bucket_name: str, object_name: str, content: bytes, content_type: str = "application/octet-stream") -> str:
    client = get_minio_client()
    ensure_bucket_exists(bucket_name)

    data_stream = io.BytesIO(content)

    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=data_stream,
        length=len(content),
        content_type=content_type,
    )

    return f"s3://{bucket_name}/{object_name}"


def upload_text(bucket_name: str, object_name: str, text: str, content_type: str = "text/plain; charset=utf-8") -> str:
    return upload_bytes(
        bucket_name=bucket_name,
        object_name=object_name,
        content=text.encode("utf-8"),
        content_type=content_type,
    )
