from infrastructure.storage.minio_client import upload_bytes

path = upload_bytes(
    bucket_name="optimization-runs",
    object_name="test/hello.txt",
    content=b"hola papu",
    content_type="text/plain",
)

print(path)