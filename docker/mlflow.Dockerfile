# syntax=docker/dockerfile:1
FROM python:3.11.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp/mlflow
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    pip install --no-cache-dir mlflow==2.21.0 SQLAlchemy==2.0.39 psycopg2-binary==2.9.10
RUN useradd --uid 10001 --create-home mlflow && mkdir /mlflow && chown mlflow:mlflow /mlflow
USER 10001:10001
WORKDIR /mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts"]
