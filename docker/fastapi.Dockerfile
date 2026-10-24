# syntax=docker/dockerfile:1
FROM python:3.11.12-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements.txt .
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    pip install pip==25.0.1 \
    && pip install torch==2.6.0 --index-url "$TORCH_INDEX_URL" \
    && pip install -r requirements.txt

FROM python:3.11.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/tmp/appuser \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    TMPDIR=/tmp
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home appuser \
    && mkdir -p /apprepo/storage \
    && chown -R appuser:appuser /apprepo
COPY --from=build /opt/venv /opt/venv
WORKDIR /apprepo
COPY --chown=10001:10001 . .
ARG CODE_VERSION=unknown
ENV CODE_VERSION=$CODE_VERSION
USER 10001:10001

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
