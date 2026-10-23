FROM python:3.11.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /apprepo

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install pip==25.0.1 && pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

COPY . .
ARG CODE_VERSION=unknown
ENV CODE_VERSION=$CODE_VERSION
RUN mkdir -p /apprepo/storage && useradd --uid 10001 --create-home appuser && chown -R appuser:appuser /apprepo
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
