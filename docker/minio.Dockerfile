# syntax=docker/dockerfile:1
# The former upstream image is no longer public. Preserve the same release by
# building its official source, pinned by commit and archive checksum.
FROM golang:1.24.2-bookworm AS build
ENV GOTOOLCHAIN=local GOMAXPROCS=2 CGO_ENABLED=0
WORKDIR /source
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca CURL_CA_BUNDLE=/run/secrets/proxy_ca; fi; \
    curl -fsSL https://codeload.github.com/minio/minio/tar.gz/0d7408fc9969caf07de6a8c3a84f9fbb10a6739e -o /tmp/minio.tar.gz \
    && echo '7eb30a913fea30f18069abf194e1e78e4983b558cc526911ae1c11396a9859a5  /tmp/minio.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/minio.tar.gz --strip-components=1 \
    && rm /tmp/minio.tar.gz \
    && go build -p 2 -trimpath -ldflags='-s -w -X github.com/minio/minio/cmd.Version=2025-04-22T22:12:26Z -X github.com/minio/minio/cmd.ReleaseTag=RELEASE.2025-04-22T22-12-26Z -X github.com/minio/minio/cmd.CommitID=0d7408fc9969caf07de6a8c3a84f9fbb10a6739e -X github.com/minio/minio/cmd.ShortCommitID=0d7408fc9969 -X github.com/minio/minio/cmd.CopyrightYear=2025' -o /out/minio .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home minio \
    && mkdir /data && chown minio:minio /data
COPY --from=build /out/minio /usr/local/bin/minio
COPY --from=build /source/LICENSE /usr/share/licenses/minio/LICENSE
LABEL org.opencontainers.image.source="https://github.com/minio/minio" \
      org.opencontainers.image.revision="0d7408fc9969caf07de6a8c3a84f9fbb10a6739e" \
      org.opencontainers.image.licenses="AGPL-3.0-only"
USER 10001:10001
EXPOSE 9000 9001
VOLUME /data
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
