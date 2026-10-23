# Despliegue de producción

`docker-compose.yml` sigue siendo el entorno local de desarrollo. Usa credenciales
locales y puertos administrativos; no publicarlo en Internet. Producción tiene un archivo
separado `docker-compose.production.yml`, secretos obligatorios y servicios administrativos
sin puertos publicados. Sólo Streamlit se enlaza a 127.0.0.1:8502. Colocar un reverse proxy
con TLS y autenticación/SSO delante: la API key protege la API, **no autentica usuarios de
Streamlit**. La interfaz con laboratorio y mantenimiento es para administradores de confianza.

## Preparación

```bash
cp .env.production.example .env.production
chmod 600 .env.production
python -c 'import secrets; print(secrets.token_hex(32))'
# Completar variables, passwords distintos y DATABASE_URL con password URL-encoded.
# CODE_VERSION debe ser el commit revisado; tags propios de imágenes deben identificarlo.
docker compose --env-file .env.production -f docker-compose.production.yml config --quiet
docker compose --env-file .env.production -f docker-compose.production.yml build --pull
docker compose --env-file .env.production -f docker-compose.production.yml up -d
```

Nunca usar valores REPLACE. El secreto de sólo lectura es opcional, distinto del administrador;
GET está permitido para lectura y POST exige administrador. Se verifica X-API-Key con comparación
constante. CORS permite sólo orígenes configurados, sin credentials wildcard. No guardar claves
en el navegador; Streamlit las agrega desde el servidor y no las reenvía a MLflow/MinIO.
La URL de API de la UI queda fija en producción. API pública para usuarios múltiples necesita
identidad, ownership y límites por usuario antes de habilitarla; esta clave es administrativa.

Los Dockerfiles instalan MLflow y dependencias durante build, usan CPU Torch y usuario 10001.
No ejecutan pip al iniciar MLflow. Imágenes base llevan versión de parche y los builds propios
se identifican por commit. Para promoción reproducible, publicar esos builds en un registry y
reemplazar los tags con sus digests SHA-256; los tags de bases por sí solos pueden ser movidos.
No se inventaron digests sin verificar en un registry. Confirmar las imágenes y sus healthchecks
en el entorno de destino antes de promoverlas.

Preparar `DATA_VOLUME_PATH` con una copia verificada del dataset o release. Serving lo monta
read-only; los refresh/train/pull se ejecutan por separado en un job/host con escritura sobre
ese mismo volumen. Ver `publication.md`. El volumen de snapshots debe poder escribirse por
UID 10001. Los volúmenes nombrados vacíos heredan la propiedad del directorio de la imagen;
si se reutilizan volúmenes anteriores, ajustar sus permisos durante mantenimiento.

MinIO usa credenciales raíz distintas de las del servicio. Antes de la primera optimización,
crear una cuenta y política restringida al bucket `optimization-runs` desde una consola/admin
accesible sólo por túnel, con permisos de crear/verificar bucket y leer/escribir objetos; provisionar
el bucket previamente permite reducir permisos. No dar credenciales raíz a FastAPI. MLflow y
PostgreSQL sólo son alcanzables en la red backend. Usar endpoints TLS y secret managers cuando
los servicios sean externos. Los defaults de MinIO se rechazan en producción.

## Salud, operación y backups

- `/health` y `/health/live`: proceso vivo. `/health/ready`: SQL y dataset disponibles, HTTP 503
  si falla alguno. No exige que tracking opcional esté sano para poder optimizar.
- Jobs persistidos: estados/duración/error técnico en SQL; ver `jobs.md`. Ejecutar un solo worker.
- Logs JSON de solicitudes con request_id, método, ruta, status y duración. No incluyen claves
  ni cuerpos. Enviar stdout/stderr a un colector; alertar por readiness, tasas 5xx, cola llena,
  duración y fallos de sincronización. Los diagnósticos de tracking conservan errores por servicio.
- Respaldar PostgreSQL con `pg_dump` y probar `pg_restore` en una base aislada. Ejemplo:
  `docker compose --env-file .env.production -f docker-compose.production.yml exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > backup.dump`.
  Guardar backups cifrados fuera del host con política de retención; no usar `down -v`.
- Respaldar los volúmenes MinIO y MLflow, `Data/releases`/puntero y `snapshots_prod` junto con SQL.
  Para MinIO preferir replicación o `mc mirror` con una cuenta de backup; para el SQLite de MLflow
  detener su servicio o usar una copia transaccional. Los snapshots y releases son inmutables;
  verificar hashes al restaurar. Versionar el puntero activo junto con el backup.
- Rotar claves en secret manager y recrear los servicios consumidores. No registrar secretos ni
  incluir `.env` en la imagen. El `.env` histórico del proyecto es sólo para desarrollo local;
  reemplazar cualquier credencial reutilizada antes de exponer servicios.

## Cloud/Kubernetes

Usar PostgreSQL administrado con PITR y almacenamiento de objetos administrado o MinIO con
versionado/replicación. Mantener API y UI en Deployments separados, imágenes por digest,
Service interno para API/MLflow, Ingress TLS con SSO para administradores, NetworkPolicies,
Secrets gestionados externamente, requests/limits, securityContext sin privilegios y PVC o
almacenamiento remoto para evidencia. Usar liveness/readiness separadas y Jobs/CronJobs para
entrenamiento/publicación con cuenta de servicio restringida. Antes de múltiples réplicas,
reemplazar el executor en memoria por cola durable con ownership de trabajos.

## Verificación realizada

Ambos archivos Compose pasaron `config --quiet`; pruebas de autenticación, roles, CORS y
readiness, lint y compilación ejecutados. **No se construyeron ni arrancaron contenedores**:
el entorno negó acceso a `/var/run/docker.sock`. Tampoco se probó restauración real ni
conectividad productiva de PostgreSQL/MLflow/MinIO. Son verificaciones necesarias en destino.
