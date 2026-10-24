# Arquitectura de la plataforma V2 y migración

Este documento describe la implementación bajo `src/platform/`, `worker/` y `/api/v1`. Los módulos financieros y entrenadores anteriores se reutilizan como algoritmos de dominio. Los documentos `jobs.md`, `publication.md` y `production.md` conservan procedimientos de la arquitectura anterior; para el despliegue vigente seguir [platform-deployment.md](platform-deployment.md).

## Responsabilidades y persistencia

| Componente | Responsabilidad |
| --- | --- |
| Streamlit | Presentación, configuración, polling y descarga mediante FastAPI |
| FastAPI | Contratos, permisos, lectura de registros/artefactos, fijación de dependencias y creación de jobs |
| Scheduler | Programación Yahoo, relay del outbox SQL y recuperación de leases vencidos |
| Celery + Redis | Entrega de IDs a procesos CPU/GPU independientes |
| Servicios `src/platform/` | Dataset, modelos, registro, optimización y backtesting compartidos |
| PostgreSQL | Fuente de verdad de jobs, eventos, universos, versiones, experimentos, modelos, referencias y auditoría |
| MinIO | Fuente canónica de datasets y bundles inmutables de modelos/resultados |
| MLflow | Parámetros, métricas, tags, relación con job/dataset y manifiesto de resultados |

El modelo de producción usa PostgreSQL. SQLite sirve para desarrollo sin servicios y pruebas unitarias; no representa una validación de bloqueo/concurrencia PostgreSQL. No se guardan datasets ni pesos grandes como blobs SQL. Los artefactos de MLflow apuntan a la publicación MinIO; MinIO conserva pesos, preprocesamiento, datos y reportes completos. El registry de modelos de la aplicación es SQL y no exige usar el MLflow Model Registry.

Los nuevos temporales de API y worker son descartables. Las rutas históricas de archivos se conservan para lectura del legado y mantenimiento, pero no son el almacenamiento activo de la UI V2. Una excepción es `MemoryArtifactStore`, utilizada explícitamente en pruebas; no se selecciona como fallback cuando MinIO falla.

## Jobs durables

Se eligió Celery con Redis por su soporte de workers separados, colas CPU/GPU, límites temporales y mantenimiento conocido. SQL conserva estado y outbox; Redis transporta IDs, no la única copia del trabajo.

1. FastAPI valida el request, resuelve el dataset a una versión concreta y captura el universo/modelos seleccionados.
2. Una transacción SQL crea el UUID del job, su configuración y la fila de outbox.
3. El scheduler publica el ID en Redis. Si cae entre envío y confirmación SQL, puede volver a entregarlo.
4. Un worker reclama el job mediante actualización condicional del estado/attempt. Sólo un propietario puede ejecutarlo.
5. El servicio trabaja en un directorio temporal exclusivo del UUID y registra checkpoints.
6. Se verifican los artefactos requeridos antes de confirmar la publicación y marcar el trabajo completado.

Los estados son `pending`, `running`, `retrying`, `completed`, `failed` y `cancelled`. PostgreSQL conserva intentos, progreso, stage, timestamps, duración, errores y uso de recursos. La consulta pública no devuelve el traceback técnico. El progreso refleja checkpoints reales; no implica precisión por batch de entrenamiento.

`Idempotency-Key` se asocia al actor y a la configuración fijada. Reutilizar una clave explícita con otro request genera HTTP 409. Sin clave se deduplica por configuración y dependencias; un request completado puede reutilizar su resultado. Reenviar una configuración que falló o se canceló puede crear otro UUID para recuperarse, conservando el historial previo.

Errores transitorios de red/servicios y de SQL admiten reintentos con backoff; los errores permanentes de validación fallan. Al perderse un worker, el scheduler recupera su lease después de vencer el timeout. Reiniciar FastAPI no invalida jobs pendientes o ejecutándose. Redis y SQL pueden provocar redelivery; la reclamación SQL evita ejecución concurrente del mismo job. Este diseño no promete entrega exactamente una vez: los efectos publicados deben ser idempotentes.

La cancelación es cooperativa en checkpoints seguros. Una vez en `committing`, se rechaza con HTTP 409 porque interrumpir la confirmación podría dejar una referencia ambigua. No se revoca bruscamente un proceso que publica objetos. Celery aplica límites suave/duro; una terminación abrupta se recupera mediante lease. Las colas CPU y GPU separan concurrencia; GPU usa un worker de concurrencia 1. Configurar límites duros de recursos en el host/orquestador según el despliegue.

## Dataset canónico

`DatasetService` resuelve registros SQL y materializa bundles verificados de MinIO. Cada versión tiene UUID, nombre de versión, hash, fechas, manifiesto, moneda, fuente y calidad. La referencia activa se mantiene en `platform_active_refs`; cambiarla no sobrescribe el dataset anterior.

El refresh sigue descarga → validación → transformación → versionado → MinIO → SQL → activación solicitada. La validación comprueba USD/log, fechas y tickers únicos, precios observados positivos, cobertura, metadata y consistencia de retornos/precios. La actualización incremental descarga un solapamiento y revisa ajustes; revisiones fuera de su alcance se recuperan mediante refresh completo periódico o explícito. Ningún proveedor externo garantiza ausencia de revisiones históricas.

Los jobs de refresh tienen identidad de versión estable para poder recuperar una publicación SQL ya confirmada después de un reinicio, sin activar de nuevo sobre una versión más reciente. Los objetos históricos usados por experimentos no se borran ni se sobrescriben.

Los universos son configuraciones reutilizables distintas de los datasets. Guardan activos habilitados, bloques y metadata; cada job captura su contenido. Editar un universo no altera corridas ya enviadas. La clasificación Yahoo conserva moneda, tipo, sector, país y origen; los valores desconocidos no se adivinan. El domicilio de un ETF no equivale a la exposición geográfica de su cartera.

## Publicación de modelos y resultados

PostgreSQL, MinIO y MLflow no comparten una transacción ACID. La implementación utiliza checkpoints de publicación e identidades estables:

1. Entrenar/evaluar en un workspace temporal; validar presencia de pesos/especificación y pronósticos requeridos.
2. Subir archivos a un prefijo propio de la corrida y del hash de contenido.
3. Subir el manifiesto al final, como marcador de bundle completo, y materializarlo verificando hashes/tamaños.
4. Registrar la publicación y sincronizar parámetros/métricas/referencia en MLflow.
5. Confirmar modelos y experimento en SQL, con estado de sincronización explícito.
6. Marcar el job completado cuando el servicio finaliza.

Un reintento puede reutilizar un bundle completo y continuar tracking/registro. MinIO es obligatorio. Para entrenamiento, inferencia, optimización y backtest, MLflow también es obligatorio por defecto; `PLATFORM_MLFLOW_REQUIRED=false` lo vuelve opcional de forma explícita. Refresh/import/activación no necesitan crear un experimento MLflow para confirmar su referencia de datos.

La evaluación de cada ticker produce evidencia y un criterio de aceptación. Los aceptados se registran `validated`, los demás `rejected`. `candidate` se representa por la publicación pendiente del experimento; no hay una etapa que active automáticamente un entrenamiento. La activación verifica artefactos y aceptación y cambia una referencia SQL; la versión previa queda `archived` y puede restaurarse. Archivar un modelo rechazado no elimina su rechazo original ni lo habilita para activación.

Cada modelo registra familia, ticker, horizonte, features, preprocesamiento, métricas de validación, corte de entrenamiento, fecha de inferencia, dataset y URI/hash. Inferencia sobre otra versión conserva el dataset de entrenamiento y genera otra publicación de pronósticos. El optimizador valida esa evidencia; no cambia etiquetas de un pronóstico antiguo para aparentar compatibilidad.

La trazabilidad incluye commit/huella de código, configuración efectiva, versiones y referencias SQL/MLflow/MinIO. Para reproducir una ejecución también se debe conservar la imagen/dependencias y sus semillas: una referencia Git por sí sola no garantiza determinismo de CUDA ni igualdad numérica entre hardware.

## Contratos de API

Las mutaciones de larga duración devuelven HTTP 202 y un job con `id`. El resultado final se consulta en el job y su experimento. Los campos completos están en `api/schemas/platform.py` y OpenAPI de desarrollo.

| Dominio | Rutas principales bajo `/api/v1` |
| --- | --- |
| Datasets | `GET /datasets`, `GET /datasets/{version}`, `GET /datasets/{version}/data`, `POST /datasets/import-legacy`, `POST /datasets/refresh`, `POST /datasets/{version}/activate`, `POST /datasets/rollback` |
| Universos | `GET/POST /universes`, `GET/PATCH/DELETE /universes/{id}`, `GET /universes/presets/original` |
| Experimentos | `POST /experiments`, `GET /experiments`, `GET /experiments/{id}` |
| Modelos | `GET /models`, `GET /models/{id}`, `POST /models/{id}/activate`, `/archive`, `/infer`, `POST /models/{family}/{ticker}/rollback` |
| Optimización | `POST /optimizations/validate`, `POST /optimizations`, `GET /optimizations`, `GET /optimizations/{id}` |
| Backtesting | `POST /backtests`, `GET /backtests`, `GET /backtests/{id}` |
| Jobs | `POST/GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/logs`, `POST /jobs/{id}/cancel` |
| Artefactos | `GET /artifacts/{id}?path=<ruta_relativa_del_bundle>` |
| Sistema | `GET /system`; fuera del prefijo, `/health/live` y `/health/ready` |

Ejemplo de entrenamiento después de disponer de un dataset publicado:

```bash
curl --fail-with-body -X POST http://localhost:8000/api/v1/experiments \
  -H 'Content-Type: application/json' \
  -d '{"family":"lstm","dataset_version":"VERSION_PUBLICADA","tickers":["AAPL"],"parameters":{"seq_len":40,"forecast_horizon":5,"epochs":50,"device":"cpu"}}'
```

En un entorno autenticado agregar `X-API-Key` usando la credencial de servicio configurada. La clave de lectura permite GET; las demás mutaciones requieren clave administrativa. La API key no es identidad individual de un usuario de Streamlit; producción necesita control de acceso en su proxy/SSO. Los logs y diagnósticos no deben publicarse como servicios administrativos abiertos.

`/health/live` sólo informa que el proceso responde. `/health/ready` comprueba SQL, broker y almacenamiento; no exige un dataset activo porque la importación inicial necesita una API disponible. `/api/v1/system` informa disponibilidad de dataset/modelos, servicios y heartbeats. Un servicio alcanzable no demuestra sincronización de una corrida concreta.

## Migración desde las mejoras anteriores

La rama `improvements/financial-validation-mlops` fue integrada en `main` por el merge `df33bb8` (PR #6); la base histórica documentada fue `f597317`. Esta evolución conserva sus algoritmos, pruebas y resultados. Las nuevas tablas `platform_*` son aditivas; no se alteran columnas ni métricas de las corridas anteriores. El arranque serializa la creación de tablas PostgreSQL entre API/scheduler/workers.

1. Respaldar SQL y artefactos históricos antes de actualizar contenedores.
2. Iniciar los servicios nuevos con PostgreSQL, Redis, MinIO y MLflow configurados.
3. Ejecutar el import explícito del dataset legado y esperar la confirmación del job.
4. Verificar versión/moneda/cobertura y crear o cargar un universo.
5. Entrenar modelos por API sobre esa versión; los modelos locales anteriores no se vuelven activos automáticamente sin evidencia verificable.
6. Validar una optimización y comparar sus convenciones/controles antes de operar con otra versión.
7. Conservar scripts y rutas históricas para investigación; usar exclusivamente el registro V2 para las páginas nuevas.

El import puede corregir metadata descriptiva de GLD en la copia migrada, sin modificar los CSV originales ni retornos guardados. La configuración original de objetivos queda disponible como preset.

Las rutas antiguas `/optimize` y `/jobs` siguen resolviendo trabajos durables. Un cliente que esperaba el resultado síncrono de `POST /optimize` debe aceptar HTTP 202 y consultar el job. `/runs` y las tablas históricas se conservan; las corridas nuevas se consultan por `/api/v1/optimizations`/`experiments`. No se inventa linaje de corridas antiguas que nunca lo guardaron.

Los workflows y scripts anteriores que escriben `Data/active_release.json` no activan referencias V2. Publicar un artifact en GitHub Actions no modifica un contenedor local. El nuevo scheduler actualiza datos; entrenamiento/validación/activación de modelos son solicitudes explícitas de API/UI.

## Backtesting y límites de interpretación

`BacktestService` compara las siete estrategias del request sobre un mismo calendario OOS. `ConstrainedModelStrategy` usa el optimizador compartido y entrena por fold con datos hasta la decisión, incluyendo validación/test internos previos al OOS. Los parámetros del request se registran, no se eligen mirando los resultados OOS.

El reporte conserva min_train solicitado y efectivo. GARCH necesita su mínimo de entrenamiento más los tramos de validación/test: con valores predeterminados son 292 observaciones, por lo que la primera decisión de una comparación que lo incluya no puede usar sólo 252. Una ventana demasiado corta falla explícitamente. El universo se fija antes de simular y los modelos temporales usan historia completa disponible dentro del fold salvo un inicio explícito.

Los pesos se aplican en la rueda siguiente, sin solapamiento de segmentos realizados. Los costos se calculan sobre nocional negociado; turnover se reporta con su convención. Buy-and-hold entra una vez y permite deriva; equal-weight rebalancea. Las restricciones se imponen al decidir pesos y no impiden deriva de exposiciones entre rebalanceos. La metadata y el universo actual no eliminan survivorship bias. Fechas sin cotización de una posición mantenida requieren una política de valuación distinta; el motor actual rechaza esa simulación.

## Verificación pendiente del entorno

Las pruebas Python incluyen entrenamiento real reducido y pruebas de regresión financiera. La integración distribuida se prepara en `scripts/platform_integration.py`, `tests/integration/` y `docker-compose.integration.yml`, con servicios reales y reinicio de API/worker. En el entorno de implementación el socket Docker denegó acceso: un archivo CI o una prueba con dobles no constituye validación end-to-end. Consultar el resultado concreto del workflow y su reporte de ambas fases antes de promocionar imágenes.

El perfil CUDA requiere hardware NVIDIA y no se considera verificado por pruebas CPU. Tampoco se declara mejor rendimiento financiero por tener un pipeline integrado: hacen falta comparaciones OOS suficientemente extensas y reproducibles para sostener esa conclusión.
