# Mejoras financieras y operativas

Rama: `improvements/financial-validation-mlops`. Base histórica: `f597317`.
Los resultados almacenados y los parámetros de entrenamiento originales no se reescriben.

## P0 implementados

1. `7282c9d`: `src/utils_portfolio.py` centraliza Sharpe diario/anual. Los motores,
   tracking, repositorio y pantallas identifican el retorno logarítmico anual y guardan
   `metrics_version=2`. Una cartera sintética verifica `Sharpe_anual=Sharpe_diario*sqrt(252)`.
2. `a6b8be3`: esquemas, `src/markowitz.py` y optimizador de bloques validan suma,
   capacidades, índices y dimensiones antes del solver. Errores de configuración: HTTP 422.
3. `859ac70`: `src/optimization/covariance.py` estima covarianza por observaciones
   coincidentes de cada par, sin imputación. Historia y solapamiento configurables;
   exclusiones, autovalores y ajuste PSD se informan en diagnósticos. Un caso sintético
   sin una sola fecha completa conserva los tres activos y obtiene una matriz válida.
4. `src/optimization/predictions.py`, `inputs_builder.py`, loaders y servicios: el
   universo elegible se conserva. Cada señal informa aceptación o fallback histórico;
   se comprueban moneda USD, versión, fecha, antigüedad, finitud y unidades diarias.
   La política `required` rechaza cualquier señal requerida ausente o inválida.
   El filtro LSTM usa exclusivamente validación. La UI permite configurar la política
   y consultar los motivos. Los loaders fijan la corrida seleccionada durante la lectura.

### Compatibilidad

- Campos antiguos se mantienen. `return_annual` continúa siendo un retorno logarítmico;
  usar `return_simple_annual` para su transformación simple.
- Corridas antiguas conservan sus métricas; no aplicar corrección silenciosa al historial.
- Predicciones antiguas sin versión, moneda o fecha verificable requieren regeneración.
  No se admite mezclar predicciones de otro dataset. Una señal individual ausente o
  caducada usa historia por defecto; configurar `required` para exigir cobertura total.
- Los mínimos por defecto son 252 observaciones por activo y 126 por par. Universos
  cortos necesitan controles explícitos menores; las exclusiones pueden volver inviables
  los objetivos por bloque y entonces se informa HTTP 422.
- No se requieren migraciones de tablas para P0. La incorporación de campos JSON es aditiva.

### Verificación

Los tres primeros commits pasaron 103 pruebas. La fase 4 agrega casos de cobertura,
fallback, señales futuras/caducadas, unidades, versión, moneda y volatilidad diaria.
La suite completa se ejecuta con `.venv/bin/python -m pytest -q` sin credenciales externas.
Los contratos HTTP necesitan sockets locales habilitados en este entorno de ejecución.

## P1

- Bloques dinámicos desde metadata; la configuración original queda disponible.
- Frontera adicional bajo las mismas restricciones, con pesos y diagnósticos de
  convergencia por punto. La frontera clásica conserva long-only sin bloques ni tope.
- Estimadores historical (pares), ledoit_wolf (fechas conjuntas observadas) y
  gjr_garch (correlación histórica regularizada y volatilidades diarias con fallback).
  Ledoit-Wolf rechaza historia conjunta insuficiente y no imputa retornos.
- Umbrales LSTM configurables, con comparación estricta `>` y valores originales
  0,52 / 0,15. El diagnóstico conserva métricas, criterio y split de validación.
- Walk-forward y ejecución real: ver `backtesting.md` y `verification/walk-forward-smoke.json`.
- Snapshots de optimización identificados por SHA-256 conservan retornos, metadata,
  mu, covarianza, resultado y configuración efectiva. `load_snapshot` verifica integridad.
  La versión de código incluye commit y huella de cambios locales. IDs de modelos
  y fecha de corte se registran sin depender del puntero activo posterior.

### Migración aditiva de trazabilidad

`python -c 'from app.db import init_db; init_db()'` crea la nueva tabla
`optimization_run_provenance` mediante `create_all`, sin alterar ni reescribir filas
históricas. El arranque también la crea. Hacer backup antes de cualquier despliegue.
Las corridas anteriores tienen `provenance=null` porque no es posible reconstruir
retroactivamente evidencia que nunca se guardó. Respaldar tanto PostgreSQL como el
volumen `storage/optimization_snapshots`. MLflow y MinIO registran estados separados;
un fallo de MinIO conserva el identificador exitoso de MLflow. Las pruebas verifican
integraciones con dobles; no se ha probado conectividad real con esos servicios.

CI agregado para pruebas offline, lint de errores y compilación. Verificación al cerrar P1:
**139 pruebas aprobadas**, Ruff sin errores y compileall correcto. La prueba HTTP ejecuta
una optimización sintética completa y comprueba HTTP 422 ante capacidad insuficiente.
Se reemplazó un módulo de servicio obsoleto que ejecutaba nombres indefinidos al importar
por una reexportación compatible. JSON convierte NaN/infinito a null en API, SQL y snapshots.

## P2 y cierre

- Riesgo: HHI, contribuciones marginales/porcentuales, VaR/ES históricos, drawdown,
  volatilidades y correlaciones observadas, con unidades/horizontes y tipo de medición.
- Publicación: bundles aislados con checksums y controles de validación, import/pull
  autenticado de Actions, activación atómica y rollback. Ver `publication.md`.
- Jobs: cola interna limitada, estados persistidos, errores recuperables y caché clásica
  dependiente de contenido/configuración/código. Ver `jobs.md` y su límite de un worker.
- Producción separada: API key/roles, CORS explícito, liveness/readiness, logs JSON,
  imágenes con versiones, MLflow instalado en build y volúmenes. Ver `production.md`.
- Diagnóstico desde el backend: dataset, señales y sus motivos, fechas y modelos, servicios
  y última sincronización registrada. README corregido: se usa validación, no test; los
  modelos no tienen superioridad financiera declarada. AppTest ejecuta la UI principal
  y la página de diagnóstico, incluido el envío/consulta de un trabajo.

La suite ampliada pasó **155 pruebas** y lint/compilación. Compose de desarrollo y producción
validan la configuración. Docker no pudo construir/arrancar por permiso denegado al daemon;
no se verificaron servicios externos reales, un workflow remoto ni un despliegue cloud.
Las pruebas de SQL usan SQLite y los contratos remotos emplean dobles.

## Verificación final y correcciones de revisión

**185 pruebas aprobadas** en un entorno limpio: Python 3.11.12, Torch 2.6.0+cpu,
ARCH 7.2.0, NumPy 2.1.3 y los demás requisitos fijados del repositorio. Ruff, compileall,
`git diff --check` y ambos `docker compose ... config --quiet` pasaron. Las advertencias
observadas son de deprecación de dependencias, no fallos de pruebas.

La revisión agregó regresiones para metadata de releases alterada, origen PR/fork de
artefactos, reactivación idempotente, rollback desde un manifiesto activo corrupto,
alineación de etiquetas en contribuciones de riesgo y recuperación del botón de envío
tras un job fallido. Los snapshots combinados conservan IDs tanto de retornos como de GARCH.

El MLP existente también seleccionaba activos con test: el composite conserva sus pesos,
pero ahora usa validación y registra política/motivo. No se reutilizan scores históricos de
test. Sus targets exigen todas las ruedas, se purgan bordes y winsorization aprende límites
sólo con entrenamiento. Los parámetros/arquitecturas y filas históricas no se reescriben.
MLP clásico sin tramo de validación y corridas antiguas sin evidencia quedan deshabilitados
para señal IA con fallback histórico; usar un nuevo laboratorio con evidencia verificable.
Los nuevos archivos exportan fecha real de inferencia, unidades diarias e identidad.
No se requiere migrar columnas existentes: se recupera evidencia adicional de los artefactos.

`verification/real-classic-optimization.json` registra una optimización ejecutada sobre los
40 activos USD, con diez puntos restringidos verificados. El metadata activo clasifica GLD
como OTHER; los objetivos legacy con GOLD se rechazaron correctamente. La verificación
asignó explícitamente 10% a OTHER, sin cambiar metadata ni resultados guardados. Sus métricas
son estimaciones ex ante, no rendimiento OOS. El backtest de demostración sigue sin ser
suficiente para declarar superioridad predictiva.

## Evolución Platform V2 — 2026-10-24

La rama de mejoras fue integrada en `main` por PR #6 (`df33bb8`); esta evolución
parte de esa versión y conserva los resultados académicos y módulos financieros.
La descripción anterior de cola en memoria corresponde al despliegue previo.

Se incorporan servicios bajo `src/platform/`, FastAPI `/api/v1`, Celery/Redis y
scheduler independiente con outbox SQL y leases. Streamlit queda como consumidor
de API. PostgreSQL mantiene nuevas tablas `platform_*` de forma aditiva; MinIO es
canónico para datasets, pesos, preprocesamiento y reportes. MLflow es obligatorio
para publicar experimentos por defecto. La activación de modelos es una operación
separada, sujeta a evidencia de validación y verificación de artefactos.

El dataset se fija al crear el job y el universo/modelos se capturan en la solicitud.
El import explícito migra una copia verificable del legado; no reescribe los CSV,
modelos ni corridas anteriores. Las rutas antiguas de optimización ahora encolan
jobs y devuelven 202. El historial anterior sigue disponible en `/runs`; el nuevo
registro se consulta bajo `/api/v1`.

La actualización Yahoo usa revisión incremental, refresh completo periódico y
publicación validada con rollback. LSTM/MLP/GARCH reutilizan los entrenadores
existentes en workers, guardando artefactos completos. El backtest compartido
compara siete estrategias con restricciones, costos y entrenamiento temporal por
fold. El frontend agrega las ocho áreas de investigación con datos de FastAPI.

La configuración Docker separa UI, API, CPU/GPU workers, scheduler, broker, SQL,
MLflow y MinIO. Adminer queda en desarrollo y producción exige secretos propios.
La integración preparada entrena modelos reducidos con servicios reales, verifica
artefactos y estados y reinicia API/worker. En este entorno el daemon Docker denegó
acceso al socket: la integración distribuida y CUDA no se consideran validadas por
la existencia de ese harness. Los resultados concretos de pruebas y CI deben
registrarse al cerrar la verificación.

Documentación vigente: [arquitectura y migración](platform-architecture.md),
[despliegue y operación](platform-deployment.md), [datos y modelos](dynamic-models.md).
