# Verificación de Platform 2.0

## Código y preservación

La rama `improvements/financial-validation-mlops` ya estaba integrada en `main`
mediante `df33bb8` (PR #6). Se continuó en `main`, preservando su historial, los
CSV académicos, las tablas existentes y los scripts CLI. No se hizo force push.

| Etapa | Commit inicial | Archivos principales |
| --- | --- | --- |
| A · arquitectura | `c1eea3c` | `src/platform/{database,jobs,registry,artifacts}.py`, `worker/`, API y Docker |
| B · datos | `9679e23` | `datasets.py`, `universes.py`, `src/data/yahoo_dataset.py` |
| C · modelos | `c761fa7` | `models.py`, entrenadores existentes y pruebas con ajustes reales |
| D · optimización | `94c492e` | `optimization.py`, señales explícitas y pruebas financieras |
| E · backtesting | `b25f5b3` | `backtesting.py`, estrategias y walk-forward |
| F · interfaz | `c919884` | `app/ui/`, páginas y transporte API |
| G · verificación | `ec6b2e8` | CI, contratos, integración y documentación |

Correcciones posteriores conservaron el release original de MinIO compilándolo
por commit/checksum, expusieron el historial académico sin recalcular métricas y
agregaron reintentos manuales con las dependencias originales fijadas.

## Pruebas ejecutadas

La suite Python local terminó con **302 pruebas aprobadas**, 29 advertencias de
bibliotecas y 38,38 segundos de ejecución, con Python 3.11 y Torch 2.6 CPU.
Incluye entrenamientos reales pequeños, inferencia después de serializar/cargar,
versionado USD, contratos HTTP, concurrencia e idempotencia, recuperación,
restricciones, matrices, selección por validación y cronología del backtesting.
Ruff (errores de ejecución), compileall y configuración Compose también pasaron.
Las advertencias incluyen deprecaciones y convergencia de MLP en ajustes breves.

La integración del backend con PostgreSQL, Redis, MinIO y MLflow reales pasó en
[Actions para d9f5362](https://github.com/Lucass12345/Markowitz/actions/runs/40050936736).
El harness entrena LSTM, MLP y GARCH para AAPL/TLT, verifica hashes y pesos,
registra métricas, comprueba aceptación/rechazo, ejecuta inferencia, optimización
y las siete estrategias de backtest, y reinicia API/worker. También comprueba
que entregar dos veces un mismo trabajo no vuelve a entrenarlo.

La CI vigente añade un recorrido Streamlit sin mocks: ocho áreas, envío de un
entrenamiento a FastAPI, ejecución por worker y recuperación de resultados. Los
reportes `integration-report.json` y `ui-integration-report.json` quedan como
artefactos de cada ejecución de [Python CI](https://github.com/Lucass12345/Markowitz/actions/workflows/ci.yml).
Revisar el estado del commit concreto; configurar un workflow no prueba su éxito.

## Límites

Los ajustes pequeños prueban integración y reproducibilidad; no demuestran que IA
supere al modelo clásico. No se validaron GPU/CUDA ni carga de producción. La
integración usa el dataset histórico importado y no depende de que Yahoo esté
accesible durante CI; fallos de proveedor, FX y refresco se prueban por separado.

Las imágenes locales se construyeron. El stack completo local no terminó por el
espacio requerido por Docker `vfs` en 32 GB; se verificó en el runner de GitHub.
No se borraron volúmenes de usuario ni se desplegó una instancia pública.

Antes de actualizar una instalación, seguir la guía de UID/GID y backups de
[despliegue](platform-deployment.md). Los modelos antiguos sin evidencia verificable
no se activan automáticamente. El primer uso requiere importar el dataset o
publicar un refresh validado. El historial anterior permanece disponible mediante
`/runs` y `/api/v1/history/legacy` y en History & Experiments.
