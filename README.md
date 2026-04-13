# Optimizador Markowitz

## 1. Resumen ejecutivo

Este proyecto implementa una **plataforma integral de optimización de carteras** basada en el marco de **Markowitz con restricciones por bloques**, extendida con señales de inteligencia artificial para dos usos distintos:

1. **predicción de retornos esperados** (`ai_returns`, `ai_returns_lstm`), y  
2. **predicción de volatilidad** (`ai_volatility` con GJR-GARCH).

El sistema no reemplaza la lógica de Markowitz; la arquitectura está pensada para que **Markowitz siga siendo el motor de optimización**, mientras que los modelos de IA funcionan como **generadores de inputs**:

- **Markowitz clásico**: usa media histórica y covarianza histórica.
- **Markowitz con retornos IA (MLP)**: reemplaza parcial o totalmente el vector de retornos esperados.
- **Markowitz con retornos IA (LSTM)**: reemplaza parcialmente el vector de retornos esperados usando el run activo del laboratorio LSTM.
- **Markowitz con volatilidad IA (GJR-GARCH)**: mantiene el retorno esperado histórico, pero reemplaza la estructura de volatilidad diagonal usando volatilidades predichas.

La plataforma está montada sobre:

- **Streamlit**: interfaz principal.
- **FastAPI**: backend de orquestación y ejecución.
- **PostgreSQL**: persistencia de corridas productivas cuando se usa Docker.
- **SQLite**: fallback local si no se define `DATABASE_URL`.
- **MLflow**: tracking experimental y trazabilidad de corridas.
- **MinIO**: object storage para artefactos exportables.
- **Docker Compose**: despliegue integrado de toda la stack.

---

## 2. Objetivo del problema

El problema de negocio/modelado que resuelve el sistema es el siguiente:

> construir portafolios óptimos bajo restricciones de diversificación por bloques, comparando un benchmark puramente histórico contra versiones híbridas donde los inputs de retorno y/o riesgo son enriquecidos mediante modelos predictivos.

En términos formales:

- el proyecto busca **maximizar la relación riesgo-retorno**;
- mantener restricciones por bloques estratégicos;
- limitar concentración por activo;
- evaluar si señales de IA mejoran la calidad del input respecto al benchmark histórico.

El objetivo académico correcto no es presentar el proyecto como “IA reemplazando a Markowitz”, sino como:

> una arquitectura modular donde la innovación se ubica en la **estimación de inputs** \((\mu, \Sigma)\), mientras que la optimización final se conserva dentro del marco clásico de media-varianza.

---

## 3. Qué datos usa el sistema

### 3.1 Fuente base operativa

La fuente de verdad del sistema es:

- `Data/returns.csv`
- `Data/assets_metadata.csv`

Adicionalmente existen:

- `Data/prices.csv`
- `Data/fx_CCL.csv`

### 3.2 Tamaño del dataset operativo actual

En la snapshot actual del repo:

- **2061 observaciones diarias**
- **40 activos**
- rango temporal: **2018-01-03 → 2025-12-31**

### 3.3 Universo de activos actual

#### Acciones argentinas (`AR`) — 15 activos
- `ALUA.BA`
- `BBAR.BA`
- `BMA.BA`
- `CEPU.BA`
- `COME.BA`
- `EDN.BA`
- `GGAL.BA`
- `LOMA.BA`
- `PAMP.BA`
- `SUPV.BA`
- `TGNO4.BA`
- `TGSU2.BA`
- `TXAR.BA`
- `VALO.BA`
- `YPFD.BA`

#### Acciones de EE.UU. (`US`) — 15 activos
- `AAPL`
- `AMZN`
- `BRK-B`
- `GOOGL`
- `JPM`
- `KO`
- `MA`
- `META`
- `MSFT`
- `NVDA`
- `PEP`
- `TSLA`
- `UNH`
- `V`
- `XOM`

#### Fondos / ETFs de equity (`FUNDS_EQUITY`) — 6 activos
- `EEM`
- `EFA`
- `IWM`
- `QQQ`
- `SPY`
- `VNQ`

#### Fondos / ETFs de renta fija (`FUNDS_BONDS`) — 3 activos
- `HYG`
- `IEF`
- `TLT`

#### Oro (`GOLD`) — 1 activo
- `GLD`

### 3.4 Metodología de construcción del dataset

La construcción histórica de la base está documentada principalmente en:

- `notebooks/Base.ipynb`

Lógica general:

1. descarga de precios con `yfinance`;
2. alineación calendaria entre mercados;
3. forward-fill para compatibilizar feriados y cierres dispares;
4. normalización monetaria a USD mediante CCL cuando corresponde;
5. cálculo de **retornos logarítmicos diarios**.

### 3.5 Notebooks y análisis exploratorio

Los notebooks relevantes son:

- `notebooks/Base.ipynb` → construcción de base, exploración inicial y series.
- `notebooks/test_garch_vs_ewma.ipynb` → comparación de enfoques de volatilidad.

Artefactos útiles para tesis/documentación:

- `notebooks/price_series.png`
- `notebooks/returns_series.png`
- `asset_distribution_stats.csv` *(si se desea usar el archivo externo que resume skewness, kurtosis, volatilidad y estadísticas por activo)*

---

## 4. Estructura conceptual del pipeline

El pipeline completo puede leerse en cinco capas.

### Capa 1 — Datos
Archivos principales:

- `Data/returns.csv`
- `Data/assets_metadata.csv`

### Capa 2 — Modelos predictivos
Modelos disponibles:

- **MLP** para retornos
- **LSTM** para retornos
- **LSTM Lab** para corridas experimentales versionadas
- **MLP Lab** para corridas experimentales versionadas
- **GJR-GARCH / EWMA fallback** para volatilidad

### Capa 3 — Ensamblado de inputs
Archivo:

- `src/optimization/inputs_builder.py`

Decide qué señales se usan según el modo seleccionado.

### Capa 4 — Optimización
Archivos principales:

- `src/optimization/optimizer_service.py`
- `src/optimization/frontier_engine.py`
- `src/optimization/optimize_blocks.py`

### Capa 5 — Plataforma
Archivos y servicios principales:

- `app/Optimizador_Markowitz.py` → home principal Streamlit
- `api/main.py` → backend FastAPI
- `app/db.py` → capa de base de datos
- `app/repository.py` → guardado / lectura de corridas
- `app/tracking.py` → tracking a MLflow + MinIO

---

## 5. Qué consume exactamente cada modo de Markowitz

La lógica actual está en `src/optimization/inputs_builder.py`.

### 5.1 `classic`
Uso exclusivo de datos historicos.

**Input:**
- `mu = returns.mean()`
- `cov = returns.cov()`

**Fuente:**
- `Data/returns.csv`

**Interpretación:** benchmark y teoria financiera pura.

---

### 5.2 `ai_returns` - (Descontinuado por mejor desempeño de LSTM)
Primero intenta consumir el **run activo de MLP Lab**. Si no hay run activo o la carga falla, hace fallback a `outputs/predictions/predictions.csv`.

**Prioridad actual:**
1. run activo de MLP Lab (`load_active_mlp_predictions()`)
2. fallback a `outputs/predictions/predictions.csv`

**Lógica:**
- los activos aprobados por la policy usan `mu_pred`
- los activos no aprobados vuelven a media histórica
- la covarianza sigue siendo histórica

**Interpretación:** sustitución parcial del vector \(\mu\).

---

### 5.3 `ai_returns_lstm`
Consume el **run activo de LSTM Lab**.

**Fuente:**
- tablas del laboratorio LSTM en DB
- rutas espejo en `outputs/lstm_lab/latest_run.json`

**Lógica actual:**
- si `dir_acc_test > 0.52`, usa `mu_pred`
- si no, el activo vuelve a media histórica
- la covarianza sigue siendo histórica

**Interpretación:** sustitución parcial y filtrada del vector \(\mu\).

---

### 5.4 `ai_volatility`
Consume directamente:

- `outputs/predictions/volatility_predictions.csv`

**No usa `predictions.csv` para volatilidad.**

**Lógica:**
- `mu`: media histórica
- `Sigma`: correlación histórica + volatilidad predicha
- construcción conceptual:
  `Sigma_AI = D * Corr_hist * D`

**Interpretación:** sustitución parcial del bloque de riesgo, manteniendo la estructura de correlaciones históricas.

---

## 6. Almacenamiento de datos

### 6.1 Datos base
- `Data/prices.csv`
- `Data/returns.csv`
- `Data/assets_metadata.csv`
- `Data/fx_CCL.csv`

### 6.2 Predicciones consolidadas
- `outputs/predictions/return_predictions.csv`
- `outputs/predictions/return_predictions_lstm.csv`
- `outputs/predictions/volatility_predictions.csv`
- `outputs/predictions/predictions.csv`

### 6.3 Modelos entrenados
#### MLP
- `models_artifacts/returns/*.joblib`

#### LSTM
- `models_artifacts/returns_lstm/*.pt`
- `models_artifacts/returns_lstm/*_meta.json`

#### LSTM Lab
- `models_artifacts/returns_lstm/runs/<run_id>/...`
- `outputs/lstm_lab/runs/<run_id>/...`

#### MLP Lab
- `models_artifacts/returns_mlp_lab/runs/<run_id>/...`
- `outputs/mlp_lab/runs/<run_id>/...`

#### GJR-GARCH
No se guarda como artefacto serializado por activo en una carpeta de modelos al estilo MLP/LSTM. El output operativo relevante es:
- `outputs/predictions/volatility_predictions.csv`

y el historial de evaluación del laboratorio:
- `outputs/garch_lab/runs/<timestamp>_metrics.csv`
- `outputs/garch_lab/runs/<timestamp>_meta.json`

### 6.4 Corridas de optimización
- `outputs/frontier_*.csv`
- `outputs/weights/*.csv`
- `outputs/metrics/*.csv`
- `outputs/figures/*.png`

### 6.5 Base de datos local
Si no se define `DATABASE_URL`, el sistema usa:
- `storage/markowitz_app.db`

### 6.6 Tracking experimental
- `mlruns/` *(si se usa MLflow local por filesystem)*
- o el backend MLflow del contenedor Docker

---

## 7. Servicios del sistema

### 7.1 Streamlit
**Archivo principal:** `app/Optimizador_Markowitz.py`

Rol:
- interfaz de usuario;
- configuración de bloques, pesos, modo y límites;
- visualización de frontera, métricas, exposiciones y pesos;
- acceso a laboratorios y monitor.

### 7.2 FastAPI
**Entrada principal:** `api/main.py`

Rol:
- exponer el backend como API REST;
- recibir el request desde Streamlit;
- correr el optimizador;
- guardar corridas;
- disparar tracking a MLflow/MinIO.

### 7.3 PostgreSQL
Rol:
- persistir corridas productivas (`optimization_runs`) y tablas de laboratorios cuando se trabaja bajo Docker.

### 7.4 SQLite
Rol:
- fallback local cuando no hay Postgres configurado.

### 7.5 MLflow
Rol:
- registrar parámetros y métricas de corridas de optimización;
- guardar artefactos como frontera, pesos y exposición;
- registrar corridas del LSTM Lab si está configurado.

### 7.6 MinIO
Rol:
- object storage para artefactos CSV de optimización;
- almacenamiento de outputs/modelos del LSTM Lab cuando la integración está activa.

### 7.7 Adminer
Rol:
- inspeccionar PostgreSQL desde navegador;
- útil para validar corridas, tablas y estados sin abrir manualmente la base.

---

## 8. Links, puertos y credenciales

> **Importante:** estas credenciales son de desarrollo local. Si el sistema se expone fuera de localhost, deben cambiarse.

## 8.1 Servicios Docker

### Streamlit
- URL: `http://localhost:8502`
- Contenedor: `markowitz_streamlit`

### FastAPI
- URL base: `http://localhost:8000`
- Docs Swagger: `http://localhost:8000/docs`
- Contenedor: `markowitz_fastapi`

### MLflow
- URL: `http://localhost:5005`
- Contenedor: `markowitz_mlflow`

### MinIO Console
- URL: `http://localhost:9001`
- Usuario: `minioadmin`
- Contraseña: `minioadmin`
- Contenedor: `markowitz_minio`

### MinIO S3 endpoint
- Endpoint: `http://localhost:9000`
- Access key: `minioadmin`
- Secret key: `minioadmin`

### PostgreSQL
- Host desde host local: `localhost`
- Puerto: `5432`
- Base: `markowitz`
- Usuario: `markowitz_user`
- Contraseña: `markowitz_pass`
- Contenedor: `markowitz_postgres`

### Adminer
- URL: `http://localhost:8081`
- Sistema: `PostgreSQL`
- Server: `postgres`
- Usuario: `markowitz_user`
- Contraseña: `markowitz_pass`
- Base: `markowitz`

---

## 9. Variables de entorno principales

Archivo real:
- `.env`

Variables clave:

```env
POSTGRES_SERVER=postgres
POSTGRES_USER=markowitz_user
POSTGRES_PASSWORD=markowitz_pass
POSTGRES_DB=markowitz

DATABASE_URL=postgresql+psycopg2://markowitz_user:markowitz_pass@postgres:5432/markowitz

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=markowitz-streamlit

MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

API_BASE_URL=http://fastapi:8000
MLFLOW_UI_URL=http://localhost:5005
MINIO_CONSOLE_URL=http://localhost:9001
```

---

## 10. Cómo levantar todo el sistema con Docker

Desde la raíz del repo:

```bash
cd /ruta/al/repo/Markowitz
docker compose up -d --build
```

Para validar:

```bash
docker compose ps
```

Servicios esperados:
- postgres
- adminer
- minio
- mlflow
- fastapi
- streamlit

Para bajar todo:

```bash
docker compose down
```

Para bajar todo y borrar volúmenes:

```bash
docker compose down -v
```

---

## 11. Cómo correrlo localmente en Windows sin Docker

### 11.1 Streamlit

```cmd
cd /d C:\Users\lucas\OneDrive\Documentos\GitHub\Markowitz
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe -m streamlit run app/Optimizador_Markowitz.py --server.port 8502
```

### 11.2 FastAPI

```cmd
cd /d C:\Users\lucas\OneDrive\Documentos\GitHub\Markowitz
set "MLFLOW_TRACKING_URI=file:///C:/Users/lucas/OneDrive/Documentos/GitHub/Markowitz/mlruns"
set "MLFLOW_EXPERIMENT_NAME=markowitz-streamlit"
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### 11.3 MLflow local por filesystem

```cmd
cd /d C:\Users\lucas\OneDrive\Documentos\GitHub\Markowitz
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe -m mlflow ui --backend-store-uri "file:///C:/Users/lucas/OneDrive/Documentos/GitHub/Markowitz/mlruns" --port 5005
```

### 11.4 Base de datos local

Si no seteás `DATABASE_URL`, el proyecto usa automáticamente:

- `storage/markowitz_app.db`

Eso permite usar la app sin Postgres, aunque para el flujo completo y más prolijo se recomienda Docker.

---

## 12. Estructura importante del repo

```text
Markowitz/
├── Data/
│   ├── prices.csv
│   ├── returns.csv
│   ├── assets_metadata.csv
│   └── fx_CCL.csv
├── api/
│   ├── main.py
│   ├── routers/
│   ├── schemas/
│   └── services/
├── app/
│   ├── Optimizador_Markowitz.py
│   ├── db.py
│   ├── repository.py
│   ├── tracking.py
│   └── pages/
├── docker/
│   ├── fastapi.Dockerfile
│   └── streamlit.Dockerfile
├── infrastructure/
│   └── storage/
├── models_artifacts/
│   ├── returns/
│   ├── returns_lstm/
│   └── returns_mlp_lab/
├── notebooks/
├── outputs/
│   ├── predictions/
│   ├── weights/
│   ├── metrics/
│   ├── figures/
│   ├── lstm_lab/
│   └── mlp_lab/
├── scripts/
├── src/
│   ├── data/
│   ├── models/
│   └── optimization/
└── storage/
    └── markowitz_app.db
```

---

## 13. Páginas principales de Streamlit

### Home
- `app/Optimizador_Markowitz.py`
- nombre visible: **Optimizador Markowitz**

### Páginas auxiliares
- `app/pages/01_Historial_de_Corridas.py`
- `app/pages/02_Monitor_del_Sistema.py`
- `app/pages/02_MLP_Lab.py`
- `app/pages/03_Laboratorio_LSTM.py`
- `app/pages/04_Laboratorio_GARCH.py`

> Nota: MLP Lab hoy queda como **laboratorio experimental / stand by**, no como motor productivo recomendado.

---

## 14. Scripts de uso frecuente

### Entrenar MLP legado de retornos
```cmd
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_train_returns.py
```

### Entrenar LSTM legado de retornos
```cmd
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_train_returns_lstm.py
```

### Entrenar MLP Lab por script
```cmd
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_train_returns_mlp_lab.py
```

### Generar volatilidades GJR-GARCH
```cmd
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_train_volatility.py
```

### Construir tabla consolidada de predicciones legacy
```cmd
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_predict_all.py
```

### Correr modos de optimización por script
```cmd
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_mode_classic.py
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_mode_ai_returns.py
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_mode_ai_returns_lstm.py
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_mode_ai_volatility.py
```

---

## 15. Cómo usar el sistema en la práctica

## 15.1 Flujo mínimo para usar el optimizador

1. levantar Streamlit + FastAPI;
2. abrir `http://localhost:8502`;
3. elegir modo (`classic`, `ai_returns`, `ai_returns_lstm`, `ai_volatility`);
4. definir pesos objetivo por bloque;
5. definir `max_weight` y tasa libre de riesgo;
6. ejecutar la corrida;
7. revisar frontera, pesos, exposición por bloque y guardar historial.

---

## 15.2 Flujo recomendado para `classic`

No requiere entrenamiento previo.

Usa:
- `Data/returns.csv`
- `Data/assets_metadata.csv`

---

## 15.3 Flujo recomendado para `ai_volatility`

### Opción A — desde GARCH Lab en Streamlit
1. ir a **Laboratorio GARCH**;
2. seleccionar fecha de inicio;
3. actualizar predicciones;
4. eso reescribe `outputs/predictions/volatility_predictions.csv`;
5. volver a la home;
6. correr `ai_volatility`.

### Opción B — por script
```cmd
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_train_volatility.py
```

Luego ejecutar `ai_volatility`.

**Importante:** hoy `ai_volatility` consume directamente `volatility_predictions.csv`.

---

## 15.4 Flujo recomendado para `ai_returns_lstm`

1. ir a **Laboratorio LSTM**;
2. entrenar una corrida;
3. revisar métricas por activo;
4. marcar el run como **activo**;
5. volver a la home;
6. correr `ai_returns_lstm`.

El optimizador tomará el run activo y solo aplicará `mu_pred` a activos con calidad suficiente.

---

## 15.5 Flujo para `ai_returns` con MLP - (Descontinuado)

Hoy existen dos caminos.

### Camino 1 — MLP Lab
1. ir a **MLP Lab**;
2. entrenar una corrida;
3. revisar métricas;
4. definir policy de selección;
5. marcar run activo;
6. volver a la home;
7. correr `ai_returns`.

### Camino 2 — legacy
1. entrenar MLP legado:
```cmd
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_train_returns.py
```
2. consolidar tabla:
```cmd
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe scripts\run_predict_all.py
```
3. correr `ai_returns`.

> Estado actual recomendado: **MLP Lab en stand by / experimental**.

---

## 16. Parámetros y estados de los modelos

## 16.1 Markowitz clásico
No hay un modelo serializado. El input se calcula desde `returns.csv`.

## 16.2 MLP legado de retornos
Ubicación:
- `models_artifacts/returns/*.joblib`

Configuración base histórica:
- `hidden_layer_sizes=(64,32)`
- `activation='relu'`
- `max_iter=500`
- `early_stopping=True`
- target a 1 día
- features de lags y rolling stats

## 16.3 LSTM legado / productivo
Ubicación:
- `models_artifacts/returns_lstm/*.pt`

Parámetros típicos históricos:
- `seq_len=20`
- `forecast_horizon=5`
- `hidden_size=64`
- `num_layers=2`
- `dropout=0.2`
- `epochs=80`
- `batch_size=32`
- `lr=0.001`

## 16.4 LSTM Lab
Ubicación de runs:
- `outputs/lstm_lab/runs/<run_id>/`
- `models_artifacts/returns_lstm/runs/<run_id>/`

Último run identificado en esta snapshot:
- `20260329_012341`

## 16.5 MLP Lab - (Descontinuado)
Ubicación de runs:
- `outputs/mlp_lab/runs/<run_id>/`
- `models_artifacts/returns_mlp_lab/runs/<run_id>/`

Último run identificado en esta snapshot:
- `20260330_003605`

Estado recomendado:
- **experimental / en pausa**

## 16.6 GJR-GARCH
Salida operativa:
- `outputs/predictions/volatility_predictions.csv`

Parámetros operativos observados:
- `start_date = 2022-01-01`
- `min_obs = 252`
- modelo principal: **GJR-GARCH(1,1,1)**
- innovación Student-t
- fallback: EWMA

---

## 17. Cómo se registra el historial

### 17.1 Corridas de optimización
Se guardan en la tabla:
- `optimization_runs`

Campos relevantes:
- timestamp
- label de corrida
- modo
- max weight
- risk-free rate
- targets
- métricas
- pesos
- frontera
- solver message

### 17.2 Corridas de LSTM Lab
Tablas principales:
- `prediction_runs`
- `prediction_run_tickers`

### 17.3 Corridas de MLP Lab - (Descontinuado)
Tablas principales:
- `mlp_prediction_runs`
- `mlp_prediction_run_tickers`

### 17.4 GARCH Lab
Se guarda por archivos, no por tablas SQL dedicadas:
- `outputs/garch_lab/runs/<run_id>_metrics.csv`
- `outputs/garch_lab/runs/<run_id>_meta.json`

---

## 18. Integración con MLflow y MinIO

## 18.1 Optimización
Cada corrida disparada por FastAPI puede:

- guardarse en DB;
- trackearse a MLflow;
- subir artefactos CSV a MinIO.

Artefactos típicos:
- `frontier.csv`
- `weights_blocks.csv`
- `block_exposure.csv`
- `comparison.csv`

Bucket típico:
- `optimization-runs`

## 18.2 LSTM Lab
El laboratorio LSTM además puede:
- loguear parámetros y métricas a MLflow;
- subir outputs y modelos a MinIO.

## 18.3 MLP Lab - (Descontinuado)
En la versión actual, el MLP Lab persiste en DB y en carpetas locales del repo, pero no tiene integración equivalente a la del LSTM Lab con MinIO/MLflow tan desarrollada.

---

## 19. Archivos clave que conviene conocer de memoria

### Frontend / UX
- `app/Optimizador_Markowitz.py`
- `app/pages/01_Historial_de_Corridas.py`
- `app/pages/02_Monitor_del_Sistema.py`
- `app/pages/03_Laboratorio_LSTM.py`
- `app/pages/04_Laboratorio_GARCH.py`
- `app/pages/02_MLP_Lab.py`

### Backend / API
- `api/main.py`
- `api/routers/optimize.py`
- `api/routers/runs.py`
- `api/services/optimization_service.py`

### Persistencia / tracking
- `app/db.py`
- `app/repository.py`
- `app/tracking.py`
- `app/lstm_lab_storage.py`
- `app/mlp_lab_storage.py`
- `app/garch_lab_storage.py`

### Optimización
- `src/optimization/inputs_builder.py`
- `src/optimization/optimizer_service.py`
- `src/optimization/frontier_engine.py`
- `src/optimization/optimize_blocks.py`

### Modelos
- `src/models/train_returns.py`
- `src/models/train_returns_lstm.py`
- `src/models/train_returns_mlp_lab.py`
- `src/models/train_volatility.py`
- `src/models/predict.py`

---

## 20. Troubleshooting rápido

### “No hay un run activo de MLP Lab”
- entrenar un run en MLP Lab;
- marcarlo como activo;
- o usar el flujo legacy con `predictions.csv`.

### “No active LSTM Lab predictions available”
- entrenar una corrida en LSTM Lab;
- marcar el run como activo.

### `ai_volatility` no refleja la última corrida
- verificar que `outputs/predictions/volatility_predictions.csv` se haya actualizado;
- revisar timestamp del archivo;
- reintentar desde GARCH Lab o script.

### Streamlit abre pero no optimiza
- verificar que FastAPI esté corriendo;
- abrir `http://localhost:8000/docs`;
- revisar `API_BASE_URL`.

### No aparecen corridas en el historial
- verificar `DATABASE_URL`;
- si no hay Postgres, confirmar que exista `storage/markowitz_app.db`;
- revisar que la corrida se haya lanzado vía FastAPI/Streamlit y no solo por script directo.

---

## 21. Disclaimer

Este sistema es únicamente informativo, experimental y educativo. **No constituye una recomendación de inversión ni asesoramiento financiero.**

---

## 22. Resumen final

**Optimizador Markowitz** es una plataforma de portfolio optimization con restricciones por bloques, donde Streamlit orquesta la experiencia de usuario, FastAPI ejecuta la lógica, PostgreSQL/SQLite persisten corridas, MLflow y MinIO trazan artefactos, y los modos `classic`, `ai_returns`, `ai_returns_lstm` y `ai_volatility` permiten comparar benchmark histórico contra inputs enriquecidos por modelos predictivos.
