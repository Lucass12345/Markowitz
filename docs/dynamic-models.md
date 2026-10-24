# Dataset dinámico y modelos a cinco ruedas

## Flujo vigente de la plataforma

En Streamlit, abrir **Data Management** para crear un universo reutilizable o ingresar tickers diarios admitidos por Yahoo Finance. La selección no está limitada a los 40 activos iniciales. La UI solicita `POST /api/v1/datasets/refresh`; FastAPI fija el dataset padre y el universo y crea un job. Un worker realiza descarga, validación y transformación, publica un bundle inmutable en MinIO y registra la versión en PostgreSQL. Consultar el UUID en `/api/v1/jobs/{id}` hasta completar la publicación.

El refresh manual crea un candidato por defecto (`activate=false`). Su activación se solicita mediante `/api/v1/datasets/{version}/activate`, también como job. El scheduler realiza refresh incremental diario por defecto, con activación sólo después de superar los controles; se configura con `YAHOO_REFRESH_INTERVAL_SECONDS` y se deshabilita con `YAHOO_REFRESH_ENABLED=false`. No se descarga al abrir Streamlit.

Cada snapshot incluye precios ajustados **en USD**, precios nativos (`prices_native.csv`), retornos logarítmicos diarios **en USD**, metadata, auditoría cambiaria (`fx_rates.csv`), acciones corporativas y manifiesto. Se guarda ID, hash, origen, fecha de generación/corte, universo, política de limpieza y parámetros de transformación. La referencia activa es SQL y se cambia atómicamente. Los archivos locales sólo se materializan como temporales verificables. Un job conserva la versión fijada aunque luego se active otra.

El modo incremental vuelve a consultar un solapamiento configurable de 30 días, compara revisiones de precios ajustados y acciones corporativas y puede ampliar a refresh completo. Por defecto se realiza un refresh completo periódico cada 30 días. Se excluye la fecha UTC en curso para evitar velas diarias incompletas. No se rellenan precios ni retornos de feriados: cada activo conserva sus propias ruedas. Fallos de tickers, monedas o calidad impiden activar el candidato y conservan el dataset anterior. Yahoo puede limitar solicitudes o no disponer de historia diaria para un símbolo.

Para migrar la base actual, usar **Importar histórico del proyecto** o `POST /api/v1/datasets/import-legacy`. Se copia y valida el dataset legado hacia MinIO y se registra/activa sin alterar los archivos originales. Las tablas y corridas históricas permanecen disponibles. Los punteros `Data/active_release.json` y `Data/active_dataset.json` son exclusivamente compatibilidad para este import y los scripts anteriores; las páginas nuevas usan `DatasetService` y el registro SQL.

La identificación guarda tipo Yahoo, tipo normalizado (incluido fondo de bonos), nombre, categoría, sector, país, bolsa y moneda. País y sector no informados quedan como desconocidos. El domicilio del emisor o la bolsa de cotización no se presentan como exposición geográfica de un fondo. Las categorías de renta fija permiten identificar ETFs de bonos como TLT/IEF/HYG. La conversión a USD ocurre antes de construir los retornos; los metadatos conservan la moneda original y documentan esa conversión. Los bloques se leen dinámicamente del metadata. Activos fuera de las categorías originales reciben `OTHER`; se pueden usar nombres personalizados en la configuración de universo. Las ediciones no reescriben versiones históricas. La UI muestra los bloques disponibles y conserva una opción histórica de cinco bloques. Revisar objetivos y capacidad (`n_activos * max_weight >= target`) antes de cambiar de mercado.

## Moneda común: USD

Todos los modelos y el optimizador reciben retornos sobre precios expresados en **USD**. El país del activo no determina el cambio: se usa la moneda de cotización informada por Yahoo. Una acción argentina en ARS se convierte; su ADR en USD permanece igual.

Para ARS, se usa **CCL vendedor en ARS por USD**: `precio_USD(t) = precio_ARS(t) / CCL(t)`. Después se calcula `log(precio_USD(t) / precio_USD(t-1))`. Por lo tanto `retorno_log_USD = retorno_log_ARS − variación_log_CCL`. Si la acción sube 10% en pesos pero el CCL sube 20%, el retorno simple en USD es `1.10 / 1.20 − 1 = −8.33%`. No alcanza con dividir el retorno porcentual por el cambio de hoy.

El CCL se descarga de [ArgentinaDatos, serie contado con liquidación](https://api.argentinadatos.com/v1/cotizaciones/dolares/contadoconliqui), campo `venta`. Se conserva el CSV local `Data/fx_CCL.csv` para extender el historial anterior al comienzo de esa serie remota (cierre local; cambio de fuente/convenio identificado en la auditoría). Si la descarga falla, sólo puede usarse la fuente local si supera los controles de cobertura y antigüedad; no se reemplaza por dólar oficial. Para fijar una serie propia y reproducible, en los scripts CLI, pasar `--ccl-file ruta/ccl.csv` (columnas `fecha,cierre` o `date,rate`, expresadas en ARS por USD). El archivo explícito se usa sin sobrescribirlo con datos remotos.

Otras monedas se convierten con la serie histórica Yahoo `<MONEDA>USD=X` (USD por unidad de moneda), multiplicando el precio local por el cambio. Se contemplan cotizaciones en peniques/agorot/centavos como `GBp`, `GBX`, `ILA` y `ZAc`, aplicando su escala antes del cambio.

Cada precio usa una tasa del mismo día o anterior, nunca posterior. La antigüedad máxima por defecto es **siete días calendario** (`max_fx_age_days` en el refresh API; `--max-fx-age-days` en CLI). No se publican activos con moneda desconocida, cambios faltantes, no positivos o vencidos; se conserva el dataset anterior. Esto permite cubrir fines de semana sin prolongar indefinidamente una cotización vieja. La moneda original, tasa usada, fecha, fuente, escala y moneda base quedan guardadas.

[Verificación de la corrección monetaria y resultados actualizados](experiments/2026-10-21-usd/README.md).

Los snapshots dinámicos anteriores sin `base_currency: USD` se rechazan al cargar precios o retornos. Las predicciones de otro snapshot también se rechazan: hay que actualizar datos y reentrenar después de aplicar esta corrección.

## LSTM

El horizonte de producción es **cinco observaciones futuras del activo**, no cinco días calendario. El modelo estima una magnitud para todo el horizonte, no cinco pronósticos diarios independientes:

- `mean`: promedio de los cinco retornos logarítmicos futuros.
- `sum`: retorno logarítmico acumulado de esas cinco ruedas.
- `mu_pred`: siempre promedio diario para mantener unidades compatibles con el optimizador.
- `forecast_log_return_5d`: acumulado a cinco ruedas.
- `forecast_simple_return_5d`: `expm1(acumulado)`.

La inferencia utiliza la última ventana **sin etiquetas futuras**, incluyendo el último dato descargado. Se corrigió el uso anterior de la última secuencia etiquetada, que dejaba el pronóstico cinco ruedas atrasado. También se eliminó el recorte del target con cuantiles de toda la serie y el recorte arbitrario de pronósticos a ±2%.

Las entradas seleccionables incluyen retorno, rezagos, medias/desviaciones móviles, momentum, sumas logarítmicas a 5/20 ruedas, volatilidad bajista, volatilidad EWMA y asimetría. Todas son causales. Las comparaciones de features usan el mismo período de calentamiento. Los escaladores de entradas y target se ajustan sólo con entrenamiento y se guardan junto al checkpoint.

La búsqueda compara 64→64 y **120→64→densa**, activaciones **ReLU, Tanh y GELU en la capa densa** (las celdas LSTM mantienen sus activaciones internas estándar) y dos configuraciones Adam:

| learning rate | betas | epsilon | weight decay |
| --- | --- | --- | --- |
| 0.001 | (0.9, 0.999) | 1e-8 | 0 |
| 0.0003 | (0.9, 0.99) | 1e-7 | 0.0001 |

Son doce candidatos por conjunto de features cuando `search=true`. El laboratorio permite seleccionar features propias y habilitar `feature_search` para comparar las entradas elegidas contra retornos solos, usando validación; por defecto esa comparación adicional está deshabilitada. Con `search=false` se conserva la especificación explícita del experimento. No hay una arquitectura ganadora impuesta para todos los activos.

Se conserva el orden temporal 70/15/15 y se purgan cuatro secuencias en cada borde para que los targets de cinco ruedas no compartan retornos entre particiones. Se usa early stopping, clipping de gradientes y semilla 42. Se selecciona el menor MSE de validación. El test sólo evalúa al ganador; también se informa RMSE contra retorno cero y media de entrenamiento. El filtro posterior del optimizador usa métricas de **validación**: dirección >0,52 y amplitud >0,15 (ambos configurables), además de que el LSTM supere el RMSE de la media de entrenamiento sobre validación (`use_for_optimizer`). Los umbrales y motivos por ticker se guardan con cada optimización. Si falla el filtro, conserva la media histórica del activo. Los errores por ticker se guardan en `metrics.csv`.

Después de evaluar el ganador sobre test, se reajusta su especificación con **todo el historial etiquetado disponible**, incluyendo los últimos datos. El presupuesto de épocas se fija con el early stopping de validación; no se vuelve a seleccionar usando test. El checkpoint de evaluación se conserva separado. Las métricas publicadas pertenecen al modelo evaluado antes del reajuste, mientras que el pronóstico futuro usa el modelo reajustado y sus nuevos escaladores.

Cada corrida guarda candidatos, métricas, curvas de pérdida, predicciones históricas, pronóstico futuro y checkpoint con `selected_config`, escaladores y orden de features. Para reconstruir un checkpoint nuevo, crear `ReturnLSTM(input_size=len(meta['selected_features']), hidden_sizes=meta['selected_config']['hidden_sizes'], activation=meta['selected_config']['activation'], feature_indices=meta['selected_config'].get('feature_indices'), dropout=meta['dropout'])` y cargar el state dict. Los checkpoints antiguos con capas homogéneas conservan sus claves si se usa el constructor original.

## GARCH

Se mantiene GJR-GARCH(1,1,1) con innovaciones Student-t y media cero. Se publican varianza/volatilidad diaria y a cinco ruedas. GARCH pronostica **riesgo**, no retorno esperado. La varianza acumulada suma las cinco varianzas condicionales bajo la especificación de media cero. La covarianza del optimizador sigue usando `var_pred`/`vol_pred` diarios.

Los diagnósticos incluyen convergencia, log-likelihood, AIC/BIC, coeficientes, errores estándar, p-values, intervalos al 95%, persistencia y condición de estacionariedad. El worker conserva los reportes de validación y test en `evaluation/{ticker}_diagnostics.json`; el CLI histórico usa `garch_diagnostics.json`. Ljung–Box evalúa residuos estandarizados y sus cuadrados; ARCH-LM busca heterocedasticidad residual (umbral orientativo 5%). Pasar estos tests no garantiza capacidad predictiva.

Por defecto se evalúan 20 orígenes móviles de validación y otros 20 de test. Cada origen reajusta sólo con datos anteriores a la observación y compara GARCH contra EWMA con QLIKE y MSE de varianza. El proxy observado es retorno logarítmico diario al cuadrado, una medición ruidosa. No se optimizan parámetros con el test. El worker exige orígenes completos y un ajuste final convergente para guardar parámetros utilizables; un ajuste incompleto falla y no activa un modelo parcial. El CLI histórico conserva su fallback EWMA explícito ante fallos de GARCH. En el optimizador, la ausencia de una señal de riesgo válida utiliza volatilidad histórica si la política lo permite, con motivo registrado.

Las predicciones nuevas incluyen versión del dataset; el optimizador rechaza predicciones LSTM/GARCH de otro dataset. Después de actualizar datos hay que entrenar modelos para esa versión o ejecutar inferencia explícita con un modelo registrado compatible. Esa inferencia conserva el dataset de entrenamiento y registra el nuevo dataset/fecha de pronóstico; nunca se reutiliza un CSV anterior cambiando sólo su etiqueta. Señales individuales faltantes o inválidas mantienen el activo con fallback histórico; la política required exige cobertura completa.

## Experimentos, publicación y automatización

**Model Laboratory** utiliza `POST /api/v1/experiments` con `family=lstm|mlp|garch`, `dataset_version`, `tickers` y `parameters`. API y worker comparten los servicios de dominio; Streamlit no entrena ni escribe modelos. El experimento, job, bundle MinIO y run MLflow conservan referencias cruzadas. Los escaladores, orden de features, pesos/parámetros, configuración, métricas y pronósticos son parte del bundle, con checksums verificados al cargarlo.

MinIO es obligatorio para estos flujos. MLflow también es obligatorio por defecto (`PLATFORM_MLFLOW_REQUIRED=true`); un fallo requerido impide completar la publicación. La excepción explícita `false` permite ejecución sin tracking en entornos que acepten esa limitación y registra el estado opcional. No existe fallback productivo que sustituya MinIO por modelos locales permanentes.

Las métricas determinan si el modelo queda `validated` o `rejected`. Activar un modelo aceptado requiere otra solicitud (`/api/v1/models/{id}/activate`); archivar y restaurar una versión no modifica sus artefactos. Inferir sobre un nuevo snapshot utiliza `/api/v1/models/{id}/infer`. El optimizador fija los registros utilizados al crear su job y valida versión, horizonte, unidades, antigüedad y estado. Tener un archivo de predicciones no implica que su señal sea admisible.

Para estudios aislados se conservan los CLI, por ejemplo:

```bash
python scripts/run_model_study.py --tickers AAPL TLT GGAL.BA --epochs 30
```

Este script compara features sobre un snapshot fijo y selecciona por validación, escribiendo resultados locales de investigación. No publica ni activa modelos en el nuevo registro. Lo mismo vale para `scripts/run_refresh_dataset.py` y el workflow histórico `.github/workflows/yahoo-models.yml`: generar un bundle como artefacto GitHub no cambia los contenedores ni las referencias SQL. `scripts/pull_release.py` mantiene compatibilidad con publicaciones locales anteriores; no sustituye a la activación de la plataforma.

El scheduler interno actualiza datasets, pero **no reentrena ni activa automáticamente las tres familias** después del refresh. Los entrenamientos y activaciones se solicitan por API/UI. El objetivo es evitar que una actualización de datos publique sin revisión señales de calidad desconocida. Ver [arquitectura y migración](platform-architecture.md) y [despliegue](platform-deployment.md).

## Compatibilidad MLP

La selección MLP también utiliza únicamente validación y registra la política efectiva.
Los scores antiguos basados en test no se reutilizan. El laboratorio nuevo exige targets
completos, purga los bordes del horizonte y ajusta límites de winsorization sólo en entrenamiento.
Las corridas antiguas sin evidencia y el MLP clásico sin validación quedan como fallback
histórico. Los resultados almacenados no se alteran. La fecha `as_of` describe la ventana
realmente inferida; no se adelanta artificialmente al corte del dataset.
