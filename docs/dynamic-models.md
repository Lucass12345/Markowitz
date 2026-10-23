# Dataset dinámico y modelos a cinco ruedas

En Streamlit, abrir **Dataset Dinámico** para elegir cualquier ticker diario admitido por Yahoo Finance. El universo se reemplaza por la selección completa; no está limitado al CSV inicial. Desde consola:

```bash
python scripts/run_refresh_dataset.py --tickers AAPL MSFT GGAL.BA TLT GLD --retrain --epochs 30
```

Sin `--tickers` se actualiza el universo activo, o el CSV original si todavía no se activó un dataset. Sin `--retrain` sólo se publican datos en modo legacy; si ya hay un release activo se exige construir otro bundle coherente. `--retrain` construye un candidato aislado, valida y activa al terminar; `--candidate-only` evita activación. Ver [publicación y rollback](publication.md). Los entrenamientos de LSTM, GARCH y MLP consumen `load_returns()`, que resuelve la versión activa.

Cada descarga completa genera `Data/dynamic/<version>/` con precios ajustados **en USD**, precios originales (`prices_native.csv`), retornos logarítmicos **en USD** por rueda, metadatos, auditoría de cambios (`fx_rates.csv`) y manifiesto. `Data/active_dataset.json` se reemplaza atómicamente después de validar todos los activos. Si falla un ticker, se conserva la versión anterior. Se vuelve a descargar el historial para incorporar revisiones de dividendos/splits. Se excluye la fecha UTC en curso para evitar velas diarias incompletas. No se rellenan precios o retornos de feriados: cada activo conserva sus propias ruedas. La descarga puede fallar por límites de Yahoo o activos sin historia diaria; no se sustituye silenciosamente un ticker.

La identificación guarda tipo Yahoo, tipo normalizado (incluido fondo de bonos), nombre, categoría, sector, país, bolsa y moneda. País y sector no informados quedan como desconocidos. El domicilio del emisor o la bolsa de cotización no se presentan como exposición geográfica de un fondo. Las categorías de renta fija permiten identificar ETFs de bonos como TLT/IEF/HYG. La conversión a USD ocurre antes de construir los retornos; los metadatos conservan la moneda original y documentan esa conversión. Los bloques se leen dinámicamente del metadata. Activos fuera de las categorías originales reciben `OTHER`; se pueden usar nombres personalizados en el CSV. La UI muestra los bloques disponibles y conserva una opción histórica de cinco bloques. Revisar objetivos y capacidad (`n_activos * max_weight >= target`) antes de cambiar de mercado.

## Moneda común: USD

Todos los modelos y el optimizador reciben retornos sobre precios expresados en **USD**. El país del activo no determina el cambio: se usa la moneda de cotización informada por Yahoo. Una acción argentina en ARS se convierte; su ADR en USD permanece igual.

Para ARS, se usa **CCL vendedor en ARS por USD**: `precio_USD(t) = precio_ARS(t) / CCL(t)`. Después se calcula `log(precio_USD(t) / precio_USD(t-1))`. Por lo tanto `retorno_log_USD = retorno_log_ARS − variación_log_CCL`. Si la acción sube 10% en pesos pero el CCL sube 20%, el retorno simple en USD es `1.10 / 1.20 − 1 = −8.33%`. No alcanza con dividir el retorno porcentual por el cambio de hoy.

El CCL se descarga de [ArgentinaDatos, serie contado con liquidación](https://api.argentinadatos.com/v1/cotizaciones/dolares/contadoconliqui), campo `venta`. Se conserva el CSV local `Data/fx_CCL.csv` para extender el historial anterior al comienzo de esa serie remota (cierre local; cambio de fuente/convenio identificado en la auditoría). Si la descarga falla, sólo puede usarse la fuente local si supera los controles de cobertura y antigüedad; no se reemplaza por dólar oficial. Para fijar una serie propia y reproducible, pasar `--ccl-file ruta/ccl.csv` (columnas `fecha,cierre` o `date,rate`, expresadas en ARS por USD). El archivo explícito se usa sin sobrescribirlo con datos remotos.

Otras monedas se convierten con la serie histórica Yahoo `<MONEDA>USD=X` (USD por unidad de moneda), multiplicando el precio local por el cambio. Se contemplan cotizaciones en peniques/agorot/centavos como `GBp`, `GBX`, `ILA` y `ZAc`, aplicando su escala antes del cambio.

Cada precio usa una tasa del mismo día o anterior, nunca posterior. La antigüedad máxima por defecto es **siete días calendario** (`--max-fx-age-days`). No se publican activos con moneda desconocida, cambios faltantes, no positivos o vencidos; se conserva el dataset anterior. Esto permite cubrir fines de semana sin prolongar indefinidamente una cotización vieja. La moneda original, tasa usada, fecha, fuente, escala y moneda base quedan guardadas.

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

Son doce candidatos por conjunto de features. La producción compara además las features completas contra retornos solos, eligiendo también las entradas por validación. El laboratorio permite habilitar esa comparación y seleccionar features propias. No hay una arquitectura ganadora impuesta para todos los activos.

Se conserva el orden temporal 70/15/15 y se purgan cuatro secuencias en cada borde para que los targets de cinco ruedas no compartan retornos entre particiones. Se usa early stopping, clipping de gradientes y semilla 42. Se selecciona el menor MSE de validación. El test sólo evalúa al ganador; también se informa RMSE contra retorno cero y media de entrenamiento. El filtro posterior del optimizador usa métricas de **validación**: dirección >0,52 y amplitud >0,15 (ambos configurables), además de que el LSTM supere el RMSE de la media de entrenamiento sobre validación (`use_for_optimizer`). Los umbrales y motivos por ticker se guardan con cada optimización. Si falla el filtro, conserva la media histórica del activo. Los errores por ticker se guardan en `metrics.csv`.

Después de evaluar el ganador sobre test, se reajusta su especificación con **todo el historial etiquetado disponible**, incluyendo los últimos datos. El presupuesto de épocas se fija con el early stopping de validación; no se vuelve a seleccionar usando test. El checkpoint de evaluación se conserva separado. Las métricas publicadas pertenecen al modelo evaluado antes del reajuste, mientras que el pronóstico futuro usa el modelo reajustado y sus nuevos escaladores.

Cada corrida guarda candidatos, métricas, curvas de pérdida, predicciones históricas, pronóstico futuro y checkpoint con `selected_config`, escaladores y orden de features. Para reconstruir un checkpoint nuevo, crear `ReturnLSTM(input_size=len(meta['selected_features']), hidden_sizes=meta['selected_config']['hidden_sizes'], activation=meta['selected_config']['activation'], feature_indices=meta['selected_config'].get('feature_indices'), dropout=meta['dropout'])` y cargar el state dict. Los checkpoints antiguos con capas homogéneas conservan sus claves si se usa el constructor original.

## GARCH

Se mantiene GJR-GARCH(1,1,1) con innovaciones Student-t y media cero. Se publican varianza/volatilidad diaria y a cinco ruedas. GARCH pronostica **riesgo**, no retorno esperado. La varianza acumulada suma las cinco varianzas condicionales bajo la especificación de media cero. La covarianza del optimizador sigue usando `var_pred`/`vol_pred` diarios.

`garch_diagnostics.json` incluye convergencia, log-likelihood, AIC/BIC, coeficientes, errores estándar, p-values, intervalos al 95%, persistencia y condición de estacionariedad. Ljung–Box evalúa residuos estandarizados y sus cuadrados; ARCH-LM busca heterocedasticidad residual (umbral orientativo 5%). Pasar estos tests no garantiza capacidad predictiva.

Un backtest de 20 orígenes móviles reajusta con datos anteriores a cada observación y compara GARCH contra EWMA con QLIKE y MSE de varianza. El proxy observado es retorno logarítmico diario al cuadrado, una medición ruidosa. Se reportan fallos y número de evaluaciones válidas. No se optimizan parámetros con el test. Si falla el ajuste final o no converge, se usa EWMA y se registra el motivo explícitamente.

Las predicciones nuevas incluyen versión del dataset; el optimizador rechaza predicciones LSTM/GARCH de otro dataset. Después de actualizar datos hay que reentrenar antes de usar predicciones de otra versión. Señales individuales faltantes o inválidas mantienen el activo con fallback histórico; la política required exige cobertura completa.

## Experimentos y automatización

```bash
python scripts/run_model_study.py --tickers AAPL TLT GGAL.BA --epochs 30
python -m pytest -q
```

El estudio compara retornos solos y estadísticas de retornos sobre un snapshot fijo. Guarda las comparaciones y selecciona por RMSE de validación en `outputs/model_search/`, sin activar corridas experimentales. Es evidencia sobre esos activos, fechas, presupuesto de épocas y semilla; no demuestra una ventaja universal. Para otro universo debe repetirse la selección.

`.github/workflows/yahoo-models.yml` prepara descarga y entrenamiento diarios a las 03:30 UTC de martes a sábado, además de ejecución manual con tickers propios. Se activará al subir el archivo a GitHub y habilitar Actions. El workflow conserva bundles candidatos como artefactos de 30 días, tras controles de calidad; **no modifica el contenedor de serving**. `scripts/pull_release.py` descarga una corrida exitosa de main, verifica contenido/calidad y opcionalmente activa el bundle sobre el volumen compartido. Ver [publicación](publication.md). Cada ejecución programada parte del universo CSV versionado del repositorio; una selección manual no cambia los futuros cron. Para persistir un universo diferente, editar el comando o el CSV versionado. En un servidor persistente puede programarse `run_refresh_dataset.py --retrain` con cron; usará el universo activo local.
