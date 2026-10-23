# Corrección de moneda: retornos en USD

El dataset activo contiene **40 activos expresados en USD**: 15 cotizaciones ARS convertidas por CCL histórico y 25 cotizaciones USD sin conversión. La última fecha es 2026-10-20. Las monedas se identifican por la cotización de Yahoo, no por país o sufijo: un ADR argentino en USD no se dolariza nuevamente.

Para ARS, `P_USD(t) = P_ARS(t) / CCL(t)` y luego `r_USD(t) = log(P_USD(t) / P_USD(t-1))`. El CCL utiliza el campo vendedor de ArgentinaDatos, expresado en ARS por USD. Cada precio se alinea con una tasa del mismo día o anterior; antigüedad máxima configurada: siete días calendario. En esta actualización los 15 activos ARS tuvieron tasas de la misma fecha (antigüedad cero).

Ejemplo real de la verificación, GGAL.BA:

| Fecha | Precio ARS | CCL ARS/USD | Precio USD | Retorno logarítmico en USD |
| --- | --- | --- | --- | --- |
| 2026-10-19 | 5690.00 | 1620.50 | 3.511262 | -0.022790 |
| 2026-10-20 | 5765.00 | 1625.60 | 3.546383 | 0.009953 |

El precio en USD refleja simultáneamente el movimiento del activo y del CCL. Las cotizaciones nativas se conservan en `prices_native.csv`; `prices.csv` y `returns.csv` están en USD. `fx_rates.csv` registra cotización, fecha, antigüedad, fuente, unidad y factor aplicado por activo. [Ejemplo auditado](currency_check.csv), [manifiesto](manifest.json), [metadatos](assets_metadata.csv).

Para otras monedas se multiplica el precio por el cambio histórico de Yahoo `<MONEDA>USD=X`, con tratamiento de peniques/agorot/centavos. Ante moneda desconocida, cambio inválido, faltante o vencido, se rechaza la publicación y se conserva el snapshot previo. No se sustituye CCL por dólar oficial. La serie local `Data/fx_CCL.csv` sólo extiende el historial anterior a la fuente remota o sirve de respaldo si cumple los controles; no se prolonga indefinidamente.

## Modelos recalculados

Se repitió la selección LSTM de **GGAL.BA con retornos USD**: 24 candidatos, dos conjuntos de features, arquitecturas 64→64/120→64, activaciones ReLU/Tanh/GELU y dos configuraciones Adam, máximo 30 épocas. Ganó por validación **120→64, ReLU, features ampliadas, Adam lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0**. Su RMSE test es 0.013270 frente a 0.013525 de la media de entrenamiento. Esta selección sustituye la anterior calculada en ARS; no demuestra una ventaja universal.

AAPL y TLT ya cotizaban en USD; se verificó que sus series no cambiaron y se conservaron sus configuraciones seleccionadas. Se reentrenaron los tres ganadores sobre todo el historial etiquetado disponible del snapshot USD, después de evaluar. Los pesos de evaluación y de inferencia permanecen separados. Los resultados de prueba describen el checkpoint evaluado antes de ese reajuste.

[Ganadores de los tres activos](selected_models.csv), [comparación de features GGAL en USD](feature_comparison.csv), [24 candidatos GGAL](lstm_candidates_GGAL_USD.csv), [configuración del estudio USD](study.json) y [pronósticos en USD](selected_forecasts.csv). Los checkpoints locales actualizados están identificados en `outputs/model_search_usd/selected_checkpoints_all.json`; el estudio permanece separado de la selección operativa de todo el universo.

También se regeneraron GARCH y su backtest para los 40 activos sobre retornos USD. Los 40 ajustes convergieron; 28 pasan conjuntamente los controles de residuos, 39 cumplen persistencia menor que uno y 19 logran menor QLIKE que EWMA en las 20 ruedas del backtest. [Resumen GARCH en USD](garch_summary.csv).

Los snapshots dinámicos previos sin moneda base verificable y las predicciones de versiones anteriores se rechazan al cargar. LSTM, GARCH y MLP comparten este criterio; MLP también conserva procedencia del snapshot en sus salidas. Para usar una corrida MLP antigua hay que volver a entrenarla con el dataset USD, no asignarle una moneda nueva al archivo.

## Verificación

**53 pruebas pasaron**, incluidas 23 pruebas de conversión, tres de publicación del dataset USD y diez de procedencia monetaria de MLP. Se verificó además la identidad `r_USD = r_ARS − Δlog(CCL)` para todas las series ARS reales y que las otras 25 conservaron sus precios USD. Se validaron división/multiplicación correctas, unidades, ausencia de tasas futuras, antigüedad máxima, rechazo de monedas desconocidas, activación atómica y bloqueo de snapshots mixtos antiguos.

Las pruebas usan las versiones fijadas del repositorio; las versiones efectivas del estudio se registran en `study.json`. No se desplegó Streamlit/Postgres ni se activó el cron remoto de GitHub Actions.
