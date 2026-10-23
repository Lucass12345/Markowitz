# Resultados iniciales del 21 de octubre de 2026

**Sustituidos para activos cotizados en ARS por la corrección monetaria posterior.** Este estudio inicial mezclaba retornos nativos de ARS y USD. Sus resultados de GGAL.BA, los diagnósticos del universo mixto y la selección de pesos no deben interpretarse como resultados en USD ni utilizarse como modelos vigentes. Los datos y predicciones correspondientes quedan rechazados por el control de versión. Los activos que ya cotizaban en USD no cambian su serie por esta conversión. Ver [resultados corregidos en USD](../2026-10-21-usd/README.md) y [criterio monetario](../../dynamic-models.md#moneda-común-usd).

Se actualizó el universo original de **40 activos** desde Yahoo Finance, con historial desde 2015 y última rueda admitida **2026-10-20**. La fecha UTC en curso se excluye para evitar una vela incompleta. El manifiesto y la identificación descargada están en [manifest.json](manifest.json) y [assets_metadata.csv](assets_metadata.csv). En diez fondos Yahoo no informó país ni sector: se conservan desconocidos y se informa su categoría, sin inventar exposición geográfica.

## LSTM: 72 ajustes comparados

AAPL, TLT y GGAL.BA representan una acción estadounidense, un ETF de bonos y una acción argentina. Se compararon dos conjuntos de entradas (retorno solo y 15 features de retornos), dos arquitecturas (64→64 y 120→64), tres activaciones densas (ReLU/Tanh/GELU) y dos configuraciones Adam. Se usaron ventanas de 40 observaciones, target promedio de los próximos cinco retornos logarítmicos, datos desde 2022-01-01, máximo 30 épocas, early stopping de ocho épocas y semilla 42.

La selección usa exclusivamente validación temporal, con cuatro secuencias purgadas en cada frontera. Los conjuntos de features usan las mismas fechas. El tramo de prueba se conserva para evaluar la selección, sin cambiar al ganador por su resultado en test.

| Activo | Entradas elegidas | Capas | Activación densa | Adam lr | RMSE validación | RMSE test | RMSE test de la media de entrenamiento |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | Retorno solo | 64→64 | GELU | 0.001 | 0.006934 | 0.007081 | 0.007176 |
| TLT | Retorno solo | 64→64 | Tanh | 0.0003 | 0.002446 | 0.002675 | 0.002657 |
| GGAL.BA | Estadísticas de retornos | 120→64 | ReLU | 0.0003 | 0.019423 | 0.014069 | 0.013644 |

Para lr 0.001 se usaron betas=(0.9,0.999), eps=1e-8 y weight_decay=0. Para lr 0.0003: betas=(0.9,0.99), eps=1e-7 y weight_decay=0.0001.

**La red 120→64 y las features ampliadas no ofrecen una mejora universal.** AAPL reduce RMSE de test un 1.32% respecto de la media de entrenamiento; TLT lo aumenta un 0.66% y GGAL.BA un 3.11%. En GGAL.BA las features ampliadas mejoran validación frente a retorno solo, pero empeoran test. No se altera la selección después de observar ese resultado: es evidencia de incertidumbre fuera de muestra, no una razón para usar test como validación.

Los tres ganadores superan a la media en validación. El optimizador conserva además sus filtros de precisión direccional y amplitud, ahora calculados sobre validación. La producción reajusta la especificación elegida sobre todo el historial etiquetado disponible después de evaluar, sin reutilizar sus pesos reajustados para reportar métricas de test.

Los pesos ganadores se reentrenaron con ese procedimiento y se guardaron localmente; `outputs/model_search/selected_checkpoints.json` identifica cada checkpoint y `selected_forecasts.csv` contiene sus pronósticos futuros. El estudio queda separado del universo operativo completo: **se comparó LSTM sobre tres activos, no sobre los 40**. El comando de producción ejecuta la selección por activo para el universo que se elija.

Esta corrida usa una sola semilla y una partición temporal, con un presupuesto acotado de épocas. Para sustentar una ventaja estable hace falta repetir en otros períodos y semillas. Los datos descargados son revisables por Yahoo; repetir sin el mismo snapshot puede cambiar resultados.

Datos completos: [todos los candidatos](lstm_candidates.csv), [ablación de features](feature_comparison.csv), [ganadores](selected_models.csv) y [configuración y versiones](study.json).

## GARCH: universo completo

Se ajustó GJR-GARCH(1,1,1)-t desde 2022-01-01 para los 40 activos y se evaluaron 20 pronósticos diarios fuera de muestra por activo, reajustando en cada origen sólo con información anterior. Los 40 ajustes finales convergieron, sin fallback a EWMA.

- 25 de 40 pasan conjuntamente Ljung–Box de residuos, Ljung–Box de residuos cuadrados y ARCH-LM al umbral de 5%.
- 39 de 40 cumplen persistencia menor que uno.
- 20 de 40 logran menor QLIKE que EWMA en las últimas 20 ruedas.

La convergencia no implica estacionariedad ni ausencia de estructura residual. QLIKE más bajo es mejor; el retorno al cuadrado es un proxy ruidoso de varianza, y 20 observaciones son una muestra corta. Se publican los diagnósticos y no se presenta GARCH como superior para todo el universo.

[Resumen por activo](garch_summary.csv). Los coeficientes, errores estándar, intervalos, pruebas de residuos y backtests detallados están en `outputs/predictions/garch_diagnostics.json`. Se mantienen las predicciones diarias del optimizador y se agregan varianzas/volatilidades a cinco ruedas.

## Verificación

Pasaron **17 pruebas** con PyTorch 2.6.0 CPU, ARCH 7.2.0, pandas 2.2.3, NumPy 2.1.3 y yfinance 1.7.0. Cubren horizonte exacto, purga temporal, causalidad de features, escalado, arquitecturas, activaciones, selección de entradas, activación atómica del dataset, feriados entre mercados, clasificación de fondos, reproducción del pronóstico desde el checkpoint, diagnósticos GARCH y carga operativa sin base de datos.

El estudio numérico se ejecutó con PyTorch 2.14.1 CPU y ARCH 8.0.0; las versiones efectivas se registran en `study.json`. Las pruebas adicionales con las versiones fijadas del repositorio verifican compatibilidad funcional, no identidad numérica entre versiones. No se verificó el despliegue de Streamlit/Postgres ni se activó GitHub Actions remotamente.
