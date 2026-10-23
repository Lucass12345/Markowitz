# Backtesting temporal

Ejemplo reproducible con el dataset USD activo:

```bash
python scripts/run_portfolio_backtest.py --tickers AAPL SPY TLT \
  --train-window 252 --oos-days 15 --rebalance-every 5 --lstm-epochs 2 \
  --commission-bps 5 --output outputs/backtesting/mi-comparacion
```

El universo se fija antes de observar el test. Se excluyen únicamente fechas en las
que ningún activo seleccionado cotizó. Una observación faltante de un activo mantenido
rechaza la simulación: para mercados distintos se necesita una política explícita de
valuación con precios, que este motor no inventa. El mismo calendario OOS se aplica a
todas las estrategias. Las ventanas incluyen el cierre de decisión; el primer retorno
aplicado es el de la rueda siguiente. No se cargan predicciones de producción actuales.

Los callbacks de clásico y GJR-GARCH actualizan estimaciones exclusivamente con historia;
LSTM reentrena una especificación fija y predice la media diaria de cinco ruedas.
Los hiperparámetros, épocas y universo no se seleccionan con el tramo OOS. El benchmark
buy-and-hold entra una vez y deja derivar los pesos; equal-weight se rebalancea.
Los retornos realizados agregan retornos simples de cada activo y después convierten
el resultado de cartera a logaritmos. Los costos se aplican al nocional negociado.
Turnover: media norma L1 por rebalanceo, entrada inicial 1. Se informa también el
nocional L1 completo que determina las comisiones.

`docs/verification/walk-forward-smoke.json` conserva la ejecución real de tres decisiones
sobre 15 ruedas, incluyendo especificaciones y métricas netas. Es una verificación del
flujo, demasiado breve para concluir superioridad financiera; no corrige survivorship
bias. Los archivos de decisiones y curvas se generan en el directorio de salida.
Las pruebas alteran datos futuros para comprobar la invariancia de decisiones anteriores,
verifican ventanas sin solapamiento, deriva, costos, drawdown inicial y rechazo de marcas
faltantes. Debe ampliarse el período OOS antes de comparar rendimiento de modelos.
