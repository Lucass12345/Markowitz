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
