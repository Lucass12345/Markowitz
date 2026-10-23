# Publicación de datos y modelos

Una corrida de Actions produce un **candidato**, no modifica un contenedor remoto.
El workflow guarda el bundle completo como artifact privado de GitHub. La cuenta que
lo descarga necesita permiso de lectura de Actions en este repositorio. No se aceptan
artefactos de PR/forks para activación automática: `pull_release.py` exige una corrida
exitosa de `main`. `gh` debe estar instalado y autenticado en el host que realiza el pull.

## Flujo

```bash
# Construcción local/entrenamiento aislado. Si falla, los punteros siguen intactos.
python scripts/run_refresh_dataset.py --tickers AAPL SPY TLT --retrain --candidate-only
# Importar una copia transportada mediante un canal autenticado (o volumen compartido).
python scripts/manage_release.py import /ruta/candidato --data-dir Data
python scripts/manage_release.py activate ID_SHA256 --data-dir Data
# Alternativa desde el host de serving: descarga autenticada + validación + activación.
python scripts/pull_release.py --run-id ID_ACTIONS --data-dir Data --activate
# Rollback atómico al bundle anterior.
python scripts/manage_release.py rollback --data-dir Data
```

Ejecutar el pull sobre el mismo volumen `Data` que lee FastAPI; por ejemplo un job
programado del host con `gh` instalado y credencial de sólo lectura de Actions, o un
contenedor de mantenimiento que monte ese volumen. El ID de la corrida exitosa puede
obtenerse mediante `gh run list --workflow yahoo-models.yml --branch main --status success`.
La descarga sin `--activate` deja el bundle instalado para revisión. No habilita acceso
público a GitHub ni a MinIO. Usar TLS y credenciales fuera del repositorio para transporte.

`--retrain` sin `--candidate-only` activa localmente sólo después de completar todas
las etapas. El entrenamiento no publica punteros intermedios ni cambia los modelos activos.
Una vez activado un release, una actualización aislada de datos se rechaza: se necesita
un nuevo bundle coherente. Los modos LSTM/GARCH usan las predicciones del release fijado
al leer el dataset; para cambiar de laboratorio se debe publicar otro candidato.
MLP conserva su selección existente y debe corresponder a la versión vigente.

## Controles y persistencia

El bundle incluye datos USD y auditoría FX, metadata, predicciones, métricas de validación,
configuración, checkpoints y diagnósticos. Antes de sellar, importar y activar se validan
historia, moneda, solapamiento, cobertura y señales finitas con fechas compatibles.
Por defecto al menos 25% de señales LSTM deben superar dirección >0,52, amplitud >0,15
y RMSE menor al baseline **en validación**. GARCH requiere cobertura de volatilidad diaria
válida; conserva el fallback EWMA ya existente. Test nunca selecciona candidatos.
El porcentaje se configura al construir y queda registrado; no demuestra superioridad OOS.

Se comprueban inventario y SHA-256; se rechazan enlaces y rutas fuera del bundle. Estos
hashes detectan corrupción, no sustituyen la autenticación del origen. No importar bundles
de terceros no confiables. `active_release.json` cambia mediante rename atómico y conserva
el ID anterior. Los loaders fijan el bundle para toda la corrida. Respaldar `Data/releases`
y el puntero junto con los snapshots de optimización; no sobrescribir bundles sellados.
Si falla descarga, entrenamiento, calidad o verificación, no se cambia el release activo.
La primera activación no tiene un release anterior: conservar el dataset/predicciones legacy
para una reversión operativa manual antes de adoptar esta modalidad.

Pruebas offline: activación y rollback, lectura de señales fijadas aunque cambie el puntero,
rechazo de mala validación pese a test perfecto, corrupción y archivos con path traversal.
No se ejecutó un workflow remoto ni se activó el dataset real de este entorno.

Los campos de ruteo y controles de calidad de `release.json` deben coincidir con
`candidate.json`, incluido en el digest. El pull sólo admite eventos schedule/workflow_dispatch,
repositorio de origen esperado, workflow Yahoo de main y commit del candidato igual al SHA
del run. Reactivar el mismo ID es idempotente y no pierde el anterior para rollback.
