# Trabajos de optimización

`POST /optimize` conserva la interfaz síncrona. `POST /jobs` acepta el mismo JSON y
responde HTTP 202 con un ID. `GET /jobs/{id}` devuelve pending, running, completed o
failed, resultado, fechas y duración. Los errores de configuración conservan su código
422 en `error_status`; una cola llena responde 503 con Retry-After. Se puede reenviar
un trabajo fallido. Los tracebacks se guardan en SQL para diagnóstico y no se exponen
por el endpoint de consulta. Streamlit envía el trabajo y consulta su estado mediante
un fragmento, sin esperar a toda la optimización en una petición bloqueante.

Decisión: ThreadPoolExecutor con dos trabajadores y cuatro pendientes, sin Redis/Celery.
Es apropiado para **un solo proceso de FastAPI** en este despliegue. No usar varios workers
ni réplicas que compartan esta cola: el arranque marca pending/running anteriores como
failed porque no sobreviven los hilos. Antes de escalar, migrar a un broker y workers
externos. No se promete cancelación de código nativo en un hilo: una operación atascada
ocupa un trabajador y necesita reinicio operativo; la UI puede seguir consultando y la
capacidad limitada evita acumulación indefinida. Training se ejecuta por separado mediante
los scripts/workflow; los trabajos de la API sólo estiman y optimizan con señales existentes.

Se reutilizan trabajos clásicos idénticos durante una hora si configuración, contenido
exacto de retornos/metadata, versión del código y dataset coinciden. La etiqueta también
forma parte de la clave, para preservar la semántica de las corridas. La comparación de
dependencias al terminar evita almacenar un resultado bajo una clave que cambió durante
la ejecución. Los modos IA y covarianza GARCH omiten el caché porque la selección activa
de modelos en SQL puede variar sin cambiar los CSV. No se cachea un error.

La tabla `optimization_jobs` se crea aditivamente con `init_db`, junto con la tabla de
provenance. Respaldar SQL. En producción, proteger ambos endpoints con la misma API key
que `/optimize`; para usuarios múltiples se necesitaría aislamiento por propietario y
un sistema de identidad. No compartir esta API key administrativa con usuarios finales.
