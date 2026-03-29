# Markowitz

Construcción y preprocesamiento del dataset
1) Universo de activos (41 activos)
1.1 Acciones argentinas (BYMA) — 15 tickers (Yahoo Finance con sufijo .BA)

GGAL.BA, YPFD.BA, PAMP.BA, TXAR.BA, ALUA.BA
CEPU.BA, LOMA.BA, BMA.BA, BBAR.BA, SUPV.BA
COME.BA, EDN.BA, TGNO4.BA, TGSU2.BA, VALO.BA

Racional: acciones locales líquidas representativas de sectores clave (financieros, energía y materiales).

1.2 Acciones de EE.UU. (universo CEDEAR) — 15 tickers (Yahoo Finance)

AAPL, MSFT, AMZN, GOOGL, META
NVDA, TSLA, BRK-B, JPM
V, MA, KO, PEP, XOM, UNH

Racional: mega caps globales con diversificación sectorial.

1.3 ETFs globales — 11 tickers (Yahoo Finance)

SPY, QQQ, IWM, EEM, EFA
GLD, SLV, TLT, IEF, HYG, VNQ

Racional: exposición a factores y clases de activos (equity global, small caps, emergentes, commodities, bonos y real estate).

2) Fuentes de datos
2.1 Precios de activos

Fuente: Yahoo Finance mediante la librería de Python yfinance.
Variable utilizada: precios ajustados diarios (auto_adjust=True).

Se construyó una matriz diaria donde:

filas = fechas

columnas = activos

2.2 Tipo de cambio

Fuente: archivos CSV históricos:

DOLAR CCL - Cotizaciones historicas.csv

DOLAR MEP - Cotizaciones historicas.csv

Se seleccionó el dólar CCL por representar el tipo de cambio financiero relevante para inversión internacional y valuación de CEDEARs.

3) Ventana temporal del estudio

El dataset final se restringió a:

01-01-2018 → 31-12-2025

Tamaño final:

2063 observaciones diarias

41 activos

Esta ventana evita problemas de liquidez previos y garantiza consistencia temporal.

4) Desalineación de calendarios de trading

Los mercados argentino y estadounidense poseen feriados distintos.
Esto generaba precios faltantes en ciertas fechas.

Sin corrección, esto produciría:

retornos artificialmente iguales a cero

volatilidad subestimada

correlaciones sesgadas

5) Sincronización de precios (alineación de calendarios)

Se aplicó el método Last Observation Carried Forward (LOCF):

prices = prices.ffill()


Interpretación:
Si un mercado está cerrado, el precio permanece igual al último valor observado.

Esto genera un calendario unificado equivalente a la unión de días de trading de ambos mercados.

6) Inconsistencia de moneda

Inicialmente:

Acciones BYMA en ARS

CEDEARs en ARS

ETFs en USD

Esta mezcla de monedas introduce sesgos en la optimización de portafolios.

7) Normalización de moneda (todos los precios en USD)

Todos los precios fueron convertidos a USD usando el dólar CCL:

Precio_USD(t) = Precio_ARS(t) / CCL(t)

El CCL fue alineado al calendario del panel y completado mediante forward fill.

Efectos:

Acciones argentinas → convertidas a USD

CEDEARs → se elimina el componente cambiario implícito

ETFs → ya estaban en USD

8) Cálculo de retornos

Se calcularon retornos logarítmicos diarios:

r_t = ln(P_t / P_{t-1})


Este enfoque es estándar en:

optimización de portafolios

modelado de riesgo

modelos de machine learning financiero

9) Archivos finales generados

Se generaron los datasets definitivos:

prices_USD.csv → precios diarios sincronizados en USD

returns_USD.csv → retornos logarítmicos diarios en USD

10) Notas metodológicas

El dataset final es:

multi-mercado y multi-activo

alineado por calendario mediante LOCF

consistente en moneda (USD)

basado en retornos logarítmicos

Se encuentra listo para:

optimización de portafolios (Markowitz)

forecasting de retornos y volatilidad

modelos de machine learning aplicados a finanzas

streamlit run: 
cd /d C:\Users\lucas\OneDrive\Documentos\GitHub\Markowitz
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe -m streamlit run app/streamlit_app.py --server.port 8502

http://127.0.0.1:8502

FAstapi: 
cd /d C:\Users\lucas\OneDrive\Documentos\GitHub\Markowitz
set "MLFLOW_TRACKING_URI=file:///C:/Users/lucas/OneDrive/Documentos/GitHub/Markowitz/mlruns"
set "MLFLOW_EXPERIMENT_NAME=markowitz-streamlit"
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe -m uvicorn api.main:app --reload

Link: http://127.0.0.1:8000/docs

mlflow:
cd /d C:\Users\lucas\OneDrive\Documentos\GitHub\Markowitz
C:\Users\lucas\anaconda3\envs\IIA_cuda_env\python.exe -m mlflow ui --backend-store-uri "file:///C:/Users/lucas/OneDrive/Documentos/GitHub/Markowitz/mlruns" --port 5005

http://127.0.0.1:5005

docker minio:
docker rm -f markowitz-minio
docker run -d --name markowitz-minio -p 9000:9000 -p 9001:9001 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin minio/minio server /data --console-address ":9001"
docker ps

a