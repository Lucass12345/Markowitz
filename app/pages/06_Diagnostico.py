import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
import requests
import streamlit as st
from app.api_client import api_headers

st.set_page_config(page_title='Diagnóstico',layout='wide')
st.title('Diagnóstico del dataset y modelos')
st.caption('Estado observado por el backend de serving. Un servicio alcanzable no confirma sincronización de artefactos.')
base=os.getenv('API_BASE_URL','http://127.0.0.1:8000')
try:
    response=requests.get(f'{base}/diagnostics',headers=api_headers(),timeout=30)
    response.raise_for_status()
    report=response.json()
except (requests.RequestException,ValueError) as exc:
    st.error(f'No se pudo consultar el diagnóstico: {exc}')
    st.stop()
st.subheader('Dataset activo')
st.json(report.get('dataset',{}))
st.subheader('Modelos vigentes y cobertura')
for name,model in report.get('models',{}).items():
    with st.expander(name.upper(),expanded=True):
        signals=model.get('signals',[])
        if signals:st.dataframe(pd.DataFrame(signals),hide_index=True,use_container_width=True)
        st.json({key:value for key,value in model.items() if key!='signals'})
st.subheader('Disponibilidad de servicios')
st.json(report.get('services',{}))
st.subheader('Última sincronización registrada')
st.json(report.get('tracking') or {'status':'no_recorded_sync'})
