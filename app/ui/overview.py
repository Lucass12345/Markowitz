from __future__ import annotations

from urllib.parse import quote
import pandas as pd
import streamlit as st

from app.api_client import APIError
from app.ui.components import artifact_downloads, dataset_selector, fetch, items, readonly, store_result


def _rows(value):
    if isinstance(value, dict) and 'items' not in value:
        return [{'name': name, **(record if isinstance(record, dict) else {'status': record})} for name, record in value.items()]
    return items(value)


def _table(value):
    rows = _rows(value)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    return rows


def render_dashboard(api, system, datasets):
    st.caption('Estado del entorno de investigación · Datos, modelos, carteras y ejecución reproducible en una sola vista.')
    active = system.get('active_dataset') or {}
    if not isinstance(active, dict):
        active = {'version': active}
    models = _rows(system.get('models') or [])
    workers = _rows(system.get('workers') or [])
    jobs = _rows(system.get('jobs') or [])
    services = _rows(system.get('services') or {})
    cols = st.columns(4)
    cols[0].metric('Dataset activo', active.get('version') or active.get('id') or 'Sin activar')
    cols[1].metric('Modelos activos', sum((row.get('state') or row.get('status')) == 'active' or bool(row.get('is_active')) for row in models))
    cols[2].metric('Workers observados', len(workers))
    cols[3].metric('Trabajos en curso', sum(row.get('status') in {'queued', 'pending', 'running', 'publishing', 'cancel_requested'} for row in jobs))
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader('Dataset y cobertura')
        if active:
            st.json({key: active.get(key) for key in ['version', 'status', 'as_of', 'generated_at', 'content_hash'] if key in active})
        versions = items(datasets)
        if versions:
            st.dataframe(pd.DataFrame([{key: row.get(key) for key in ['version', 'status', 'as_of', 'generated_at']} for row in versions[:8]]), hide_index=True, use_container_width=True)
        else:
            st.info('Importá el histórico o actualizá Yahoo Finance desde Data Management.')
        st.subheader('Actividad reciente')
        if jobs:
            st.dataframe(pd.DataFrame([{key: row.get(key) for key in ['id', 'kind', 'status', 'progress', 'stage', 'created_at']} for row in jobs[:8]]), hide_index=True, use_container_width=True)
        else:
            st.caption('No hay actividad reciente registrada.')
    with right:
        st.subheader('Servicios y alertas')
        if services:
            st.dataframe(pd.DataFrame(services), hide_index=True, use_container_width=True)
        for alert in system.get('alerts') or []:
            message = alert.get('message') or str(alert) if isinstance(alert, dict) else str(alert)
            st.warning(message)
        if not system.get('alerts'):
            st.caption('La API no reporta alertas activas.')
        st.subheader('Ruta de investigación')
        st.markdown('**01** Versioná datos y universo.\n\n**02** Entrená y validá modelos.\n\n**03** Construí carteras con restricciones.\n\n**04** Evaluá fuera de muestra con costos.')
    st.caption('Retornos: logarítmicos diarios · Moneda base: USD · Las predicciones y estimaciones no garantizan resultados futuros.')


def render_history(api, system, datasets):
    st.caption('Registro persistente de configuraciones, métricas, estado de publicación y artefactos. Las consultas se resuelven por API.')
    version = dataset_selector(datasets, system, key='history_dataset')
    params = {'dataset_version': version} if version else {}
    tabs = st.tabs(['Optimizaciones', 'Experimentos', 'Backtests', 'Trabajos', 'Histórico académico'])
    for tab, path, label in zip(tabs, ['/optimizations', '/experiments', '/backtests', '/jobs','/history/legacy'], ['optimización', 'experimento', 'backtest', 'trabajo','histórico']):
        with tab:
            rows = items(fetch(api, path, {'items': []}, **params))
            if not rows:
                st.info(f'No hay registros de {label} para esta selección.')
                continue
            flat = [{key: row.get(key) for key in ['id', 'kind', 'family', 'model_type', 'status', 'dataset_version', 'created_at', 'mlflow_run_id']} for row in rows]
            st.dataframe(pd.DataFrame(flat), hide_index=True, use_container_width=True)
            selected = st.selectbox(f'Inspeccionar {label}', [row['id'] for row in rows], key=f'history_{label}')
            detail = fetch(api, f'{path}/{quote(str(selected), safe="")}', {}) or {}
            st.json(detail)
            if path == '/optimizations' and st.button('Cargar en Portfolio Optimizer', key='history_load_optimization'):
                store_result('optimization', detail)
                st.success('Resultado cargado. Abrí Portfolio Optimizer para analizarlo.')
            if path == '/jobs':
                if st.button('Consultar logs del trabajo', key='history_job_logs'):
                    _table(fetch(api, f'/jobs/{quote(str(selected), safe="")}/logs', {'items': []}))
            elif path=='/history/legacy':
                st.caption('Valores históricos conservados sin recalcular. Las nuevas corridas guardan Sharpe diario y anual por separado.')
            else:
                artifact_downloads(api, detail, key=f'history_{label}_artifacts')


def render_system(api, system, datasets):
    st.caption('Estado observado por FastAPI. Un servicio disponible no confirma la publicación ni la sincronización de una corrida.')
    if st.button('Actualizar diagnóstico'):
        st.rerun()
    active = system.get('active_dataset') or system.get('dataset') or {}
    st.subheader('Dataset activo')
    st.json(active)
    st.subheader('Disponibilidad de servicios')
    st.json(system.get('services') or {})
    st.subheader('Última sincronización registrada')
    st.json(system.get('tracking') or {'status': 'no_recorded_sync'})
    tabs = st.tabs(['Workers y recursos', 'Modelos y cobertura', 'Trabajos y cancelación', 'Alertas'])
    with tabs[0]:
        workers = _table(system.get('workers') or [])
        if not workers:
            st.info('No hay workers observados. Los trabajos permanecerán en cola hasta que exista un worker compatible.')
        if any(str(row.get('device', '')).lower() == 'cuda' or row.get('gpu') for row in workers):
            st.caption('Existe capacidad GPU reportada por el backend.')
        else:
            st.caption('Sin capacidad GPU reportada. Seleccioná CPU o auto para entrenar.')
    with tabs[1]:
        _table(system.get('models') or [])
    with tabs[2]:
        rows = items(fetch(api, '/jobs', {'items': []}))
        st.dataframe(pd.DataFrame([{key: row.get(key) for key in ['id', 'kind', 'status', 'progress', 'stage', 'worker_id', 'updated_at']} for row in rows]), hide_index=True, use_container_width=True)
        if rows:
            selected = st.selectbox('Trabajo', [row['id'] for row in rows], key='system_job')
            job = next(row for row in rows if row['id'] == selected)
            c1, c2 = st.columns(2)
            if c1.button('Consultar logs', key='system_logs'):
                _table(fetch(api, f'/jobs/{quote(str(selected), safe="")}/logs', {'items': []}))
            if c2.button('Solicitar cancelación', disabled=readonly(system) or job.get('status') in {'completed', 'failed', 'cancelled', 'timed_out'}, key='system_cancel'):
                try:
                    api.post(f'/jobs/{quote(str(selected), safe="")}/cancel')
                    st.info('Cancelación solicitada; el worker la aplicará en un punto seguro.')
                except APIError as exc:
                    st.error(str(exc))
            if st.button('Reintentar con las mismas versiones',disabled=readonly(system) or job.get('status') not in {'failed','cancelled'},key='system_retry'):
                try:
                    api.post(f'/jobs/{quote(str(selected),safe="")}/retry')
                    st.success('Reintento encolado con el dataset y los modelos originales.')
                except APIError as exc:
                    st.error(str(exc))
    with tabs[3]:
        alerts = system.get('alerts') or []
        if alerts:
            _table(alerts)
        else:
            st.info('No hay alertas reportadas por FastAPI.')
