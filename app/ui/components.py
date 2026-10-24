from __future__ import annotations

from datetime import date
import json
import os
from urllib.parse import quote
import pandas as pd
import streamlit as st
from app.api_client import APIError

SECTIONS = ['Dashboard', 'Data Management', 'Model Laboratory', 'Portfolio Optimizer',
            'Portfolio Analytics', 'Backtesting', 'History & Experiments', 'System & Diagnostics']
ICONS = ['◈', '▦', '◉', '⌁', '◫', '↗', '≡', '◌']
TERMINAL = {'completed', 'failed', 'cancelled', 'canceled', 'timed_out'}


def items(payload):
    if isinstance(payload, list):
        return payload
    return payload.get('items', []) if isinstance(payload, dict) else []


def fetch(api, path, default=None, **params):
    try:
        return api.get(path, **params)
    except APIError as exc:
        st.warning(str(exc))
        return default


def readonly(system=None):
    role = ((system or {}).get('actor') or {}).get('role')
    return role in {'reader', 'read_only'} or bool(os.getenv('MARKOWITZ_READ_API_KEY') and not os.getenv('MARKOWITZ_API_KEY'))


def shell(section='Dashboard'):
    st.markdown('''<style>
    .block-container {max-width:1600px;padding-top:1.6rem;padding-bottom:2.5rem}
    [data-testid="stSidebar"] {border-right:1px solid #23364d}
    [data-testid="stMetric"] {background:#101e30;border:1px solid #23364d;border-radius:12px;padding:1rem}
    [data-testid="stMetricLabel"] {color:#94a3b8;font-size:.83rem;letter-spacing:.02em}
    [data-testid="stMetricValue"] {font-size:1.7rem}
    h1 {letter-spacing:-.04em} h2 {letter-spacing:-.02em}
    .terminal-eyebrow {color:#38bdf8;font-size:.78rem;letter-spacing:.18em;font-weight:700}
    .terminal-muted {color:#94a3b8;font-size:.9rem}
    </style>''', unsafe_allow_html=True)
    st.sidebar.markdown('### ◈ MARKOWITZ\nResearch Terminal · 2.0')
    selected = st.sidebar.selectbox('Espacio de trabajo', SECTIONS,
        index=SECTIONS.index(section) if section in SECTIONS else 0,
        format_func=lambda value: f'{ICONS[SECTIONS.index(value)]}  {value}', key='workspace_section')
    st.sidebar.caption('Datos USD · Targets diarios logarítmicos · Ejecución por workers')
    st.sidebar.caption('Uso informativo y de investigación. No constituye asesoramiento de inversión.')
    st.markdown('<div class="terminal-eyebrow">QUANTITATIVE RESEARCH · MLOPS</div>', unsafe_allow_html=True)
    st.title(selected)
    return selected


def dataset_selector(datasets, system, key='dataset_version'):
    rows = items(datasets)
    options = [str(row.get('version') or row.get('id')) for row in rows]
    active = system.get('active_dataset') or {}
    active_version = str(active.get('version') or active.get('id') or '') if isinstance(active, dict) else str(active)
    if not options:
        st.info('No hay datasets registrados. Importá el histórico o creá una actualización desde Data Management.')
        return None
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    chosen = st.selectbox('Dataset versionado', options,
        index=options.index(active_version) if active_version in options else 0, key=key,
        help='La versión enviada queda fijada al crear cada trabajo.')
    record = next(row for row in rows if str(row.get('version') or row.get('id')) == chosen)
    as_of = record.get('as_of') or (record.get('manifest_json') or record.get('manifest') or {}).get('latest_observation')
    st.caption(f'Versión fijada: {chosen} · Última observación: {as_of or "no informada"}')
    return chosen


def enqueue(api, path, payload, kind, label):
    try:
        job = api.post(path, payload)
    except APIError as exc:
        st.error(str(exc))
        return None
    status = job.get('status')
    job_id = job.get('job_id') or (job.get('job') or {}).get('id')
    if status in TERMINAL | {'queued', 'pending', 'running', 'retrying', 'publishing', 'cancel_requested'}:
        job_id = job_id or job.get('id')
    if job_id:
        tracked = dict(st.session_state.get('tracked_jobs', {}))
        tracked[str(job_id)] = {'kind': kind, 'label': label, 'response': job}
        st.session_state['tracked_jobs'] = tracked
        st.success(f'Trabajo recibido: {job_id}. El worker ejecutará la operación.')
    else:
        st.success('La API confirmó la operación.')
        st.json(job)
    return job


def store_result(kind, result):
    results = dict(st.session_state.get('research_results', {}))
    results[kind] = result
    st.session_state['research_results'] = results
    if kind == 'optimization':
        st.session_state['last_api_response'] = result
        st.session_state['last_result'] = result.get('result', result)
        st.session_state['last_run_id'] = result.get('run_id') or result.get('optimization_id') or result.get('experiment_id') or result.get('id')
        st.session_state['last_mlflow_run_id'] = result.get('mlflow_run_id')
        st.session_state['last_mlflow_error'] = result.get('mlflow_error')


@st.fragment(run_every='2s')
def job_monitor(api, system=None):
    tracked = dict(st.session_state.get('tracked_jobs', {}))
    if not tracked:
        return
    changed = False
    with st.expander('Actividad y trabajos', expanded=True):
        for job_id, entry in list(tracked.items())[-8:]:
            previous = entry.get('response', {})
            if previous.get('status') in TERMINAL and entry.get('observed_terminal'):
                job = previous
            else:
                try:
                    job = api.get(f'/jobs/{quote(job_id, safe="")}')
                except APIError as exc:
                    st.warning(f'{entry["label"]}: {exc}')
                    continue
            status = str(job.get('status', 'pending'))
            st.markdown(f'**{entry["label"]}** · `{job_id}` · **{status}**')
            progress = job.get('progress', 0) or 0
            st.progress(min(100, max(0, int(progress))) / 100, text=job.get('stage') or status)
            if status not in TERMINAL:
                if st.button('Solicitar cancelación', key=f'cancel_{job_id}', disabled=readonly(system)):
                    try:
                        api.post(f'/jobs/{quote(job_id, safe="")}/cancel')
                        st.info('Cancelación solicitada. Se aplicará en un punto seguro del trabajo.')
                    except APIError as exc:
                        st.error(str(exc))
            elif not entry.get('observed_terminal'):
                entry['observed_terminal'] = True
                changed = True
                if status == 'completed':
                    store_result(entry['kind'], job.get('result_json') or job.get('result') or {})
                    st.session_state.pop('last_job_error', None)
                else:
                    st.session_state['last_job_error'] = job.get('error') or f'Trabajo {status}. Podés reenviar la solicitud.'
            if status in {'failed', 'timed_out'}:
                st.error(job.get('error') or 'Falló el trabajo; consultá sus logs.')
            if st.button('Consultar logs', key=f'logs_{job_id}'):
                try:
                    logs = api.get(f'/jobs/{quote(job_id, safe="")}/logs')
                    st.dataframe(pd.DataFrame(items(logs)), hide_index=True, use_container_width=True)
                except APIError as exc:
                    st.error(str(exc))
            entry['response'] = job
            tracked[job_id] = entry
    st.session_state['tracked_jobs'] = tracked
    if changed:
        st.rerun()


def artifact_downloads(api, record, key='artifacts'):
    artifacts = record.get('artifacts') or []
    if isinstance(artifacts, dict):
        artifacts = artifacts.get('items') or [artifacts]
    if record.get('artifact_id'):
        artifacts = [{'id': record['artifact_id'], 'name': 'Bundle de la corrida'}, *artifacts]
    if not artifacts and record.get('artifact_uri') and (record.get('id') or record.get('experiment_id')):
        artifacts = [{'id': record.get('id') or record['experiment_id'], 'name': 'Reporte reproducible de la corrida',
                      'filename': 'platform_result.json'}]
    available = [entry for entry in artifacts if isinstance(entry, dict) and (entry.get('id') or entry.get('artifact_id'))]
    if not available:
        st.caption('La API no publicó referencias descargables para esta vista.')
        return
    selected = st.selectbox('Artefacto', range(len(available)),
        format_func=lambda index: available[index].get('name') or available[index].get('path') or available[index].get('filename') or available[index].get('id'), key=f'{key}_selection')
    artifact = available[selected]
    artifact_id = artifact.get('id') or artifact.get('artifact_id')
    if st.button('Preparar descarga autorizada', key=f'{key}_fetch'):
        try:
            content, content_type = api.download(artifact_id, artifact.get('path'))
            st.download_button('Descargar artefacto', content,
                file_name=artifact.get('filename') or f'artifact_{artifact_id}.bin', mime=content_type,
                key=f'{key}_download')
        except APIError as exc:
            st.error(str(exc))


def json_editor(label, value, key):
    raw = st.text_area(label, json.dumps(value, indent=2), height=180, key=key)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError('La configuración debe ser un objeto JSON')
        return parsed
    except (ValueError, TypeError) as exc:
        st.error(f'JSON inválido: {exc}')
        return None


def date_text(value):
    return value.isoformat() if isinstance(value, date) else value
