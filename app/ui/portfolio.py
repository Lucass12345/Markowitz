from __future__ import annotations

import json
from urllib.parse import quote
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.api_client import APIError
from app.ui.charts import (build_block_exposure_chart, build_comparison_df,
    build_frontier_figure, build_weights_bar, chart_layout, frame, pretty_pct)
from app.ui.components import artifact_downloads, dataset_selector, enqueue, fetch, items, readonly

MODES = {'classic': 'Histórico · Markowitz', 'ai_returns': 'MLP · retornos esperados',
         'ai_returns_lstm': 'LSTM · retornos esperados', 'ai_volatility': 'GJR-GARCH · riesgo',
         'combined': 'LSTM + GJR-GARCH · retornos y riesgo'}


def configuration(api, system, datasets, prefix='optimizer'):
    version = dataset_selector(datasets, system, key=f'{prefix}_dataset')
    if not version:
        return None
    detail = fetch(api, f'/datasets/{quote(version, safe="")}', {}) or {}
    manifest = detail.get('manifest_json') or detail.get('manifest') or {}
    defaults = detail.get('default_targets') or manifest.get('default_targets') or {}
    blocks = detail.get('blocks') or manifest.get('blocks') or {}
    if not defaults and blocks:
        defaults = {block: 1. / len(blocks) for block in blocks}
    c1, c2 = st.columns(2)
    mode = c1.selectbox('Estimador de retornos y riesgo', list(MODES), format_func=MODES.get, key=f'{prefix}_mode')
    universes = items(fetch(api, '/universes', {'items': []}))
    universe_id = c2.selectbox('Universo guardado', [None, *[row['id'] for row in universes]],
        format_func=lambda value: next((row.get('name', value) for row in universes if row['id'] == value), 'Todos los activos del dataset'), key=f'{prefix}_universe')
    universe = next((row for row in universes if row['id'] == universe_id), {})
    if universe:
        settings = universe.get('config') or universe.get('config_json') or {}
        assets = universe.get('assets') or universe.get('assets_json') or []
        dataset_assets = detail.get('assets') or manifest.get('assets') or []
        original_blocks = {asset.get('ticker'): asset.get('block') for asset in dataset_assets if isinstance(asset, dict)}
        selected_blocks = sorted({asset.get('block') or original_blocks.get(asset.get('ticker')) or 'OTHER'
            for asset in assets if isinstance(asset, dict) and asset.get('enabled', True)})
        saved_targets = settings.get('targets') or defaults
        if selected_blocks:
            defaults = {block: saved_targets.get(block, 0.) for block in selected_blocks}
            if not sum(defaults.values()):
                defaults = {block: 1. / len(selected_blocks) for block in selected_blocks}
    if not defaults:
        st.warning('El dataset no publica bloques ni objetivos. Revisá su metadata en Data Management.')
        return None
    as_of_default = detail.get('as_of') or manifest.get('latest_observation') or ''
    as_of = c1.text_input('Fecha de corte (as of)', value=str(as_of_default)[:10],
        placeholder='YYYY-MM-DD; vacío: última observación de la versión', key=f'{prefix}_as_of_{version}')
    label = c2.text_input('Etiqueta de corrida', placeholder='Hipótesis o nombre del experimento', key=f'{prefix}_label')
    st.markdown('#### Objetivos por bloque')
    columns = st.columns(min(4, len(defaults)))
    raw = {block: float(columns[index % len(columns)].number_input(str(block), min_value=0., max_value=1.,
        value=float(weight), step=.01, format='%.3f', key=f'{prefix}_{version}_{universe_id}_{block}')) for index, (block, weight) in enumerate(defaults.items())}
    normalize = st.checkbox('Normalizar automáticamente a 100%', value=True, key=f'{prefix}_normalize')
    total = sum(raw.values())
    targets = {name: value / total for name, value in raw.items()} if normalize and total > 0 else raw
    valid = total > 0 and (normalize or abs(total - 1.) < 1e-6)
    if valid:
        st.caption('Objetivos enviados: ' + ' · '.join(f'{key} {value:.1%}' for key, value in targets.items()))
    else:
        st.error('Los objetivos deben sumar 100% y contener al menos un peso positivo.')
    col1, col2, col3 = st.columns(3)
    cap = col1.slider('Peso máximo por activo', .01, 1., .15, .01, key=f'{prefix}_cap')
    risk_free = col2.number_input('Tasa libre de riesgo anual simple', min_value=0., max_value=1., value=0., step=.01, key=f'{prefix}_rf')
    covariance = col3.selectbox('Estimador de covarianza', ['historical', 'ledoit_wolf', 'gjr_garch'],
        index=2 if mode in {'ai_volatility','combined'} else 0,disabled=mode=='combined', key=f'{prefix}_covariance_{mode}')
    options = {}
    with st.expander('Calidad, señales y precisión'):
        col1, col2, col3 = st.columns(3)
        options['min_history'] = int(col1.number_input('Observaciones mínimas por activo', min_value=2, value=252, key=f'{prefix}_history'))
        options['min_overlap'] = int(col2.number_input('Solapamiento mínimo de pares', min_value=2, value=126, key=f'{prefix}_overlap'))
        options['n_frontier_points'] = int(col3.number_input('Puntos de frontera', min_value=10, max_value=500, value=80, key=f'{prefix}_frontier'))
        options['prediction_fallback'] = col1.selectbox('Señal faltante o vencida', ['historical', 'required'],
            format_func=lambda value: 'Fallback histórico explícito' if value == 'historical' else 'Exigir señal validada', key=f'{prefix}_fallback')
        options['max_prediction_age_days'] = int(col2.number_input('Antigüedad máxima de señal (días)', min_value=0, value=7, key=f'{prefix}_age'))
        options['lstm_direction_min'] = float(col3.number_input('Acierto direccional mínimo LSTM', min_value=0., max_value=1., value=.52, key=f'{prefix}_direction'))
        options['lstm_amplitude_min'] = float(col3.number_input('Amplitud mínima LSTM', min_value=0., value=.15, key=f'{prefix}_amplitude'))
    model_ids = {'return_model_ids': [], 'volatility_model_ids': []}
    if mode != 'classic' or covariance == 'gjr_garch':
        models = items(fetch(api, '/models', {'items': []}, dataset_version=version))
        available = [row for row in models if (row.get('state') or row.get('status')) not in {'archived', 'failed', 'rejected'}]
        names = {row['id']: f"{row.get('family')} · {row.get('ticker') or 'global'} · {row.get('state') or row.get('status')} · {row['id']}" for row in available}
        with st.expander('Versiones de modelos fijadas en la solicitud', expanded=True):
            return_families = {'lstm'} if mode in {'ai_returns_lstm','combined'} else {'mlp'} if mode == 'ai_returns' else set()
            for field, label_text, families in [('return_model_ids', 'Modelos de retorno', return_families),
                                                ('volatility_model_ids', 'Modelos de volatilidad', {'garch', 'gjr_garch'})]:
                candidates = [row['id'] for row in available if row.get('family') in families]
                model_ids[field] = st.multiselect(label_text, candidates, format_func=lambda value: names[value], key=f'{prefix}_{field}_{version}') or None
            st.caption('La API registra las versiones elegidas y verifica temporalidad, dataset y artefactos; una selección vacía se resuelve según la política de señales.')
    return {'dataset_version': version, 'universe_id': universe_id, 'as_of': as_of or None,
            'model_type': mode, 'targets': targets, 'max_weight': cap, 'rf_annual_simple': risk_free,
            'covariance_estimator': covariance, 'run_label': label or None, **options, **model_ids}, valid


def render_optimizer(api, system, datasets):
    st.caption('Construí carteras long-only con objetivos por bloque, límites y modelos verificables. Las restricciones se validan antes de ejecutar el solver.')
    configured = configuration(api, system, datasets)
    if configured:
        payload, valid = configured
        with st.expander('Solicitud reproducible'):
            st.json(payload)
        left, right = st.columns(2)
        if left.button('Validar configuración', disabled=not valid or readonly(system)):
            try:
                st.json(api.post('/optimizations/validate', payload))
            except APIError as exc:
                st.error(str(exc))
        if right.button('Correr optimización', type='primary', disabled=not valid or readonly(system)):
            enqueue(api, '/optimizations', payload, 'optimization', 'Optimización de cartera')
    if st.session_state.get('last_job_error'):
        st.error(st.session_state['last_job_error'])
    if st.session_state.get('last_api_response'):
        render_portfolio_result(api, st.session_state['last_api_response'], key='optimizer_result')
    else:
        st.info('La configuración se envía como trabajo asíncrono. Los resultados aparecen al confirmar su publicación.')


def render_portfolio_result(api, record, key='portfolio'):
    result = record.get('result') or record.get('result_json') or record
    if isinstance(result, dict) and 'result' in result:
        result = result['result']
    st.subheader('Resultado de cartera')
    stats = result.get('blocks', {}).get('stats', {})
    cols = st.columns(4)
    cols[0].metric('Retorno logarítmico anual estimado', pretty_pct(stats.get('return_annual')))
    cols[1].metric('Volatilidad anual estimada', pretty_pct(stats.get('vol_annual')))
    sharpe = stats.get('sharpe_annual')
    cols[2].metric('Sharpe anualizado estimado', f'{sharpe:.3f}' if isinstance(sharpe, (int, float)) else '—')
    cols[3].metric('Corrida registrada', str(record.get('run_id') or record.get('optimization_id') or record.get('experiment_id') or record.get('id') or '—'))
    tracking = record.get('tracking') or result.get('tracking') or {}
    mlflow_id = record.get('mlflow_run_id') or tracking.get('mlflow_run_id')
    mlflow_error = record.get('mlflow_error') or tracking.get('mlflow_error')
    if mlflow_error:
        st.warning(f'MLflow: {mlflow_error}')
    elif mlflow_id:
        st.success(f'MLflow registrado: {mlflow_id}')
    else:
        st.info('MLflow sin registro confirmado. La disponibilidad del servicio no confirma sincronización de esta corrida.')
    st.caption('Estimaciones ex ante en USD, sobre retornos diarios logarítmicos. El desempeño realizado se evalúa en Backtesting.')
    st.plotly_chart(build_frontier_figure(result), use_container_width=True, key=f'{key}_frontier')
    weights = frame(result.get('blocks', {}).get('weights_df', []))
    exposure = frame(result.get('blocks', {}).get('exposure_by_block', []))
    c1, c2 = st.columns([1.15, 1])
    c1.plotly_chart(build_weights_bar(weights), use_container_width=True, key=f'{key}_weights')
    c2.plotly_chart(build_block_exposure_chart(exposure), use_container_width=True, key=f'{key}_exposure')
    tab1, tab2, tab3, tab4 = st.tabs(['Pesos y comparación', 'Riesgo y escenarios', 'Señales y diagnósticos', 'Trazabilidad'])
    with tab1:
        st.dataframe(build_comparison_df(result), hide_index=True, use_container_width=True)
        st.dataframe(weights, hide_index=True, use_container_width=True)
        st.dataframe(exposure, hide_index=True, use_container_width=True)
        st.download_button('Exportar pesos CSV', weights.to_csv(index=False), 'portfolio_weights.csv', 'text/csv', key=f'{key}_weights_csv')
    with tab2:
        risk = result.get('portfolio_risk') or {}
        st.caption('La covarianza estima riesgo ex ante. Los escenarios históricos mantienen pesos fijos; no representan una estrategia realizada.')
        contributions = frame(risk.get('risk_contributions', []))
        if not contributions.empty:
            field = next((name for name in ['percent_of_variance_risk', 'component_vol_daily'] if name in contributions), None)
            if field and 'ticker' in contributions:
                fig = go.Figure(go.Bar(x=contributions.ticker, y=contributions[field], marker_color='#2dd4bf'))
                st.plotly_chart(chart_layout(fig, 'Contribución al riesgo'), use_container_width=True, key=f'{key}_risk')
            st.dataframe(contributions, hide_index=True, use_container_width=True)
        st.json(risk or {'status': 'not_reported'})
    with tab3:
        diagnostics = result.get('diagnostics') or {}
        signals = diagnostics.get('signals') or (diagnostics.get('risk') or {}).get('signals') or []
        if signals:
            st.dataframe(pd.DataFrame(signals), hide_index=True, use_container_width=True)
        st.json(diagnostics)
    with tab4:
        st.json({'provenance': result.get('provenance', {}), 'tracking': tracking or {'status': 'not_recorded'}})
        st.download_button('Exportar respuesta JSON', json.dumps(record, ensure_ascii=False, indent=2),
            'optimization.json', 'application/json', key=f'{key}_json')
        artifact_downloads(api, record, key=f'{key}_artifacts')


def render_analytics(api, system, datasets):
    st.caption('Composición, fronteras y contribuciones al riesgo sobre una corrida registrada. Cada resultado conserva su dataset y configuración.')
    version = dataset_selector(datasets, system, key='analytics_dataset')
    rows = items(fetch(api, '/optimizations', {'items': []}, **({'dataset_version': version} if version else {})))
    if rows:
        selected = st.selectbox('Corrida a analizar', [row['id'] for row in rows],
            format_func=lambda value: next((f"{row.get('run_label') or row.get('model_type') or 'Optimización'} · {row.get('created_at', '')} · {value}" for row in rows if row['id'] == value), str(value)))
        detail = fetch(api, f'/optimizations/{quote(str(selected), safe="")}', {})
        if detail:
            render_portfolio_result(api, detail, key='analytics_result')
    elif st.session_state.get('last_api_response'):
        render_portfolio_result(api, st.session_state['last_api_response'], key='analytics_result')
    else:
        st.info('Ejecutá una optimización o seleccioná un dataset con corridas registradas.')
