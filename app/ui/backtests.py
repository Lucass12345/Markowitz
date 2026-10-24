from __future__ import annotations

from urllib.parse import quote
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui.charts import chart_layout, frame
from app.ui.components import artifact_downloads, enqueue, fetch, items, readonly
from app.ui.portfolio import configuration

STRATEGIES = {'classic': 'Markowitz histórico', 'lstm': 'Retornos LSTM', 'mlp': 'Retornos MLP',
    'gjr_garch': 'Riesgo GJR-GARCH', 'combined': 'Retornos IA + riesgo IA',
    'equal_weight': 'Equal weight', 'buy_and_hold': 'Buy & hold'}


def render_backtest_result(api, record, key='backtest'):
    result = record.get('result') or record.get('result_json') or record
    if 'result' in result:
        result = result['result']
    st.subheader('Evaluación fuera de muestra')
    st.caption(f"Dataset: {result.get('dataset_version', '—')} · OOS: {result.get('oos_start', '—')} → {result.get('oos_end', '—')}")
    curves = frame(result.get('wealth_curves', []))
    if not curves.empty:
        date_field = next((name for name in ['date', 'Date', 'index'] if name in curves), None)
        fig = go.Figure()
        for column in curves:
            if column != date_field:
                fig.add_trace(go.Scatter(x=curves[date_field] if date_field else curves.index, y=curves[column], name=STRATEGIES.get(column, column), mode='lines'))
        st.plotly_chart(chart_layout(fig, 'Crecimiento de USD 1 · neto de costos', 500), use_container_width=True, key=f'{key}_wealth')
    metrics = result.get('metrics') or {}
    table = result.get('summary') or [{'strategy': strategy, **values} for strategy, values in metrics.items()]
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    with st.expander('Diferencias frente a equal weight, metodología y limitaciones'):
        st.json({field: result.get(field) for field in ['differences_vs_equal_weight', 'benchmark_definition', 'methodology', 'calendar_policy', 'constraints_timing', 'limitations']})
    with st.expander('Configuración reproducible'):
        st.json(result.get('configuration') or {})
    artifact_downloads(api, record, key=f'{key}_artifacts')


def render_backtests(api, system, datasets):
    st.caption('Walk-forward causal: cada decisión usa sólo el pasado y comienza a rendir en la rueda siguiente. Compará estrategias sobre el mismo calendario OOS.')
    configure, results = st.tabs(['Configurar evaluación', 'Resultados registrados'])
    with configure:
        configured = configuration(api, system, datasets, prefix='backtest')
        if configured:
            optimization, valid = configured
            selected = st.multiselect('Estrategias a comparar', list(STRATEGIES), default=['classic', 'equal_weight', 'buy_and_hold'], format_func=STRATEGIES.get)
            c1, c2, c3 = st.columns(3)
            train_window = int(c1.number_input('Ventana de entrenamiento (ruedas)', min_value=30, value=504))
            min_train = int(c2.number_input('Historia mínima de entrenamiento', min_value=2, value=252))
            rebalance = int(c3.number_input('Rebalancear cada (ruedas)', min_value=1, value=5))
            oos_sessions = int(c1.number_input('Ruedas fuera de muestra', min_value=2, value=126))
            start = c2.text_input('Inicio OOS opcional', placeholder='YYYY-MM-DD')
            end = c3.text_input('Fin OOS opcional', placeholder='YYYY-MM-DD')
            commission = float(c1.number_input('Comisión (bps por nocional)', min_value=0., max_value=999., value=0.))
            cost = float(c2.number_input('Otros costos (bps)', min_value=0., max_value=999., value=0.))
            slippage = float(c3.number_input('Slippage (bps)', min_value=0., max_value=999., value=0.))
            matched = st.checkbox('Benchmarks con las mismas restricciones por bloque y activo', value=True)
            st.caption('Los modelos se reentrenan dentro de cada fold. Se conserva el universo elegido: la simulación no corrige sesgo de supervivencia ni metadata histórica.')
            payload = {'dataset_version': optimization['dataset_version'], 'universe_id': optimization['universe_id'],
                'optimization': optimization, 'strategies': selected, 'train_window': train_window,
                'min_train': min_train, 'rebalance_every': rebalance, 'oos_sessions': oos_sessions,
                'commission_bps': commission, 'transaction_cost_bps': cost, 'slippage_bps': slippage,
                'benchmark_constraints': matched}
            if start:
                payload['oos_start'] = start
            if end:
                payload['oos_end'] = end
            with st.expander('Solicitud completa'):
                st.json(payload)
            if st.button('Ejecutar backtest', type='primary', disabled=not valid or not selected or min_train > train_window or readonly(system)):
                enqueue(api, '/backtests', payload, 'backtest', 'Backtest walk-forward')
    with results:
        rows = items(fetch(api, '/backtests', {'items': []}))
        if rows:
            selected_id = st.selectbox('Backtest registrado', [row['id'] for row in rows])
            detail = fetch(api, f'/backtests/{quote(str(selected_id), safe="")}', {})
            if detail:
                render_backtest_result(api, detail, key='saved_backtest')
        latest = st.session_state.get('research_results', {}).get('backtest')
        if latest:
            render_backtest_result(api, latest, key='latest_backtest')
        elif not rows:
            st.info('Aún no hay backtests publicados.')
