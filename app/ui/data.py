from __future__ import annotations

from datetime import date
from urllib.parse import quote
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app.api_client import APIError
from app.ui.charts import chart_layout
from app.ui.components import dataset_selector, enqueue, fetch, items, readonly


def render_data(api, system, datasets):
    st.caption('Universos reutilizables, datos USD versionados y publicación controlada. Toda actualización se ejecuta en workers.')
    version = dataset_selector(datasets, system)
    tab_refresh, tab_universe, tab_versions, tab_quality = st.tabs(['Actualizar Yahoo', 'Universo y metadata', 'Versiones y rollback', 'Datos y calidad'])
    with tab_refresh:
        universes = items(fetch(api, '/universes', {'items': []}))
        universe_ids = [row['id'] for row in universes]
        names = {row['id']: row.get('name', row['id']) for row in universes}
        universe_id = st.selectbox('Universo de actualización', [None, *universe_ids],
            format_func=lambda value: names.get(value, 'Tickers ingresados manualmente'), key='refresh_universe')
        symbols = st.text_area('Tickers Yahoo Finance', placeholder='AAPL MSFT GGAL.BA TLT GLD',
            help='Se validan símbolos, duplicados, monedas y cobertura antes de publicar.')
        c1, c2 = st.columns(2)
        start = c1.date_input('Inicio de la historia', value=date(2015, 1, 1))
        full = c2.checkbox('Refresh completo', value=False, help='El modo incremental conserva historia y vuelve a consultar un solapamiento para revisar ajustes.')
        overlap = c2.number_input('Solapamiento de revisión (días)', min_value=5, max_value=365, value=30)
        full_interval = c1.number_input('Refresh completo periódico (días)', min_value=1, max_value=365, value=30)
        activate = st.checkbox('Activar dataset al superar controles', value=False)
        st.info('USD es la moneda base. Las cotizaciones ARS usan CCL histórico antes de calcular retornos; no se rellenan feriados.')
        if st.button('Validar y actualizar Yahoo Finance', type='primary', disabled=readonly(system), key='refresh_dataset'):
            payload = {'start': start.isoformat(), 'refresh_mode': 'full' if full else 'incremental',
                       'revision_overlap_days': int(overlap), 'full_refresh_every_days': int(full_interval),
                       'activate': activate}
            if universe_id:
                payload['universe_id'] = universe_id
            else:
                payload['tickers'] = symbols.replace(',', ' ').split()
            if version:
                payload['dataset_version'] = version
            enqueue(api, '/datasets/refresh', payload, 'dataset', 'Actualización Yahoo Finance')
        with st.expander('Migrar datos existentes una sola vez'):
            st.caption('Importa versiones históricas verificables al registro y MinIO, sin sobrescribir archivos ni resultados académicos.')
            if st.button('Importar histórico del proyecto', disabled=readonly(system)):
                enqueue(api, '/datasets/import-legacy', {}, 'dataset', 'Importación histórica')
    with tab_universe:
        universes = items(fetch(api, '/universes', {'items': []}))
        options = [None, *[row['id'] for row in universes]]
        selected = st.selectbox('Universo guardado', options,
            format_func=lambda value: next((row.get('name', value) for row in universes if row['id'] == value), 'Crear universo'), key='universe_editor')
        record = next((row for row in universes if row['id'] == selected), {})
        name = st.text_input('Nombre del universo', value=record.get('name', ''), key=f'universe_name_{selected}')
        assets = record.get('assets') or record.get('assets_json') or []
        if st.button('Cargar preset histórico original', key='load_original_preset'):
            preset = fetch(api, '/universes/presets/original', {})
            if preset:
                st.session_state['universe_preset'] = preset.get('assets') or preset.get('assets_json') or []
        if not selected and st.session_state.get('universe_preset'):
            assets = st.session_state['universe_preset']
        table = pd.DataFrame(assets) if assets else pd.DataFrame(columns=['ticker', 'block', 'enabled', 'country', 'currency', 'sector'])
        if 'enabled' not in table:
            table['enabled'] = True
        edited = st.data_editor(table, num_rows='dynamic', hide_index=True, use_container_width=True,
            disabled=readonly(system), key=f'asset_editor_{selected}',
            column_config={'ticker': st.column_config.TextColumn('Ticker', required=True),
                           'block': st.column_config.TextColumn('Bloque', help='Admite GOLD, OTHER y bloques personalizados.'),
                           'enabled': st.column_config.CheckboxColumn('Activo', default=True)})
        st.caption('Agregá o eliminá filas, editá bloques y activá/desactivá activos. Los cambios crean configuración de universo; no reescriben datasets históricos.')
        if st.button('Guardar universo', disabled=readonly(system) or not name.strip()):
            rows = edited.where(pd.notna(edited), None).to_dict(orient='records')
            payload = {'name': name.strip(), 'assets': rows, 'config': record.get('config') or record.get('config_json') or {}}
            try:
                result = api.patch(f'/universes/{quote(str(selected), safe="")}', payload) if selected else api.post('/universes', payload)
                st.success(f'Universo guardado: {result.get("name", name)}')
            except APIError as exc:
                st.error(str(exc))
    with tab_versions:
        rows = items(datasets)
        st.dataframe(pd.DataFrame([{key: row.get(key) for key in ['id', 'version', 'status', 'as_of', 'generated_at', 'content_hash']} for row in rows]),
                     hide_index=True, use_container_width=True)
        st.caption('Las versiones usadas por experimentos no se sobrescriben ni se eliminan.')
        if version and st.button('Activar esta versión', disabled=readonly(system)):
            enqueue(api, f'/datasets/{quote(version, safe="")}/activate', {}, 'dataset', f'Activar {version}')
        if st.button('Restaurar dataset anterior', disabled=readonly(system)):
            enqueue(api, '/datasets/rollback', {}, 'dataset', 'Rollback de dataset')
    with tab_quality:
        if not version:
            return
        detail = fetch(api, f'/datasets/{quote(version, safe="")}', {})
        manifest = detail.get('manifest_json') or detail.get('manifest') or detail
        st.json(detail.get('quality') or manifest.get('quality') or {'status': 'quality_not_reported'})
        assets = detail.get('assets') or detail.get('metadata') or manifest.get('assets') or []
        if assets:
            st.dataframe(pd.DataFrame(assets), hide_index=True, use_container_width=True)
        kind = st.radio('Serie a visualizar', ['prices', 'returns'], horizontal=True)
        limit = st.slider('Observaciones por consulta', 50, 2000, 500, 50)
        if st.button('Consultar series desde API'):
            payload = fetch(api, f'/datasets/{quote(version, safe="")}/data', {}, kind=kind, limit=limit)
            rows = payload.get('rows', items(payload)) if isinstance(payload, dict) else payload
            data = pd.DataFrame(rows or [])
            if data.empty:
                st.info('La API no devolvió observaciones para esa consulta.')
            else:
                date_column = next((name for name in ['date', 'Date', 'index'] if name in data), None)
                fig = go.Figure()
                for column in data:
                    if column != date_column:
                        fig.add_trace(go.Scatter(x=data[date_column] if date_column else data.index,
                            y=data[column], mode='lines', name=column))
                st.plotly_chart(chart_layout(fig, 'Precios ajustados USD' if kind == 'prices' else 'Retornos logarítmicos diarios USD'), use_container_width=True)
                st.dataframe(data, hide_index=True, use_container_width=True)
