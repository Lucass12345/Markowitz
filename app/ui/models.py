from __future__ import annotations

from urllib.parse import quote
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app.api_client import APIError
from app.ui.charts import chart_layout
from app.ui.components import artifact_downloads, dataset_selector, enqueue, fetch, items, readonly

FEATURES = ['return', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_5', 'rolling_std_5', 'rolling_mean_20',
            'rolling_std_20', 'momentum_5', 'momentum_20', 'log_return_sum_5', 'log_return_sum_20',
            'downside_vol_20', 'ewma_vol_20', 'rolling_skew_20']


def render_models(api, system, datasets, initial_family='lstm'):
    st.caption('Un único flujo para entrenar, comparar y activar modelos. FastAPI crea el trabajo; el worker publica artefactos verificables.')
    version = dataset_selector(datasets, system)
    configure, experiments, registry = st.tabs(['Configurar experimento', 'Resultados y comparación', 'Model Registry'])
    with configure:
        family = st.selectbox('Familia de modelo', ['lstm', 'mlp', 'garch'], index=['lstm', 'mlp', 'garch'].index(initial_family),
                              format_func=lambda value: {'lstm': 'LSTM · retornos', 'mlp': 'MLP · retornos', 'garch': 'GJR-GARCH · riesgo'}[value])
        symbols = st.text_input('Tickers del experimento', placeholder='Vacío: universo completo del dataset')
        parameters = {}
        c1, c2, c3 = st.columns(3)
        parameters['seed'] = int(c1.number_input('Semilla', min_value=0, value=42))
        parameters['device'] = c2.selectbox('Dispositivo', ['auto', 'cpu', 'cuda'],
            help='CUDA sólo se admite si existe un worker GPU. CPU no requiere CUDA.')
        parameters['forecast_horizon'] = int(c3.number_input('Horizonte (ruedas)', min_value=5, max_value=5, value=5))
        if family in {'lstm', 'mlp'}:
            parameters['start_date'] = st.text_input('Inicio de entrenamiento', value='2022-01-01',
                help='YYYY-MM-DD o full para usar toda la historia disponible.', key=f'{family}_start_date')
            parameters['target_aggregation'] = st.selectbox('Target', ['mean', 'sum'],
                format_func=lambda value: 'Media diaria logarítmica a cinco ruedas' if value == 'mean' else 'Acumulado logarítmico a cinco ruedas; mu_pred se convierte a diario')
        if family == 'lstm':
            parameters['seq_len'] = int(c1.number_input('Ventana temporal', min_value=10, max_value=252, value=40))
            parameters['epochs'] = int(c2.number_input('Épocas máximas', min_value=1, max_value=1000, value=50))
            parameters['batch_size'] = int(c3.number_input('Batch size', min_value=1, max_value=1024, value=32))
            search = st.checkbox('Seleccionar arquitectura y Adam por validación', value=True)
            parameters['search'] = search
            parameters['hidden_sizes'] = list(st.selectbox('Preset de arquitectura', [(64, 64), (120, 64)],
                format_func=lambda value: ' → '.join(map(str, value)) + ' → densa', disabled=search))
            parameters['activation'] = c1.selectbox('Activación de capa densa', ['relu', 'tanh', 'gelu'], disabled=search)
            parameters['lr'] = float(c2.number_input('Adam learning rate', min_value=.000001, max_value=.1, value=.001, format='%.6f', disabled=search))
            parameters['dropout'] = float(c3.number_input('Dropout', min_value=0., max_value=.9, value=.2))
            parameters['patience'] = int(c1.number_input('Early stopping patience', min_value=1, max_value=100, value=8))
            parameters['betas'] = [float(c2.number_input('Adam beta 1', min_value=0., max_value=.9999, value=.9)),
                                   float(c3.number_input('Adam beta 2', min_value=0., max_value=.99999, value=.999))]
            parameters['eps'] = float(c2.number_input('Adam epsilon', min_value=1e-10, max_value=.001, value=1e-8, format='%.10f'))
            parameters['weight_decay'] = float(c3.number_input('Weight decay', min_value=0., max_value=1., value=0., format='%.6f'))
            presets = (system.get('presets') or {}).get('lstm') or {}
            features = presets.get('features') or FEATURES
            parameters['selected_features'] = st.multiselect('Variables de entrada', features, default=features)
            parameters['feature_search'] = st.checkbox('Comparar subsets de variables sólo por validación', value=False)
            st.info('Preset académico: ventana 40, horizonte 5 y retorno logarítmico diario promedio. La grilla mantiene 64→64 / 120→64 y selecciona sólo con validación.')
        elif family == 'mlp':
            parameters['hidden_layers'] = c1.text_input('Capas ocultas', value='256,128,64')
            parameters['activation'] = c2.selectbox('Activación', ['relu', 'tanh', 'logistic'])
            parameters['scaler'] = c3.selectbox('Escalador', ['standard', 'robust', 'minmax', 'none'])
            parameters['max_iter'] = int(c1.number_input('Iteraciones máximas', min_value=10, max_value=5000, value=500))
            with st.expander('Features, regularización y optimización MLP'):
                a, b, c = st.columns(3)
                parameters['lags'] = int(a.number_input('Rezagos MLP', min_value=1, max_value=252, value=20))
                parameters['short_window'] = int(b.number_input('Ventana corta MLP', min_value=2, value=5))
                parameters['long_window'] = int(c.number_input('Ventana larga MLP', min_value=2, value=20))
                parameters['alpha'] = float(a.number_input('Regularización L2', min_value=0., value=.0001, format='%.6f'))
                parameters['learning_rate_init'] = float(b.number_input('Learning rate MLP', min_value=.000001, value=.001, format='%.6f'))
                parameters['solver'] = c.selectbox('Optimizador MLP', ['adam', 'lbfgs', 'sgd'])
                parameters['early_stopping'] = a.checkbox('Early stopping MLP', value=True)
                parameters['validation_fraction'] = float(b.number_input('Fracción para early stopping', min_value=.05, max_value=.4, value=.15))
                parameters['n_iter_no_change'] = int(c.number_input('Paciencia MLP', min_value=1, value=20))
                parameters['clip_pred'] = float(a.number_input('Recorte diario de predicción', min_value=.001, max_value=1., value=.02))
                parameters['winsorize_quantile'] = float(b.number_input('Cuantil de winsorización', min_value=0., max_value=.1, value=.01))
                parameters['include_abs_features'] = a.checkbox('Features de magnitud', value=True)
                parameters['include_square_features'] = b.checkbox('Features cuadráticas', value=True)
                parameters['include_sign_features'] = c.checkbox('Features de signo', value=True)
            st.info('Los modelos MLP sin evidencia de validación no son señales activables. El test sólo informa evaluación final.')
        else:
            parameters['min_obs'] = int(c1.number_input('Historia mínima GARCH', min_value=252, value=252))
            parameters['validation_days'] = int(c2.number_input('Orígenes de validación', min_value=5, max_value=252, value=20))
            parameters['test_days'] = int(c3.number_input('Orígenes de test', min_value=5, max_value=252, value=20))
            parameters['dist'] = c1.selectbox('Innovaciones', ['t', 'normal'])
            parameters['mean'] = 'Zero'
            parameters['maxiter'] = int(c2.number_input('Iteraciones de ajuste', min_value=100, max_value=5000, value=300))
            st.info('GJR-GARCH(1,1,1) estima riesgo diario y a cinco ruedas. Se conservan diagnósticos y fallback EWMA explícito.')
        with st.expander('Configuración enviada'):
            st.json({'family': family, 'dataset_version': version, 'parameters': parameters})
        if st.button('Lanzar entrenamiento', type='primary', disabled=not version or readonly(system)):
            enqueue(api, '/experiments', {'family': family, 'dataset_version': version,
                'tickers': symbols.replace(',', ' ').split() or None, 'parameters': parameters}, 'experiment', f'Entrenamiento {family.upper()}')
    with experiments:
        rows = items(fetch(api, '/experiments', {'items': []}, dataset_version=version))
        st.dataframe(pd.DataFrame([{key: row.get(key) for key in ['id', 'family', 'status', 'dataset_version', 'created_at', 'duration_seconds', 'mlflow_run_id']} for row in rows]), hide_index=True, use_container_width=True)
        ids = [row['id'] for row in rows]
        if ids:
            selected = st.selectbox('Inspeccionar experimento', ids)
            detail = fetch(api, f'/experiments/{quote(str(selected), safe="")}', {})
            metrics = detail.get('metrics') or detail.get('metrics_json') or {}
            st.json(metrics)
            curves = detail.get('curves') or []
            curve_files=[item for item in detail.get('artifacts',[]) if item.get('path','').endswith('_losses.csv')]
            if curve_files:
                selected_curve=st.selectbox('Curva por ticker',curve_files,format_func=lambda item:item['filename'])
                if st.button('Cargar curva del experimento'):
                    try:
                        content,_=api.download(selected_curve['id'],selected_curve['path'])
                        curves=pd.read_csv(BytesIO(content)).to_dict('records')
                    except (APIError,ValueError) as exc:
                        st.error(str(exc))
            if curves:
                data = pd.DataFrame(curves)
                if 'epoch' in data:
                    fig = go.Figure()
                    for column in ['train_loss', 'val_loss']:
                        if column in data:
                            fig.add_trace(go.Scatter(x=data.epoch, y=data[column], mode='lines', name=column))
                    st.plotly_chart(chart_layout(fig, 'Curvas de entrenamiento y validación'), use_container_width=True)
            with st.expander('Configuración, temporalidad y sincronización'):
                st.json(detail)
            artifact_downloads(api, detail, key='experiment_artifacts')
        last = st.session_state.get('research_results', {}).get('experiment')
        if last:
            st.subheader('Último trabajo completado')
            st.json(last)
    with registry:
        rows = items(fetch(api, '/models', {'items': []}, dataset_version=version))
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        ids = [row['id'] for row in rows]
        if ids:
            selected = st.selectbox('Versión de modelo', ids, key='registry_model',
                format_func=lambda value: next((f"{row.get('family')} · {row.get('ticker')} · {row.get('state') or row.get('status')} · {value}" for row in rows if row['id'] == value), value))
            col1, col2, col3 = st.columns(3)
            if col1.button('Activar versión validada', disabled=readonly(system)):
                enqueue(api, f'/models/{quote(str(selected), safe="")}/activate', {}, 'model_activation', 'Activar modelo')
            if col2.button('Archivar versión', disabled=readonly(system)):
                enqueue(api, f'/models/{quote(str(selected), safe="")}/archive', {}, 'model_activation', 'Archivar modelo')
            if col3.button('Ejecutar inferencia', disabled=readonly(system) or not version):
                enqueue(api, f'/models/{quote(str(selected), safe="")}/infer', {'dataset_version': version}, 'inference', 'Inferencia verificada')
            if st.button('Inspeccionar artefactos del modelo'):
                detail = fetch(api, f'/models/{quote(str(selected), safe="")}', {})
                st.session_state['registry_model_detail'] = detail
            detail = st.session_state.get('registry_model_detail') or {}
            if detail.get('id') == selected:
                artifact_downloads(api, detail, key='model_artifacts')
            st.caption('La activación verifica calidad y artefactos. La optimización verifica además compatibilidad con el dataset. Para restaurar un modelo anterior seleccioná su versión validada y activala.')
