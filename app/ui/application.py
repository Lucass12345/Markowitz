from __future__ import annotations

import streamlit as st

from app.api_client import PlatformAPI
from app.ui.components import fetch, job_monitor, shell
from app.ui.data import render_data
from app.ui.models import render_models
from app.ui.portfolio import render_analytics, render_optimizer
from app.ui.backtests import render_backtests
from app.ui.overview import render_dashboard, render_history, render_system


def run(initial_section='Dashboard', initial_family='lstm'):
    st.set_page_config(page_title='Markowitz · Research Terminal', page_icon='◈',
                       layout='wide', initial_sidebar_state='expanded')
    section = shell(initial_section)
    api = PlatformAPI()
    system = fetch(api, '/system', {}) or {}
    datasets = fetch(api, '/datasets', {'items': []}) or {'items': []}
    if not system:
        st.error('El backend no está disponible. Las operaciones requieren una respuesta de FastAPI.')
    renderers = {
        'Dashboard': render_dashboard,
        'Data Management': render_data,
        'Portfolio Optimizer': render_optimizer,
        'Portfolio Analytics': render_analytics,
        'Backtesting': render_backtests,
        'History & Experiments': render_history,
        'System & Diagnostics': render_system,
    }
    if section == 'Model Laboratory':
        render_models(api, system, datasets, initial_family)
    else:
        renderers[section](api, system, datasets)
    job_monitor(api, system)
