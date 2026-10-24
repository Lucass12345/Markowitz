from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

COLORS = ['#38bdf8', '#a78bfa', '#2dd4bf', '#fbbf24', '#fb7185', '#94a3b8']


def frame(value):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return pd.DataFrame(value)
    return pd.DataFrame(value or [])


def pretty_pct(value, digits=2):
    try:
        return f'{float(value):.{digits}%}'
    except (TypeError, ValueError):
        return '—'


def chart_layout(fig, title, height=420):
    fig.update_layout(template='plotly_dark', title=title, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', height=height, margin=dict(l=20, r=20, t=60, b=25),
                      font=dict(color='#cbd5e1'), legend=dict(orientation='h', y=1.1))
    return fig


def build_frontier_figure(result):
    fig = go.Figure()
    for key, label, color in [('frontier', 'Clásica · long-only', COLORS[0]),
                              ('constrained_frontier', 'Restringida · bloques y tope', COLORS[1])]:
        value = result.get(key, [])
        if isinstance(value, dict) and 'points' in value:
            value = value['points']
        points = frame(value)
        if 'return_log_annual' in points and 'return_annual' not in points:
            points['return_annual'] = points['return_log_annual']
        if not points.empty and {'vol_annual', 'return_annual'}.issubset(points):
            fig.add_trace(go.Scatter(x=points.vol_annual, y=points.return_annual, mode='lines', name=label,
                                    line=dict(color=color, width=3)))
    for key, label, symbol, color in [('minvar', 'MinVar', 'diamond', COLORS[2]),
                                      ('maxsharpe', 'MaxSharpe', 'star', COLORS[3]),
                                      ('blocks', 'Cartera con bloques', 'x', COLORS[4])]:
        stats = result.get(key, {}).get('stats', {})
        if stats.get('vol_annual') is not None and stats.get('return_annual') is not None:
            fig.add_trace(go.Scatter(x=[stats['vol_annual']], y=[stats['return_annual']], mode='markers',
                                    name=label, marker=dict(size=13, symbol=symbol, color=color)))
    fig.update_xaxes(title='Volatilidad anual', tickformat='.1%')
    fig.update_yaxes(title='Retorno logarítmico anual', tickformat='.1%')
    return chart_layout(fig, 'Fronteras eficientes', 520)


def build_weights_bar(weights, top_n=20):
    data = frame(weights)
    fig = go.Figure()
    if not data.empty and {'ticker', 'weight'}.issubset(data):
        data = data.sort_values('weight', ascending=False).head(top_n)
        fig.add_trace(go.Bar(x=data.ticker, y=data.weight, marker_color=COLORS[0]))
    fig.update_yaxes(title='Peso', tickformat='.1%')
    return chart_layout(fig, 'Composición de cartera')


def build_block_exposure_chart(exposures):
    data = frame(exposures)
    fig = go.Figure()
    if not data.empty and {'block', 'target', 'achieved'}.issubset(data):
        fig.add_trace(go.Bar(x=data.block, y=data.target, name='Objetivo', marker_color=COLORS[1]))
        fig.add_trace(go.Bar(x=data.block, y=data.achieved, name='Logrado', marker_color=COLORS[2]))
    fig.update_yaxes(tickformat='.1%')
    return chart_layout(fig, 'Exposición por bloque')


def build_comparison_df(result):
    return pd.DataFrame([{'portfolio': label, **result.get(name, {}).get('stats', {})}
                         for name, label in [('minvar', 'MinVar'), ('maxsharpe', 'MaxSharpe'), ('blocks', 'Bloques')]])
