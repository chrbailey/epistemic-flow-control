"""
Visualization components for the Streamlit demo.

Beautiful, interactive charts demonstrating the statistical concepts.
"""

import math
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px


def wilson_score_bounds(successes: int, total: int, z: float = 1.96) -> tuple:
    """Calculate Wilson score interval bounds."""
    if total == 0:
        return 0.0, 1.0

    p = successes / total
    denominator = 1 + z*z/total
    center = p + z*z/(2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)

    lower = max(0.0, (center - spread) / denominator)
    upper = min(1.0, (center + spread) / denominator)
    return lower, upper


def plot_bayesian_update(
    observations: List[Dict],
    title: str = "Pattern Weight Evolution"
) -> go.Figure:
    """
    Plot how a pattern's weight evolves with observations.

    Args:
        observations: List of dicts with 'observation_num', 'successes', 'total'
        title: Chart title

    Returns:
        Plotly figure
    """
    if not observations:
        fig = go.Figure()
        fig.add_annotation(text="No observations yet", x=0.5, y=0.5, showarrow=False)
        return fig

    x_vals = []
    weights = []
    lower_bounds = []
    upper_bounds = []

    for obs in observations:
        x_vals.append(obs['observation_num'])
        weight = obs['successes'] / obs['total'] if obs['total'] > 0 else 0.5
        weights.append(weight)
        lower, upper = wilson_score_bounds(obs['successes'], obs['total'])
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    fig = go.Figure()

    # Confidence interval fill
    fig.add_trace(go.Scatter(
        x=x_vals + x_vals[::-1],
        y=upper_bounds + lower_bounds[::-1],
        fill='toself',
        fillcolor='rgba(74, 144, 226, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))

    # Main weight line
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=weights,
        mode='lines+markers',
        name='Observed Rate',
        line=dict(color='#4A90E2', width=3),
        marker=dict(size=8, color='#4A90E2'),
        hovertemplate='<b>Observation %{x}</b><br>Rate: %{y:.1%}<extra></extra>'
    ))

    # Wilson lower bound line
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=lower_bounds,
        mode='lines',
        name='Conservative Estimate (Wilson)',
        line=dict(color='#F5A623', width=2, dash='dash'),
        hovertemplate='Wilson Lower: %{y:.1%}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Number of Observations",
        yaxis_title="Pattern Weight",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=400
    )

    return fig


def plot_confidence_gauge(
    confidence: float,
    thresholds: Dict[str, float],
    title: str = "Confidence & Gate Decision"
) -> go.Figure:
    """
    Display a confidence gauge with gate decision visualization.

    Args:
        confidence: Confidence score (0-1)
        thresholds: Dict with 'auto_pass', 'review', 'block' thresholds
        title: Chart title

    Returns:
        Plotly figure
    """
    # Determine gate decision
    if confidence >= thresholds.get('auto_pass', 0.92):
        decision = "AUTO PASS"
        color = "#7ED321"
    elif confidence >= thresholds.get('review', 0.70):
        decision = "REVIEW REQUIRED"
        color = "#F5A623"
    elif confidence >= thresholds.get('block', 0.50):
        decision = "BLOCKED"
        color = "#D0021B"
    else:
        decision = "REJECTED"
        color = "#8B0000"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={'valueformat': '.1%'},
        title={'text': f"<b>{decision}</b>", 'font': {'color': color, 'size': 20}},
        gauge={
            'axis': {'range': [0, 1], 'tickformat': '.0%'},
            'bar': {'color': color, 'thickness': 0.75},
            'steps': [
                {'range': [0, thresholds.get('block', 0.50)], 'color': "rgba(208, 2, 27, 0.2)"},
                {'range': [thresholds.get('block', 0.50), thresholds.get('review', 0.70)], 'color': "rgba(245, 166, 35, 0.2)"},
                {'range': [thresholds.get('review', 0.70), thresholds.get('auto_pass', 0.92)], 'color': "rgba(74, 144, 226, 0.2)"},
                {'range': [thresholds.get('auto_pass', 0.92), 1], 'color': "rgba(126, 211, 33, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.8,
                'value': confidence
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def plot_calibration_curve(
    calibration_data: Dict,
    title: str = "Calibration Curve"
) -> go.Figure:
    """
    Plot calibration curve with ECE visualization.

    Args:
        calibration_data: Dict with 'calibration_curve' list and 'expected_calibration_error'
        title: Chart title

    Returns:
        Plotly figure
    """
    from plotly.subplots import make_subplots

    curve = calibration_data.get('calibration_curve', [])
    ece = calibration_data.get('expected_calibration_error', 0)

    if not curve:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for calibration", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Calibration Curve', 'Calibration Error by Bucket'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
    )

    expected = [b.get('expected_accuracy', b.get('confidence_midpoint', 0)) for b in curve]
    actual = [b.get('actual_accuracy', 0) for b in curve]
    sizes = [b.get('sample_size', 10) for b in curve]
    errors = [b.get('calibration_error', actual[i] - expected[i]) for i, b in enumerate(curve)]

    # Calibration curve (left)
    fig.add_trace(
        go.Scatter(
            x=expected,
            y=actual,
            mode='markers+lines',
            name='Actual Accuracy',
            marker=dict(size=[max(8, min(30, s/2)) for s in sizes], color='#4A90E2'),
            line=dict(color='#4A90E2', width=2),
            hovertemplate='Expected: %{x:.1%}<br>Actual: %{y:.1%}<extra></extra>'
        ),
        row=1, col=1
    )

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray', width=1),
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # Calibration error bars (right)
    colors = ['#D0021B' if e < 0 else '#7ED321' for e in errors]
    fig.add_trace(
        go.Bar(
            x=list(range(len(errors))),
            y=errors,
            name='Calibration Error',
            marker_color=colors,
            hovertemplate='Bucket %{x}<br>Error: %{y:.1%}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Expected Accuracy", tickformat='.0%', row=1, col=1)
    fig.update_yaxes(title_text="Actual Accuracy", tickformat='.0%', row=1, col=1)
    fig.update_xaxes(title_text="Confidence Bucket", row=1, col=2)
    fig.update_yaxes(title_text="Calibration Error", tickformat='.0%', row=1, col=2)

    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.1, xanchor='center', x=0.5)
    )

    # Add ECE annotation
    fig.add_annotation(
        text=f"<b>ECE: {ece:.1%}</b>",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(size=16)
    )

    return fig


def plot_pattern_evolution(
    judge_name: str,
    historical_rates: List[Dict],
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot a judge's pattern evolution over time.

    Args:
        judge_name: Name of the judge
        historical_rates: List of dicts with 'period', 'rate', 'cases', 'note'
        title: Optional custom title

    Returns:
        Plotly figure
    """
    if not historical_rates:
        fig = go.Figure()
        fig.add_annotation(text="No historical data", x=0.5, y=0.5, showarrow=False)
        return fig

    periods = [h['period'] for h in historical_rates]
    rates = [h['rate'] for h in historical_rates]
    cases = [h.get('cases', 10) for h in historical_rates]
    notes = [h.get('note', '') for h in historical_rates]

    # Calculate Wilson bounds
    bounds = []
    for i, h in enumerate(historical_rates):
        successes = int(h['rate'] * h.get('cases', 10))
        total = h.get('cases', 10)
        lower, upper = wilson_score_bounds(successes, total)
        bounds.append((lower, upper))

    lowers = [b[0] for b in bounds]
    uppers = [b[1] for b in bounds]

    fig = go.Figure()

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=periods + periods[::-1],
        y=uppers + lowers[::-1],
        fill='toself',
        fillcolor='rgba(74, 144, 226, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        hoverinfo='skip'
    ))

    # Main rate line
    fig.add_trace(go.Scatter(
        x=periods,
        y=rates,
        mode='lines+markers',
        name='Grant Rate',
        line=dict(color='#4A90E2', width=3),
        marker=dict(size=[max(8, min(20, c/2)) for c in cases], color='#4A90E2'),
        text=notes,
        hovertemplate='<b>%{x}</b><br>Rate: %{y:.0%}<br>Cases: ' +
                      '<br>'.join([str(c) for c in cases]) + '<br>%{text}<extra></extra>'
    ))

    fig.update_layout(
        title=title or f"{judge_name} - Summary Judgment Grant Rate Evolution",
        xaxis_title="Time Period",
        yaxis_title="Grant Rate",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        hovermode='x unified',
        height=400
    )

    return fig


def plot_source_reliability_spectrum(sources: List[Dict]) -> go.Figure:
    """
    Plot source reliability as a horizontal spectrum.

    Args:
        sources: List of dicts with 'name', 'reliability', 'source_type'

    Returns:
        Plotly figure
    """
    if not sources:
        fig = go.Figure()
        fig.add_annotation(text="No sources", x=0.5, y=0.5, showarrow=False)
        return fig

    # Sort by reliability
    sorted_sources = sorted(sources, key=lambda x: x['reliability'], reverse=True)

    names = [s['name'] for s in sorted_sources]
    reliabilities = [s['reliability'] for s in sorted_sources]
    types = [s.get('source_type', 'unknown') for s in sorted_sources]

    # Color by type
    color_map = {
        'official': '#7ED321',
        'professional': '#4A90E2',
        'news': '#F5A623',
        'blog': '#F8E71C',
        'social': '#D0021B',
        'unknown': '#9B9B9B'
    }
    colors = [color_map.get(t, '#9B9B9B') for t in types]

    fig = go.Figure(go.Bar(
        x=reliabilities,
        y=names,
        orientation='h',
        marker_color=colors,
        text=[f"{r:.0%}" for r in reliabilities],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Reliability: %{x:.1%}<extra></extra>'
    ))

    fig.update_layout(
        title="Source Reliability Spectrum",
        xaxis_title="Reliability Score",
        xaxis=dict(range=[0, 1.1], tickformat='.0%'),
        yaxis_title="",
        height=max(300, len(sources) * 40),
        margin=dict(l=200)
    )

    return fig
