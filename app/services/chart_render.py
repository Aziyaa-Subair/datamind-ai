"""Turn a chart spec (see charts.py) into a Plotly figure.

Styling follows the data-viz guidance: one hue for a single series, thin marks,
a recessive grid, no chart junk, and direct labels only where they help (the
largest bar, the last point) instead of a number on everything. Backgrounds are
transparent so the figure inherits Streamlit's light or dark theme.

Plotly is imported inside the function so the rest of the project (and its
tests) never needs it.
"""

ACCENT = "#2a78d6"                     # single-series blue
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order
GRID = "rgba(128,128,128,0.25)"
TRANSPARENT = "rgba(0,0,0,0)"


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _label_peak(values: list[float]) -> list[str]:
    """Value text on the largest bar only."""
    if not values:
        return []
    peak = max(range(len(values)), key=lambda i: values[i])
    return [_fmt(v) if i == peak else "" for i, v in enumerate(values)]


def _label_last(values: list[float]) -> list[str]:
    """Value text on the final point only."""
    return [""] * (len(values) - 1) + [_fmt(values[-1])] if values else []


def to_figure(spec: dict):
    import plotly.graph_objects as go

    kind, x, y = spec["chart_type"], spec["x"], spec["y"]

    if kind == "bar":
        fig = go.Figure(go.Bar(
            x=x, y=y, marker_color=ACCENT, text=_label_peak(y),
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        ))
        fig.update_layout(bargap=0.4)
    elif kind == "histogram":
        fig = go.Figure(go.Bar(
            x=x, y=y, marker_color=ACCENT,
            hovertemplate="%{x}<br>%{y:,.0f} rows<extra></extra>",
        ))
        fig.update_layout(bargap=0.05)
    elif kind == "line":
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="lines+markers+text" if len(x) <= 24 else "lines+text",
            line=dict(color=ACCENT, width=2), marker=dict(size=8, color=ACCENT),
            text=_label_last(y), textposition="top center",
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        ))
    elif kind == "scatter":
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=8, color=ACCENT, opacity=0.75),
            hovertemplate="%{x:,.2f}, %{y:,.2f}<extra></extra>",
        ))
    elif kind == "pie":
        fig = go.Figure(go.Pie(
            labels=x, values=y, hole=0.5, sort=False,
            marker=dict(colors=CATEGORICAL[: len(x)], line=dict(color=TRANSPARENT, width=2)),
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,.2f}<extra></extra>",
        ))
    else:
        raise ValueError(f"Unknown chart_type: {kind}")

    fig.update_layout(
        title=dict(text=spec["title"], x=0, xanchor="left", font=dict(size=16)),
        margin=dict(l=10, r=10, t=50, b=10), height=360,
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
        showlegend=(kind == "pie"),
    )
    if kind != "pie":
        fig.update_xaxes(title_text=spec["x_label"], showgrid=False, zeroline=False,
                         type="linear" if kind == "scatter" else "category")
        fig.update_yaxes(title_text=spec["y_label"], showgrid=True, gridcolor=GRID,
                         zeroline=False, separatethousands=True)
    return fig
