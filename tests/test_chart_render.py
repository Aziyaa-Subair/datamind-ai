"""Renderer tests. Skipped automatically if Plotly is not installed."""

import pytest

pytest.importorskip("plotly")

from app import config  # noqa: E402
from app.services import charts  # noqa: E402
from app.services.chart_render import to_figure  # noqa: E402
from app.services.dataset_service import load_dataset  # noqa: E402


def test_every_chart_type_renders_one_trace():
    df = load_dataset(config.DEFAULT_DATASET)
    specs = [
        charts.build_chart(df, "bar", "Product", "Revenue"),
        charts.build_chart(df, "line", "OrderDate", "Revenue"),
        charts.build_chart(df, "pie", "Category", "Revenue"),
        charts.build_chart(df, "histogram", "Revenue"),
        charts.build_chart(df, "scatter", "Quantity", "Revenue"),
    ]
    for spec in specs:
        fig = to_figure(spec)
        assert len(fig.data) == 1, spec["chart_type"]
        assert fig.layout.title.text == spec["title"]


def test_bar_figure_keeps_the_computed_numbers():
    df = load_dataset(config.DEFAULT_DATASET)
    fig = to_figure(charts.build_chart(df, "bar", "Product", "Revenue"))
    assert list(fig.data[0].y)[0] == 20400.0
    assert list(fig.data[0].x)[0] == "Laptop"
    # Only the largest bar carries a value label.
    assert [t for t in fig.data[0].text if t] == ["20,400"]


def test_unknown_chart_type_is_rejected():
    with pytest.raises(ValueError):
        to_figure({"chart_type": "donut", "title": "", "x_label": "", "y_label": "",
                   "x": [], "y": [], "notes": []})
