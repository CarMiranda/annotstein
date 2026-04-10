"""Tests for annotstein.viz.plots — interactive visualization functions."""

import pytest

from tests.conftest import make_coco_dataset
from annotstein.analysis.eda import (
    spatial_distribution,
    bbox_aspect_ratio_buckets,
    class_distribution,
    annotation_density,
    summary,
)


@pytest.fixture
def sample_ds():
    return make_coco_dataset(n_images=20, n_categories=3, n_annotations_per_image=2)


@pytest.fixture
def heatmap_data(sample_ds):
    return spatial_distribution(sample_ds)


@pytest.fixture
def aspect_ratio_data(sample_ds):
    return bbox_aspect_ratio_buckets(sample_ds)


@pytest.fixture
def class_dist_data(sample_ds):
    return class_distribution(sample_ds)


@pytest.fixture
def density_data(sample_ds):
    return annotation_density(sample_ds)


@pytest.fixture
def eda_data(sample_ds):
    return summary(sample_ds)


@pytest.fixture
def cluster_data():
    """Hand-crafted cluster result dict — avoids sklearn dependency."""
    return {
        "labels": [0, 0, 1, 1, 2, -1],
        "annotation_ids": [1, 2, 3, 4, 5, 6],
        "centroids": {
            "0": [100.0, 80.0, 1.25, 8000.0],
            "1": [50.0, 50.0, 1.0, 2500.0],
            "2": [200.0, 150.0, 1.33, 30000.0],
        },
        "n_clusters": 3,
        "inertia": 12345.6,
        "feature_names": ["width", "height", "aspect_ratio", "area"],
        "features": [
            [100.0, 80.0, 1.25, 8000.0],
            [110.0, 85.0, 1.29, 9350.0],
            [50.0, 50.0, 1.0, 2500.0],
            [55.0, 52.0, 1.06, 2860.0],
            [200.0, 150.0, 1.33, 30000.0],
            [10.0, 5.0, 2.0, 50.0],
        ],
    }


def test_plot_heatmap_returns_figure(heatmap_data):
    from annotstein.viz.plots import plot_heatmap
    import plotly.graph_objects as go

    fig = plot_heatmap(heatmap_data)
    assert isinstance(fig, go.Figure)


def test_plot_heatmap_has_heatmap_trace(heatmap_data):
    from annotstein.viz.plots import plot_heatmap
    import plotly.graph_objects as go

    fig = plot_heatmap(heatmap_data)
    assert any(isinstance(t, go.Heatmap) for t in fig.data)


def test_plot_heatmap_title_contains_grid_size(heatmap_data):
    from annotstein.viz.plots import plot_heatmap

    fig = plot_heatmap(heatmap_data)
    assert "10" in fig.layout.title.text  # default grid_size=10


def test_plot_clusters_returns_figure(cluster_data):
    from annotstein.viz.plots import plot_clusters
    import plotly.graph_objects as go

    fig = plot_clusters(cluster_data)
    assert isinstance(fig, go.Figure)


def test_plot_clusters_has_scatter_traces(cluster_data):
    from annotstein.viz.plots import plot_clusters
    import plotly.graph_objects as go

    fig = plot_clusters(cluster_data)
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) > 0


def test_plot_clusters_custom_features(cluster_data):
    from annotstein.viz.plots import plot_clusters

    feature_names = cluster_data["feature_names"]
    fig = plot_clusters(cluster_data, x_feature=feature_names[0], y_feature=feature_names[-1])
    assert fig.layout.xaxis.title.text == feature_names[0]


def test_plot_clusters_missing_features_raises():
    from annotstein.viz.plots import plot_clusters

    bad_data = {
        "labels": [0, 1],
        "annotation_ids": [1, 2],
        "centroids": {},
        "n_clusters": 2,
        "feature_names": ["w"],
        "features": None,
    }
    with pytest.raises(ValueError, match="features"):
        plot_clusters(bad_data)


def test_plot_aspect_ratios_returns_figure(aspect_ratio_data):
    from annotstein.viz.plots import plot_aspect_ratios
    import plotly.graph_objects as go

    fig = plot_aspect_ratios(aspect_ratio_data)
    assert isinstance(fig, go.Figure)


def test_plot_aspect_ratios_has_bar_traces(aspect_ratio_data):
    from annotstein.viz.plots import plot_aspect_ratios
    import plotly.graph_objects as go

    fig = plot_aspect_ratios(aspect_ratio_data)
    assert any(isinstance(t, go.Bar) for t in fig.data)


def test_plot_aspect_ratios_empty_data():
    from annotstein.viz.plots import plot_aspect_ratios

    fig = plot_aspect_ratios({"overall": {"edges": [], "counts": []}})
    assert fig is not None


def test_plot_class_distribution_returns_figure(class_dist_data):
    from annotstein.viz.plots import plot_class_distribution
    import plotly.graph_objects as go

    fig = plot_class_distribution(class_dist_data)
    assert isinstance(fig, go.Figure)


def test_plot_class_distribution_has_bar(class_dist_data):
    from annotstein.viz.plots import plot_class_distribution
    import plotly.graph_objects as go

    fig = plot_class_distribution(class_dist_data)
    assert any(isinstance(t, go.Bar) for t in fig.data)


def test_plot_class_distribution_imbalance_in_title(class_dist_data):
    from annotstein.viz.plots import plot_class_distribution

    fig = plot_class_distribution(class_dist_data)
    if class_dist_data.get("imbalance_ratio") is not None:
        assert "imbalance" in fig.layout.title.text


def test_plot_annotation_density_returns_figure(density_data):
    from annotstein.viz.plots import plot_annotation_density
    import plotly.graph_objects as go

    fig = plot_annotation_density(density_data)
    assert isinstance(fig, go.Figure)


def test_plot_annotation_density_has_bar(density_data):
    from annotstein.viz.plots import plot_annotation_density
    import plotly.graph_objects as go

    fig = plot_annotation_density(density_data)
    assert any(isinstance(t, go.Bar) for t in fig.data)


def test_plot_eda_dashboard_returns_figure(eda_data):
    from annotstein.viz.plots import plot_eda_dashboard
    import plotly.graph_objects as go

    fig = plot_eda_dashboard(eda_data)
    assert isinstance(fig, go.Figure)


def test_plot_eda_dashboard_has_multiple_traces(eda_data):
    from annotstein.viz.plots import plot_eda_dashboard

    fig = plot_eda_dashboard(eda_data)
    assert len(fig.data) >= 3  # at least class dist + density + aspect ratios


def test_plot_eda_dashboard_title_contains_counts(eda_data):
    from annotstein.viz.plots import plot_eda_dashboard

    fig = plot_eda_dashboard(eda_data)
    title = fig.layout.title.text
    assert "images" in title
    assert "annotations" in title


def test_cli_heatmap_writes_html(heatmap_data, tmp_path):
    import json
    from annotstein.cli.viz import heatmap

    in_path = tmp_path / "heatmap.json"
    out_path = tmp_path / "heatmap.html"
    in_path.write_text(json.dumps(heatmap_data))
    heatmap(input_path=in_path, output_path=out_path)
    assert out_path.exists()
    assert "<html" in out_path.read_text().lower()


def test_cli_clusters_writes_html(cluster_data, tmp_path):
    import json
    from annotstein.cli.viz import clusters

    in_path = tmp_path / "clusters.json"
    out_path = tmp_path / "clusters.html"
    in_path.write_text(json.dumps(cluster_data))
    clusters(input_path=in_path, output_path=out_path)
    assert out_path.exists()
    assert "<html" in out_path.read_text().lower()


def test_cli_eda_writes_html(eda_data, tmp_path):
    import json
    from annotstein.cli.viz import eda

    in_path = tmp_path / "eda.json"
    out_path = tmp_path / "eda.html"
    in_path.write_text(json.dumps(eda_data, default=str))
    eda(input_path=in_path, output_path=out_path)
    assert out_path.exists()
    assert "<html" in out_path.read_text().lower()
