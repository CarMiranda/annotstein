"""Plotly-based interactive visualizations for annotstein analysis outputs.

Every public function:

* Accepts the **dict** produced by the corresponding analysis function
  (i.e. the JSON you get from ``annotstein analysis heatmap``, etc.).
* Returns a :class:`plotly.graph_objects.Figure` that can be saved with
  ``.write_html(path)`` or rendered in a notebook.

Import guard
------------
All imports of ``plotly`` are local so that the rest of the library still
works even when plotly is not installed.  A clear ``ImportError`` is raised
at call time if plotly is missing.
"""

from __future__ import annotations

import typing as t


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:
        raise ImportError("plotly is required for visualizations. Install it with:  pip install 'annotstein[viz]'") from exc


def plot_heatmap(data: t.Dict[str, t.Any]) -> t.Any:
    """Interactive heatmap of bbox centre spatial distribution.

    Args:
        data: Dict produced by :func:`annotstein.analysis.eda.spatial_distribution`
              (keys: ``grid``, ``grid_size``, ``skipped_images``).

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.
    """
    _require_plotly()
    import plotly.graph_objects as go

    grid = data["grid"]
    grid_size = data.get("grid_size", len(grid))
    skipped = data.get("skipped_images", 0)

    tick_vals = [i / grid_size for i in range(grid_size + 1)]
    tick_text = [f"{v:.1f}" for v in tick_vals]

    fig = go.Figure(
        go.Heatmap(
            z=grid,
            colorscale="Viridis",
            colorbar={"title": "Count"},
            hovertemplate="col: %{x}<br>row: %{y}<br>count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Bbox Centre Spatial Distribution (grid {grid_size}×{grid_size}, {skipped} skipped)",
        xaxis=dict(title="Normalised X", tickvals=list(range(grid_size)), ticktext=tick_text[:-1]),
        yaxis=dict(title="Normalised Y", tickvals=list(range(grid_size)), ticktext=tick_text[:-1], autorange="reversed"),
        template="plotly_white",
    )
    return fig


def plot_clusters(
    data: t.Dict[str, t.Any],
    x_feature: t.Optional[str] = None,
    y_feature: t.Optional[str] = None,
) -> t.Any:
    """Interactive scatter plot of clustering results.

    Each point is one annotation, coloured by its cluster label.  Cluster
    centroids are drawn as larger marker symbols on top.

    Args:
        data: Dict produced by :func:`annotstein.analysis.clustering.cluster_bboxes`
              or :func:`~annotstein.analysis.clustering.cluster_crops`
              (keys: ``labels``, ``annotation_ids``, ``centroids``,
              ``feature_names``, ``features``).
        x_feature: Name of the feature to use on the x-axis.  Defaults to
            the first feature in ``feature_names``.
        y_feature: Name of the feature to use on the y-axis.  Defaults to
            the second feature in ``feature_names``.

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.

    Raises:
        ValueError: If ``features`` is absent from *data* (run the cluster
            command with a recent version of annotstein that includes raw
            feature values in the output).
    """
    _require_plotly()
    import plotly.graph_objects as go

    if not data.get("features"):
        raise ValueError(
            "'features' key is missing from cluster data. Re-run 'annotstein analysis cluster' with this version of annotstein."
        )

    feature_names: t.List[str] = data["feature_names"]
    features: t.List[t.List[float]] = data["features"]
    labels: t.List[int] = data["labels"]
    centroids: t.Dict[str, t.List[float]] = data["centroids"]

    x_feat = x_feature or (feature_names[0] if feature_names else "x")
    y_feat = y_feature or (feature_names[1] if len(feature_names) > 1 else feature_names[0])

    x_idx = feature_names.index(x_feat) if x_feat in feature_names else 0
    y_idx = feature_names.index(y_feat) if y_feat in feature_names else min(1, len(feature_names) - 1)

    xs = [row[x_idx] for row in features]
    ys = [row[y_idx] for row in features]
    ann_ids = data["annotation_ids"]

    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    colorscale = "Turbo"

    import plotly.colors as pc

    colours = pc.sample_colorscale(colorscale, [i / max(n_labels - 1, 1) for i in range(n_labels)])
    label_to_colour = {lbl: colours[i] for i, lbl in enumerate(unique_labels)}

    fig = go.Figure()

    for lbl in unique_labels:
        mask = [i for i, lbl_i in enumerate(labels) if lbl_i == lbl]
        label_str = "noise" if lbl == -1 else f"cluster {lbl}"
        colour = label_to_colour[lbl]
        fig.add_trace(
            go.Scatter(
                x=[xs[i] for i in mask],
                y=[ys[i] for i in mask],
                mode="markers",
                name=label_str,
                marker=dict(color=colour, size=5, opacity=0.65),
                text=[f"ann_id={ann_ids[i]}" for i in mask],
                hovertemplate=f"<b>{label_str}</b><br>{x_feat}: %{{x:.2f}}<br>{y_feat}: %{{y:.2f}}<br>%{{text}}<extra></extra>",
            )
        )

    # Overlay centroids
    for lbl_str, centroid in centroids.items():
        lbl_int = int(lbl_str)
        cx = centroid[x_idx] if len(centroid) > x_idx else 0
        cy = centroid[y_idx] if len(centroid) > y_idx else 0
        colour = label_to_colour.get(lbl_int, "black")
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                name=f"centroid {lbl_int}",
                marker=dict(symbol="x", size=14, color=colour, line=dict(width=2, color="black")),
                showlegend=False,
                hovertemplate=f"centroid {lbl_int}<br>{x_feat}: {cx:.2f}<br>{y_feat}: {cy:.2f}<extra></extra>",
            )
        )

    n_clusters = data.get("n_clusters", len(centroids))
    inertia = data.get("inertia")
    title = f"Cluster Scatter — {n_clusters} clusters"
    if inertia is not None:
        title += f" (inertia: {inertia:.1f})"

    fig.update_layout(
        title=title,
        xaxis_title=x_feat,
        yaxis_title=y_feat,
        template="plotly_white",
        legend=dict(title="Cluster"),
    )
    return fig


def plot_aspect_ratios(data: t.Dict[str, t.Any]) -> t.Any:
    """Interactive histogram of bbox aspect ratios.

    Args:
        data: Dict produced by
              :func:`annotstein.analysis.eda.bbox_aspect_ratio_buckets`
              (keys: ``overall``, optionally ``per_category``).

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.
    """
    _require_plotly()
    import plotly.graph_objects as go

    fig = go.Figure()

    def _add_bar(hist: t.Dict[str, t.Any], name: str) -> None:
        edges: t.List[float] = hist.get("edges", [])
        counts: t.List[int] = hist.get("counts", [])
        if not edges or not counts:
            return
        midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        fig.add_trace(
            go.Bar(
                x=midpoints,
                y=counts,
                width=widths,
                name=name,
                opacity=0.75,
                hovertemplate=f"<b>{name}</b><br>aspect ratio: %{{x:.2f}}<br>count: %{{y}}<extra></extra>",
            )
        )

    _add_bar(data.get("overall", {}), "overall")
    for cat_name, hist in data.get("per_category", {}).items():
        _add_bar(hist, cat_name)

    fig.update_layout(
        title="Bbox Aspect Ratio Distribution (width / height)",
        xaxis_title="Aspect ratio (w/h)",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        legend=dict(title="Category"),
    )
    return fig


def plot_class_distribution(data: t.Dict[str, t.Any]) -> t.Any:
    """Interactive bar chart of annotation counts per category.

    Args:
        data: Dict produced by
              :func:`annotstein.analysis.eda.class_distribution`
              (keys: ``counts``, ``imbalance_ratio``).

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.
    """
    _require_plotly()
    import plotly.graph_objects as go

    counts: t.Dict[str, int] = data.get("counts", {})
    imbalance = data.get("imbalance_ratio")

    cats = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    vals = [counts[c] for c in cats]

    title = "Annotation Count per Category"
    if imbalance is not None:
        title += f"  (imbalance ratio: {imbalance:.2f}×)"

    fig = go.Figure(
        go.Bar(
            x=cats,
            y=vals,
            marker_color=vals,
            marker_colorscale="Blues",
            hovertemplate="<b>%{x}</b><br>annotations: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Category",
        yaxis_title="Annotation count",
        template="plotly_white",
    )
    return fig


def plot_annotation_density(data: t.Dict[str, t.Any]) -> t.Any:
    """Interactive histogram of annotations per image.

    Args:
        data: Dict produced by
              :func:`annotstein.analysis.eda.annotation_density`
              (keys: ``stats``, ``histogram``).

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.
    """
    _require_plotly()
    import plotly.graph_objects as go

    hist = data.get("histogram", {})
    edges: t.List[float] = hist.get("edges", [])
    counts: t.List[int] = hist.get("counts", [])
    stats = data.get("stats", {})

    if not edges or not counts:
        fig = go.Figure()
        fig.update_layout(title="Annotation Density (no data)", template="plotly_white")
        return fig

    midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]

    mean = stats.get("mean")
    subtitle = f"mean={mean:.2f}" if mean is not None else ""

    fig = go.Figure(
        go.Bar(
            x=midpoints,
            y=counts,
            width=widths,
            name="images",
            marker_color="steelblue",
            opacity=0.8,
            hovertemplate="annotations/image: %{x:.1f}<br>image count: %{y}<extra></extra>",
        )
    )

    if mean is not None:
        fig.add_vline(x=mean, line_dash="dash", line_color="firebrick", annotation_text=f"mean={mean:.2f}")

    fig.update_layout(
        title=f"Annotations per Image  ({subtitle})",
        xaxis_title="Annotations per image",
        yaxis_title="Number of images",
        template="plotly_white",
    )
    return fig


def plot_eda_dashboard(data: t.Dict[str, t.Any]) -> t.Any:
    """Multi-panel interactive EDA dashboard.

    Args:
        data: Dict produced by :func:`annotstein.analysis.eda.summary`
              (the full EDA report).

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure` with four panels:
        class distribution, annotation density, aspect ratio distribution,
        and spatial heatmap.
    """
    _require_plotly()
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    dataset_info = data.get("dataset", {})
    title = (
        f"EDA Dashboard — {dataset_info.get('n_images', '?')} images, "
        f"{dataset_info.get('n_annotations', '?')} annotations, "
        f"{dataset_info.get('n_categories', '?')} categories"
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Class Distribution",
            "Annotations per Image",
            "Bbox Aspect Ratios",
            "Spatial Heatmap",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    counts: t.Dict[str, int] = data.get("class_distribution", {}).get("counts", {})
    cats = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    fig.add_trace(
        go.Bar(
            x=cats,
            y=[counts[c] for c in cats],
            name="annotations",
            marker_color="steelblue",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>count: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    density = data.get("annotation_density", {})
    hist = density.get("histogram", {})
    edges: t.List[float] = hist.get("edges", [])
    ann_counts: t.List[int] = hist.get("counts", [])
    if edges and ann_counts:
        midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        fig.add_trace(
            go.Bar(
                x=midpoints,
                y=ann_counts,
                width=widths,
                name="images",
                marker_color="mediumseagreen",
                showlegend=False,
                hovertemplate="ann/img: %{x:.1f}<br>images: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        mean = density.get("stats", {}).get("mean")
        if mean is not None:
            fig.add_vline(x=mean, line_dash="dash", line_color="firebrick", row=1, col=2)

    ar_data = data.get("bbox_aspect_ratio_buckets", {})
    overall_hist = ar_data.get("overall", {})
    ar_edges: t.List[float] = overall_hist.get("edges", [])
    ar_counts: t.List[int] = overall_hist.get("counts", [])
    if ar_edges and ar_counts:
        ar_mid = [(ar_edges[i] + ar_edges[i + 1]) / 2 for i in range(len(ar_edges) - 1)]
        ar_widths = [ar_edges[i + 1] - ar_edges[i] for i in range(len(ar_edges) - 1)]
        fig.add_trace(
            go.Bar(
                x=ar_mid,
                y=ar_counts,
                width=ar_widths,
                name="aspect ratios",
                marker_color="mediumpurple",
                showlegend=False,
                hovertemplate="w/h: %{x:.2f}<br>count: %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    sd = data.get("spatial_distribution", {})
    grid = sd.get("grid")
    if grid:
        fig.add_trace(
            go.Heatmap(
                z=grid,
                colorscale="Viridis",
                showscale=False,
                hovertemplate="col: %{x}<br>row: %{y}<br>count: %{z}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        height=700,
    )
    fig.update_xaxes(title_text="Category", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Annotations / image", row=1, col=2)
    fig.update_yaxes(title_text="Images", row=1, col=2)
    fig.update_xaxes(title_text="Aspect ratio (w/h)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Normalised X", row=2, col=2)
    fig.update_yaxes(title_text="Normalised Y", autorange="reversed", row=2, col=2)

    return fig
