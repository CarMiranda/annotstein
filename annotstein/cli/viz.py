"""CLI sub-commands for interactive visualizations of COCO analysis outputs.

Each command reads the JSON output of a corresponding ``annotstein analysis``
command and writes a self-contained interactive HTML file.

Requires the ``viz`` optional dependency group::

    pip install 'annotstein[viz]'
"""

from __future__ import annotations

import json
import pathlib
import typing as t

import typer

app = typer.Typer(help="Interactive HTML visualizations for COCO analysis outputs.")


def _load_json(path: pathlib.Path) -> t.Dict[str, t.Any]:
    return json.loads(path.read_text())


def _write_html(fig: t.Any, output_path: pathlib.Path) -> None:
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    typer.echo(f"Written to {output_path}", err=True)


@app.command(help="Visualize a spatial heatmap from 'annotstein analysis heatmap' output.")
def heatmap(
    *,
    input_path: t.Annotated[
        pathlib.Path,
        typer.Option(help="JSON file produced by 'annotstein analysis heatmap'."),
    ],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Output HTML file.")] = pathlib.Path("heatmap.html"),
) -> None:
    from annotstein.viz.plots import plot_heatmap

    _write_html(plot_heatmap(_load_json(input_path)), output_path)


@app.command(help="Visualize clustering results from 'annotstein analysis cluster' output.")
def clusters(
    *,
    input_path: t.Annotated[
        pathlib.Path,
        typer.Option(help="JSON file produced by 'annotstein analysis cluster'."),
    ],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Output HTML file.")] = pathlib.Path("clusters.html"),
    x_feature: t.Annotated[
        t.Optional[str],
        typer.Option(help="Feature name for the x-axis (default: first feature)."),
    ] = None,
    y_feature: t.Annotated[
        t.Optional[str],
        typer.Option(help="Feature name for the y-axis (default: second feature)."),
    ] = None,
) -> None:
    from annotstein.viz.plots import plot_clusters

    _write_html(plot_clusters(_load_json(input_path), x_feature=x_feature, y_feature=y_feature), output_path)


@app.command(name="aspect-ratios", help="Visualize bbox aspect ratio histogram from 'annotstein analysis aspect-ratios' output.")
def aspect_ratios(
    *,
    input_path: t.Annotated[
        pathlib.Path,
        typer.Option(help="JSON file produced by 'annotstein analysis aspect-ratios'."),
    ],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Output HTML file.")] = pathlib.Path("aspect_ratios.html"),
) -> None:
    from annotstein.viz.plots import plot_aspect_ratios

    _write_html(plot_aspect_ratios(_load_json(input_path)), output_path)


@app.command(name="class-distribution", help="Visualize annotation counts per category from 'annotstein analysis eda' output.")
def class_distribution(
    *,
    input_path: t.Annotated[
        pathlib.Path,
        typer.Option(
            help="JSON file with 'counts' and 'imbalance_ratio' keys "
            "(e.g. the 'class_distribution' field from 'annotstein analysis eda')."
        ),
    ],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Output HTML file.")] = pathlib.Path("class_distribution.html"),
) -> None:
    from annotstein.viz.plots import plot_class_distribution

    data = _load_json(input_path)
    # Accept either the full EDA report or the class_distribution sub-dict directly
    if "class_distribution" in data:
        data = data["class_distribution"]
    _write_html(plot_class_distribution(data), output_path)


@app.command(name="annotation-density", help="Visualize annotations-per-image histogram from 'annotstein analysis eda' output.")
def annotation_density(
    *,
    input_path: t.Annotated[
        pathlib.Path,
        typer.Option(
            help="JSON file with 'stats' and 'histogram' keys "
            "(e.g. the 'annotation_density' field from 'annotstein analysis eda')."
        ),
    ],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Output HTML file.")] = pathlib.Path("annotation_density.html"),
) -> None:
    from annotstein.viz.plots import plot_annotation_density

    data = _load_json(input_path)
    if "annotation_density" in data:
        data = data["annotation_density"]
    _write_html(plot_annotation_density(data), output_path)


@app.command(help="Generate a full EDA dashboard from 'annotstein analysis eda' output.")
def eda(
    *,
    input_path: t.Annotated[
        pathlib.Path,
        typer.Option(help="JSON file produced by 'annotstein analysis eda'."),
    ],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Output HTML file.")] = pathlib.Path("eda_dashboard.html"),
) -> None:
    from annotstein.viz.plots import plot_eda_dashboard

    _write_html(plot_eda_dashboard(_load_json(input_path)), output_path)
