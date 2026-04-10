"""CLI sub-commands for EDA and clustering analysis of COCO datasets."""

import json
import pathlib
import typing as t

import typer

from annotstein.coco.ops import COCO

app = typer.Typer(help="Exploratory data analysis and clustering for COCO datasets.")


def _output(data: t.Any, output_path: t.Optional[pathlib.Path]) -> None:
    serialised = json.dumps(data, indent=2, default=str)
    if output_path is None:
        typer.echo(serialised)
    else:
        output_path.write_text(serialised)
        typer.echo(f"Written to {output_path}", err=True)


@app.command(help="Print a full EDA summary report for a COCO dataset.")
def eda(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="COCO annotation JSON file.")],
    output_path: t.Annotated[t.Optional[pathlib.Path], typer.Option(help="Write JSON to this file (default: stdout).")] = None,
) -> None:
    coco = COCO.read_from(input_path)
    _output(coco.eda(), output_path)


@app.command(help="Cluster annotations by bbox geometry or image-crop features.")
def cluster(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="COCO annotation JSON file.")],
    n_clusters: t.Annotated[int, typer.Option(help="Number of clusters (KMeans only).")] = 8,
    method: t.Annotated[str, typer.Option(help="Clustering algorithm: 'kmeans' or 'dbscan'.")] = "kmeans",
    images_dir: t.Annotated[
        t.Optional[pathlib.Path],
        typer.Option(help="Root directory for images. If provided, cluster by crop colour histograms."),
    ] = None,
    output_path: t.Annotated[t.Optional[pathlib.Path], typer.Option(help="Write JSON to this file (default: stdout).")] = None,
) -> None:
    from dataclasses import asdict

    coco = COCO.read_from(input_path)
    if images_dir is not None:
        result = coco.cluster_crops(images_root=images_dir, n_clusters=n_clusters, method=method)
    else:
        result = coco.cluster_bboxes(n_clusters=n_clusters, method=method)

    _output(asdict(result), output_path)


@app.command(help="Print a 2D spatial heatmap of bbox centre coordinates.")
def heatmap(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="COCO annotation JSON file.")],
    grid_size: t.Annotated[int, typer.Option(help="Grid resolution (grid_size × grid_size cells).")] = 10,
    output_path: t.Annotated[t.Optional[pathlib.Path], typer.Option(help="Write JSON to this file (default: stdout).")] = None,
) -> None:
    coco = COCO.read_from(input_path)
    _output(coco.spatial_distribution(grid_size=grid_size), output_path)


@app.command(help="Print aspect ratio histogram for bboxes in a COCO dataset.")
def aspect_ratios(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="COCO annotation JSON file.")],
    n_buckets: t.Annotated[int, typer.Option(help="Number of histogram bins.")] = 10,
    output_path: t.Annotated[t.Optional[pathlib.Path], typer.Option(help="Write JSON to this file (default: stdout).")] = None,
) -> None:
    coco = COCO.read_from(input_path)
    _output(coco.bbox_aspect_ratio_buckets(n_buckets=n_buckets), output_path)
