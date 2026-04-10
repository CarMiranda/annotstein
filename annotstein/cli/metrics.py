"""CLI sub-commands for annotation quality and detection metrics."""

import json
import pathlib
import typing as t

import typer

from annotstein.coco.ops import COCO

app = typer.Typer(help="Annotation quality and detection metrics for COCO datasets.")


def _output(data: t.Any, output_path: t.Optional[pathlib.Path]) -> None:
    serialised = json.dumps(data, indent=2, default=str)
    if output_path is None:
        typer.echo(serialised)
    else:
        output_path.write_text(serialised)
        typer.echo(f"Written to {output_path}", err=True)


@app.command(help="Run all annotation quality checks and print a report.")
def quality(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="COCO annotation JSON file.")],
    dup_iou: t.Annotated[float, typer.Option(help="IoU threshold for duplicate detection.")] = 0.9,
    conflict_iou: t.Annotated[float, typer.Option(help="IoU threshold for label conflict detection.")] = 0.5,
    output_path: t.Annotated[t.Optional[pathlib.Path], typer.Option(help="Write JSON to this file (default: stdout).")] = None,
) -> None:
    coco = COCO.read_from(input_path)
    _output(coco.quality_report(dup_iou=dup_iou, conflict_iou=conflict_iou), output_path)


@app.command(help="Compute COCO detection metrics (AP/AR) from ground-truth and predictions.")
def ap(
    *,
    gt_path: t.Annotated[pathlib.Path, typer.Option(help="Ground-truth COCO annotation JSON file.")],
    predictions_path: t.Annotated[
        pathlib.Path,
        typer.Option(help="Predictions COCO JSON file; annotations must have a 'score' field."),
    ],
    output_path: t.Annotated[t.Optional[pathlib.Path], typer.Option(help="Write JSON to this file (default: stdout).")] = None,
) -> None:
    gt = COCO.read_from(gt_path)
    preds = COCO.read_from(predictions_path)
    _output(gt.coco_metrics(preds), output_path)
