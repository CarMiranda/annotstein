import pathlib
import typing as t
import enum
import typer

from annotstein.cli import coco
from annotstein.coco.ops import COCO
from annotstein.voc.ops import VOC
from annotstein.yolo.common import DetectionTask, SegmentationTask, DetectionDataset, SegmentationDataset

app = typer.Typer()
app.add_typer(coco.app, name="coco", help="Perform operations on COCO datasets.")


class Formats(str, enum.Enum):
    coco = "coco"
    voc = "voc"
    yolo_od_task = "yolo-od-task"
    yolo_od_split = "yolo-od-split"
    yolo_seg_task = "yolo-seg-task"
    yolo_seg_split = "yolo-seg-split"


@app.command(help="Convert from one annotation format into another.")
def convert(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="Source path to read from.")],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Target path to write into.")],
    source: t.Annotated[Formats, typer.Option(help="Input path format.")],
    target: t.Annotated[Formats, typer.Option(help="Output path format.")],
):
    if source == "voc" and target == "coco":
        ds = VOC.parse_xml(input_path)
        COCO.from_dict(ds).write(output_path)

    elif source == "yolo-od-split" and target == "coco":
        ags = DetectionDataset.parse_directory(input_path)
        ds = DetectionDataset.model_construct(annotations=ags).to_coco()
        COCO(ds).write(output_path)

    elif source == "yolo-od-task" and target == "coco":
        ds = DetectionTask.parse_yaml(input_path)
        train_ds, val_ds, test_ds = ds.to_coco()
        COCO(train_ds).write(output_path.stem + "_train.json")
        COCO(val_ds).write(output_path.stem + "_val.json")
        if test_ds is not None:
            COCO(test_ds).write(output_path.stem + "_test.json")

    elif source == "yolo-seg-split" and target == "coco":
        ags = SegmentationDataset.parse_directory(input_path)
        ds = SegmentationDataset(annotations=ags).to_coco()
        COCO(ds).write(output_path)

    elif source == "yolo-seg-task" and target == "coco":
        ds = SegmentationTask.parse_yaml(input_path)
        train_ds, val_ds, test_ds = ds.to_coco()
        COCO(train_ds).write(output_path.stem + "_train.json")
        COCO(val_ds).write(output_path.stem + "_val.json")
        if test_ds is not None:
            COCO(test_ds).write(output_path.stem + "_test.json")

    else:
        raise NotImplementedError()
