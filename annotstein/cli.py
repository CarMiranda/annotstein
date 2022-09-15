import enum
import pathlib
import typing as t
import typer
from collections import Counter

from annotstein.coco import COCO
from annotstein.voc import VOC


class CoordinateKind(str, enum.Enum):
    relative = "rel"
    absolute = "abs"


class Formats(str, enum.Enum):
    coco = "coco"
    voc = "voc"


app = typer.Typer()


@app.command(help="Convert from one annotation format into another.")
def convert(
    *,
    input_path: pathlib.Path = typer.Option(..., help="Source path to read from."),
    output_path: pathlib.Path = typer.Option(..., help="Target path to write into."),
    source: Formats = typer.Option(..., help="Input path format."),
    target: Formats = typer.Option(..., help="Output path format."),
):
    if source == Formats.voc and target == Formats.coco:
        voc = VOC.parse_xml(input_path)
        COCO.from_dict(voc).write(output_path)
    else:
        raise NotImplementedError()


@app.command(help="Transform coordinates from absolute to relative, or vice versa.")
def coordinates(
    *,
    input_path: pathlib.Path = typer.Option(..., help="Source files to read from."),
    images_dir: pathlib.Path = typer.Option(
        ...,
        help="Directory containing images referenced in the annotations. Image dimensions will be read to transform coordinates.",
    ),
    output_path: pathlib.Path = typer.Option(..., help="Target files to write into."),
    to: CoordinateKind = typer.Option(..., help="Coordinate system to transform into."),
):
    index = COCO.from_file(input_path)
    if to == CoordinateKind.relative:
        index.to_relative_coordinates(images_dir).write(output_path)
    elif to == CoordinateKind.absolute:
        index.to_absolute_coordinates(images_dir).write(output_path)


@app.command(help="Extract image patches into a class-base file tree.")
def extract(
    *,
    input_path: pathlib.Path = typer.Option(..., help="Source file to read from."),
    images_dir: pathlib.Path = typer.Option(
        ...,
        help="Directory containing images referenced in the annotations. Image dimensions will be read to transform coordinates.",
    ),
    output_path: pathlib.Path = typer.Option(
        ..., help="Target directory to write into."
    ),
):
    COCO.from_file(input_path).to_classtree(images_dir, output_path)


@app.command(help="Merge multiple COCO annotations into a single one.")
def merge(
    *,
    input_paths: t.List[pathlib.Path] = typer.Option(
        ..., help="Source files to read from."
    ),
    output_file: pathlib.Path = typer.Option(..., help="Target file to write into."),
):
    indices = []
    for input_path in input_paths:
        indices.append(COCO.from_file(input_path))
    COCO.merge(*indices).write(output_file)


@app.command(help="Rebase image file names using the given prefix.")
def rebase(
    *,
    input_path: pathlib.Path = typer.Option(..., help="Source file to read from."),
    prefix: pathlib.Path = typer.Option(
        ..., help="Prefix to use for each image entry."
    ),
    output_path: pathlib.Path = typer.Option(..., help="Target file to write to."),
):
    COCO.from_file(input_path).rebase_filenames(prefix).write(output_path)

@app.command(help="Print statistics about the given annotation file.")
def stats(
    *,
    input_path: pathlib.Path = typer.Option(..., help="Source file to read from.")
):
    coco = COCO.from_file(input_path)
    category_counts = Counter([coco.category_index[c.category_id].name for c in coco.annotations])

    print("Category counts:")
    for name, count in category_counts.items():
        print(f"- {name}: {count}")

@app.command(help="Split a given dataset index into train and test indices.")
def split(
    *,
    input_path: pathlib.Path = typer.Option(..., help="Source file to read from."),
    output_path: pathlib.Path = typer.Option(..., help="Target file to write to."),
    test_ratio: float = typer.Option(..., help="Ratio (between 0 and 1) for the test split."),
    train_ratio: t.Optional[float] = typer.Option(None, help="Ratio (between 0 and 1) for the train split. Use in with test_ratio in order to also get a validation split.")
):
    coco = COCO.from_file(input_path)
    if train_ratio is None:
        coco_train, coco_test = coco.split(test_ratio)
    else:
        coco_train, coco_val = coco.split(test_ratio=(1 - train_ratio))
        coco_val, coco_test = coco_val.split(test_ratio=test_ratio / (1 - train_ratio))


    if output_path.is_dir():
        train_path = output_path / (input_path.stem + "_train.json")
        val_path = output_path / (input_path.stem + "_val.json")
        test_path = output_path / (input_path.stem + "_test.json")
    else:
        train_path = output_path.parent / (output_path.stem + "_train.json")
        val_path = output_path.parent / (output_path.stem + "_val.json")
        test_path = output_path.parent / (output_path.stem + "_test.json")

    coco_train.write(train_path)
    coco_test.write(test_path)

    if train_ratio is not None:
        coco_val.write(val_path)


