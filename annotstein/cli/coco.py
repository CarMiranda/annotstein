import enum
import typer
import typing as t
import pathlib
from collections import Counter

from annotstein.coco.ops import COCO


app = typer.Typer()


class CoordinateKind(str, enum.Enum):
    relative = "rel"
    absolute = "abs"


@app.command(help="Transform coordinates from absolute to relative, or vice versa.")
def coordinates(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="Source files to read from.")],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Target files to write into.")],
    to: t.Annotated[CoordinateKind, typer.Option(help="Coordinate system to transform into.")],
):
    index = COCO.read_from(input_path)
    if to == CoordinateKind.relative:
        index.to_relative_coordinates().write(output_path)
    elif to == CoordinateKind.absolute:
        index.to_absolute_coordinates().write(output_path)


@app.command(help="Extract image patches into a class-base file tree.")
def crop(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="Source file to read from.")],
    images_dir: t.Annotated[
        pathlib.Path,
        typer.Option(
            help="""Directory containing images referenced in the annotations. \
                    Image dimensions will be read to transform coordinates.""",
        ),
    ],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Target directory to write into.")],
):
    COCO.read_from(input_path).to_classtree(output_path, images_dir)


@app.command(help="Merge multiple COCO annotations into a single one.")
def merge(
    *,
    input_paths: t.Annotated[t.List[pathlib.Path], typer.Option(help="Source files to read from.")],
    output_file: t.Annotated[pathlib.Path, typer.Option(help="Target file to write into.")],
):
    indices = []
    for input_path in input_paths:
        indices.append(COCO.read_from(input_path))
    COCO.merge(*indices).write(output_file)


@app.command(help="Rebase image file names using the given prefix.")
def rebase(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="Source file to read from.")],
    prefix: t.Annotated[pathlib.Path, typer.Option(help="Prefix to use for each image entry.")],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Target file to write to.")],
):
    COCO.read_from(input_path).rebase_filenames(prefix).write(output_path)


@app.command(help="Split a given dataset index into train and test indices.")
def split(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="Source file to read from.")],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Target file to write to.")],
    test_ratio: t.Annotated[float, typer.Option(help="Ratio (between 0 and 1) for the test split.")],
    train_ratio: t.Annotated[
        t.Optional[float],
        typer.Option(
            help="""Ratio (between 0 and 1) for the train split. \
                                                  Use it with test_ratio in order to also get a validation split."""
        ),
    ] = None,
    shuffle: t.Annotated[bool, typer.Option(help="Whether to shuffle the dataset prior to splitting.")] = True,
    stratify: t.Annotated[bool, typer.Option(help="Whether to stratify the split based on categories.")] = True,
    keep_empty: t.Annotated[bool, typer.Option(help="Whether to keep images without annotations.")] = True,
):
    coco = COCO.read_from(input_path)
    if train_ratio is None:
        coco_train, coco_test = coco.train_test_split(test_ratio, shuffle, stratify, keep_empty)
        coco_val = None
    else:
        coco_train, coco_val = coco.train_test_split((1 - train_ratio), shuffle, stratify, keep_empty)
        coco_val, coco_test = coco_val.train_test_split(test_ratio / (1 - train_ratio), shuffle, stratify, keep_empty)

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

    if coco_val is not None:
        coco_val.write(val_path)


@app.command(help="Print statistics about the given annotation file.")
def stats(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="Source file to read from.")],
):
    coco = COCO.read_from(input_path)
    category_counts = Counter([coco.category_index[c.category_id].name for c in coco.annotations])

    print("Category counts:")
    for name, count in category_counts.items():
        print(f"- {name}: {count}")
