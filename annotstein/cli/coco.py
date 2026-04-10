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


@app.command(help="Split a given dataset index into train, validation, and test indices.")
def split(
    *,
    input_path: t.Annotated[pathlib.Path, typer.Option(help="Source file to read from.")],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Target file or directory to write to.")],
    val_ratio: t.Annotated[float, typer.Option(help="Fraction of the dataset to use for validation (0–1).")] = 0.0,
    test_ratio: t.Annotated[float, typer.Option(help="Fraction of the dataset to use for testing (0–1).")],
    shuffle: t.Annotated[bool, typer.Option(help="Whether to shuffle the dataset prior to splitting.")] = True,
    stratify: t.Annotated[bool, typer.Option(help="Whether to stratify the split based on categories.")] = True,
    keep_empty: t.Annotated[bool, typer.Option(help="Whether to keep images without annotations.")] = True,
):
    coco = COCO.read_from(input_path)

    if output_path.is_dir():
        stem = input_path.stem
        parent = output_path
    else:
        stem = output_path.stem
        parent = output_path.parent

    if val_ratio > 0.0:
        coco_train, coco_val, coco_test = coco.train_val_test_split(val_ratio, test_ratio, shuffle, stratify, keep_empty)
        coco_val.write(parent / f"{stem}_val.json")
    else:
        coco_train, coco_test = coco.train_test_split(1.0 - test_ratio, shuffle, stratify, keep_empty)

    coco_train.write(parent / f"{stem}_train.json")
    coco_test.write(parent / f"{stem}_test.json")


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


@app.command(name="from-dir", help="Create a COCO annotation file from a classification directory tree.")
def from_dir(
    *,
    input_path: t.Annotated[
        pathlib.Path,
        typer.Option(help="Root directory; each immediate subdirectory is a class."),
    ],
    output_path: t.Annotated[pathlib.Path, typer.Option(help="Target COCO JSON file to write.")],
    recursive: t.Annotated[
        bool,
        typer.Option(help="Recurse into subdirectories of each class folder."),
    ] = False,
):
    COCO.from_classification_dir(input_path, recursive=recursive).write(output_path)
    print(f"Written to {output_path}")
