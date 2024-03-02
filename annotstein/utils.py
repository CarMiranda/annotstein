import typing as t
import pathlib
from annotstein.common import IMAGE_SUFFIXES


def glob_images(path: pathlib.Path, recursive: bool = True, suffixes: t.List[str] = IMAGE_SUFFIXES):
    if not path.is_dir():
        raise ValueError("`path` should be a directory containing images.")

    if recursive:
        images_list = list(sorted(path.glob("**/*")))
    else:
        images_list = list(sorted(path.glob("*")))

    available_images = [f for f in images_list if f.suffix in suffixes]

    return available_images
