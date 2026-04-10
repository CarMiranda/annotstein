"""Shared fixtures for annotstein tests."""

import typing as t

import pytest

from annotstein.coco.schemas import Annotation, Category, Dataset, Image


def make_coco_dataset(
    n_images: int = 5,
    n_categories: int = 3,
    n_annotations_per_image: int = 4,
    image_width: int = 640,
    image_height: int = 480,
) -> Dataset:
    """Build a synthetic COCO Dataset for testing."""
    categories = [Category(id=i, name=f"cat_{i}", supercategory="object") for i in range(n_categories)]
    images = [Image(id=i, file_name=f"image_{i:04d}.jpg", width=image_width, height=image_height) for i in range(n_images)]

    annotations: t.List[Annotation] = []
    ann_id = 0
    for img in images:
        for j in range(n_annotations_per_image):
            cat_id = j % n_categories
            x = float(10 + j * 50)
            y = float(10 + j * 30)
            w = float(80 + j * 10)
            h = float(60 + j * 10)
            annotations.append(
                Annotation(
                    id=ann_id,
                    image_id=img.id,
                    category_id=cat_id,
                    bbox=[x, y, w, h],
                    segmentation=[[x, y, x + w, y, x + w, y + h, x, y + h]],
                )
            )
            ann_id += 1

    return Dataset(categories=categories, images=images, annotations=annotations)


@pytest.fixture
def sample_dataset() -> Dataset:
    return make_coco_dataset()
