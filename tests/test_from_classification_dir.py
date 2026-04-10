"""Tests for COCO.from_classification_dir / from_classification_dir."""

import pathlib

import pytest
from PIL import Image as PImage

from annotstein.coco.ops import COCO, from_classification_dir


@pytest.fixture
def classification_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a minimal classification directory tree with real images."""
    classes = {"cat": 3, "dog": 2, "bird": 1}
    for class_name, n_images in classes.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(n_images):
            img = PImage.new("RGB", (100 + i * 10, 80 + i * 5), color=(i * 50, 0, 0))
            img.save(class_dir / f"img_{i:04d}.jpg")
    return tmp_path


def test_categories_match_subdirs(classification_dir):
    ds = from_classification_dir(classification_dir)
    category_names = {c.name for c in ds.categories}
    assert category_names == {"cat", "dog", "bird"}


def test_category_ids_are_unique(classification_dir):
    ds = from_classification_dir(classification_dir)
    ids = [c.id for c in ds.categories]
    assert len(ids) == len(set(ids))


def test_image_count(classification_dir):
    ds = from_classification_dir(classification_dir)
    assert len(ds.images) == 6  # 3 + 2 + 1


def test_one_annotation_per_image(classification_dir):
    ds = from_classification_dir(classification_dir)
    assert len(ds.annotations) == len(ds.images)


def test_annotation_category_matches_dir(classification_dir):
    ds = from_classification_dir(classification_dir)
    cat_index = {c.id: c.name for c in ds.categories}
    image_index = {img.id: img for img in ds.images}
    for ann in ds.annotations:
        cat_name = cat_index[ann.category_id]
        file_name = image_index[ann.image_id].file_name
        assert cat_name in file_name


def test_bbox_is_full_image(classification_dir):
    ds = from_classification_dir(classification_dir)
    image_index = {img.id: img for img in ds.images}
    for ann in ds.annotations:
        img = image_index[ann.image_id]
        x, y, w, h = ann.bbox
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert w == pytest.approx(float(img.width))
        assert h == pytest.approx(float(img.height))


def test_image_dimensions_recorded(classification_dir):
    ds = from_classification_dir(classification_dir)
    for img in ds.images:
        assert img.width is not None
        assert img.height is not None
        assert img.width > 0
        assert img.height > 0


def test_segmentation_is_full_image_polygon(classification_dir):
    ds = from_classification_dir(classification_dir)
    for ann in ds.annotations:
        assert len(ann.segmentation) == 1
        poly = ann.segmentation[0]
        assert len(poly) == 8  # 4 corners × (x, y)


def test_coco_classmethod(classification_dir):
    coco = COCO.from_classification_dir(classification_dir)
    assert isinstance(coco, COCO)
    assert len(coco.images) == 6


def test_roundtrip_write_read(classification_dir, tmp_path):
    out = tmp_path / "out.json"
    COCO.from_classification_dir(classification_dir).write(out)
    assert out.exists()
    reloaded = COCO.read_from(out)
    assert len(reloaded.images) == 6
    assert len(reloaded.categories) == 3


def test_empty_root_raises(tmp_path):
    with pytest.raises(ValueError, match="No subdirectories"):
        from_classification_dir(tmp_path)


def test_recursive_flag(tmp_path):
    """Recursive=True should pick up images nested inside class subdirs."""
    class_dir = tmp_path / "cat"
    nested = class_dir / "subset"
    nested.mkdir(parents=True)
    PImage.new("RGB", (50, 50)).save(nested / "img.jpg")

    ds_flat = from_classification_dir(tmp_path, recursive=False)
    ds_recursive = from_classification_dir(tmp_path, recursive=True)

    assert len(ds_flat.images) == 0
    assert len(ds_recursive.images) == 1
