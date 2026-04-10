"""Tests for annotstein.metrics.quality."""

import pytest

from annotstein.coco.schemas import Annotation, Category, Dataset, Image
from annotstein.metrics.quality import (
    duplicate_annotations,
    label_consistency,
    quality_report,
    validate_bboxes,
    validate_references,
    validate_segmentations,
)
from tests.conftest import make_coco_dataset


def _make_ds(**kwargs) -> Dataset:
    return make_coco_dataset(**kwargs)


# ---------------------------------------------------------------------------
# validate_bboxes
# ---------------------------------------------------------------------------


def test_validate_bboxes_clean(sample_dataset):
    issues = validate_bboxes(sample_dataset)
    assert issues == []


def test_validate_bboxes_negative_width():
    ds = Dataset(
        categories=[Category(id=0, name="a", supercategory="")],
        images=[Image(id=0, file_name="x.jpg", width=100, height=100)],
        annotations=[Annotation(id=0, image_id=0, category_id=0, bbox=[10, 10, -5, 20], segmentation=[])],
    )
    issues = validate_bboxes(ds)
    assert any("non-positive width" in i.issue for i in issues)


def test_validate_bboxes_out_of_bounds():
    ds = Dataset(
        categories=[Category(id=0, name="a", supercategory="")],
        images=[Image(id=0, file_name="x.jpg", width=100, height=100)],
        annotations=[Annotation(id=0, image_id=0, category_id=0, bbox=[90, 90, 50, 50], segmentation=[])],
    )
    issues = validate_bboxes(ds)
    assert len(issues) == 2  # exceeds both width and height


# ---------------------------------------------------------------------------
# validate_segmentations
# ---------------------------------------------------------------------------


def test_validate_segmentations_clean(sample_dataset):
    issues = validate_segmentations(sample_dataset)
    assert issues == []


def test_validate_segmentations_too_few_points():
    ds = Dataset(
        categories=[Category(id=0, name="a", supercategory="")],
        images=[Image(id=0, file_name="x.jpg")],
        annotations=[Annotation(id=0, image_id=0, category_id=0, bbox=[0, 0, 10, 10], segmentation=[[1, 2, 3, 4]])],
    )
    issues = validate_segmentations(ds)
    assert any("fewer than 3 points" in i.issue for i in issues)


def test_validate_segmentations_odd_coords():
    ds = Dataset(
        categories=[Category(id=0, name="a", supercategory="")],
        images=[Image(id=0, file_name="x.jpg")],
        annotations=[Annotation(id=0, image_id=0, category_id=0, bbox=[0, 0, 10, 10], segmentation=[[1, 2, 3]])],
    )
    issues = validate_segmentations(ds)
    assert any("odd" in i.issue for i in issues)


# ---------------------------------------------------------------------------
# validate_references
# ---------------------------------------------------------------------------


def test_validate_references_clean(sample_dataset):
    issues = validate_references(sample_dataset)
    assert issues == []


def test_validate_references_bad_image_id():
    ds = Dataset(
        categories=[Category(id=0, name="a", supercategory="")],
        images=[Image(id=0, file_name="x.jpg")],
        annotations=[Annotation(id=0, image_id=999, category_id=0, bbox=[0, 0, 10, 10], segmentation=[])],
    )
    issues = validate_references(ds)
    assert any("image_id" in i.issue for i in issues)


def test_validate_references_bad_category_id():
    ds = Dataset(
        categories=[Category(id=0, name="a", supercategory="")],
        images=[Image(id=0, file_name="x.jpg")],
        annotations=[Annotation(id=0, image_id=0, category_id=999, bbox=[0, 0, 10, 10], segmentation=[])],
    )
    issues = validate_references(ds)
    assert any("category_id" in i.issue for i in issues)


# ---------------------------------------------------------------------------
# duplicate_annotations
# ---------------------------------------------------------------------------


def test_no_duplicates(sample_dataset):
    dups = duplicate_annotations(sample_dataset, iou_threshold=0.9)
    assert dups == []


def test_exact_duplicate_detected():
    ds = Dataset(
        categories=[Category(id=0, name="a", supercategory="")],
        images=[Image(id=0, file_name="x.jpg")],
        annotations=[
            Annotation(id=0, image_id=0, category_id=0, bbox=[10, 10, 50, 50], segmentation=[]),
            Annotation(id=1, image_id=0, category_id=0, bbox=[10, 10, 50, 50], segmentation=[]),
        ],
    )
    dups = duplicate_annotations(ds, iou_threshold=0.9)
    assert len(dups) == 1
    assert dups[0].iou == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# label_consistency
# ---------------------------------------------------------------------------


def test_no_label_conflicts(sample_dataset):
    conflicts = label_consistency(sample_dataset, iou_threshold=0.5)
    assert conflicts == []


def test_label_conflict_detected():
    ds = Dataset(
        categories=[Category(id=0, name="a", supercategory=""), Category(id=1, name="b", supercategory="")],
        images=[Image(id=0, file_name="x.jpg")],
        annotations=[
            Annotation(id=0, image_id=0, category_id=0, bbox=[10, 10, 50, 50], segmentation=[]),
            Annotation(id=1, image_id=0, category_id=1, bbox=[10, 10, 50, 50], segmentation=[]),
        ],
    )
    conflicts = label_consistency(ds, iou_threshold=0.5)
    assert len(conflicts) == 1


# ---------------------------------------------------------------------------
# quality_report
# ---------------------------------------------------------------------------


def test_quality_report_structure(sample_dataset):
    report = quality_report(sample_dataset)
    expected = {"counts", "bbox_issues", "segmentation_issues", "reference_issues", "duplicates", "label_conflicts"}
    assert set(report.keys()) == expected
    assert report["counts"]["bbox_issues"] == 0
