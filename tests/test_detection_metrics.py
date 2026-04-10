"""Tests for annotstein.metrics.detection."""

import math

import pytest

from annotstein.coco.schemas import Annotation, Category, Dataset, Image, Prediction
from annotstein.metrics.detection import (
    COCO_IOU_THRESHOLDS,
    compute_ap,
    compute_ar,
    compute_coco_metrics,
    compute_iou,
)


def _make_gt() -> Dataset:
    categories = [Category(id=0, name="cat", supercategory="object")]
    images = [Image(id=i, file_name=f"{i}.jpg", width=640, height=480) for i in range(3)]
    annotations = [
        Annotation(id=0, image_id=0, category_id=0, bbox=[10, 10, 50, 50], segmentation=[]),
        Annotation(id=1, image_id=1, category_id=0, bbox=[20, 20, 60, 60], segmentation=[]),
        Annotation(id=2, image_id=2, category_id=0, bbox=[30, 30, 70, 70], segmentation=[]),
    ]
    return Dataset(categories=categories, images=images, annotations=annotations)


def _make_perfect_preds(gt: Dataset) -> Dataset:
    """Predictions that exactly match ground truth (score=1.0)."""
    preds = [
        Prediction(id=ann.id, image_id=ann.image_id, category_id=ann.category_id, bbox=ann.bbox, segmentation=[], score=1.0)
        for ann in gt.annotations
    ]
    return Dataset(categories=gt.categories, images=gt.images, annotations=preds)  # type: ignore[arg-type]


def _make_empty_preds(gt: Dataset) -> Dataset:
    return Dataset(categories=gt.categories, images=gt.images, annotations=[])


def test_iou_identical_boxes():
    assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert compute_iou([0, 0, 10, 10], [20, 20, 10, 10]) == pytest.approx(0.0)


def test_iou_partial_overlap():
    iou = compute_iou([0, 0, 10, 10], [5, 5, 10, 10])
    assert 0.0 < iou < 1.0


def test_iou_zero_area_box():
    assert compute_iou([0, 0, 0, 10], [0, 0, 10, 10]) == pytest.approx(0.0)


def test_ap_perfect_predictions():
    gt = _make_gt()
    preds = _make_perfect_preds(gt)
    ap = compute_ap(gt, preds, iou_threshold=0.5)
    assert ap == pytest.approx(1.0, abs=1e-6)


def test_ap_no_predictions():
    gt = _make_gt()
    preds = _make_empty_preds(gt)
    ap = compute_ap(gt, preds, iou_threshold=0.5)
    assert ap == pytest.approx(0.0)


def test_ap_no_ground_truth():
    gt = _make_gt()
    preds = _make_perfect_preds(gt)
    empty_gt = Dataset(categories=gt.categories, images=gt.images, annotations=[])
    ap = compute_ap(empty_gt, preds, iou_threshold=0.5)
    assert math.isnan(ap)  # undefined AP when no ground truth exists


def test_ap_category_filter():
    gt = _make_gt()
    preds = _make_perfect_preds(gt)
    ap = compute_ap(gt, preds, iou_threshold=0.5, category_id=0)
    assert ap == pytest.approx(1.0, abs=1e-6)


def test_ap_wrong_category():
    gt = _make_gt()
    preds = _make_perfect_preds(gt)
    ap = compute_ap(gt, preds, iou_threshold=0.5, category_id=99)
    assert ap == pytest.approx(0.0)


def test_ar_perfect_predictions():
    gt = _make_gt()
    preds = _make_perfect_preds(gt)
    ar = compute_ar(gt, preds, iou_thresholds=[0.5], max_dets=100)
    assert ar == pytest.approx(1.0, abs=1e-6)


def test_ar_no_predictions():
    gt = _make_gt()
    preds = _make_empty_preds(gt)
    ar = compute_ar(gt, preds, iou_thresholds=[0.5], max_dets=100)
    assert ar == pytest.approx(0.0)


def test_coco_metrics_perfect():
    gt = _make_gt()
    preds = _make_perfect_preds(gt)
    metrics = compute_coco_metrics(gt, preds)

    assert "overall" in metrics
    assert "per_category" in metrics

    overall = metrics["overall"]
    assert overall["AP50"] == pytest.approx(1.0, abs=1e-4)
    assert overall["AR100"] == pytest.approx(1.0, abs=1e-4)


def test_coco_metrics_zero():
    gt = _make_gt()
    preds = _make_empty_preds(gt)
    metrics = compute_coco_metrics(gt, preds)
    assert metrics["overall"]["AP50"] == pytest.approx(0.0)
    assert metrics["overall"]["AR100"] == pytest.approx(0.0)


def test_coco_metrics_per_category_keys():
    gt = _make_gt()
    preds = _make_perfect_preds(gt)
    metrics = compute_coco_metrics(gt, preds)
    assert "cat" in metrics["per_category"]
    cat_metrics = metrics["per_category"]["cat"]
    for key in ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR100"):
        assert key in cat_metrics


def test_coco_iou_thresholds():
    assert len(COCO_IOU_THRESHOLDS) == 10
    assert COCO_IOU_THRESHOLDS[0] == pytest.approx(0.5)
    assert COCO_IOU_THRESHOLDS[-1] == pytest.approx(0.95)
