"""Tests for annotstein.analysis.eda."""

import pytest

from annotstein.analysis import eda
from annotstein.coco.schemas import Dataset
from tests.conftest import make_coco_dataset


def test_bbox_stats_keys(sample_dataset):
    result = eda.bbox_stats(sample_dataset)
    assert len(result) == 3  # n_categories
    for cat_name, stats in result.items():
        for key in ("width", "height", "area", "aspect_ratio"):
            assert key in stats
            assert stats[key]["count"] > 0


def test_bbox_aspect_ratio_buckets_overall(sample_dataset):
    result = eda.bbox_aspect_ratio_buckets(sample_dataset, n_buckets=5)
    assert "overall" in result
    assert "per_category" in result
    hist = result["overall"]
    assert len(hist["edges"]) == 6  # n_buckets + 1
    assert len(hist["counts"]) == 5
    assert sum(hist["counts"]) == len(sample_dataset.annotations)


def test_bbox_size_distribution(sample_dataset):
    result = eda.bbox_size_distribution(sample_dataset)
    for cat_name, counts in result.items():
        assert set(counts.keys()) == {"small", "medium", "large"}
        assert sum(counts.values()) > 0


def test_class_distribution(sample_dataset):
    result = eda.class_distribution(sample_dataset)
    assert "counts" in result
    assert "imbalance_ratio" in result
    assert len(result["counts"]) == 3
    total = sum(result["counts"].values())
    assert total == len(sample_dataset.annotations)


def test_annotation_density(sample_dataset):
    result = eda.annotation_density(sample_dataset)
    assert "stats" in result
    assert "histogram" in result
    assert result["stats"]["mean"] == pytest.approx(4.0)  # 4 annotations per image


def test_spatial_distribution(sample_dataset):
    result = eda.spatial_distribution(sample_dataset, grid_size=5)
    assert result["grid_size"] == 5
    assert len(result["grid"]) == 5
    total = sum(sum(row) for row in result["grid"])
    expected = len(sample_dataset.annotations) - result["skipped_images"]
    assert total == expected


def test_image_coverage(sample_dataset):
    result = eda.image_coverage(sample_dataset)
    assert "per_image" in result
    assert "per_category" in result
    assert "overall" in result
    for v in result["per_image"].values():
        assert 0.0 <= v


def test_segmentation_complexity(sample_dataset):
    result = eda.segmentation_complexity(sample_dataset)
    assert "overall" in result
    assert result["overall"]["count"] == len(sample_dataset.annotations)
    assert result["overall"]["mean"] == pytest.approx(4.0)  # 4 points per bbox polygon


def test_images_without_annotations():
    ds = make_coco_dataset(n_images=3, n_annotations_per_image=0)
    # Manually keep images but remove annotations
    ds_sparse = Dataset(categories=ds.categories, images=ds.images, annotations=[])
    result = eda.images_without_annotations(ds_sparse)
    assert result["count"] == 3


def test_annotations_per_category_per_image(sample_dataset):
    result = eda.annotations_per_category_per_image(sample_dataset)
    assert "table" in result
    assert "stats" in result


def test_summary_structure(sample_dataset):
    result = eda.summary(sample_dataset)
    expected_keys = {
        "dataset",
        "class_distribution",
        "bbox_stats",
        "bbox_size_distribution",
        "bbox_aspect_ratio_buckets",
        "annotation_density",
        "spatial_distribution",
        "image_coverage",
        "segmentation_complexity",
        "images_without_annotations",
        "annotations_per_category_per_image",
    }
    assert expected_keys == set(result.keys())
    assert result["dataset"]["n_images"] == 5
    assert result["dataset"]["n_annotations"] == 20


def test_empty_dataset():
    ds = Dataset(categories=[], images=[], annotations=[])
    result = eda.summary(ds)
    assert result["dataset"]["n_images"] == 0
    assert result["class_distribution"]["counts"] == {}
