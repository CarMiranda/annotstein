"""Tests for annotstein.analysis.clustering."""

import pytest

from annotstein.analysis.clustering import ClusterResult, cluster_bboxes
from tests.conftest import make_coco_dataset


def test_cluster_bboxes_kmeans_basic(sample_dataset):
    result = cluster_bboxes(sample_dataset, n_clusters=3, method="kmeans")
    assert isinstance(result, ClusterResult)
    assert len(result.labels) == len(sample_dataset.annotations)
    assert len(result.annotation_ids) == len(sample_dataset.annotations)
    assert result.n_clusters <= 3
    assert result.inertia is not None
    assert result.inertia >= 0.0


def test_cluster_bboxes_dbscan(sample_dataset):
    result = cluster_bboxes(sample_dataset, method="dbscan", dbscan_eps=2.0, dbscan_min_samples=2)
    assert isinstance(result, ClusterResult)
    assert len(result.labels) == len(sample_dataset.annotations)


def test_cluster_bboxes_feature_selection(sample_dataset):
    result = cluster_bboxes(sample_dataset, n_clusters=2, features=("width", "height"))
    assert result.feature_names == ["width", "height"]


def test_cluster_bboxes_invalid_feature(sample_dataset):
    with pytest.raises(ValueError, match="Unknown features"):
        cluster_bboxes(sample_dataset, features=("nonexistent",))


def test_cluster_bboxes_invalid_method(sample_dataset):
    with pytest.raises(ValueError, match="Unknown method"):
        cluster_bboxes(sample_dataset, method="spectral")


def test_cluster_bboxes_n_clusters_capped():
    """n_clusters should be capped to the number of annotations."""
    ds = make_coco_dataset(n_images=1, n_annotations_per_image=2, n_categories=1)
    result = cluster_bboxes(ds, n_clusters=100)
    assert result.n_clusters <= 2


def test_cluster_bboxes_empty_dataset():
    from annotstein.coco.schemas import Dataset

    ds = Dataset(categories=[], images=[], annotations=[])
    result = cluster_bboxes(ds)
    assert result.labels == []
    assert result.n_clusters == 0


def test_centroids_keys_match_unique_labels(sample_dataset):
    result = cluster_bboxes(sample_dataset, n_clusters=3)
    unique_labels = set(result.labels) - {-1}
    assert set(result.centroids.keys()) == unique_labels


def test_annotation_ids_match(sample_dataset):
    result = cluster_bboxes(sample_dataset)
    expected_ids = [ann.id for ann in sample_dataset.annotations]
    assert result.annotation_ids == expected_ids
