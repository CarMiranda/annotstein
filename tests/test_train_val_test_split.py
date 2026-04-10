"""Tests for train_val_test_split / split_dataset_three_way / stratified_split_three_way."""

import pytest

from annotstein.coco.ops import COCO, split_dataset_three_way, stratified_split_three_way
from tests.conftest import make_coco_dataset


@pytest.fixture
def large_dataset():
    return make_coco_dataset(n_images=100, n_categories=4, n_annotations_per_image=2)


def test_split_dataset_three_way_counts(large_dataset):
    train, val, test = split_dataset_three_way(large_dataset, val_ratio=0.1, test_ratio=0.2)
    total = len(train.images) + len(val.images) + len(test.images)
    assert total == len(large_dataset.images)


def test_split_dataset_three_way_no_overlap(large_dataset):
    train, val, test = split_dataset_three_way(large_dataset, val_ratio=0.1, test_ratio=0.2)
    train_ids = {i.id for i in train.images}
    val_ids = {i.id for i in val.images}
    test_ids = {i.id for i in test.images}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_dataset_three_way_covers_all(large_dataset):
    train, val, test = split_dataset_three_way(large_dataset, val_ratio=0.1, test_ratio=0.2)
    all_ids = {i.id for i in large_dataset.images}
    split_ids = {i.id for i in train.images} | {i.id for i in val.images} | {i.id for i in test.images}
    assert all_ids == split_ids


def test_split_dataset_three_way_ratios_approximate(large_dataset):
    n = len(large_dataset.images)
    train, val, test = split_dataset_three_way(large_dataset, val_ratio=0.1, test_ratio=0.2)
    assert abs(len(val.images) / n - 0.1) <= 0.05
    assert abs(len(test.images) / n - 0.2) <= 0.05


def test_split_dataset_three_way_invalid_ratio(large_dataset):
    with pytest.raises(ValueError, match="val_ratio.*test_ratio"):
        split_dataset_three_way(large_dataset, val_ratio=0.6, test_ratio=0.6)


def test_split_dataset_three_way_categories_preserved(large_dataset):
    train, val, test = split_dataset_three_way(large_dataset, val_ratio=0.1, test_ratio=0.2)
    for ds in (train, val, test):
        assert {c.id for c in ds.categories} == {c.id for c in large_dataset.categories}


def test_stratified_split_three_way_counts(large_dataset):
    train, val, test = stratified_split_three_way(large_dataset, val_ratio=0.1, test_ratio=0.2)
    total = len(train.images) + len(val.images) + len(test.images)
    assert total == len(large_dataset.images)


def test_stratified_split_three_way_no_overlap(large_dataset):
    train, val, test = stratified_split_three_way(large_dataset, val_ratio=0.1, test_ratio=0.2)
    train_ids = {i.id for i in train.images}
    val_ids = {i.id for i in val.images}
    test_ids = {i.id for i in test.images}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_stratified_split_three_way_covers_all(large_dataset):
    train, val, test = stratified_split_three_way(large_dataset, val_ratio=0.1, test_ratio=0.2)
    all_ids = {i.id for i in large_dataset.images}
    split_ids = {i.id for i in train.images} | {i.id for i in val.images} | {i.id for i in test.images}
    assert all_ids == split_ids


def test_stratified_split_three_way_invalid_ratio(large_dataset):
    with pytest.raises(ValueError, match="val_ratio.*test_ratio"):
        stratified_split_three_way(large_dataset, val_ratio=0.5, test_ratio=0.6)


def test_coco_train_val_test_split_returns_three(large_dataset):
    coco = COCO(large_dataset)
    result = coco.train_val_test_split(val_ratio=0.1, test_ratio=0.2)
    assert len(result) == 3
    train, val, test = result
    assert isinstance(train, COCO)
    assert isinstance(val, COCO)
    assert isinstance(test, COCO)


def test_coco_train_val_test_split_total(large_dataset):
    coco = COCO(large_dataset)
    train, val, test = coco.train_val_test_split(val_ratio=0.1, test_ratio=0.2)
    total = len(train.images) + len(val.images) + len(test.images)
    assert total == len(large_dataset.images)


def test_coco_train_val_test_split_no_overlap(large_dataset):
    coco = COCO(large_dataset)
    train, val, test = coco.train_val_test_split(val_ratio=0.1, test_ratio=0.2)
    train_ids = {i.id for i in train.images}
    val_ids = {i.id for i in val.images}
    test_ids = {i.id for i in test.images}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_coco_train_val_test_split_no_stratify(large_dataset):
    coco = COCO(large_dataset)
    train, val, test = coco.train_val_test_split(val_ratio=0.15, test_ratio=0.15, stratify=False)
    total = len(train.images) + len(val.images) + len(test.images)
    assert total == len(large_dataset.images)


def test_coco_train_val_test_split_invalid_ratios(large_dataset):
    coco = COCO(large_dataset)
    with pytest.raises(ValueError):
        coco.train_val_test_split(val_ratio=0.5, test_ratio=0.6)


def test_coco_train_val_test_split_roundtrip(large_dataset, tmp_path):
    coco = COCO(large_dataset)
    train, val, test = coco.train_val_test_split(val_ratio=0.1, test_ratio=0.2)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = tmp_path / f"{name}.json"
        split.write(path)
        reloaded = COCO.read_from(path)
        assert len(reloaded.images) == len(split.images)
