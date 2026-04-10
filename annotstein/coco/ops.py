import json
import pathlib
import typing as t
import itertools
from PIL import Image as PImage
import random
import os
from collections import defaultdict

from annotstein import utils
from annotstein.common import IMAGE_SUFFIXES
from annotstein.coco.schemas import Annotation, Category, Dataset, Image

FilePath = t.Union[str, os.PathLike[str], pathlib.Path]

TDataset = t.TypeVar("TDataset", bound=Dataset)


class COCO(t.Generic[TDataset]):
    def __init__(self, ds: TDataset):
        self.info = ds.info
        self.images = ds.images
        self.annotations = ds.annotations
        self.categories = ds.categories
        self.licenses = ds.licenses
        self.images_index: t.Dict[int, Image]

        self.__create_indices()

    def __create_indices(self):
        self.images_to_annotations = {}
        self.categories_to_annotations = {}
        self.category_keys_to_id = {}
        self.images_index = {image.id: image for image in self.images}
        self.category_index = {category.id: category for category in self.categories}

        for annotation in self.annotations:
            if annotation.image_id in self.images_to_annotations:
                self.images_to_annotations[annotation.image_id].append(annotation.id)
            else:
                self.images_to_annotations[annotation.image_id] = [annotation.id]

            if annotation.category_id in self.categories_to_annotations:
                self.categories_to_annotations[annotation.category_id].append(annotation.id)
            else:
                self.categories_to_annotations[annotation.category_id] = [annotation.id]

        for category in self.categories:
            category_key = (category.supercategory, category.name)
            self.category_keys_to_id[category_key] = category.id

    @classmethod
    def read_from(cls, path: FilePath) -> "COCO":
        with open(path) as f:
            index = json.load(f)

        return cls.from_dict(index)

    def write(self, path: FilePath):
        with open(path, "w") as f:
            f.write(self.to_dataset().model_dump_json())

    @classmethod
    def from_dict(cls, index: t.Dict[str, t.Any]) -> "COCO":
        return cls(Dataset(**index))

    @classmethod
    def merge(cls, *indices: "COCO") -> "COCO":
        datasets = [i.to_dataset() for i in indices]

        return cls(merge_datasets(*datasets))

    def to_dataset(self):
        return Dataset(
            categories=self.categories,
            images=self.images,
            annotations=self.annotations,
            info=self.info,
            licenses=self.licenses,
        )

    def rebase_filenames(self, prefix: FilePath) -> "COCO":
        prefix_p = pathlib.Path(prefix)
        for image in self.images:
            image.file_name = (prefix_p / pathlib.Path(image.file_name).name).as_posix()

        return self

    def to_classtree(self, output_dir: FilePath, root_dir: t.Optional[FilePath] = None):
        ds = self.to_dataset()
        return to_classtree(ds, output_dir, root_dir)

    def to_relative_coordinates(self) -> "COCO":
        for annotation in self.annotations:
            bbox = annotation.bbox
            image = self.images_index[annotation.image_id]

            if image.width is None or image.height is None:
                raise ValueError(f"Cannot convert coordinates image with empty dimensions (id={image.id})")

            rel_bbox = [
                bbox[0] / image.width,
                bbox[1] / image.height,
                bbox[2] / image.width,
                bbox[3] / image.height,
            ]
            annotation.bbox = rel_bbox

        return self

    def to_absolute_coordinates(self) -> "COCO":
        for annotation in self.annotations:
            bbox = annotation.bbox
            image = self.images_index[annotation.image_id]

            if image.width is None or image.height is None:
                raise ValueError(f"Cannot convert coordinates image with empty dimensions (id={image.id})")

            abs_bbox = [
                bbox[0] * image.width,
                bbox[1] * image.height,
                bbox[2] * image.width,
                bbox[3] * image.height,
            ]
            annotation.bbox = abs_bbox

        return self

    def train_test_split(
        self,
        split: float,
        shuffle: bool = True,
        stratify: bool = True,
        keep_empty_images: bool = True,
    ) -> t.Tuple["COCO", "COCO"]:
        ds = Dataset(
            categories=self.categories,
            images=self.images,
            annotations=self.annotations,
            info=self.info,
            licenses=self.licenses,
        )

        if stratify:
            train, test = stratified_split(ds, split, shuffle, keep_empty_images)
        else:
            train, test = split_dataset(ds, split, shuffle, keep_empty_images)

        return COCO(train), COCO(test)

    def train_val_test_split(
        self,
        val_ratio: float,
        test_ratio: float,
        shuffle: bool = True,
        stratify: bool = True,
        keep_empty_images: bool = True,
    ) -> t.Tuple["COCO", "COCO", "COCO"]:
        """Split the dataset into train, validation, and test subsets.

        Args:
            val_ratio: Fraction of the total dataset to use for validation (0–1).
            test_ratio: Fraction of the total dataset to use for testing (0–1).
                ``val_ratio + test_ratio`` must be < 1; the remainder becomes the
                training set.
            shuffle: Shuffle images before partitioning.
            stratify: If ``True``, preserve per-category proportions across splits.
            keep_empty_images: Include images without annotations, distributed
                proportionally across splits.

        Returns:
            A ``(train, val, test)`` tuple of :class:`COCO` instances.
        """
        ds = Dataset(
            categories=self.categories,
            images=self.images,
            annotations=self.annotations,
            info=self.info,
            licenses=self.licenses,
        )

        if stratify:
            train, val, test = stratified_split_three_way(ds, val_ratio, test_ratio, shuffle, keep_empty_images)
        else:
            train, val, test = split_dataset_three_way(ds, val_ratio, test_ratio, shuffle, keep_empty_images)

        return COCO(train), COCO(val), COCO(test)

    @classmethod
    def from_classification_dir(
        cls,
        path: FilePath,
        recursive: bool = False,
    ) -> "COCO":
        """Create a COCO instance from a classification directory tree.

        Expected layout::

            root/
              class_a/
                image1.jpg
                image2.jpg
              class_b/
                image3.jpg

        Each immediate subdirectory is treated as a category.  Every image
        inside that subdirectory receives a full-image bounding-box annotation
        for the corresponding category.  Images for which dimensions cannot be
        determined are still included (with ``width=None``, ``height=None``)
        and receive a zero bbox.

        Args:
            path: Root directory of the classification dataset.
            recursive: If ``True``, search subdirectories of each class folder
                for additional images.

        Returns:
            A :class:`COCO` instance.
        """
        return cls(from_classification_dir(path, recursive=recursive))

    def eda(self) -> t.Dict[str, t.Any]:
        """Return a full EDA summary report for this dataset."""
        from annotstein.analysis import eda as _eda

        return _eda.summary(self.to_dataset())

    def bbox_aspect_ratio_buckets(self, n_buckets: int = 10, per_category: bool = True) -> t.Dict[str, t.Any]:
        """Histogram of bbox aspect ratios."""
        from annotstein.analysis import eda as _eda

        return _eda.bbox_aspect_ratio_buckets(self.to_dataset(), n_buckets, per_category)

    def spatial_distribution(self, grid_size: int = 10) -> t.Dict[str, t.Any]:
        """2D heatmap of normalised bbox centre coordinates."""
        from annotstein.analysis import eda as _eda

        return _eda.spatial_distribution(self.to_dataset(), grid_size)

    def cluster_bboxes(
        self,
        n_clusters: int = 8,
        method: str = "kmeans",
        features: t.Sequence[str] = ("width", "height", "aspect_ratio", "area"),
    ):
        """Cluster annotations by geometry features.

        Returns a :class:`annotstein.analysis.clustering.ClusterResult`.
        """
        from annotstein.analysis.clustering import cluster_bboxes as _cluster

        return _cluster(self.to_dataset(), n_clusters=n_clusters, method=method, features=features)

    def cluster_crops(
        self,
        images_root: FilePath,
        n_clusters: int = 8,
        method: str = "kmeans",
    ):
        """Cluster annotations by HSV colour histogram of their image crops.

        Returns a :class:`annotstein.analysis.clustering.ClusterResult`.
        """
        from annotstein.analysis.clustering import cluster_crops as _cluster

        return _cluster(self.to_dataset(), images_root=images_root, n_clusters=n_clusters, method=method)

    def quality_report(self, dup_iou: float = 0.9, conflict_iou: float = 0.5) -> t.Dict[str, t.Any]:
        """Aggregate annotation quality report."""
        from annotstein.metrics.quality import quality_report as _quality

        return _quality(self.to_dataset(), dup_iou=dup_iou, conflict_iou=conflict_iou)

    def coco_metrics(self, predictions: "COCO") -> t.Dict[str, t.Any]:
        """Compute COCO detection metrics against a predictions COCO object.

        Args:
            predictions: A :class:`COCO` instance whose annotations carry a
                ``score`` attribute.

        Returns:
            Dict with ``overall`` and ``per_category`` metric breakdowns.
        """
        from annotstein.metrics.detection import compute_coco_metrics

        return compute_coco_metrics(self.to_dataset(), predictions.to_dataset())


def glob_images(path: FilePath, recursive: bool = True, suffixes: t.List[str] = IMAGE_SUFFIXES):
    path = pathlib.Path(path)
    available_images = utils.glob_images(path, recursive, suffixes)
    if len(available_images) == 0:
        return []

    coco_images: t.List[Image] = []
    current_id = 0
    for image_path in available_images:
        try:
            img = PImage.open(image_path)
            width, height = img.size
            coco_images.append(
                Image(
                    id=current_id,
                    file_name=image_path.as_posix(),
                    height=height,
                    width=width,
                )
            )
            current_id += 1
        except Exception as e:
            print(f"Could not read image at {image_path}. {e}")

    return coco_images


def to_classtree(ds: Dataset, output_dir: FilePath, root_dir: t.Optional[FilePath] = None):
    images_index = {i.id: i for i in ds.images}
    category_index = {c.id: c for c in ds.categories}

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    if root_dir is not None:
        root_dir = pathlib.Path(root_dir)

    for category in ds.categories:
        (output_dir / str(category.id)).mkdir(exist_ok=True)

    groups = {c.id: [] for c in ds.categories}

    for annotation in ds.annotations:
        bbox = annotation.bbox
        image = images_index[annotation.image_id]
        category = category_index[annotation.category_id]

        try:
            if root_dir:
                p = root_dir / image.file_name
            else:
                p = image.file_name
            img = PImage.open(p)
        except Exception as e:
            print(e)
            continue

        if all(0 <= b <= 1 for b in bbox):
            width, height = img.size
            bbox = (int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height))
        else:
            bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

        crop = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        dest_path = output_dir / str(category.id) / image.file_name.replace("/", "_")
        crop.save(dest_path)

        groups[annotation.category_id].append(dest_path)

    return groups


def get_category(name: str, ds: Dataset):
    category_names = get_categories([name], ds)
    return category_names.get(name)


def get_categories(names: t.List[str], ds: Dataset):
    return {c.name: c for c in ds.categories if c.name in names}


def get_supercategory(name: str, ds: Dataset):
    category_names = get_supercategories([name], ds)

    return category_names.get(name)


def get_supercategories(names: t.List[str], ds: Dataset):
    category_names: t.Dict[str, t.List[Category]] = dict()
    for c in ds.categories:
        if c.supercategory in category_names:
            category_names[c.supercategory].append(c)
        else:
            category_names[c.supercategory] = [c]

    return {name: category_names.get(name) for name in names}


def filter_categories(ds: TDataset, categories: t.List[int]) -> TDataset:
    kept_categories = {c.id: c for c in ds.categories if c.id in categories}
    kept_annotations = [a for a in ds.annotations if a.category_id in kept_categories]
    kept_image_ids = set(a.image_id for a in kept_annotations)
    kept_images = [i for i in ds.images if i.id in kept_image_ids]

    return ds.__class__(
        categories=list(kept_categories.values()),
        images=kept_images,
        annotations=kept_annotations,
        info=ds.info,
        licenses=ds.licenses,
    )


def filter_images(ds: TDataset, images: t.List[int]) -> TDataset:
    kept_images = {i.id: i for i in ds.images if i.id in images}
    kept_annotations = [a for a in ds.annotations if a.image_id in kept_images]
    kept_categories_ids = set(a.category_id for a in kept_annotations)
    kept_categories = [c for c in ds.categories if c.id in kept_categories_ids]

    return ds.__class__(
        categories=kept_categories,
        images=list(kept_images.values()),
        annotations=kept_annotations,
        info=ds.info,
        licenses=ds.licenses,
    )


def group_by_category(ds: Dataset):
    groups = {c.id: [] for c in ds.categories}

    for annotation in ds.annotations:
        groups[annotation.category_id].append(annotation)

    return groups


def group_by_image(ds: Dataset):
    groups = {i.id: [] for i in ds.images}
    annotated_images = set()

    for annotation in ds.annotations:
        groups[annotation.image_id].append(annotation)
        annotated_images.add(annotation.image_id)

    all_images = set(i.id for i in ds.images)
    non_annotated_images = all_images - annotated_images

    return groups, non_annotated_images


def group_by_categories_images(ds: Dataset) -> t.Tuple[t.Dict[int, t.Dict[int, t.List[Annotation]]], t.Set[int]]:
    groups = {c.id: defaultdict(list) for c in ds.categories}
    annotated_images = set()

    for annotation in ds.annotations:
        groups[annotation.category_id][annotation.image_id].append(annotation)
        annotated_images.add(annotation.image_id)

    all_images = set(i.id for i in ds.images)
    non_annotated_images = all_images - annotated_images

    return groups, non_annotated_images


def map_categories(ds: Dataset, category_map: t.Dict[int, int], inplace: bool = True):
    category_ids = set(c.id for c in ds.categories)

    if not inplace:
        ds = ds.model_copy(deep=True)

    if any(tc not in category_ids for tc in category_map.values()):
        raise ValueError(f"At least one of the target categories {category_ids} does not exist in the dataset.")

    for annotation in ds.annotations:
        annotation.category_id = category_map[annotation.category_id]

    for category in ds.categories:
        category.id = category_map[category.id]

    return ds


def split_dataset(
    ds: TDataset,
    split: float,
    shuffle: bool = True,
    keep_empty_images: bool = True,
) -> t.Tuple[TDataset, TDataset]:
    images_index = {i.id: i for i in ds.images}

    annotations_by_image, non_annotated_images = group_by_image(ds)
    indices = list(annotations_by_image.keys())
    if shuffle:
        random.shuffle(indices)

    n_train = int(split * len(annotations_by_image))
    n_test = len(annotations_by_image) - n_train

    train_split = {i: annotations_by_image[i] for i in indices[:n_train]}
    test_split = {i: annotations_by_image[i] for i in indices[-n_test:]}

    train_annotations = list(itertools.chain(*train_split.values()))
    test_annotations = list(itertools.chain(*test_split.values()))

    train_images = [images_index[i] for i in train_split]
    test_images = [images_index[i] for i in test_split]

    if keep_empty_images:
        n_train = int(split * len(non_annotated_images))
        n_test = len(non_annotated_images) - n_train

        train_images.extend([images_index[non_annotated_images.pop()] for _ in range(n_train)])
        test_images.extend([images_index[non_annotated_images.pop()] for _ in range(n_test)])

    train_ds = ds.__class__(
        categories=ds.categories,
        images=train_images,
        annotations=train_annotations,
        info=ds.info,
        licenses=ds.licenses,
    )

    test_ds = ds.__class__(
        categories=ds.categories,
        images=test_images,
        annotations=test_annotations,
        info=ds.info,
        licenses=ds.licenses,
    )

    return train_ds, test_ds


def stratified_split(
    ds: TDataset,
    split: float,
    shuffle: bool = True,
    keep_empty_images: bool = True,
) -> t.Tuple[TDataset, TDataset]:
    images_index = {i.id: i for i in ds.images}
    groups, non_annotated_images = group_by_categories_images(ds)

    train_images = []
    train_annotations = []

    test_images = []
    test_annotations = []

    for annotations_by_image in groups.values():
        indices = list(annotations_by_image.keys())
        if shuffle:
            random.shuffle(indices)

        n_train = int(split * len(indices))
        n_test = len(indices) - n_train

        train_split = {i: annotations_by_image[i] for i in indices[:n_train]}
        test_split = {i: annotations_by_image[i] for i in indices[-n_test:]}

        train_annotations.extend(itertools.chain(*train_split.values()))
        test_annotations.extend(itertools.chain(*test_split.values()))

        train_images.extend([images_index[i] for i in train_split])
        test_images.extend([images_index[i] for i in test_split])

    if keep_empty_images:
        n_train = int(split * len(non_annotated_images))
        n_test = len(non_annotated_images) - n_train

        train_images.extend([images_index[non_annotated_images.pop()] for _ in range(n_train)])
        test_images.extend([images_index[non_annotated_images.pop()] for _ in range(n_test)])

    train_ds = ds.__class__(
        categories=ds.categories,
        images=train_images,
        annotations=train_annotations,
        info=ds.info,
        licenses=ds.licenses,
    )

    test_ds = ds.__class__(
        categories=ds.categories,
        images=test_images,
        annotations=test_annotations,
        info=ds.info,
        licenses=ds.licenses,
    )

    return train_ds, test_ds


def split_dataset_three_way(
    ds: TDataset,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool = True,
    keep_empty_images: bool = True,
) -> t.Tuple[TDataset, TDataset, TDataset]:
    """Split *ds* into train / val / test subsets (no stratification).

    Args:
        ds: Source dataset.
        val_ratio: Fraction of the dataset to use for validation (0–1).
        test_ratio: Fraction of the dataset to use for testing (0–1).
            ``val_ratio + test_ratio`` must be < 1.
        shuffle: Shuffle images before splitting.
        keep_empty_images: Distribute images without annotations proportionally.

    Returns:
        ``(train_ds, val_ds, test_ds)``
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio:.4f}")

    images_index = {i.id: i for i in ds.images}
    annotations_by_image, non_annotated_images = group_by_image(ds)
    indices = list(annotations_by_image.keys())
    if shuffle:
        random.shuffle(indices)

    n = len(indices)
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def _make_ds(idx: t.List[int]) -> TDataset:
        anns = list(itertools.chain(*[annotations_by_image[i] for i in idx]))
        imgs = [images_index[i] for i in idx]
        return ds.__class__(categories=ds.categories, images=imgs, annotations=anns, info=ds.info, licenses=ds.licenses)

    train_ds = _make_ds(train_idx)
    val_ds = _make_ds(val_idx)
    test_ds = _make_ds(test_idx)

    if keep_empty_images:
        empty = list(non_annotated_images)
        if shuffle:
            random.shuffle(empty)
        n_e = len(empty)
        n_e_val = int(val_ratio * n_e)
        n_e_test = int(test_ratio * n_e)
        n_e_train = n_e - n_e_val - n_e_test
        for attr, e_idx in [
            (train_ds, empty[:n_e_train]),
            (val_ds, empty[n_e_train : n_e_train + n_e_val]),
            (test_ds, empty[n_e_train + n_e_val :]),
        ]:
            attr.images.extend([images_index[i] for i in e_idx])

    return train_ds, val_ds, test_ds


def stratified_split_three_way(
    ds: TDataset,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool = True,
    keep_empty_images: bool = True,
) -> t.Tuple[TDataset, TDataset, TDataset]:
    """Split *ds* into train / val / test with per-category stratification.

    Args:
        ds: Source dataset.
        val_ratio: Fraction of the dataset to use for validation (0–1).
        test_ratio: Fraction of the dataset to use for testing (0–1).
            ``val_ratio + test_ratio`` must be < 1.
        shuffle: Shuffle images within each category group before splitting.
        keep_empty_images: Distribute images without annotations proportionally.

    Returns:
        ``(train_ds, val_ds, test_ds)``
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio:.4f}")

    images_index = {i.id: i for i in ds.images}
    groups, non_annotated_images = group_by_categories_images(ds)

    # Assign each image to exactly one split (first-seen-per-category wins).
    # This preserves category proportions while handling multi-label images.
    image_split: t.Dict[int, str] = {}  # image_id -> "train" | "val" | "test"

    for annotations_by_image in groups.values():
        indices = [i for i in annotations_by_image.keys() if i not in image_split]
        if shuffle:
            random.shuffle(indices)

        n = len(indices)
        n_val = int(val_ratio * n)
        n_test = int(test_ratio * n)
        n_train = n - n_val - n_test

        for img_id in indices[:n_train]:
            image_split[img_id] = "train"
        for img_id in indices[n_train : n_train + n_val]:
            image_split[img_id] = "val"
        for img_id in indices[n_train + n_val :]:
            image_split[img_id] = "test"

    # Build per-split image and annotation lists using all annotations per image.
    all_annotations_by_image, _ = group_by_image(ds)

    train_images: t.List[Image] = []
    train_annotations: t.List[Annotation] = []
    val_images: t.List[Image] = []
    val_annotations: t.List[Annotation] = []
    test_images: t.List[Image] = []
    test_annotations: t.List[Annotation] = []

    for img_id, split_name in image_split.items():
        img = images_index[img_id]
        anns = all_annotations_by_image.get(img_id, [])
        if split_name == "train":
            train_images.append(img)
            train_annotations.extend(anns)
        elif split_name == "val":
            val_images.append(img)
            val_annotations.extend(anns)
        else:
            test_images.append(img)
            test_annotations.extend(anns)

    if keep_empty_images:
        empty = list(non_annotated_images)
        if shuffle:
            random.shuffle(empty)
        n_e = len(empty)
        n_e_val = int(val_ratio * n_e)
        n_e_test = int(test_ratio * n_e)
        n_e_train = n_e - n_e_val - n_e_test
        train_images.extend([images_index[i] for i in empty[:n_e_train]])
        val_images.extend([images_index[i] for i in empty[n_e_train : n_e_train + n_e_val]])
        test_images.extend([images_index[i] for i in empty[n_e_train + n_e_val :]])

    def _ds(imgs: t.List[Image], anns: t.List[Annotation]) -> TDataset:
        return ds.__class__(categories=ds.categories, images=imgs, annotations=anns, info=ds.info, licenses=ds.licenses)

    return _ds(train_images, train_annotations), _ds(val_images, val_annotations), _ds(test_images, test_annotations)


def merge_datasets(*datasets: TDataset) -> TDataset:
    n_images = 0
    n_annotations = 0
    n_categories = 0
    for dataset in datasets:
        n_images += len(dataset.images)
        n_annotations += len(dataset.annotations)
        n_categories += len(dataset.categories)

    c_image = 0
    c_annotation = 0
    c_category = 0
    category_keys = {}  # name to new category dataset
    categories = []

    for dataset in datasets:
        c2c = {}  # old categories to new categories
        for category in dataset.categories:
            category_key = (category.name, category.supercategory)
            if category_key not in category_keys:
                c2c[category.id] = c_category
                category.id = c_category
                category_keys[category_key] = c_category

                categories.append(category)
                c_category += 1
            else:
                c2c[category.id] = category_keys[category_key]

        i2i = {}  # old images to new images
        for image in dataset.images:
            i2i[image.id] = c_image
            image.id = c_image
            c_image += 1

        for annotation in dataset.annotations:
            annotation.id = c_annotation
            annotation.image_id = i2i[annotation.image_id]
            annotation.category_id = c2c[annotation.category_id]
            c_annotation += 1

    contributors = ", ".join([i.info.contributor for i in datasets if i.info is not None])
    merged_dataset = {
        "info": {
            "description": "Merged dataset",
            "contributor": contributors,
        },
        "images": list(itertools.chain(*[dataset.images for dataset in datasets])),
        "annotations": list(itertools.chain(*[dataset.annotations for dataset in datasets])),
        "categories": categories,
    }

    return datasets[0].__class__(**merged_dataset)


def from_classification_dir(
    path: FilePath,
    recursive: bool = False,
) -> Dataset:
    """Build a COCO :class:`Dataset` from a classification directory tree.

    Expected layout::

        root/
          class_a/
            image1.jpg
            image2.jpg
          class_b/
            image3.jpg

    Each immediate subdirectory name becomes a category.  Every image inside
    receives a full-image bounding-box annotation for that category.  Images
    whose dimensions cannot be read are still included (width/height ``None``)
    with a zero-area bbox ``[0, 0, 0, 0]``.

    Args:
        path: Root directory of the classification dataset.
        recursive: If ``True``, search subdirectories of each class folder for
            additional images (useful for nested splits like ``train/class_a/``
            under a common root).

    Returns:
        A :class:`~annotstein.coco.schemas.Dataset`.
    """
    root = pathlib.Path(path)

    # Collect class subdirectories (sorted for deterministic IDs)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No subdirectories found in {root}. Expected one subfolder per class.")

    categories: t.List[Category] = []
    images: t.List[Image] = []
    annotations: t.List[Annotation] = []

    cat_id = 0
    image_id = 0
    annotation_id = 0

    for class_dir in class_dirs:
        category = Category(id=cat_id, name=class_dir.name, supercategory=class_dir.name)
        categories.append(category)

        image_paths = utils.glob_images(class_dir, recursive=recursive, suffixes=IMAGE_SUFFIXES)

        for img_path in sorted(image_paths):
            width: t.Optional[int] = None
            height: t.Optional[int] = None
            try:
                pil_img = PImage.open(img_path)
                width, height = pil_img.size
            except Exception as e:
                print(f"Could not read image dimensions for {img_path}: {e}")

            bbox = [0.0, 0.0, float(width or 0), float(height or 0)]
            segmentation: t.List[t.List[float]] = []
            if width and height:
                segmentation = [[0.0, 0.0, float(width), 0.0, float(width), float(height), 0.0, float(height)]]

            images.append(Image(id=image_id, file_name=img_path.as_posix(), width=width, height=height))
            annotations.append(
                Annotation(
                    id=annotation_id,
                    image_id=image_id,
                    category_id=cat_id,
                    bbox=bbox,
                    segmentation=segmentation,
                )
            )
            image_id += 1
            annotation_id += 1

        cat_id += 1

    return Dataset(categories=categories, images=images, annotations=annotations)
