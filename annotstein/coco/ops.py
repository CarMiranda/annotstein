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
