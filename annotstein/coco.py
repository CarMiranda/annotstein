import json
import pathlib
import typing as t
import itertools
from datetime import datetime
from PIL import Image
from pydantic import BaseModel, Field, Extra, validator
import random


class COCOImage(BaseModel):
    id: int
    file_name: str
    height: t.Optional[int] = None
    width: t.Optional[int] = None
    flickr_url: t.Optional[str] = None
    coco_url: t.Optional[str] = None
    license: t.Optional[int] = None


class COCOAnnotation(BaseModel):
    id: t.Optional[int] = None
    area: t.Optional[float] = None
    bbox: t.List[float] 
    category_id: int
    image_id: int
    iscrowd: t.Optional[int] = None
    segmentation: t.Optional[t.List[t.List[float]]] = None
    attributes: t.Dict[str, str] = dict()


class COCOCategory(BaseModel):
    id: int
    name: str
    supercategory: t.Optional[str] = None


class COCOInfo(BaseModel):
    contributor: str = ""
    date_created: str = ""
    description: str = ""
    url: str = ""
    version: str = "1.0"
    year: int = 2023


class COCOLicense(BaseModel):
    id: int
    name: str = ""
    url: str = ""


class COCOModel(BaseModel, extra=Extra.allow):
    annotations: t.List[COCOAnnotation]
    categories: t.List[COCOCategory]
    licenses: t.Optional[t.List[COCOLicense]] = None
    images: t.List[COCOImage]
    info: t.Optional[COCOInfo] = None


class COCO:
    def __init__(self, index: COCOModel):
        self.info = index.info
        self.images = index.images
        self.annotations = index.annotations
        self.categories = index.categories
        self.images_index: t.Dict[int, COCOImage]

        self.create_indices()

    def create_indices(self):
        self.images_to_annotations = {}
        self.categories_to_annotations = {}
        self.category_keys_to_id = {}
        self.images_index = {image.id: image for image in self.images}
        self.category_index = {category.id: category for category in self.categories}

        for annotation in self.annotations:
            if annotation.image_id in self.images_to_annotations:
                self.images_to_annotations[annotation.image_id].append(
                    annotation.id
                )
            else:
                self.images_to_annotations[annotation.image_id] = [annotation.id]

            if annotation.category_id in self.categories_to_annotations:
                self.categories_to_annotations[annotation.category_id].append(
                    annotation.id
                )
            else:
                self.categories_to_annotations[annotation.category_id] = [
                    annotation.id
                ]

        for category in self.categories:
            category_key = (category.supercategory, category.name)
            self.category_keys_to_id[category_key] = category.id

    @staticmethod
    def from_file(path: pathlib.Path) -> "COCO":
        with open(path) as f:
            index = json.load(f)

        return COCO.from_dict(index)

    @staticmethod
    def from_dict(index: t.Dict[str, t.Any]) -> "COCO":

        return COCO(COCOModel(**index))

    @staticmethod
    def merge(*indices: "COCO") -> "COCO":

        n_images = 0
        n_annotations = 0
        n_categories = 0
        for index in indices:
            n_images += len(index.images)
            n_annotations += len(index.annotations)
            n_categories += len(index.categories)

        c_image = 0
        c_annotation = 0
        c_category = 0
        category_keys = {}  # name to new category index
        categories = []

        for index in indices:
            c2c = {}  # old categories to new categories
            for category in index.categories:
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
            for image in index.images:
                i2i[image.id] = c_image
                image.id = c_image
                c_image += 1

            for annotation in index.annotations:
                annotation.id = c_annotation
                annotation.image_id = i2i[annotation.image_id]
                annotation.category_id = c2c[annotation.category_id]
                c_annotation += 1

        now = datetime.now()

        contributors = ", ".join([i.info.contributor for i in indices if i.info is not None])
        new_index = {
            "info": {
                "description": "",
                "url": "",
                "version": "1.0.0",
                "year": now.year,
                "contributor": contributors,
                "date_created": str(now),
            },
            "images": list(itertools.chain(*[index.images for index in indices])),
            "annotations": list(
                itertools.chain(*[index.annotations for index in indices])
            ),
            "categories": categories,
        }

        return COCO.from_dict(new_index)

    def write(self, path: pathlib.Path):
        index_dict = self.to_dict()
        with open(path, "w") as f:
            json.dump(index_dict, f, indent=4)

        return index_dict

    def rebase_filenames(self, prefix: t.Union[str, pathlib.Path]) -> "COCO":

        prefix_p = pathlib.Path(prefix)
        for image in self.images:
            image.file_name = (prefix_p / pathlib.Path(image.file_name).name).as_posix()

        return self

    def to_dict(self):
        d = {}
        if self.info is not None:
            d["info"] = self.info.model_dump(exclude_unset=True)
        # if self.licenses is not None:
        #    d["licenses"] = self.licenses.model_dump(exclude_unset=True)
        return {
            "images": [i.model_dump(exclude_unset=True) for i in self.images],
            "annotations": [a.model_dump(exclude_unset=True) for a in self.annotations],
            "categories": [c.model_dump(exclude_unset=True) for c in self.categories],
            **d,
        }

    def to_classtree(self, root_dir: pathlib.Path, output_dir: pathlib.Path):
        output_dir.mkdir(exist_ok=True)

        for category in self.categories:
            (output_dir / category.name).mkdir(exist_ok=True)

        for annotation in self.annotations:
            bbox = annotation.bbox
            image = self.images_index[annotation.image_id]
            category = self.category_index[annotation.category_id]

            try:
                img = Image.open(root_dir / image.file_name)
            except Exception as e:
                print(e)
                continue

            if all(0 <= b <= 1 for b in bbox):
                width, height = img.size
                bbox = (
                    int(bbox[0] * width),
                    int(bbox[1] * height),
                    int(bbox[2] * width),
                    int(bbox[3] * height)
                )
            else:
                bbox = (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3])
                )

            crop = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            dest_path = (
                output_dir / category.name / image.file_name.replace("/", "_")
            )
            crop.save(dest_path)

    def to_relative_coordinates(self) -> "COCO":
        for annotation in self.annotations:
            bbox = annotation.bbox
            image = self.images_index[annotation.image_id]

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
            abs_bbox = [
                bbox[0] * image.width,
                bbox[1] * image.height,
                bbox[2] * image.width,
                bbox[3] * image.height,
            ]
            annotation.bbox = abs_bbox

        return self

    def split(self, test_ratio: float, stratify: bool = False) -> t.Tuple["COCO", "COCO"]:
        n_samples = len(self.annotations)
        n_test = int(n_samples * test_ratio)
        all_indices = list(range(n_samples))
        random.shuffle(all_indices)
        test_indices = all_indices[:n_test]
        train_indices = all_indices[n_test:]

        train_annotations = [self.annotations[i] for i in train_indices]
        train_images_ids = set(a.image_id for a in train_annotations)
        train_images = [self.images_index[image_id] for image_id in train_images_ids]

        test_annotations = [self.annotations[i] for i in test_indices]
        test_images_ids = set(a.image_id for a in test_annotations)
        test_images = [self.images_index[image_id] for image_id in test_images_ids]

        train_index = COCO(
            COCOModel(
                info=self.info,
                images=train_images,
                annotations=train_annotations,
                categories=self.categories
            )
        )
        test_index = COCO(
            COCOModel(
                info=self.info,
                images=test_images,
                annotations=test_annotations,
                categories=self.categories,
            )
        )

        return train_index, test_index

    @staticmethod
    def get_images_from_folder(path: pathlib.Path, recursive: bool = True):
        if not path.is_dir():
            raise ValueError("`path` should be a directory containing images.")

        if recursive:
            images_list = list(path.glob("**/*"))
        else:
            images_list = list(path.glob("*"))

        available_images = [f for f in images_list if f.suffix in [".jpg", ".png", ".jpeg", ".tiff"]]
        if len(available_images) == 0:
            raise ValueError(f"Given path ({path}) does not contain images.")

        coco_images: t.List[COCOImage] = []
        current_id = 0
        for image_path in available_images:
            try:
                img = Image.open(image_path)
                width, height = img.size
                coco_images.append(COCOImage(
                    id=current_id,
                    file_name=image_path.as_posix(),
                    height=height,
                    width=width,
                ))
                current_id += 1
            except Exception as e:
                print(f"Could not read image at {image_path}. {e}")

        return coco_images
