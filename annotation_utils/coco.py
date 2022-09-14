import json
import pathlib
from typing import Dict, Any
import itertools
from datetime import datetime
from PIL import Image


class COCO:
    def __init__(self, index: Dict[str, Any]):
        self.info = index["info"]
        self.images = index["images"]
        self.annotations = index["annotations"]
        self.categories = index["categories"]

        self.create_indices()

    def create_indices(self):
        self.images_to_annotations = {}
        self.categories_to_annotations = {}
        self.category_keys_to_id = {}
        self.images_index = {image["id"]: image for image in self.images}
        self.annotation_index = {
            annotation["id"]: annotation for annotation in self.annotations
        }
        self.category_index = {category["id"]: category for category in self.categories}

        for annotation in self.annotations:
            if annotation["image_id"] in self.images_to_annotations:
                self.images_to_annotations[annotation["image_id"]].append(
                    annotation["id"]
                )
            else:
                self.images_to_annotations[annotation["image_id"]] = [annotation["id"]]

            if annotation["category_id"] in self.categories_to_annotations:
                self.categories_to_annotations[annotation["category_id"]].append(
                    annotation["id"]
                )
            else:
                self.categories_to_annotations[annotation["category_id"]] = [
                    annotation["id"]
                ]

        for category in self.categories:
            category_key = (category["supercategory"], category["name"])
            self.category_keys_to_id[category_key] = category["id"]

    @staticmethod
    def from_file(path: pathlib.Path) -> "COCO":
        with open(path) as f:
            index = json.load(f)

        return COCO.from_dict(index)

    @staticmethod
    def validate(index: Dict[str, Any]) -> None:
        top = all(
            [
                entry in index
                for entry in ["info", "images", "annotations", "categories"]
            ]
        )

        if not top:
            pass  # raise Exception("Top-level structure is not valid.")

        info = all(
            [
                entry in index["info"]
                for entry in [
                    "description",
                    "url",
                    "version",
                    "year",
                    "contributor",
                    "date_created",
                ]
            ]
        )

        if not info:
            pass  # raise Exception("Info structure is not valid.")

    @staticmethod
    def from_dict(index: Dict[str, Any]) -> "COCO":
        COCO.validate(index)

        return COCO(index)

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
                category_key = (category["name"], category["supercategory"])
                if category_key not in category_keys:
                    c2c[category["id"]] = c_category
                    category["id"] = c_category
                    category_keys[category_key] = c_category

                    categories.append(category)
                    c_category += 1
                else:
                    c2c[category["id"]] = category_keys[category_key]

            i2i = {}  # old images to new images
            for image in index.images:
                i2i[image["id"]] = c_image
                image["id"] = c_image
                c_image += 1

            for annotation in index.annotations:
                annotation["id"] = c_annotation
                annotation["image_id"] = i2i[annotation["image_id"]]
                annotation["category_id"] = c2c[annotation["category_id"]]
                c_annotation += 1

        now = datetime.now()

        new_index = {
            "info": {
                "description": "",
                "url": "",
                "version": "1.0.0",
                "year": now.year,
                "contributor": "Buawei",
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

    def rebase_filenames(self, prefix: str) -> "COCO":

        prefix_p = pathlib.Path(prefix)
        for image in self.images:
            image["file_name"] = str(prefix_p / pathlib.Path(image["file_name"]).name)

        return self

    def to_dict(self):
        return {
            "info": self.info,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def to_classtree(self, root_dir: pathlib.Path, output_dir: pathlib.Path):
        output_dir.mkdir(exist_ok=True)

        for category in self.categories:
            (output_dir / category["name"]).mkdir(exist_ok=True)

        for annotation in self.annotations:
            bbox = annotation["bbox"]
            image = self.images_index[annotation["image_id"]]
            category = self.category_index[annotation["category_id"]]

            img = Image.open(root_dir / image["file_name"])
            crop = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            dest_path = (
                output_dir / category["name"] / image["file_name"].replace("/", "_")
            )
            crop.save(dest_path)

    def to_relative_coordinates(self, root_dir: pathlib.Path) -> "COCO":
        for annotation in self.annotations:
            bbox = annotation["bbox"]
            image = self.images_index[annotation["image_id"]]
            img = Image.open(root_dir / image["file_name"])
            width, height = img.size
            rel_bbox = (
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height,
            )
            annotation["bbox"] = rel_bbox

        return self

    def to_absolute_coordinates(self, root_dir: pathlib.Path) -> "COCO":
        for annotation in self.annotations:
            bbox = annotation["bbox"]
            image = self.images_index[annotation["image_id"]]
            img = Image.open(root_dir / image["file_name"])
            width, height = img.size
            abs_bbox = (
                bbox[0] * width,
                bbox[1] * height,
                bbox[2] * width,
                bbox[3] * height,
            )
            annotation["bbox"] = abs_bbox

        return self
