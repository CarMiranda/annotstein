import csv
from io import TextIOWrapper
from itertools import chain
import pathlib
import typing as t
from typing_extensions import Annotated
from pydantic import AnyHttpUrl, BaseModel, Field

from PIL import Image
from annotstein.coco import ops as coco
from annotstein.common import IMAGE_SUFFIXES
from annotstein.yolo.common import xywh_to_xcycwh

SplitSet = t.Union[pathlib.Path, t.List[pathlib.Path]]


class Annotation(BaseModel):
    # Category identifier
    category_id: int

    # Segmentation with format [x0, y0, x1, y1, ..., xn, yn]
    segmentation: Annotated[t.List[float], Field(min_length=6)]


class AnnotationGroup(BaseModel):
    # Image file concerned by these annotations
    file_name: str

    # Collection of annotations
    annotations: t.List[Annotation]

    @classmethod
    def parse_row(cls, *row):
        return Annotation(category_id=int(row[0]), segmentation=list(map(float, row[1:])))

    @classmethod
    def parse_annotation_file(cls, path: pathlib.Path, delimiter: str = " ") -> t.Self:
        annotations = []
        with open(path) as fp:
            reader = csv.reader(fp, delimiter=delimiter)
            for row in reader:
                try:
                    annotations.append(cls.parse_row(*row))
                except Exception:
                    pass

        return cls(file_name=str(path.resolve()), annotations=annotations)

    def dump(self, fp: TextIOWrapper):
        writer = csv.writer(fp, delimiter=" ")
        for annotation in self.annotations:
            writer.writerow([annotation.category_id, *annotation.segmentation])

    def dumps(self):
        res = ""
        for annotation in self.annotations:
            res += " ".join(map(str, [annotation.category_id, *annotation.segmentation]))

        return res


class Dataset(BaseModel):
    annotations: t.List[AnnotationGroup]

    @classmethod
    def from_coco(cls, coco_dataset: coco.Dataset) -> "t.Self":
        images_index = {i.id: i for i in coco_dataset.images}
        grouped_by_image: t.Dict[str, t.List[coco.Annotation]] = {i.file_name: [] for i in coco_dataset.images}
        for annotation in coco_dataset.annotations:
            file_name = images_index[annotation.image_id].file_name
            grouped_by_image[file_name].append(annotation)

        yolo_annotations = []
        for file_name, annotation_group in grouped_by_image.items():
            annotations = [Annotation(category_id=a.category_id, bbox=list(xywh_to_xcycwh(a.bbox))) for a in annotation_group]
            yolo_annotations.append(AnnotationGroup(file_name=file_name, annotations=annotations))

        return cls(annotations=yolo_annotations)


class TrainTask(BaseModel):
    path: pathlib.Path
    train: SplitSet
    val: SplitSet
    test: t.Optional[SplitSet] = None

    classes: t.Dict[int, str]

    download: t.Optional[AnyHttpUrl] = None

    def parse_directory(self, path: pathlib.Path):
        images: t.List[coco.Image] = []
        image_index: t.Dict[str, coco.Image] = dict()
        annotations: t.List[coco.Annotation] = []
        for file in chain(*[path.glob(f"*{s}") for s in IMAGE_SUFFIXES]):
            image = None
            try:
                width, height = Image.open(file).size
            except Exception:
                continue

            image = coco.Image(
                id=1 + len(images),
                file_name=str(file.resolve()),
                height=height,
                width=width,
            )

            images.append(image)
            image_index[file.stem] = image

        for file in path.glob("*.txt"):
            if file.stem not in image_index:
                continue

            image = image_index[file.stem]
            try:
                yolo_annotations = AnnotationGroup.parse_annotation_file(file)
            except Exception:
                continue

            coco_annotations = []
            try:
                for annotation in yolo_annotations.annotations:
                    bbox = xywh_to_xcycwh(annotation.bbox)
                    coco_annotations.append(
                        coco.Annotation(
                            id=len(annotations),
                            image_id=image.id,
                            category_id=annotation.category_id,
                            area=bbox[2] * bbox[3],
                            iscrowd=0,
                            bbox=list(bbox),
                            segmentation=[],
                            attributes=dict(),
                        )
                    )
            except Exception:
                pass

            annotations.extend(coco_annotations)

        categories = [coco.Category(id=i, supercategory=name, name=name) for i, name in self.classes.items()]

        return coco.Dataset(
            images=images,
            annotations=annotations,
            categories=categories,
        )


class SplitsDataset(BaseModel):
    train_annotations: t.List[AnnotationGroup]
    val_annotations: t.List[AnnotationGroup]
    test_annotations: t.Optional[AnnotationGroup] = None

    def to_coco(self):
        datasets: t.List[coco.Dataset] = []
        return datasets
