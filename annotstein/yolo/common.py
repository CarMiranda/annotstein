from abc import abstractmethod
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

SplitSet = t.Union[pathlib.Path, t.List[pathlib.Path]]


class Annotation(BaseModel):
    category_id: int

    @classmethod
    @abstractmethod
    def parse_row(cls, *row) -> t.Self:
        pass

    @classmethod
    @abstractmethod
    def from_coco(
        cls,
        coco_annotation: coco.Annotation,
    ) -> t.Self:
        pass


TAnnotation = t.TypeVar("TAnnotation", bound=Annotation)


class AnnotationGroup(BaseModel, t.Generic[TAnnotation]):
    # Kind of annotation for this annotation group
    annotation_type: t.ClassVar[t.Type[Annotation]]

    # Image file concerned by these annotations
    file_name: str

    # Collection of annotations
    annotations: t.Sequence[TAnnotation]

    @abstractmethod
    def dump(self, fp: TextIOWrapper):
        pass

    @abstractmethod
    def dumps(self, delimiter: str = " ") -> str:
        pass

    @classmethod
    def parse_annotation_file(cls, path: pathlib.Path, delimiter: str = " ") -> t.Self:
        annotations = []
        with open(path) as fp:
            reader = csv.reader(fp, delimiter=delimiter)
            for row in reader:
                try:
                    annotations.append(cls.annotation_type.parse_row(*row))
                except Exception:
                    pass

        return cls(file_name=str(path.resolve()), annotations=annotations)

    @classmethod
    def from_coco(
        cls,
        file_name: str,
        coco_annotations: t.Sequence[coco.Annotation],
    ):
        return cls(file_name=file_name, annotations=[cls.annotation_type.from_coco(a) for a in coco_annotations])


class Dataset(BaseModel, t.Generic[TAnnotation]):
    # Kind of annotation group for this dataset
    annotation_type: t.ClassVar[t.Type[AnnotationGroup]]

    # Collection of annotation groups
    annotations: t.Sequence[AnnotationGroup[TAnnotation]]

    @classmethod
    def from_coco(cls, coco_dataset: coco.Dataset) -> t.Self:
        images_index = {i.id: i for i in coco_dataset.images}
        grouped_by_image: t.Dict[str, t.List[coco.Annotation]] = {i.file_name: [] for i in coco_dataset.images}
        for annotation in coco_dataset.annotations:
            file_name = images_index[annotation.image_id].file_name
            grouped_by_image[file_name].append(annotation)

        yolo_annotations = []
        for file_name, annotation_group in grouped_by_image.items():
            yolo_annotations.append(cls.annotation_type.from_coco(file_name, annotation_group))

        return cls(annotations=yolo_annotations)

    @classmethod
    def parse_directory(cls, path: pathlib.Path):
        annotations = []
        for file in path.glob("*.txt"):
            try:
                image_annotations = cls.annotation_type.parse_annotation_file(file)
            except Exception:
                continue

            annotations.extend(image_annotations)

        return cls(annotations=annotations)

    @classmethod
    def parse_to_coco(cls, path: pathlib.Path):
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
                yolo_annotations = cls.annotation_type.parse_annotation_file(file)
            except Exception:
                continue

            coco_annotations = []
            try:
                for annotation in yolo_annotations.annotations:
                    bbox = xcycwh_to_xywh(annotation.bbox)
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

        categories = set(a.category_id for a in annotations)
        categories = [coco.Category(id=i, supercategory=str(i), name=str(i)) for i in categories]

        return coco.Dataset(
            images=images,
            annotations=annotations,
            categories=categories,
        )


class TrainTask(BaseModel):
    dataset_type: t.ClassVar[t.Type[Dataset]]

    root_path: pathlib.Path
    train: SplitSet
    val: SplitSet
    test: t.Optional[SplitSet] = None

    names: t.Dict[int, str]

    download: t.Optional[AnyHttpUrl] = None

    def load(self) -> "SplitsDataset":
        train_annotations: t.Sequence[AnnotationGroup]
        if isinstance(self.train, list):
            train_annotations = list(chain(*[TrainTask.dataset_type.parse_directory(td).annotations for td in self.train]))
        elif isinstance(self.train, pathlib.Path):
            train_annotations = TrainTask.dataset_type.parse_directory(self.train).annotations
        else:
            raise ValueError(f"Found type {type(self.train)} for train annotations, expected path or list of paths.")

        val_annotations: t.Sequence[AnnotationGroup]
        if isinstance(self.val, list):
            val_annotations = list(chain(*[TrainTask.dataset_type.parse_directory(td).annotations for td in self.val]))
        elif isinstance(self.val, pathlib.Path):
            val_annotations = TrainTask.dataset_type.parse_directory(self.val).annotations
        else:
            raise ValueError(f"Found type {type(self.val)} for val annotations, expected path or list of paths.")

        test_annotations: t.Optional[t.Sequence[AnnotationGroup]]
        if self.test is None:
            test_annotations = None
        elif isinstance(self.test, list):
            test_annotations = list(chain(*[TrainTask.dataset_type.parse_directory(td).annotations for td in self.test]))
        elif isinstance(self.test, pathlib.Path):
            test_annotations = TrainTask.dataset_type.parse_directory(self.test).annotations
        else:
            raise ValueError(f"Found type {type(self.test)} for test annotations, expected path or list of paths.")

        return SplitsDataset(
            train_annotations=TrainTask.dataset_type(annotations=train_annotations),
            val_annotations=TrainTask.dataset_type(annotations=val_annotations),
            test_annotations=TrainTask.dataset_type(annotations=test_annotations) if test_annotations is not None else None,
        )


class SplitsDataset(BaseModel, t.Generic[TAnnotation]):
    # Train split
    train_annotations: Dataset[TAnnotation]

    # Validation split
    val_annotations: Dataset[TAnnotation]

    # Optional test split
    test_annotations: t.Optional[Dataset[TAnnotation]] = None

    def to_coco(self):
        datasets: t.List[coco.Dataset] = []
        return datasets

    @classmethod
    def from_coco(cls):
        pass


# Specifics


def xcycwh_to_xywh(bbox: t.Tuple[float, float, float, float]) -> t.Tuple[float, float, float, float]:
    return bbox[0] - bbox[2] * 0.5, bbox[1] - bbox[3] * 0.5, bbox[2], bbox[3]


def xywh_to_xcycwh(bbox: t.Tuple[float, float, float, float]) -> t.Tuple[float, float, float, float]:
    return bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5, bbox[2], bbox[3]


class BBoxAnnotation(Annotation):
    bbox: Annotated[t.Sequence[float], Field(min_length=4, max_length=4)]

    @classmethod
    def parse_row(cls, *row) -> "BBoxAnnotation":
        return cls(category_id=int(row[0]), bbox=list(map(float, row[1:])))

    @classmethod
    def from_coco(
        cls,
        coco_annotation: coco.Annotation,
    ) -> t.Self:
        return cls(category_id=coco_annotation.category_id, bbox=xywh_to_xcycwh(coco_annotation.bbox))


class SegmentationAnnotation(Annotation):
    segmentation: Annotated[t.Sequence[float], Field(min_length=6)]

    @classmethod
    def parse_row(cls, *row) -> "SegmentationAnnotation":
        return cls(category_id=int(row[0]), segmentation=list(map(float, row[1:])))

    @classmethod
    def from_coco(
        cls,
        coco_annotation: coco.Annotation,
    ) -> t.Self:
        return cls(
            category_id=coco_annotation.category_id,
            segmentation=coco_annotation.segmentation[0] if coco_annotation.segmentation is not None else [],
        )


class BBoxAnnotationGroup(AnnotationGroup[BBoxAnnotation]):
    annotation_type = BBoxAnnotation


class SegmentationAnnotationGroup(AnnotationGroup[SegmentationAnnotation]):
    annotation_type = SegmentationAnnotation
