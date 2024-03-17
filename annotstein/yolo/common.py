from abc import abstractmethod
import csv
from io import TextIOWrapper
from itertools import chain
import pathlib
import typing as t
from typing_extensions import Annotated, Self
from pydantic import AnyHttpUrl, BaseModel, Field
import yaml

from PIL import Image
from annotstein.coco import ops as coco
from annotstein.common import IMAGE_SUFFIXES

SplitSet = t.Union[pathlib.Path, t.List[pathlib.Path]]


class Annotation(BaseModel):
    category_id: int

    @classmethod
    @abstractmethod
    def parse_row(cls, *row) -> Self:
        pass

    @abstractmethod
    def dumps_row(self, delimiter: str = " ") -> str:
        pass

    @abstractmethod
    def dump_row(self) -> t.List[t.Any]:
        pass

    @classmethod
    @abstractmethod
    def from_coco(
        cls,
        coco_annotation: coco.Annotation,
    ) -> Self:
        pass

    @abstractmethod
    def to_coco(self) -> t.Tuple[t.List[float], t.List[t.List[float]]]:
        pass


TAnnotation = t.TypeVar("TAnnotation", bound=Annotation)


class AnnotationGroup(BaseModel, t.Generic[TAnnotation]):
    # Kind of annotation for this annotation group
    annotation_type: t.ClassVar[t.Type[Annotation]]

    # Image file concerned by these annotations
    file_name: str

    # Collection of annotations
    annotations: t.Sequence[TAnnotation]

    def dump(self, fp: TextIOWrapper):
        writer = csv.writer(fp, delimiter=" ")
        for annotation in self.annotations:
            writer.writerow(annotation.dump_row())

    def dumps(self):
        res = ""
        for annotation in self.annotations:
            res += annotation.dumps_row()

    @classmethod
    def parse_annotation_file(cls, path: pathlib.Path, delimiter: str = " ") -> t.Sequence[TAnnotation]:
        annotations = []
        with open(path) as fp:
            reader = csv.reader(fp, delimiter=delimiter)
            for row in reader:
                try:
                    annotations.append(cls.annotation_type.parse_row(*row))
                except Exception:
                    pass

        return annotations

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

    # Category names
    names: t.Optional[t.Dict[int, str]] = None

    @classmethod
    def from_coco(cls, coco_dataset: coco.Dataset) -> Self:
        images_index = {i.id: i for i in coco_dataset.images}
        grouped_by_image: t.Dict[str, t.List[coco.Annotation]] = {i.file_name: [] for i in coco_dataset.images}
        for annotation in coco_dataset.annotations:
            file_name = images_index[annotation.image_id].file_name
            grouped_by_image[file_name].append(annotation)

        yolo_annotations = []
        for file_name, annotation_group in grouped_by_image.items():
            yolo_annotations.append(cls.annotation_type.from_coco(file_name, annotation_group))

        names = {c.id: c.name for c in coco_dataset.categories}

        return cls(annotations=yolo_annotations, names=names)

    def to_coco(self):
        images: t.List[coco.Image] = []
        annotations: t.List[coco.Annotation] = []
        found_categories = set()

        for i, annotation_group in enumerate(self.annotations):
            images.append(
                coco.Image(
                    id=i,
                    file_name=annotation_group.file_name,
                    height=None,
                    width=None,
                )
            )

            for j, annotation in enumerate(annotation_group.annotations):
                bbox, segmentation = annotation.to_coco()
                annotations.append(
                    coco.Annotation(
                        id=j,
                        image_id=i,
                        category_id=annotation.category_id,
                        bbox=bbox,
                        segmentation=segmentation,
                    )
                )
                found_categories.add(annotation.category_id)

        if self.names is not None:
            categories = [coco.Category(id=i, name=name, supercategory=name) for i, name in self.names.items()]
        else:
            categories = [coco.Category(id=i, name=str(i), supercategory=str(i)) for i in found_categories]

        return coco.Dataset(categories=categories, images=images, annotations=annotations)

    @classmethod
    def parse_directory(cls, path: pathlib.Path):
        images_path = path / "images"
        labels_path = path / "labels"
        if not (images_path.exists() and labels_path.exists()):
            images_path = labels_path = path

        images: t.Dict[str, pathlib.Path] = dict()
        annotations: t.Sequence[AnnotationGroup[TAnnotation]] = []

        images_lst = list(chain(*[images_path.glob(f"*{s}") for s in IMAGE_SUFFIXES]))
        images = {p.stem: p for p in images_lst}

        for file in labels_path.glob("*.txt"):
            try:
                image_annotations = cls.annotation_type.parse_annotation_file(file)
            except Exception as e:
                print(e)
                continue

            image_name = images.pop(file.stem)
            annotation_group = AnnotationGroup(file_name=str(image_name.resolve()), annotations=image_annotations)
            annotations.append(annotation_group)

        # Images without annotations
        for i in images.values():
            annotations.append(AnnotationGroup(file_name=str(i.resolve()), annotations=[]))

        return annotations

    @classmethod
    def parse_to_coco(cls, path: pathlib.Path):
        images_path = path / "images"
        labels_path = path / "labels"
        if not (images_path.exists() and labels_path.exists()):
            images_path = labels_path = path

        images: t.List[coco.Image] = []
        image_index: t.Dict[str, coco.Image] = dict()
        annotations: t.List[coco.Annotation] = []
        for file in chain(*[images_path.glob(f"*{s}") for s in IMAGE_SUFFIXES]):
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

        for file in labels_path.glob("*.txt"):
            if file.stem not in image_index:
                continue

            image = image_index[file.stem]
            try:
                yolo_annotations = cls.annotation_type.parse_annotation_file(file)
            except Exception:
                continue

            annotation_group = AnnotationGroup(file_name=image.file_name, annotations=yolo_annotations)

            coco_annotations = []
            try:
                for annotation in annotation_group.annotations:
                    bbox, segmentation = annotation.to_coco()
                    coco_annotations.append(
                        coco.Annotation(
                            id=len(annotations),
                            image_id=image.id,
                            category_id=annotation.category_id,
                            area=bbox[2] * bbox[3],
                            iscrowd=0,
                            bbox=bbox,
                            segmentation=segmentation,
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

    path: pathlib.Path
    train: SplitSet
    val: SplitSet
    test: t.Optional[SplitSet] = None

    names: t.Dict[int, str]

    download: t.Optional[AnyHttpUrl] = None

    @classmethod
    def parse_yaml(cls, path: pathlib.Path) -> Self:
        with open(path) as fp:
            d = yaml.safe_load(fp)
        return cls.model_validate(d)

    def load(self) -> "SplitsDataset":
        train_annotations: t.Sequence[AnnotationGroup]
        val_annotations: t.Sequence[AnnotationGroup]
        test_annotations: t.Optional[t.Sequence[AnnotationGroup]] = None

        train_paths = self.train if isinstance(self.train, list) else [self.train]
        train_annotations = list(chain(*[TrainTask.dataset_type.parse_directory(td) for td in train_paths]))

        val_paths = self.val if isinstance(self.val, list) else [self.val]
        val_annotations = list(chain(*[TrainTask.dataset_type.parse_directory(td) for td in val_paths]))

        if self.test is not None:
            test_paths = self.test if isinstance(self.test, list) else [self.test]
            test_annotations = list(chain(*[TrainTask.dataset_type.parse_directory(td) for td in test_paths]))

        return SplitsDataset(
            train_annotations=TrainTask.dataset_type(annotations=train_annotations),
            val_annotations=TrainTask.dataset_type(annotations=val_annotations),
            test_annotations=TrainTask.dataset_type(annotations=test_annotations) if test_annotations is not None else None,
        )

    def to_coco(self) -> t.Tuple[coco.Dataset, coco.Dataset, t.Optional[coco.Dataset]]:
        train_paths = self.train if isinstance(self.train, list) else [self.train]
        train_ags = [self.dataset_type.parse_directory(self.path / pt) for pt in train_paths]
        train_dss = [Dataset(annotations=ags, names=self.names).to_coco() for ags in train_ags]
        train_ds = coco.merge_datasets(*train_dss)

        val_paths = self.val if isinstance(self.val, list) else [self.val]
        val_ags = [self.dataset_type.parse_directory(self.path / pt) for pt in val_paths]
        val_dss = [Dataset(annotations=ags, names=self.names).to_coco() for ags in val_ags]
        val_ds = coco.merge_datasets(*val_dss)

        test_ds = None
        if self.test is not None:
            test_paths = self.test if isinstance(self.test, list) else [self.test]
            test_ags = [self.dataset_type.parse_directory(self.path / pt) for pt in test_paths]
            test_dss = [Dataset(annotations=ags, names=self.names).to_coco() for ags in test_ags]
            test_ds = coco.merge_datasets(*test_dss)

        return train_ds, val_ds, test_ds


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


class DetectionAnnotation(Annotation):
    bbox: Annotated[t.Sequence[float], Field(min_length=4, max_length=4)]

    @classmethod
    def parse_row(cls, *row) -> "DetectionAnnotation":
        return cls(category_id=int(row[0]), bbox=list(map(float, row[1:])))

    def dump_row(self) -> t.List[t.Any]:
        return [self.category_id, *self.bbox]

    def dumps_row(self, delimiter: str = " ") -> str:
        return delimiter.join(map(str, [self.category_id, *self.bbox]))

    @classmethod
    def from_coco(
        cls,
        coco_annotation: coco.Annotation,
    ) -> Self:
        return cls(category_id=coco_annotation.category_id, bbox=xywh_to_xcycwh(coco_annotation.bbox))

    def to_coco(self):
        x, y, w, h = xcycwh_to_xywh(self.bbox)
        return [x, y, w, h], [[x, y, x + w, y, x + w, y + h, x, y + h]]


class SegmentationAnnotation(Annotation):
    segmentation: Annotated[t.Sequence[float], Field(min_length=6)]

    @classmethod
    def parse_row(cls, *row) -> "SegmentationAnnotation":
        return cls(category_id=int(row[0]), segmentation=list(map(float, row[1:])))

    def dump_row(self) -> t.List[t.Any]:
        return [self.category_id, *self.segmentation]

    def dumps_row(self, delimiter: str = " ") -> str:
        return delimiter.join(map(str, [self.category_id, *self.segmentation]))

    @classmethod
    def from_coco(
        cls,
        coco_annotation: coco.Annotation,
    ) -> Self:
        return cls(
            category_id=coco_annotation.category_id,
            segmentation=coco_annotation.segmentation[0] if coco_annotation.segmentation is not None else [],
        )

    def to_coco(self):
        xmin = min(self.segmentation[::2])
        ymin = min(self.segmentation[1::2])
        xmax = max(self.segmentation[::2])
        ymax = max(self.segmentation[1::2])
        return [xmin, ymin, xmax - xmin, ymax - ymin], [list(self.segmentation)]


class DetectionAnnotationGroup(AnnotationGroup[DetectionAnnotation]):
    annotation_type = DetectionAnnotation


class SegmentationAnnotationGroup(AnnotationGroup[SegmentationAnnotation]):
    annotation_type = SegmentationAnnotation


class DetectionDataset(Dataset[DetectionAnnotation]):
    annotation_type = DetectionAnnotationGroup


class SegmentationDataset(Dataset[SegmentationAnnotation]):
    annotation_type = DetectionAnnotationGroup


class DetectionTask(TrainTask):
    dataset_type = DetectionDataset


class SegmentationTask(TrainTask):
    dataset_type = SegmentationDataset
