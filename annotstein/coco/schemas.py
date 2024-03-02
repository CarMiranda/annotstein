from datetime import datetime as dt
from pydantic import BaseModel, Extra, Field, model_validator
import typing as t

try:
    # python > 3.10
    from typing import Annotated
except ImportError:
    # python < 3.10
    from typing_extensions import Annotated


def compute_area(bbox: t.Tuple[float, float, float, float]):
    return bbox[2] * bbox[3]


class Image(BaseModel):
    id: int
    file_name: str
    height: t.Optional[int] = None
    width: t.Optional[int] = None
    flickr_url: t.Optional[str] = None
    coco_url: t.Optional[str] = None
    license: t.Optional[int] = None


class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: t.List[float]
    segmentation: t.List[t.List[float]]

    area: Annotated[float, Field(validate_default=True)] = 0.0
    iscrowd: int = 0
    attributes: t.Dict[str, t.Any] = dict()

    @model_validator(mode="after")
    def compute_area(self):
        self.area = compute_area(self.bbox)
        return self


class Prediction(Annotation):
    score: float

    def to_annotation(self):
        return Annotation(**self.model_dump(exclude={"score"}))


class Category(BaseModel):
    id: int
    name: str
    supercategory: str


class Info(BaseModel):
    contributor: str = ""
    date_created: Annotated[str, Field(validate_default=True)] = ""
    description: str = ""
    url: str = ""
    version: str = "1.0"
    year: int = dt.now().year

    @model_validator(mode="after")
    def add_date(self):
        if self.date_created == "":
            self.date_created = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
        return self


class License(BaseModel):
    id: int
    name: str = ""
    url: str = ""


class Dataset(BaseModel, extra=Extra.allow):
    categories: t.List[Category]
    images: t.List[Image]
    annotations: t.List[Annotation]

    info: Info = Info()
    licenses: t.List[License] = []
