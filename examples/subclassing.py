from pydantic import model_validator
from annotstein.coco.schemas import Category, Dataset, Annotation, Image


class AtLeastOneAnnotation(Dataset):
    @model_validator(mode="after")
    def at_least_one(self):
        image_ids = set(i.id for i in self.images)
        annotation_image_ids = set(a.image_id for a in self.annotations)

        assert image_ids.issubset(
            annotation_image_ids
        ), f"Some images are missing annotations: {image_ids - annotation_image_ids}"


AtLeastOneAnnotation(
    images=[Image(id=i, file_name=str(i)) for i in range(10)],
    annotations=[Annotation(id=i, image_id=i, category_id=0, bbox=[0, 1, 2, 3], segmentation=[[]]) for i in range(9)],
    categories=[Category(id=0, name="0", supercategory="0")],
)
