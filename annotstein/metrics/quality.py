"""Annotation quality checks for COCO datasets.

All functions accept a :class:`annotstein.coco.schemas.Dataset` and return
lists of issue dataclasses so callers can inspect, filter, or serialise them.
"""

import typing as t
from dataclasses import dataclass

from annotstein.coco.schemas import Dataset


# ---------------------------------------------------------------------------
# Issue types
# ---------------------------------------------------------------------------


@dataclass
class BboxIssue:
    annotation_id: int
    image_id: int
    issue: str


@dataclass
class SegmentationIssue:
    annotation_id: int
    image_id: int
    issue: str


@dataclass
class ReferenceIssue:
    annotation_id: int
    issue: str
    missing_id: int


@dataclass
class DuplicateAnnotation:
    annotation_id_a: int
    annotation_id_b: int
    image_id: int
    category_id: int
    iou: float


@dataclass
class LabelConflict:
    annotation_id_a: int
    annotation_id_b: int
    image_id: int
    category_id_a: int
    category_id_b: int
    iou: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iou_xywh(a: t.List[float], b: t.List[float]) -> float:
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah

    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_bboxes(ds: Dataset) -> t.List[BboxIssue]:
    """Detect invalid bounding boxes.

    Checks for:
    - Non-positive width or height.
    - Coordinates that exceed the recorded image dimensions.

    Images without recorded dimensions are skipped for bounds checking.

    Returns:
        List of :class:`BboxIssue` instances.
    """
    image_index = {img.id: img for img in ds.images}
    issues: t.List[BboxIssue] = []

    for ann in ds.annotations:
        if len(ann.bbox) != 4:
            issues.append(BboxIssue(ann.id, ann.image_id, "bbox must have exactly 4 values"))
            continue

        x, y, w, h = ann.bbox

        if w <= 0:
            issues.append(BboxIssue(ann.id, ann.image_id, f"non-positive width ({w})"))
        if h <= 0:
            issues.append(BboxIssue(ann.id, ann.image_id, f"non-positive height ({h})"))
        if x < 0 or y < 0:
            issues.append(BboxIssue(ann.id, ann.image_id, f"negative origin ({x}, {y})"))

        image = image_index.get(ann.image_id)
        if image and image.width and image.height:
            if x + w > image.width:
                issues.append(BboxIssue(ann.id, ann.image_id, f"bbox exceeds image width ({x + w} > {image.width})"))
            if y + h > image.height:
                issues.append(BboxIssue(ann.id, ann.image_id, f"bbox exceeds image height ({y + h} > {image.height})"))

    return issues


def validate_segmentations(ds: Dataset) -> t.List[SegmentationIssue]:
    """Detect malformed segmentation polygons.

    Checks for:
    - Polygons with fewer than 6 coordinate values (3 points minimum).
    - Odd number of coordinates (unpaired x/y).

    Returns:
        List of :class:`SegmentationIssue` instances.
    """
    issues: t.List[SegmentationIssue] = []

    for ann in ds.annotations:
        for i, poly in enumerate(ann.segmentation):
            if len(poly) % 2 != 0:
                issues.append(SegmentationIssue(ann.id, ann.image_id, f"polygon {i} has odd coordinate count ({len(poly)})"))
            elif len(poly) < 6:
                issues.append(SegmentationIssue(ann.id, ann.image_id, f"polygon {i} has fewer than 3 points ({len(poly) // 2})"))

    return issues


def validate_references(ds: Dataset) -> t.List[ReferenceIssue]:
    """Detect annotations with dangling image_id or category_id references.

    Returns:
        List of :class:`ReferenceIssue` instances.
    """
    image_ids = {img.id for img in ds.images}
    category_ids = {cat.id for cat in ds.categories}
    issues: t.List[ReferenceIssue] = []

    for ann in ds.annotations:
        if ann.image_id not in image_ids:
            issues.append(ReferenceIssue(ann.id, f"image_id {ann.image_id} not found in dataset", ann.image_id))
        if ann.category_id not in category_ids:
            issues.append(ReferenceIssue(ann.id, f"category_id {ann.category_id} not found in dataset", ann.category_id))

    return issues


def duplicate_annotations(
    ds: Dataset,
    iou_threshold: float = 0.9,
) -> t.List[DuplicateAnnotation]:
    """Detect near-duplicate annotations (same image, same category, high IoU).

    Args:
        ds: COCO dataset.
        iou_threshold: Minimum IoU to consider two annotations duplicates.

    Returns:
        List of :class:`DuplicateAnnotation` instances.
    """
    from collections import defaultdict

    by_image_category: t.Dict[t.Tuple[int, int], t.List] = defaultdict(list)
    for ann in ds.annotations:
        by_image_category[(ann.image_id, ann.category_id)].append(ann)

    duplicates: t.List[DuplicateAnnotation] = []

    for (image_id, category_id), anns in by_image_category.items():
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                iou = _iou_xywh(anns[i].bbox, anns[j].bbox)
                if iou >= iou_threshold:
                    duplicates.append(DuplicateAnnotation(anns[i].id, anns[j].id, image_id, category_id, round(iou, 4)))

    return duplicates


def label_consistency(
    ds: Dataset,
    iou_threshold: float = 0.5,
) -> t.List[LabelConflict]:
    """Detect same-location annotations with different category labels.

    Finds pairs of annotations on the same image with IoU >= ``iou_threshold``
    but different category IDs.

    Args:
        ds: COCO dataset.
        iou_threshold: Minimum IoU to consider two annotations co-located.

    Returns:
        List of :class:`LabelConflict` instances.
    """
    from collections import defaultdict

    by_image: t.Dict[int, t.List] = defaultdict(list)
    for ann in ds.annotations:
        by_image[ann.image_id].append(ann)

    conflicts: t.List[LabelConflict] = []

    for image_id, anns in by_image.items():
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if anns[i].category_id == anns[j].category_id:
                    continue
                iou = _iou_xywh(anns[i].bbox, anns[j].bbox)
                if iou >= iou_threshold:
                    conflicts.append(
                        LabelConflict(
                            anns[i].id,
                            anns[j].id,
                            image_id,
                            anns[i].category_id,
                            anns[j].category_id,
                            round(iou, 4),
                        )
                    )

    return conflicts


def quality_report(ds: Dataset, dup_iou: float = 0.9, conflict_iou: float = 0.5) -> t.Dict[str, t.Any]:
    """Aggregate quality report combining all validation checks.

    Returns:
        Dict with keys ``bbox_issues``, ``segmentation_issues``,
        ``reference_issues``, ``duplicates``, ``label_conflicts``, and
        a summary ``counts`` sub-dict.
    """
    from dataclasses import asdict

    bbox = validate_bboxes(ds)
    seg = validate_segmentations(ds)
    ref = validate_references(ds)
    dups = duplicate_annotations(ds, dup_iou)
    conflicts = label_consistency(ds, conflict_iou)

    return {
        "counts": {
            "bbox_issues": len(bbox),
            "segmentation_issues": len(seg),
            "reference_issues": len(ref),
            "duplicates": len(dups),
            "label_conflicts": len(conflicts),
        },
        "bbox_issues": [asdict(i) for i in bbox],
        "segmentation_issues": [asdict(i) for i in seg],
        "reference_issues": [asdict(i) for i in ref],
        "duplicates": [asdict(i) for i in dups],
        "label_conflicts": [asdict(i) for i in conflicts],
    }
