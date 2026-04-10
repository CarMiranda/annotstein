"""Exploratory data analysis functions for COCO datasets.

All functions accept a :class:`annotstein.coco.schemas.Dataset` and return
plain dicts so results can be serialised to JSON directly.
"""

import typing as t
from collections import defaultdict

import numpy as np

from annotstein.coco.schemas import Annotation, Dataset


def _bbox_features(annotation: Annotation) -> t.Dict[str, float]:
    x, y, w, h = annotation.bbox
    aspect_ratio = w / h if h > 0 else float("nan")
    return {
        "width": w,
        "height": h,
        "area": w * h,
        "aspect_ratio": aspect_ratio,
        "cx": x + w / 2,
        "cy": y + h / 2,
    }


def _describe(values: t.List[float]) -> t.Dict[str, float]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "median": None}
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "median": None}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
    }


def bbox_stats(ds: Dataset) -> t.Dict[str, t.Dict[str, t.Dict[str, float]]]:
    """Per-category descriptive statistics for bbox width, height, area and aspect ratio.

    Returns a nested dict:
        ``{category_name: {width: {...}, height: {...}, area: {...}, aspect_ratio: {...}}}``
    """
    category_index = {c.id: c.name for c in ds.categories}
    buckets: t.Dict[str, t.Dict[str, t.List[float]]] = {}

    for annotation in ds.annotations:
        name = category_index.get(annotation.category_id, str(annotation.category_id))
        if name not in buckets:
            buckets[name] = {"width": [], "height": [], "area": [], "aspect_ratio": []}
        feats = _bbox_features(annotation)
        for key in ("width", "height", "area", "aspect_ratio"):
            buckets[name][key].append(feats[key])

    return {name: {k: _describe(v) for k, v in fields.items()} for name, fields in buckets.items()}


def bbox_aspect_ratio_buckets(
    ds: Dataset,
    n_buckets: int = 10,
    per_category: bool = True,
) -> t.Dict[str, t.Any]:
    """Histogram of bbox aspect ratios (width / height).

    Args:
        ds: COCO dataset.
        n_buckets: Number of histogram bins.
        per_category: If True, also returns per-category histograms.

    Returns:
        Dict with keys ``overall`` and (optionally) per-category dicts, each
        containing ``edges`` and ``counts`` lists.
    """
    category_index = {c.id: c.name for c in ds.categories}
    all_ratios: t.List[float] = []
    per_cat: t.Dict[str, t.List[float]] = defaultdict(list)

    for annotation in ds.annotations:
        ratio = _bbox_features(annotation)["aspect_ratio"]
        if np.isfinite(ratio):
            all_ratios.append(ratio)
            if per_category:
                name = category_index.get(annotation.category_id, str(annotation.category_id))
                per_cat[name].append(ratio)

    def _histogram(values: t.List[float]) -> t.Dict[str, t.Any]:
        if not values:
            return {"edges": [], "counts": []}
        counts, edges = np.histogram(values, bins=n_buckets)
        return {"edges": edges.tolist(), "counts": counts.tolist()}

    result: t.Dict[str, t.Any] = {"overall": _histogram(all_ratios)}
    if per_category:
        result["per_category"] = {name: _histogram(ratios) for name, ratios in per_cat.items()}
    return result


def bbox_size_distribution(
    ds: Dataset,
    thresholds: t.Tuple[float, float] = (32**2, 96**2),
) -> t.Dict[str, t.Dict[str, int]]:
    """Per-category count of small / medium / large bboxes.

    Uses COCO-standard area thresholds by default (small < 32², medium 32²–96²,
    large > 96²).  Pass custom ``thresholds=(t1, t2)`` to override.
    """
    t_small, t_large = thresholds
    category_index = {c.id: c.name for c in ds.categories}
    counts: t.Dict[str, t.Dict[str, int]] = {}

    for annotation in ds.annotations:
        name = category_index.get(annotation.category_id, str(annotation.category_id))
        if name not in counts:
            counts[name] = {"small": 0, "medium": 0, "large": 0}
        area = _bbox_features(annotation)["area"]
        if area < t_small:
            counts[name]["small"] += 1
        elif area <= t_large:
            counts[name]["medium"] += 1
        else:
            counts[name]["large"] += 1

    return counts


def class_distribution(ds: Dataset) -> t.Dict[str, t.Any]:
    """Instance counts per category and overall imbalance ratio.

    Returns:
        Dict with ``counts`` (category → int) and ``imbalance_ratio``
        (max_count / min_count, or None if fewer than 2 categories).
    """
    category_index = {c.id: c.name for c in ds.categories}
    counts: t.Dict[str, int] = defaultdict(int)

    for annotation in ds.annotations:
        name = category_index.get(annotation.category_id, str(annotation.category_id))
        counts[name] += 1

    counts_dict = dict(counts)
    values = list(counts_dict.values())
    imbalance = float(max(values)) / float(min(values)) if len(values) >= 2 and min(values) > 0 else None
    return {"counts": counts_dict, "imbalance_ratio": imbalance}


def annotation_density(ds: Dataset, n_buckets: int = 10) -> t.Dict[str, t.Any]:
    """Annotations per image: descriptive stats and histogram.

    Returns:
        Dict with ``stats`` (mean/std/min/max/median) and ``histogram``
        (edges + counts).
    """
    per_image: t.Dict[int, int] = {img.id: 0 for img in ds.images}
    for annotation in ds.annotations:
        if annotation.image_id in per_image:
            per_image[annotation.image_id] += 1

    values = list(per_image.values())
    counts, edges = np.histogram(values, bins=n_buckets)
    return {
        "stats": _describe(values),
        "histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
    }


def spatial_distribution(
    ds: Dataset,
    grid_size: int = 10,
) -> t.Dict[str, t.Any]:
    """2D heatmap of normalised bbox centre coordinates.

    Centres are expressed in [0, 1] coordinates (relative to image dimensions).
    Images without dimensions recorded are skipped.

    Returns:
        Dict with ``grid`` (2D list of counts, shape grid_size × grid_size)
        and ``skipped_images`` count.
    """
    image_index = {img.id: img for img in ds.images}
    grid = np.zeros((grid_size, grid_size), dtype=int)
    skipped = 0

    for annotation in ds.annotations:
        image = image_index.get(annotation.image_id)
        if image is None or image.width is None or image.height is None:
            skipped += 1
            continue

        feats = _bbox_features(annotation)
        cx_rel = feats["cx"] / image.width
        cy_rel = feats["cy"] / image.height

        col = min(int(cx_rel * grid_size), grid_size - 1)
        row = min(int(cy_rel * grid_size), grid_size - 1)
        grid[row, col] += 1

    return {"grid": grid.tolist(), "grid_size": grid_size, "skipped_images": skipped}


def image_coverage(ds: Dataset) -> t.Dict[str, t.Any]:
    """Ratio of image area covered by bboxes, per image and per category.

    Images without recorded dimensions are skipped.

    Returns:
        Dict with ``per_image`` (image_id → coverage ratio) and
        ``per_category`` (category_name → stats dict).
    """
    image_index = {img.id: img for img in ds.images}
    category_index = {c.id: c.name for c in ds.categories}

    per_image: t.Dict[int, float] = {}
    per_category: t.Dict[str, t.List[float]] = defaultdict(list)

    image_area_used: t.Dict[int, float] = defaultdict(float)
    image_area_total: t.Dict[int, float] = {}

    for image in ds.images:
        if image.width and image.height:
            image_area_total[image.id] = float(image.width * image.height)

    for annotation in ds.annotations:
        if annotation.image_id not in image_area_total:
            continue
        _, _, w, h = annotation.bbox
        image_area_used[annotation.image_id] += w * h

        name = category_index.get(annotation.category_id, str(annotation.category_id))
        image = image_index[annotation.image_id]
        per_category[name].append((w * h) / image_area_total[annotation.image_id])

    for image_id, total in image_area_total.items():
        per_image[image_id] = image_area_used[image_id] / total if total > 0 else 0.0

    return {
        "per_image": {str(k): v for k, v in per_image.items()},
        "per_category": {name: _describe(values) for name, values in per_category.items()},
        "overall": _describe(list(per_image.values())),
    }


def segmentation_complexity(ds: Dataset) -> t.Dict[str, t.Any]:
    """Per-annotation polygon vertex count statistics.

    A segmentation polygon with *k* coordinate pairs has *k/2* vertices.

    Returns:
        Dict with ``overall`` stats and ``per_category`` stats.
    """
    category_index = {c.id: c.name for c in ds.categories}
    all_counts: t.List[float] = []
    per_category: t.Dict[str, t.List[float]] = defaultdict(list)

    for annotation in ds.annotations:
        if not annotation.segmentation:
            continue
        vertex_count = sum(len(poly) / 2 for poly in annotation.segmentation)
        all_counts.append(vertex_count)
        name = category_index.get(annotation.category_id, str(annotation.category_id))
        per_category[name].append(vertex_count)

    return {
        "overall": _describe(all_counts),
        "per_category": {name: _describe(counts) for name, counts in per_category.items()},
    }


def images_without_annotations(ds: Dataset) -> t.Dict[str, t.Any]:
    """Returns images that have no annotations.

    Returns:
        Dict with ``count`` and ``image_ids`` list.
    """
    annotated_ids = {a.image_id for a in ds.annotations}
    unannotated = [img for img in ds.images if img.id not in annotated_ids]
    return {
        "count": len(unannotated),
        "image_ids": [img.id for img in unannotated],
        "file_names": [img.file_name for img in unannotated],
    }


def annotations_per_category_per_image(ds: Dataset) -> t.Dict[str, t.Any]:
    """Cross-tabulation of annotation counts by category × image.

    Returns:
        Dict with ``table`` (category_name → {image_id → count}) and
        per-category ``stats`` (mean/std/min/max/median annotations per image).
    """
    category_index = {c.id: c.name for c in ds.categories}
    table: t.Dict[str, t.Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for annotation in ds.annotations:
        name = category_index.get(annotation.category_id, str(annotation.category_id))
        table[name][str(annotation.image_id)] += 1

    stats = {name: _describe(list(counts.values())) for name, counts in table.items()}
    return {
        "table": {name: dict(counts) for name, counts in table.items()},
        "stats": stats,
    }


def summary(ds: Dataset) -> t.Dict[str, t.Any]:
    """Aggregate EDA report combining all analysis functions.

    Returns a single nested dict suitable for JSON serialisation.
    """
    return {
        "dataset": {
            "n_images": len(ds.images),
            "n_annotations": len(ds.annotations),
            "n_categories": len(ds.categories),
        },
        "class_distribution": class_distribution(ds),
        "bbox_stats": bbox_stats(ds),
        "bbox_size_distribution": bbox_size_distribution(ds),
        "bbox_aspect_ratio_buckets": bbox_aspect_ratio_buckets(ds),
        "annotation_density": annotation_density(ds),
        "spatial_distribution": spatial_distribution(ds),
        "image_coverage": image_coverage(ds),
        "segmentation_complexity": segmentation_complexity(ds),
        "images_without_annotations": images_without_annotations(ds),
        "annotations_per_category_per_image": annotations_per_category_per_image(ds),
    }
