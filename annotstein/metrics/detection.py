"""Standard COCO detection metrics (AP / AR).

Implements AP@[.50:.05:.95], AP50, AP75, APsmall/medium/large,
AR@1/10/100, and per-category breakdowns — matching the official COCO
evaluation protocol.

Ground-truth annotations are represented by
:class:`annotstein.coco.schemas.Dataset`; predictions by a dataset whose
annotations carry a ``score`` field (:class:`annotstein.coco.schemas.Prediction`
already models this).
"""

import typing as t
from collections import defaultdict

import numpy as np

from annotstein.coco.schemas import Annotation, Dataset

# COCO standard IoU thresholds: 0.50, 0.55, ..., 0.95
COCO_IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]

# COCO area ranges (in squared pixels)
AREA_RANGES: t.Dict[str, t.Tuple[float, float]] = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32**2),
    "medium": (32**2, 96**2),
    "large": (96**2, float("inf")),
}


# ---------------------------------------------------------------------------
# Core IoU utilities
# ---------------------------------------------------------------------------


def compute_iou(box_a: t.List[float], box_b: t.List[float]) -> float:
    """Compute IoU between two bboxes in [x, y, w, h] format."""
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah

    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _iou_matrix(gt_boxes: t.List[t.List[float]], pred_boxes: t.List[t.List[float]]) -> np.ndarray:
    """Compute IoU matrix of shape (n_gt, n_pred)."""
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    mat = np.zeros((n_gt, n_pred), dtype=float)
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            mat[i, j] = compute_iou(g, p)
    return mat


# ---------------------------------------------------------------------------
# Per-image matching
# ---------------------------------------------------------------------------


def _match_single_image(
    gt_anns: t.List[Annotation],
    pred_anns: t.List[t.Any],
    iou_threshold: float,
) -> t.Tuple[t.List[float], t.List[bool], int]:
    """Match predictions to ground-truth for one image at one IoU threshold.

    Returns:
        ``(scores, tp_flags, n_gt)`` where ``scores`` and ``tp_flags`` are
        aligned lists sorted by descending score.
    """
    if not pred_anns:
        return [], [], len(gt_anns)

    scores = [getattr(p, "score", 1.0) for p in pred_anns]
    order = sorted(range(len(pred_anns)), key=lambda i: -scores[i])
    pred_sorted = [pred_anns[i] for i in order]
    scores_sorted = [scores[i] for i in order]

    if not gt_anns:
        return scores_sorted, [False] * len(pred_sorted), 0

    gt_boxes = [ann.bbox for ann in gt_anns]
    pred_boxes = [ann.bbox for ann in pred_sorted]
    iou_mat = _iou_matrix(gt_boxes, pred_boxes)

    gt_matched = [False] * len(gt_anns)
    tp_flags: t.List[bool] = []

    for j in range(len(pred_sorted)):
        best_iou = iou_threshold - 1e-9
        best_gt = -1
        for i in range(len(gt_anns)):
            if gt_matched[i]:
                continue
            if iou_mat[i, j] > best_iou:
                best_iou = iou_mat[i, j]
                best_gt = i
        if best_gt >= 0:
            gt_matched[best_gt] = True
            tp_flags.append(True)
        else:
            tp_flags.append(False)

    return scores_sorted, tp_flags, len(gt_anns)


# ---------------------------------------------------------------------------
# AP / AR computation
# ---------------------------------------------------------------------------


def _compute_ap_from_tp_fp(tp_flags: t.List[bool], n_gt: int) -> float:
    """Compute average precision using the 101-point COCO interpolation."""
    if n_gt == 0:
        return float("nan")
    if not tp_flags:
        return 0.0

    tp_cumsum = np.cumsum(tp_flags).astype(float)
    fp_cumsum = np.cumsum([not f for f in tp_flags]).astype(float)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / n_gt

    # 101-point interpolation
    ap = 0.0
    for t_r in np.linspace(0, 1, 101):
        p = precisions[recalls >= t_r]
        ap += p.max() if p.size > 0 else 0.0
    return ap / 101.0


def compute_ap(
    gt_ds: Dataset,
    pred_ds: Dataset,
    iou_threshold: float = 0.5,
    category_id: t.Optional[int] = None,
    area_range: t.Tuple[float, float] = AREA_RANGES["all"],
    max_dets: int = 100,
) -> float:
    """Compute average precision for one category at one IoU threshold.

    Args:
        gt_ds: Ground-truth COCO dataset.
        pred_ds: Predictions dataset (annotations should carry a ``score``
            attribute; defaults to 1.0 if absent).
        iou_threshold: IoU threshold for a true positive match.
        category_id: If provided, only evaluate this category.  If ``None``,
            evaluates across all categories.
        area_range: ``(min_area, max_area)`` filter on gt bbox area.
        max_dets: Maximum detections per image to consider.

    Returns:
        AP value in [0, 1].
    """
    gt_by_image: t.Dict[int, t.List[Annotation]] = defaultdict(list)
    for ann in gt_ds.annotations:
        if category_id is not None and ann.category_id != category_id:
            continue
        area = ann.bbox[2] * ann.bbox[3]
        if area_range[0] <= area <= area_range[1]:
            gt_by_image[ann.image_id].append(ann)

    pred_by_image: t.Dict[int, t.List] = defaultdict(list)
    for ann in pred_ds.annotations:
        if category_id is not None and ann.category_id != category_id:
            continue
        area = ann.bbox[2] * ann.bbox[3]
        if area_range[0] <= area <= area_range[1]:
            pred_by_image[ann.image_id].append(ann)

    all_scores: t.List[float] = []
    all_tp: t.List[bool] = []
    total_gt = 0

    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    for image_id in all_image_ids:
        gt_anns = gt_by_image.get(image_id, [])
        pred_anns = pred_by_image.get(image_id, [])

        if max_dets < len(pred_anns):
            pred_anns = sorted(pred_anns, key=lambda a: -getattr(a, "score", 1.0))[:max_dets]

        scores, tp_flags, n_gt = _match_single_image(gt_anns, pred_anns, iou_threshold)
        all_scores.extend(scores)
        all_tp.extend(tp_flags)
        total_gt += n_gt

    if not all_scores:
        return 0.0

    order = sorted(range(len(all_scores)), key=lambda i: -all_scores[i])
    all_tp_sorted = [all_tp[i] for i in order]

    return _compute_ap_from_tp_fp(all_tp_sorted, total_gt)


def compute_ar(
    gt_ds: Dataset,
    pred_ds: Dataset,
    iou_thresholds: t.Optional[t.List[float]] = None,
    max_dets: int = 100,
    category_id: t.Optional[int] = None,
    area_range: t.Tuple[float, float] = AREA_RANGES["all"],
) -> float:
    """Compute average recall averaged over IoU thresholds.

    Args:
        gt_ds: Ground-truth COCO dataset.
        pred_ds: Predictions dataset.
        iou_thresholds: IoU thresholds to average over.  Defaults to
            ``[0.50, 0.55, …, 0.95]``.
        max_dets: Maximum detections per image.
        category_id: Restrict to this category, or ``None`` for all.
        area_range: Bbox area filter.

    Returns:
        AR value in [0, 1].
    """
    if iou_thresholds is None:
        iou_thresholds = COCO_IOU_THRESHOLDS

    recalls = []
    for iou_thr in iou_thresholds:
        gt_by_image: t.Dict[int, t.List[Annotation]] = defaultdict(list)
        for ann in gt_ds.annotations:
            if category_id is not None and ann.category_id != category_id:
                continue
            area = ann.bbox[2] * ann.bbox[3]
            if area_range[0] <= area <= area_range[1]:
                gt_by_image[ann.image_id].append(ann)

        pred_by_image: t.Dict[int, t.List] = defaultdict(list)
        for ann in pred_ds.annotations:
            if category_id is not None and ann.category_id != category_id:
                continue
            area = ann.bbox[2] * ann.bbox[3]
            if area_range[0] <= area <= area_range[1]:
                pred_by_image[ann.image_id].append(ann)

        tp_total = 0
        gt_total = 0
        for image_id in set(gt_by_image.keys()) | set(pred_by_image.keys()):
            gt_anns = gt_by_image.get(image_id, [])
            pred_anns = pred_by_image.get(image_id, [])
            if max_dets < len(pred_anns):
                pred_anns = sorted(pred_anns, key=lambda a: -getattr(a, "score", 1.0))[:max_dets]
            _, tp_flags, n_gt = _match_single_image(gt_anns, pred_anns, iou_thr)
            tp_total += sum(tp_flags)
            gt_total += n_gt

        recalls.append(tp_total / gt_total if gt_total > 0 else 0.0)

    return float(np.mean(recalls))


def compute_coco_metrics(
    gt_ds: Dataset,
    pred_ds: Dataset,
) -> t.Dict[str, t.Any]:
    """Compute the full suite of COCO detection metrics.

    Metrics computed:
    - AP @ IoU=0.50:0.05:0.95  (primary metric)
    - AP @ IoU=0.50  (AP50)
    - AP @ IoU=0.75  (AP75)
    - AP for small / medium / large objects
    - AR @ max 1 / 10 / 100 detections
    - All of the above per category

    Args:
        gt_ds: Ground-truth COCO dataset.
        pred_ds: Predictions dataset.  Annotation objects must have a
            ``score`` attribute (float).

    Returns:
        Nested dict with keys ``overall`` and ``per_category``.
    """
    categories = {c.id: c.name for c in gt_ds.categories}

    def _overall_ap(iou_thr: float, area: str = "all", max_dets: int = 100) -> float:
        aps = []
        for cat_id in categories:
            ap = compute_ap(gt_ds, pred_ds, iou_thr, cat_id, AREA_RANGES[area], max_dets)
            if not np.isnan(ap):
                aps.append(ap)
        return float(np.mean(aps)) if aps else float("nan")

    def _overall_ar(max_dets: int) -> float:
        ars = []
        for cat_id in categories:
            ar = compute_ar(gt_ds, pred_ds, COCO_IOU_THRESHOLDS, max_dets, cat_id)
            ars.append(ar)
        return float(np.mean(ars)) if ars else float("nan")

    def _overall_ap_range() -> float:
        aps = []
        for iou_thr in COCO_IOU_THRESHOLDS:
            for cat_id in categories:
                ap = compute_ap(gt_ds, pred_ds, iou_thr, cat_id)
                if not np.isnan(ap):
                    aps.append(ap)
        return float(np.mean(aps)) if aps else float("nan")

    overall = {
        "AP": _overall_ap_range(),
        "AP50": _overall_ap(0.50),
        "AP75": _overall_ap(0.75),
        "APsmall": _overall_ap(0.50, "small"),
        "APmedium": _overall_ap(0.50, "medium"),
        "APlarge": _overall_ap(0.50, "large"),
        "AR1": _overall_ar(1),
        "AR10": _overall_ar(10),
        "AR100": _overall_ar(100),
    }

    per_category: t.Dict[str, t.Dict[str, float]] = {}
    for cat_id, cat_name in categories.items():
        cat_aps = [compute_ap(gt_ds, pred_ds, t, cat_id) for t in COCO_IOU_THRESHOLDS]
        cat_aps_valid = [v for v in cat_aps if not np.isnan(v)]
        per_category[cat_name] = {
            "AP": float(np.mean(cat_aps_valid)) if cat_aps_valid else float("nan"),
            "AP50": compute_ap(gt_ds, pred_ds, 0.50, cat_id),
            "AP75": compute_ap(gt_ds, pred_ds, 0.75, cat_id),
            "APsmall": compute_ap(gt_ds, pred_ds, 0.50, cat_id, AREA_RANGES["small"]),
            "APmedium": compute_ap(gt_ds, pred_ds, 0.50, cat_id, AREA_RANGES["medium"]),
            "APlarge": compute_ap(gt_ds, pred_ds, 0.50, cat_id, AREA_RANGES["large"]),
            "AR100": compute_ar(gt_ds, pred_ds, COCO_IOU_THRESHOLDS, 100, cat_id),
        }

    return {"overall": overall, "per_category": per_category}
