"""Clustering analysis for COCO datasets.

Provides geometry-based clustering (no images needed) and image-crop–based
clustering (requires images on disk).

All functions return a :class:`ClusterResult` dataclass.
"""

import pathlib
import typing as t
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from annotstein.coco.schemas import Annotation, Dataset

FilePath = t.Union[str, pathlib.Path]

GEOMETRY_FEATURES = ("width", "height", "aspect_ratio", "area", "cx", "cy")


@dataclass
class ClusterResult:
    """Result of a clustering operation.

    Attributes:
        labels: Cluster label for each annotation, in the same order as
            ``annotation_ids``.
        annotation_ids: Annotation IDs corresponding to each label entry.
        centroids: Cluster centroid feature vectors keyed by cluster label.
            For DBSCAN the centroids are computed as the mean of each cluster.
            Noise points (label -1) are excluded from centroids.
        n_clusters: Number of clusters found (excludes DBSCAN noise label -1).
        inertia: KMeans inertia (within-cluster sum of squares).  ``None`` for
            DBSCAN.
        feature_names: Names of the features used for clustering.
    """

    labels: t.List[int]
    annotation_ids: t.List[int]
    centroids: t.Dict[int, t.List[float]]
    n_clusters: int
    inertia: t.Optional[float]
    feature_names: t.List[str]


def _bbox_feature_matrix(
    annotations: t.List[Annotation],
    features: t.Sequence[str],
) -> np.ndarray:
    rows = []
    for ann in annotations:
        x, y, w, h = ann.bbox
        aspect_ratio = w / h if h > 0 else 0.0
        feat_map = {
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "area": w * h,
            "cx": x + w / 2,
            "cy": y + h / 2,
        }
        rows.append([feat_map[f] for f in features])
    return np.array(rows, dtype=float)


def _build_cluster_result(
    annotations: t.List[Annotation],
    labels: np.ndarray,
    X_original: np.ndarray,
    feature_names: t.List[str],
    inertia: t.Optional[float],
) -> ClusterResult:
    unique_labels = sorted(set(int(lbl) for lbl in labels) - {-1})
    centroids: t.Dict[int, t.List[float]] = {}
    for lbl in unique_labels:
        mask = labels == lbl
        centroids[lbl] = X_original[mask].mean(axis=0).tolist()

    return ClusterResult(
        labels=labels.tolist(),
        annotation_ids=[ann.id for ann in annotations],
        centroids=centroids,
        n_clusters=len(unique_labels),
        inertia=inertia,
        feature_names=feature_names,
    )


def cluster_bboxes(
    ds: Dataset,
    n_clusters: int = 8,
    method: str = "kmeans",
    features: t.Sequence[str] = ("width", "height", "aspect_ratio", "area"),
    random_state: int = 42,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> ClusterResult:
    """Cluster annotations by their bbox geometry features.

    No image files are required.

    Args:
        ds: COCO dataset.
        n_clusters: Number of clusters (KMeans only; ignored for DBSCAN).
        method: ``"kmeans"`` or ``"dbscan"``.
        features: Subset of ``("width", "height", "aspect_ratio", "area",
            "cx", "cy")``.
        random_state: Random seed for KMeans.
        dbscan_eps: DBSCAN neighbourhood radius (applied after StandardScaler).
        dbscan_min_samples: DBSCAN minimum neighbourhood size.

    Returns:
        :class:`ClusterResult` with labels aligned to ``ds.annotations``.
    """
    unknown = set(features) - set(GEOMETRY_FEATURES)
    if unknown:
        raise ValueError(f"Unknown features: {unknown}. Valid choices: {GEOMETRY_FEATURES}")

    annotations = ds.annotations
    if not annotations:
        return ClusterResult(labels=[], annotation_ids=[], centroids={}, n_clusters=0, inertia=None, feature_names=list(features))

    X = _bbox_feature_matrix(annotations, list(features))
    X_scaled = StandardScaler().fit_transform(X)

    if method == "kmeans":
        n = min(n_clusters, len(annotations))
        model = KMeans(n_clusters=n, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X_scaled)
        inertia: t.Optional[float] = float(model.inertia_)
    elif method == "dbscan":
        model_db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model_db.fit_predict(X_scaled)
        inertia = None
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'kmeans' or 'dbscan'.")

    return _build_cluster_result(annotations, labels, X, list(features), inertia)


def cluster_crops(
    ds: Dataset,
    images_root: FilePath,
    n_clusters: int = 8,
    method: str = "kmeans",
    hist_bins: int = 16,
    random_state: int = 42,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> ClusterResult:
    """Cluster annotations by HSV colour histogram features of their image crops.

    Requires image files to be present on disk.

    Args:
        ds: COCO dataset.
        images_root: Root directory to resolve relative ``file_name`` paths.
        n_clusters: Number of clusters (KMeans only).
        method: ``"kmeans"`` or ``"dbscan"``.
        hist_bins: Number of bins per HSV channel (total feature dim = 3 × bins).
        random_state: Random seed for KMeans.
        dbscan_eps: DBSCAN neighbourhood radius (after StandardScaler).
        dbscan_min_samples: DBSCAN minimum neighbourhood size.

    Returns:
        :class:`ClusterResult`.  Annotations for which the crop could not be
        loaded receive label ``-1``.
    """
    import cv2
    from PIL import Image as PImage

    images_root = pathlib.Path(images_root)
    image_index = {img.id: img for img in ds.images}

    features_list: t.List[np.ndarray] = []
    valid_annotations: t.List[Annotation] = []
    failed_ids: t.List[int] = []

    for ann in ds.annotations:
        img_meta = image_index.get(ann.image_id)
        if img_meta is None:
            failed_ids.append(ann.id)
            continue

        img_path = images_root / img_meta.file_name
        try:
            pil_img = PImage.open(img_path).convert("RGB")
        except Exception:
            failed_ids.append(ann.id)
            continue

        x, y, w, h = [int(v) for v in ann.bbox]
        crop = pil_img.crop((x, y, x + w, y + h))
        if crop.width == 0 or crop.height == 0:
            failed_ids.append(ann.id)
            continue

        crop_np = np.array(crop, dtype=np.uint8)
        crop_hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)

        hist = np.concatenate(
            [np.histogram(crop_hsv[:, :, ch].ravel(), bins=hist_bins, range=(0, 256))[0].astype(float) for ch in range(3)]
        )
        hist /= hist.sum() + 1e-8
        features_list.append(hist)
        valid_annotations.append(ann)

    feature_names = [f"hsv_ch{ch}_bin{b}" for ch in range(3) for b in range(hist_bins)]

    if not valid_annotations:
        all_annotations = ds.annotations
        return ClusterResult(
            labels=[-1] * len(all_annotations),
            annotation_ids=[ann.id for ann in all_annotations],
            centroids={},
            n_clusters=0,
            inertia=None,
            feature_names=feature_names,
        )

    X = np.stack(features_list)
    X_scaled = StandardScaler().fit_transform(X)

    if method == "kmeans":
        n = min(n_clusters, len(valid_annotations))
        model = KMeans(n_clusters=n, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X_scaled)
        inertia: t.Optional[float] = float(model.inertia_)
    elif method == "dbscan":
        model_db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model_db.fit_predict(X_scaled)
        inertia = None
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'kmeans' or 'dbscan'.")

    result = _build_cluster_result(valid_annotations, labels, X, feature_names, inertia)

    if failed_ids:
        all_ids = {ann.id for ann in valid_annotations}
        extended_labels = []
        extended_ann_ids = []
        valid_iter = iter(zip(result.annotation_ids, result.labels))
        for ann in ds.annotations:
            if ann.id in all_ids:
                aid, lbl = next(valid_iter)
                extended_ann_ids.append(aid)
                extended_labels.append(lbl)
            else:
                extended_ann_ids.append(ann.id)
                extended_labels.append(-1)
        result.labels = extended_labels
        result.annotation_ids = extended_ann_ids

    return result
