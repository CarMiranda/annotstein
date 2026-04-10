"""Exploratory data analysis and clustering for COCO datasets.

Submodules:
    eda: Descriptive statistics and distribution analysis.
    clustering: Geometry- and image-based annotation clustering.
"""

from __future__ import annotations
import importlib
import typing as t

__all__ = ["eda", "clustering"]


def __getattr__(name: str) -> t.Any:
    if name in __all__:
        return importlib.import_module(f"annotstein.analysis.{name}")
    raise AttributeError(f"module 'annotstein.analysis' has no attribute {name!r}")
