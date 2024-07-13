import pathlib
import typing as t

import gcsfs
from pydantic import AnyUrl
from annotstein.coco.schemas import Dataset

from annotstein.io.common import download_files, group_urls_by_protocol


def download_gcs_files(
    urls: t.List[AnyUrl],
    output_paths: t.List[pathlib.Path],
    max_workers: int,
    project: str = "",
    token: t.Optional[str] = None,
) -> t.List[t.Tuple[int, pathlib.Path]]:
    """Download files using HTTP.

    Args:
        urls (t.List[AnyUrl]): urls to download from
        output_paths (t.List[pathlib.Path]): local paths to download to
        project (t.Optional[str]): project_id to work under
        token (t.Optional[str]): auth method for GCP, see https://gcsfs.readthedocs.io/en/latest/api.html#gcsfs.core.GCSFileSystem

    Returns:
        A list of (i, local_path), where i is the position in ds.images and local_path the downloaded file location.
    """
    fs = gcsfs.GCSFileSystem(project=project, access="read_only", token=token)
    return download_files(fs, urls, output_paths, max_workers)


def download_gcs_dataset(
    ds: Dataset,
    output_dir: pathlib.Path,
    project: str = "",
    token: t.Optional[str] = None,
    replace: bool = True,
) -> t.List[t.Tuple[int, pathlib.Path]]:
    """Download images referenced in the input :class:`annotstein.coco.schemas.Dataset`.

    Args:
        ds (Dataset): a COCODataset to download. images[].file_name will be used as urls
        output_dir (pathlib.Path): directory in which files will be stored
        project (t.Optional[str]): project_id to work under
        token (t.Optional[str]): auth method for GCP, see https://gcsfs.readthedocs.io/en/latest/api.html#gcsfs.core.GCSFileSystem
        replace (bool): whether to replace retrieved urls for local paths in the input dataset

    Returns:
        A list of (i, local_path), where i is the position in ds.images and local_path the downloaded file location.
    """
    urls = group_urls_by_protocol([i.file_name for i in ds.images])
    output_paths = [output_dir / (url.path or "").split("/")[-1] for url in urls["gcs"]]
    retrieved_files = download_gcs_files(urls["gcs"], output_paths, 8, project=project, token=token)

    if replace:
        for i, local_path in retrieved_files:
            ds.images[i].file_name = str(local_path.resolve())

    return retrieved_files
