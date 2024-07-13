import pathlib
import typing as t

import fsspec.implementations.http
from pydantic import AnyUrl
from annotstein.coco.schemas import Dataset

from annotstein.io.common import download_files, group_urls_by_protocol


def download_http_files(
    urls: t.List[AnyUrl],
    output_paths: t.List[pathlib.Path],
    max_workers: int,
    headers: t.Dict[str, str],
) -> t.List[t.Tuple[int, pathlib.Path]]:
    """Download files using HTTP.

    Args:
        urls (t.List[AnyUrl]): urls to download from
        output_paths (t.List[pathlib.Path]): local paths to download to
        headers (t.Dict[str, str]): Passed to :class:`aiohttp.ClientSession`, see https://docs.aiohttp.org/en/stable/client_reference.html

    Returns:
        A list of (i, local_path), where i is the position in ds.images and local_path the downloaded file location.
    """
    fs = fsspec.implementations.http.HTTPFileSystem()
    fs.client_kwargs["headers"] = headers
    return download_files(fs, urls, output_paths, max_workers)


def download_http_dataset(
    ds: Dataset,
    output_dir: pathlib.Path,
    headers: t.Dict[str, str],
    replace: bool = True,
) -> t.List[t.Tuple[int, pathlib.Path]]:
    """Download images referenced in the input :class:`annotstein.coco.schemas.Dataset`.

    Args:
        ds (Dataset): a COCODataset to download. images[].file_name will be used as urls
        output_dir (pathlib.Path): directory in which files will be stored
        headers (t.Dict[str, str]): Passed to :class:`aiohttp.ClientSession`, see https://docs.aiohttp.org/en/stable/client_reference.html
        replace (bool): whether to replace retrieved urls for local paths in the input dataset

    Returns:
        A list of (i, local_path), where i is the position in ds.images and local_path the downloaded file location.
    """
    urls = group_urls_by_protocol([i.file_name for i in ds.images])
    output_paths = [output_dir / (url.path or "").split("/")[-1] for url in urls["http"]]
    retrieved_files = download_http_files(urls["http"], output_paths, 8, headers=headers)

    if replace:
        for i, local_path in retrieved_files:
            ds.images[i].file_name = str(local_path.resolve())

    return retrieved_files
