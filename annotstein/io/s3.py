import pathlib
import typing as t

import s3fs
from pydantic import AnyUrl
from annotstein.coco.schemas import Dataset

from annotstein.io.common import download_files, group_urls_by_protocol


def download_s3_files(
    urls: t.List[AnyUrl],
    output_paths: t.List[pathlib.Path],
    max_workers: int,
    key: t.Optional[str] = None,
    secret: t.Optional[str] = None,
    endpoint_url: t.Optional[str] = None,
) -> t.List[t.Tuple[int, pathlib.Path]]:
    """Download files from s3.

    Args:
        urls (t.List[AnyUrl]): urls to download from
        output_paths (t.List[pathlib.Path]): local paths to download to
        key (t.Optional[str]): s3 access key id
        secret (t.Optional[str]): s3 secret access key
        endpoint_url (t.Optional[str]): s3 endpoint url

    Returns:
        A list of (i, local_path), where i is the position in ds.images and local_path the downloaded file location.
    """
    fs = s3fs.S3FileSystem(key=key, secret=secret, endpoint_url=endpoint_url)
    return download_files(fs, urls, output_paths, max_workers)


def download_prefix(
    bucket: str,
    prefix: str,
    output_path: pathlib.Path,
    key: t.Optional[str] = None,
    secret: t.Optional[str] = None,
    endpoint_url: t.Optional[str] = None,
):
    fs = s3fs.S3FileSystem(key=key, secret=secret, url=endpoint_url)
    fs.get(f"s3://{bucket.rstrip('/')}/{prefix.rstrip('/')}/", output_path, recursive=True)


def download_s3_dataset(
    ds: Dataset,
    output_dir: pathlib.Path,
    key: t.Optional[str] = None,
    secret: t.Optional[str] = None,
    endpoint_url: t.Optional[str] = None,
    replace: bool = True,
) -> t.List[t.Tuple[int, pathlib.Path]]:
    """Download images referenced in the input :class:`annotstein.coco.schemas.Dataset`.

    Args:
        ds (Dataset): a COCODataset to download. images[].file_name will be used as urls
        output_dir (pathlib.Path): directory in which files will be stored
        key (t.Optional[str]): s3 access key id
        secret (t.Optional[str]): s3 secret access key
        endpoint_url (t.Optional[str]): s3 endpoint url
        replace (bool): whether to replace retrieved urls for local paths in the input dataset

    Returns:
        A list of (i, local_path), where i is the position in ds.images and local_path the downloaded file location.
    """
    urls = group_urls_by_protocol([i.file_name for i in ds.images])
    output_paths = [output_dir / (url.path or "").split("/")[-1] for url in urls["s3"]]
    retrieved_files = download_s3_files(urls["s3"], output_paths, 8, key=key, secret=secret, endpoint_url=endpoint_url)

    if replace:
        for i, local_path in retrieved_files:
            ds.images[i].file_name = str(local_path.resolve())

    return retrieved_files
