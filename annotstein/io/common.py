from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pathlib
import typing as t

import fsspec
import fsspec.generic
from pydantic import AnyUrl, ValidationError

from annotstein.logging import logger


def download_file(
    fs: fsspec.AbstractFileSystem,
    url: AnyUrl,
    output_path: pathlib.Path,
):
    with fs.open(str(url), "rb") as source_file:
        with open(output_path, "wb") as dest_file:
            dest_file.write(source_file.read())


def download_files(
    fs: fsspec.AbstractFileSystem,
    urls: t.List[AnyUrl],
    output_paths: t.List[pathlib.Path],
    max_workers: int,
) -> t.List[t.Tuple[int, pathlib.Path]]:
    # [(index, url, local path)]
    retrieved_urls: t.List[t.Tuple[int, AnyUrl, pathlib.Path]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures_to_args = {pool.submit(download_file, fs, *args): (i, *args) for i, args in enumerate(zip(urls, output_paths))}
        for future in as_completed(futures_to_args):
            i, url, local_path = futures_to_args[future]
            try:
                future.result()
                retrieved_urls.append((i, url, local_path))
            except Exception:
                logger.error("Exception raised while downloading %s.", str(url), exc_info=True)

    return [(i, p) for i, _, p in retrieved_urls]


def group_urls_by_protocol(urls: t.List[str]):
    # Group images by protocol
    grouped_urls: t.Dict[str, t.List[AnyUrl]] = defaultdict(list)
    for url in urls:
        try:
            url = AnyUrl(url)
            grouped_urls[url.scheme].append(url)
        except ValidationError:
            pass

    found_protocols = set(grouped_urls.keys())
    available_protocols = set(p.lower() for p in fsspec.available_protocols())
    missing_protocols = found_protocols - available_protocols
    if len(missing_protocols) > 0:
        logger.warning("The following protocols are not supported: %s", str(missing_protocols))

    return grouped_urls
