from gzip import GzipFile
from io import TextIOWrapper
from pathlib import Path
from typing import TextIO, BinaryIO

import google
from cloudpathlib import GSPath, AnyPath, CloudPath


def gcs_blob(path: GSPath | CloudPath | str) -> google.cloud.storage.Blob:
    path = GSPath(path)
    return google.cloud.storage.Client().get_bucket(path.bucket).blob(path.blob)


def any_file_open(file: str, mode='r') -> TextIO | BinaryIO:
    file = AnyPath(file)
    if isinstance(file, GSPath):
        return gcs_blob(file).open(mode)
    elif isinstance(file, Path):
        return file.open(mode)
    else:
        assert False


def any_file_read_bytes(file: str) -> bytes:
    path = AnyPath(file)
    if isinstance(path, Path):
        return path.read_bytes()
    elif isinstance(path, GSPath):
        return gcs_blob(file).download_as_bytes()


def any_file_open_gz(file: str, mode='r'):
    if file.endswith('.gz'):
        buffer = GzipFile(fileobj=any_file_open(file, mode='rb'), mode=mode)
        if mode == 'r':
            # noinspection PyTypeChecker
            return TextIOWrapper(buffer)
        else:
            return buffer
    else:
        return any_file_open(file, mode=mode)
