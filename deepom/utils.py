from contextlib import contextmanager
import math
import os
import pickle
from pathlib import Path
from time import strftime
from typing import TypeVar, Type

import click
import coolname
import joblib
import numpy
import scipy
import torch
from git import InvalidGitRepositoryError
from matplotlib import pyplot
from numba import njit
from scipy.stats import gaussian_kde
from skimage.transform import warp, EuclideanTransform
from torch import nn
from tqdm.auto import tqdm

from deepom.config import Config, Consts

sqrt2pi = math.sqrt(2 * math.pi)
T = TypeVar('T')
tqdm = tqdm
pyplot = pyplot


def path_mkdir(*args) -> Path:
    path = Path(*args)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Paths:
    @property
    def home(self):
        return Config.HOME_DIR

    @property
    def data(self):
        return Config.DATA_DIR

    @property
    def out(self):
        return Config.OUT_DIR

    def data_path(self, *args):
        path = Path(*args)

        if not path.is_absolute():
            path = self.data / path
        return path

    def out_path(self, *args):
        path = Path(*args)

        if not path.is_absolute():
            path = self.out / path
        return path

    def generate_out_dir_mkdir(self, *args):
        return path_mkdir(self.out, *args, generate_name())

    def generate_out_file_mkdir(self, *args):
        return path_mkdir(self.out, *args) / generate_name()

    def out_file_mkdir(self, *args):
        path = self.out_path(*args)
        return path_mkdir(path.parent) / path.name

    def out_path_mkdir(self, *args) -> Path:
        return path_mkdir(self.out_path(*args))


paths_config = Paths()


def find_file(pattern=None, search_dir=None, suffix=None, recursive=True):
    if pattern is None:
        pattern = '*'

    if search_dir is None:
        search_dir = paths_config.out

    if not pattern.endswith('*'):
        pattern = pattern + '*'

    if not pattern.startswith('*'):
        pattern = '*' + pattern

    if suffix is not None:
        pattern = pattern + suffix

    search_dir = paths_config.out_path(search_dir)
    files = search_dir.rglob(pattern) if recursive else search_dir.glob(pattern)
    paths = [file for file in files if file.is_file()]
    if not paths:
        raise ValueError(f'empty glob: {search_dir=} {pattern=}')

    most_recent_path = max(paths, key=os.path.getctime)
    return most_recent_path


def asdict_recursive(x, include_modules=(), prefix=''):
    if isinstance(x, tuple) and hasattr(x, '_fields'):
        d = type(x)(*[asdict_recursive(v) for v in x])
    elif isinstance(x, (list, tuple)):
        d = type(x)(asdict_recursive(v) for v in x)
    elif isinstance(x, dict):
        d = type(x)((asdict_recursive(k),
                     asdict_recursive(v))
                    for k, v in x.items())
    elif hasattr(x, '__dict__'):
        if type(x).__module__ in include_modules:
            d = {(prefix + name): asdict_recursive(val) for name, val in vars(x).items()}
        else:
            d = vars(x.__class__) | vars(x)
        d = dict(__TYPE__=type(x).__name__) | d
    else:
        d = x
    return d


def nested_dict_filter_types(d, types=(bool, str, float, int, Path), limit_list_len=10):
    def _inner(x):
        if isinstance(x, dict):
            return {str(key): _inner(val) for key, val in x.items()}
        elif isinstance(x, tuple) or isinstance(x, list):
            n0 = len(x)
            x = x[:limit_list_len]
            n1 = len(x)
            return {
                f'{type(x).__name__}({n1} out of {n0})':
                    [_inner(val) for val in x]
            }
        elif isinstance(x, types) or x is None:
            if isinstance(x, Path):
                x = str(x)
            return x
        else:
            return type(x).__name__

    return _inner(d)


def num_module_params(module):
    params = list(module.parameters())
    numel = sum(param.numel() for param in params)
    return numel


def pickle_dump(file, obj):
    with Path(file).open(mode='wb') as file_object:
        pickle.dump(obj=obj, file=file_object)


def pickle_load(file, cls: Type[T] = None) -> T:
    with Path(file).open(mode='rb') as file_object:
        obj = pickle.load(file=file_object)
        if cls is not None:
            assert isinstance(obj, cls)
        return obj


def timestamp_str_iso_8601() -> str:
    return strftime('%Y%m%dT%H%M%SZ')


def datestamp_str() -> str:
    return strftime('%Y%m%d')


def generate_name(*prefix):
    return '-'.join([
        *prefix,
        datestamp_str(),
        coolname.generate_slug(2)
    ])


def cached_func(func, enable=True, **kw):
    return get_memory(**kw).cache(func) if enable else func


def get_memory(verbose=True, **kw):
    return joblib.Memory(paths_config.out / 'joblib', verbose=verbose, **kw)


def min_max(x):
    return numpy.min(x), numpy.max(x)


def git_repo():
    try:
        import git
        return git.Repo(path=__file__, search_parent_directories=True)
    except (InvalidGitRepositoryError, ImportError):
        return None


def git_commit(confirm=True, commit=False):
    repo = git_repo()
    if repo is None:
        return

    if commit and repo.is_dirty():
        if not confirm or click.confirm('do git commit?', default=True):
            repo.git.commit('-am', '.')

    return str(repo.head.object)


@contextmanager
def inference_eval(module: nn.Module):
    module.eval()
    with torch.inference_mode():
        yield
    module.train()


@njit
def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def numpy_sigmoid(x):
    return torch.sigmoid(torch.from_numpy(x)).numpy()


def set_names(cls):
    for key, val in vars(cls).items():
        if val is None:
            setattr(cls, key, key.lower())
        elif isinstance(val, type):
            set_names(val)
    return cls


def ndargmax(arr):
    return numpy.unravel_index(numpy.argmax(arr), arr.shape)


def uniform_range(low, high):
    assert high > low
    return scipy.stats.uniform(low, high - low)


def uniform_central_range(center, margin):
    return uniform_range(center - margin, center + margin)


def extract_segment_from_endpoints(image_tseq, endpoints, segment_width):
    (y0, x0), (y1, x1) = endpoints

    segment_angle = numpy.arctan2(y1 - y0, x1 - x0)
    segment_length = numpy.sqrt((y0 - y1) ** 2 + (x0 - x1) ** 2)
    segment_center = numpy.stack([x0 + x1, y0 + y1]) / 2

    T = EuclideanTransform(translation=-segment_center)
    R = EuclideanTransform(rotation=-segment_angle)
    T2 = EuclideanTransform(translation=[segment_length / 2, segment_width / 2])
    M = EuclideanTransform(matrix=T2.params @ R.params @ T.params)

    segment_image = warp(
        image_tseq.astype(float).transpose(1, 2, 0),
        inverse_map=M.inverse,
        output_shape=(int(segment_width), int(segment_length))
    ).transpose(2, 0, 1)
    return segment_image


def gaussian_density(coords, intensity, shape, sigma, truncate_sigmas=4):
    row, col = coords.T
    kernel_radius = max(1, int(truncate_sigmas * sigma))
    kernel_size = int(kernel_radius) * 2 + 1
    kernel_coords = numpy.indices((kernel_size, kernel_size)).astype(float) - kernel_size // 2

    row = row[:, None, None]
    col = col[:, None, None]
    intensity = intensity[:, None, None]

    i = (row + kernel_coords[0]).astype(int)
    j = (col + kernel_coords[1]).astype(int)

    z = (i - row) ** 2 + (j - col) ** 2
    v = intensity * numpy.exp(-z / 2 / sigma ** 2) / Consts.SQRT_2PI / sigma

    i = i.ravel()
    j = j.ravel()
    v = v.ravel()

    mask = (i >= 0) & (i < shape[0]) & (j < shape[1]) & (j >= 0)
    i = i[mask]
    j = j[mask]
    v = v[mask]

    ind = numpy.ravel_multi_index((i, j), dims=shape)
    res = numpy.bincount(ind, weights=v, minlength=shape[0] * shape[1]).reshape(shape)
    return res


def filter_valid_coords(coords, shape):
    valid = (
            (coords.T[0] >= 0) &
            (coords.T[0] < shape[0]) &
            (coords.T[1] >= 0) &
            (coords.T[1] < shape[1])
    )
    coords = coords[valid, :]
    return coords
