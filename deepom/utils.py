import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
import dataclasses
import math
import os
import pickle
from functools import partial
from inspect import currentframe
from pathlib import Path
from time import strftime
from typing import Callable, TypeVar, Type, Union, NamedTuple, Optional, Generic, Sequence

import click
import coolname
import imageio
import joblib
import matplotlib
import numba
import numpy
import pandas
import scipy
import skimage
import torch
import yaml
from git import InvalidGitRepositoryError
from matplotlib import pyplot
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter
from monai.data import is_supported_format, NumpyReader
from monai.transforms import LoadImaged
from monai.utils import ensure_tuple
from more_itertools import pairwise
from numba import jit, njit
from numpy import ndarray, array2string
from numpy.random import default_rng
from pandas import Series, DataFrame
from scipy.stats import gaussian_kde
from skimage import draw, transform
from skimage.transform import warp, EuclideanTransform
from sklearn.utils import murmurhash3_32
from torch import nn
from torch.utils.data import Dataset, IterableDataset
from tqdm.auto import tqdm
from nd2reader import ND2Reader

from deepom.config import Config

sqrt2pi = math.sqrt(2 * math.pi)
T = TypeVar('T')
tqdm = tqdm
pyplot = pyplot


def get_hash(data) -> int:
    return murmurhash3_32(bytes(data))


def get_hash_str(data):
    return '%032x' % get_hash(data)


def path_mkdir(*args) -> Path:
    path = Path(*args)
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_func_name():
    return currentframe().f_back.f_code.co_name


def get_chromo_name(s):
    match, = re.compile("(chromosome \w+)").findall(s)
    return match


def segment_iou(seg1, seg2):
    (x, y), (a, b) = sorted(seg1), sorted(seg2)
    union = max(b, y) - min(a, x)
    intersection = max(0, min(b, y) - max(a, x))
    iou = intersection / union
    return iou


def set_ax_bp_format(ax=None, which='x', places=3, base=None, unit='b'):
    if ax is None:
        ax = pyplot.gca()
    sub_ax = dict(x=ax.xaxis, y=ax.yaxis)[which]

    if base is not None:
        locator = matplotlib.ticker.MultipleLocator(base=base)
        sub_ax.set_major_locator(locator)

    formatter = matplotlib.ticker.EngFormatter(unit=unit, places=places, sep="")
    sub_ax.set_major_formatter(formatter)


class Paths:
    @property
    def home(self):
        return Config.HOME

    @property
    def data(self):
        return self.home / Config.DATA_DIR

    @property
    def out(self):
        return self.home / Config.OUT_DIR

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


def glob_recent_dir(path: Path, pattern: str):
    if not pattern.endswith('*'):
        pattern = pattern + '*'

    if not pattern.startswith('*'):
        pattern = '*' + pattern
    files = (_ for _ in path.glob(pattern) if not _.is_file())
    return max(files, key=os.path.getctime)


def _flatten_dict(obj: dict, name=''):
    for key, val in obj.items():
        if isinstance(val, dict):
            yield from _flatten_dict(val, name=name + '.' + key)
        else:
            yield name + '.' + key, val


def flatten_dict(obj: dict):
    return dict(_flatten_dict(obj))


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


#   todo: leave tuples, but encode to enumerate dict
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


def obj_to_yaml(obj):
    data = asdict_recursive(obj, include_modules=[obj.__module__])
    data = nested_dict_filter_types(data)
    return yaml.dump(data)


def dict_to_tensorboard_text(d, filter_dtypes=True):
    if dataclasses.is_dataclass(d):
        d = dataclasses.asdict(d)
    if filter_dtypes:
        d = nested_dict_filter_types(d)
    d = flatten_dict(d)
    text = pandas.DataFrame([d]).T.to_markdown()
    return text


def _extra_repr(module, extra_repr, enable_extra=True):
    base = f'params={num_module_params(module)}'
    if enable_extra:
        return base + f', {extra_repr()}'
    else:
        return base


def _set_extra_repr(module):
    module.extra_repr = partial(_extra_repr, module=module, extra_repr=module.extra_repr)


def add_num_params_to_repr(module):
    return module.apply(_set_extra_repr)


def num_module_params(module):
    params = list(module.parameters())
    numel = sum(param.numel() for param in params)
    return numel


def num_params_dict(module: nn.Module, max_level: int = 1):
    def _items():
        for name, sub_module in module.named_children():
            numel = num_module_params(sub_module)
            info = f'{sub_module.__class__.__name__}(name={name}, numel={numel})'
            if max_level < 1:
                yield info, {}
            else:
                yield info, num_params_dict(sub_module, max_level=max_level - 1)

    return dict(_items())


def tensor_to_numpy(obj) -> ndarray:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    else:
        return obj


def figure_to_tensorboard_image(figure: Figure = None, close=True):
    if figure is None:
        figure = pyplot.gcf()
    canvas = figure.canvas
    canvas.draw()
    data = numpy.frombuffer(canvas.buffer_rgba(), dtype=numpy.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    image_chw = numpy.moveaxis(image_hwc, source=2, destination=0)
    if close:
        pyplot.close(figure)
    return image_chw


#
# #   todo: from obj not cls. and replace
#
# def kw_from_cli(cls: Type[T]):
#     fields = dataclasses.fields(cls)
#     return {
#         field.name: (
#             click.prompt(field.name, default=field.default, type=field.type, show_default=True)
#         ) if not dataclasses.is_dataclass(field.type) else (init_from_cli(field.type, default=field.default))
#         for field in fields
#         if hasattr(cls, '__cli_args__') and field.name in cls.__cli_args__
#     }
#
#   todo: inplace version
def replace_from_cli(obj: T, cli_args=None) -> T:
    if not hasattr(obj.__class__, '__cli_args__'):
        return obj
    if cli_args is None:
        cli_args = obj.__cli_args__

    return replace(
        obj,
        **{
            name:
                replace_from_cli(attr) if dataclasses.is_dataclass(attr)
                else click.prompt(name, default=attr, type=type(attr), show_default=True)
            for name, attr in vars(obj).items()
            if name in cli_args
        }
    )


#
# def init_from_cli(cls: Type[T], default=None) -> T:
#     kw = kw_from_cli(cls)
#
#     if default is None:
#         return cls(**kw)
#     else:
#         return replace(default, **kw)
#

def num_params_text(module: nn.Module, max_level: int = 1):
    return yaml.dump(num_params_dict(module, max_level=max_level))


def log2_int_assert(x):
    y = int(numpy.log2(x))
    assert 2 ** y == x
    return y


def cat_channels(tensors):
    return torch.cat(tensors=tensors, dim=1)


#   todo: cutoff optimization
def gaussian_function_sqr(x_sqr, sigma):
    arg = x_sqr / (2 * sigma ** 2)
    norm = sqrt2pi * sigma
    return numpy.exp(-arg) / norm


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


@dataclass(frozen=True)
class ImageReader:
    file: str = None
    images: ndarray = None
    slice: slice = None
    reader_metadata: dict = None
    extra_metadata: dict = None
    frame0_metadata: dict = None
    x0: float = None
    y0: float = None
    times: ndarray = None
    date: str = None
    channel: str = None
    camera_exposure_time: float = None

    def read_image(self, file, slice_=None):
        file = str(file)
        if slice_ is None:
            slice_ = self.slice

        if slice_ is None:
            slice_ = slice(None)

        if Path(file).suffix == '.nd2':
            #   todo add cache class here
            items, reader = cached_func(read_nd2)(file, slice_)
            reader: ND2Reader
            raw_metadata = reader.parser._raw_metadata
            x0, y0 = raw_metadata.x_data[0], raw_metadata.y_data[0]
            reader_metadata = raw_metadata.get_parsed_metadata()

            frame0_metadata = dict()
            for name in dir(raw_metadata):
                if not name.startswith('_') and hasattr(raw_metadata, name):
                    value = getattr(raw_metadata, name)
                    if hasattr(value, '__len__') and len(value) == len(reader):
                        frame0_metadata[name] = value[0]

        else:
            items, reader = cached_func(read_imageio)(file, slice_)
            reader_metadata = {}
            frame0_metadata = {}
            x0, y0 = None, None

        ms_per_sec = 1e3
        return dataclasses.replace(
            self,
            file=file,
            images=numpy.asarray(items),
            reader_metadata=reader_metadata,
            frame0_metadata=frame0_metadata,
            x0=x0,
            y0=y0,
            times=reader.timesteps / ms_per_sec,
            slice=slice_,
            date=str(reader_metadata['date']),
            channel=str(reader_metadata['channels'][0]),
            camera_exposure_time=float(frame0_metadata['camera_exposure_time']),
        )


def read_nd2(file: Union[Path, str], slice_=None):
    reader = ND2Reader(str(file))
    if slice_ is not None:
        items = numpy.asarray(reader[slice_])
    else:
        items = numpy.asarray(reader)
    return items, reader


def read_nd2_image(file: Union[Path, str]) -> ndarray:
    reader = ND2Reader(str(file))
    image = numpy.asarray(reader)
    return image


def cached_func(func, enable=True, **kw):
    return get_memory(**kw).cache(func) if enable else func


def read_imageio(file, slice_):
    reader = imageio.get_reader(file)
    items = list(reader)[slice_]
    return items, reader


def get_memory(verbose=True, **kw):
    return joblib.Memory(paths_config.out / 'joblib', verbose=verbose, **kw)


def matplotlib_use_qt():
    matplotlib.use('Qt5Agg')


def autoset_attr_name_field(cls, name_field='name'):
    items = vars(cls).items()
    for var_name, obj in items:
        if isinstance(obj, list):
            for index, item in enumerate(obj):
                setattr(item, name_field, f'{cls.__name__}.{var_name}.{index}')
        elif not isinstance(obj, type) and hasattr(obj, name_field):
            setattr(obj, name_field, f'{cls.__name__}.{var_name}')
    return cls


class MapDataset(Dataset):
    def __init__(self, map_func: Callable, args):
        super().__init__()
        self.map_func = map_func
        self.args = args

    def __len__(self):
        return len(self.args)

    def __getitem__(self, item):
        return self.map_func(self.args[item])


class FuncDataset(IterableDataset):
    def __init__(self, iter_func: Callable):
        super().__init__()
        self.iter_func = iter_func

    def __iter__(self):
        yield from self.iter_func()


def grid_indices(shape):
    n, m = shape
    i, j = numpy.mgrid[:n, :m]
    ij = numpy.stack([i.ravel(), j.ravel()]).T
    return ij


@dataclass(frozen=True)
class RangeRandom(Generic[T]):
    range: tuple[T, T] = None
    size: Union[int, tuple] = None
    value: T = None

    @property
    def max(self):
        if self.range is None:
            return self.value
        else:
            return self.range[1]

    @property
    def min(self):
        if self.range is None:
            return self.value
        else:
            return self.range[0]

    def sample(self, rng: numpy.random.Generator):
        raise NotImplementedError


@dataclass(frozen=True)
class IntRangeRandom(RangeRandom[int]):
    def sample(self, rng=default_rng()):
        if self.value is not None:
            return self.value
        if self.range[0] == self.range[1]:
            value = self.range[0]
        else:
            value = rng.integers(*self.range, size=self.size)
        return value


@dataclass(frozen=True)
class FloatRangeRandom(RangeRandom[float]):
    def sample(self, rng=default_rng()):
        if self.value is not None:
            return self.value

        return rng.uniform(*self.range, size=self.size)


def raster_line_coords(coords: ndarray):
    coords = numpy.stack(coords).astype(int)
    line_points = [
        p
        for a, b in pairwise(coords)
        for p in zip(*draw.line(*a, *b))
    ]
    return numpy.unique(line_points, axis=0)


def bernoulli_mask(rng, size, p):
    return rng.binomial(n=1, size=size, p=p).astype(bool)


class ImageShape(NamedTuple):
    time: Optional[int]
    lat: Optional[int]
    lon: Optional[int]


def make_random_values(self: T, rng: numpy.random.Generator = None) -> T:
    if rng is None:
        rng = self.rng
    return replace(
        self,
        rng=rng,
        **{
            key: val.sample(rng)
            for key, val in vars(self).items()
            if isinstance(val, RangeRandom)
        }
    )


def data_to_md(data, name=''):
    if not isinstance(data, DataFrame):
        data = Series(data).to_frame(name=name)
    return data.to_markdown(tablefmt='grid')


def min_max(x):
    return numpy.min(x), numpy.max(x)


def quantiles_dict(x, q=(0, 1)):
    return dict(zip(q, numpy.quantile(x, q)))


def list_format(x, fmt):
    return [fmt.format(_) for _ in x]


def dict_format(x, fmt, key_fmt='{}'):
    return {key_fmt.format(k): fmt.format(v) for k, v in x.items()}


def value_range_str(x, q=(0, 1), precision=4):
    return array2string(numpy.quantile(x, q=q), precision=precision, floatmode='maxprec')


def min_max_dict(x):
    return dict(zip(('min', 'max'), min_max(x)))


# @numba.jit(nopython=True)
def padded_slice(x: ndarray, start: int, stop: int):
    n = len(x)
    pad_width = max(0, -start), max(0, stop - n)
    x = numpy.pad(x, pad_width=pad_width)
    return x[start + pad_width[0]: stop + pad_width[0]]


def standardize(x: ndarray):
    return (x - numpy.mean(x, axis=0)) / numpy.std(x, axis=0)


def reverse_if(x, reverse):
    return x[::-1] if reverse else x


def gaussian_kde_grid(centers, shape, bw_method):
    grid = numpy.stack(numpy.meshgrid(
        numpy.arange(shape[-2]),
        numpy.arange(shape[-1]),
        indexing='ij'
    ))
    shape_in = grid.shape[0], -1
    shape_out = grid.shape[1:]
    density_func = gaussian_kde(centers, bw_method=bw_method)
    return density_func(grid.reshape(shape_in)).reshape(shape_out)


def git_repo():
    try:
        import git
        return git.Repo(path=__file__, search_parent_directories=True)
    except (InvalidGitRepositoryError, ImportError):
        return None


def git_commit(confirm=True):
    repo = git_repo()
    if repo is None:
        return

    if repo.is_dirty():
        if not confirm or click.confirm('commit?', default=True):
            repo.git.commit('-am', '.')

    return str(repo.head.object)


def downscale(x, factor):
    return skimage.transform.rescale(x, 1 / factor,
                                     # anti_aliasing=True,
                                     # anti_aliasing_sigma=factor,
                                     # preserve_range=True,
                                     mode='constant')


@contextmanager
def pyplot_show_window():
    pyplot_switch_backend_qt()
    yield
    pyplot_show_block()


def pyplot_show_block():
    pyplot.show(block=True)


def pyplot_switch_backend_qt():
    pyplot.switch_backend('Qt5Agg')


class ND2ImageReader(NumpyReader):
    def verify_suffix(self, filename):
        suffixes: Sequence[str] = ["nd2"]
        return is_supported_format(filename, suffixes)

    def read(self, data, **kwargs):
        filenames = ensure_tuple(data)
        func = cached_func(read_nd2_image, enable=False)
        return [func(path) for path in filenames]


class ImageIOReader(NumpyReader):
    def verify_suffix(self, filename):
        suffixes: Sequence[str] = ["tiff", "tif"]
        return is_supported_format(filename, suffixes)

    def read(self, data, **kwargs):
        filenames = ensure_tuple(data)
        func = cached_func(imageio.imread, enable=False)
        return [numpy.asarray(func(str(path))) for path in filenames]


@contextmanager
def inference_eval(module: nn.Module):
    module.eval()
    with torch.inference_mode():
        yield
    module.train()


class LoadImageMic(LoadImaged):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(ND2ImageReader())
        self.register(ImageIOReader())


@njit
def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def filter_valid_coords(coords, shape):
    valid = (
            (coords.T[0] >= 0) &
            (coords.T[0] < shape[0]) &
            (coords.T[1] >= 0) &
            (coords.T[1] < shape[1])
    )
    coords = coords[valid, :]
    return coords


def numpy_sigmoid(x):
    return torch.sigmoid(torch.from_numpy(x)).numpy()


def rasterize(coords: ndarray, shape: tuple):
    assert coords.shape[-1] == len(shape)
    coords = coords.astype(int)
    target = numpy.zeros(shape, dtype='float')
    coords = filter_valid_coords(coords, target.shape)
    target[tuple(coords.T)] = 1
    return target


def set_names(cls):
    for key, val in vars(cls).items():
        if val is None:
            setattr(cls, key, key.lower())
        elif isinstance(val, type):
            set_names(val)
    return cls


def collate_simple_list(batch: list[dict]) -> dict[str, list]:
    batch = {
        key: [item[key] for item in batch]
        for key in batch[0].keys()
    }
    return batch


def ndargmax(arr):
    return numpy.unravel_index(numpy.argmax(arr), arr.shape)


def enumerate_key(items, key: str):
    yield from (
        {key: i} | item
        for i, item in enumerate(items)
    )


def uniform_range(low, high):
    assert high > low
    return scipy.stats.uniform(low, high - low)


def uniform_central_range(center, margin):
    return uniform_range(center - margin, center + margin)


def title_from_dict(data, keys=None):
    ser = Series(data)
    if keys is not None:
        ser = ser[keys]
    pyplot.title(ser.to_string(), horizontalalignment='left', loc='left')


def plot_dist(dist, size=10000):
    _, b, _ = pyplot.hist(dist.rvs(size=size), bins='auto', density=True)
    pyplot.plot(b, dist.pdf(b))


class Consts:
    PI = numpy.pi
    _2PI = PI * 2
    PI_HALF = PI / 2
    SQRT_2PI = numpy.sqrt(_2PI)
    PI_QUARTER = PI / 4


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
