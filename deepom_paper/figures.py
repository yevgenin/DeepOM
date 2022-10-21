import itertools
import shutil
from copy import copy
from inspect import currentframe
from traceback import print_exc
from typing import NamedTuple, Union

import matplotlib
import monai
import more_itertools
import numpy
import torch
import wandb
from matplotlib import pyplot
from monai.data import list_data_collate, IterableDataset, DataLoader
from monai.losses import DiceCELoss
from monai.networks.nets import BasicUNet
from monai.transforms import Compose, SelectItemsd, SpatialPad, SpatialCrop, AddChannel, ScaleIntensity, SqueezeDim, \
    DivisiblePad, ToNumpy
from monai.utils import convert_to_tensor, Method
from numpy import ndarray
from numpy.random import default_rng
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.stats import gamma
from scipy.stats._distn_infrastructure import rv_generic
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from torch import Tensor
from torch.optim import Adam
from tqdm.auto import tqdm

def set_formatter(fmt):
    pyplot.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))

def set_only_x_visible():
    ax = pyplot.gca()    
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

def plot_bipartite_match(ref, qry, ypos=(0, 1)):
    pyplot.plot(
        numpy.stack([
            ref,
            qry,
        ]),
        numpy.stack([ypos] * len(ref)).T,
        c="b",
        lw=2,
        ms=1,
        marker=".", 
        ls=":", 
        alpha=1.0
    )


class Images_FigureData:
    file = "../data/images_figure.pickle"

    def make_figure_data(self):
        compare = BionanoCompare()
        compare.data_prep.num_sizes = 1
        compare.data_prep.num_crops_per_size = 1
        compare.data_prep.crop_size_range_bp = 600 * 1000, 600 * 1000
        compare.data_prep.selector.molecule_ids = [34]
        compare.read_cmap()
        compare.make_refs()
        compare.data_prep.make_crops()
        compare.run_spaligner()
            
        self.item = compare.spaligner_items[0]
        self.item.crop_item.parent_bnx_item.bionano_image.segment_width = 201
        self.item.crop_item.parent_bnx_item.bionano_image.read_segment_image()

        compare.localizer_module.rng = default_rng(seed=6)
        compare.localizer_module.genome_data.sparsity = 3500
        
        self.refs = compare.refs
        self.data_item = compare.localizer_module.make_data()
        self.inference_item = compare.localizer_module.inference_item(self.data_item.image)
        joblib.dump(self, self.file)

    @classmethod
    def load_figure_data(cls):
        return joblib.load(cls.file)

if __name__ == '__main__':
    from fire import Fire
    Fire()
