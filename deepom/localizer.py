import itertools
import shutil
from copy import copy
from inspect import currentframe
from traceback import print_exc
from typing import NamedTuple, Union

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
from scipy.stats import gamma
from scipy.stats._distn_infrastructure import rv_generic
from skimage.exposure import rescale_intensity
from torch import Tensor
from torch.optim import Adam
from tqdm.auto import tqdm

from deepom.utils import uniform_range, Config, uniform_central_range, is_sorted, gaussian_density, filter_valid_coords, \
    Paths, generate_name, path_mkdir, asdict_recursive, nested_dict_filter_types, find_file, num_module_params, \
    inference_eval, numpy_sigmoid, set_names


@set_names
class Keys:
    MODULE_INPUT = None
    MODULE_OUTPUT: str = None
    MODULE_TARGET = None
    LOSS_TENSOR = None
    LOSSES = None
    LOSS = None
    TRAIN: str = None
    BATCH_ITEMS = None
    DATA_ITEM = None
    DATA_ITEM_INDEX = None


class DataItem:
    def __init__(self):
        self.labeled_coords = None
        self.align_locs = None
        self.align_sites = None
        self.fragment_coords = None
        self.locvec = None
        self.sitevec = None
        self.image = None
        self.nominal_scale = None
        self.segment_name = None


class RandomSample:
    def __init__(self, distrib: rv_generic = None, size=None):
        self.distrib = distrib
        self.size = size

    def random_sample(self, random_state, size=None):
        if size is None:
            size = self.size

        return self.distrib.rvs(random_state=random_state, size=size)


class SimulatedDataItem(DataItem):
    def __init__(self):
        super().__init__()
        self.rng = default_rng()

        self.segment_name = f'{type(self).__name__}'

        self.time_size = 1
        self.emitter_p = uniform_range(0.5, 1)
        self.time_sigma = 1

        self.nominal_scale = Config.BIONANO_NOMINAL_SCALE
        self.offset = None
        self.lon_size_min = 1
        self.lat_size_min = 3
        self.label_eff = RandomSample(uniform_range(0.9, 1))
        self.angle = 0
        self.scale = RandomSample(uniform_central_range(center=self.nominal_scale, margin=5))
        self.strand_image_offset = RandomSample(uniform_range(5, 10), size=2)
        self.psf_sigma = RandomSample(uniform_range(1.2, 1.5))

        self.stray_density = 1e-3
        self.emitter_intensity_distrib = gamma(2, .5, 1e-1)
        self.bg_distrib = gamma(100, 0, .3e-2)

        self.fragment_len = RandomSample(uniform_range(self.nominal_scale * 100, self.nominal_scale * 150))

        self.stray_coords = None
        self.visible_mask = None
        self.fragment_coords = None
        self.image_fg = None
        self.image_bg = None
        self.emitter_intensity = None
        self.emitter_field = None
        self.intensity_field = None
        self.labeled_coords = None
        self.emitter_coords = None
        self.labeled_mask = None
        self.fragment_sitevec = None
        self.fragment_sitevec_indices = None
        self.lon_size = None
        self.lat_size = None

    def random_sample(self, x, size):
        return x.rvs(random_state=self.rng, size=size)

    def make_params(self):
        for key, val in vars(self).items():
            if isinstance(val, RandomSample):
                setattr(self, key, val.random_sample(self.rng))

    def make_image(self):
        self.make_params()
        self.make_fragment()
        self.make_fragment_image()

    def make_fragment_image(self):
        self.make_emitter_coords()
        self.make_emitter_field()
        self.make_image_readout()

    def make_fragment(self):
        assert self.sitevec is not None
        assert is_sorted(self.sitevec)
        assert len(self.sitevec) >= 2

        if self.offset is None:
            offset_start, offset_stop = self.sitevec.min(), self.sitevec.max() - self.fragment_len

            if offset_start < offset_stop:
                self.offset = self.rng.integers(offset_start, offset_stop)
            else:
                self.offset = offset_start

        start, stop = self.offset, self.offset + self.fragment_len
        start_i, stop_i = self.sitevec.searchsorted([start, stop])
        self.fragment_sitevec_indices = numpy.arange(start_i, stop_i)
        self.fragment_sitevec = self.sitevec[self.fragment_sitevec_indices]

        assert is_sorted(self.fragment_sitevec)


    def make_emitter_coords(self):
        self.make_labeled_coords()
        self.make_stray_coords()
        self.emitter_coords = numpy.concatenate([
            self.labeled_coords,
            self.stray_coords,
        ])


    def make_labeled_coords(self):
        assert len(self.fragment_sitevec)
        start_offset, end_offset = self.strand_image_offset

        #   todo: add off-target labels
        self.labeled_mask = self.rng.binomial(n=1, p=self.label_eff, size=len(self.fragment_sitevec)).astype(bool)

        r = (self.fragment_sitevec - self.fragment_sitevec[0]) / self.scale
        assert is_sorted(r)

        x = r * numpy.cos(self.angle) + start_offset
        y = r * numpy.sin(self.angle)

        self.lon_size = int(max(self.lon_size_min, x.max() + start_offset + end_offset))
        assert self.lon_size > 0

        self.lat_size = int(max(self.lat_size_min, y.max()))
        y = y + self.lat_size / 2
        assert self.lat_size > 0

        self.fragment_coords = numpy.stack([y, x]).T
        shape = (self.lat_size, self.lon_size)
        assert (
                (y >= 0) &
                (y < shape[0]) &
                (x >= 0) &
                (x < shape[1])
        ).all()

        self.visible_mask = self.labeled_mask

        assert len(self.fragment_coords) == len(self.fragment_sitevec) == len(self.visible_mask)

        self.labeled_coords = self.fragment_coords[self.visible_mask]

    def make_emitter_field(self):
        emitter_field = numpy.zeros((int(self.time_size), int(self.lat_size), int(self.lon_size)), dtype=float)

        assert len(self.emitter_coords)

        num_emitters = len(self.emitter_coords)

        emitter_intensity = self.random_sample(
            self.emitter_intensity_distrib,
            size=(self.time_size, num_emitters)
        )

        #   todo: add exponential bleaching
        if self.time_size > 1:
            emitter_p = self.random_sample(self.emitter_p, size=num_emitters)
            time_mask = self.rng.binomial(n=1, p=emitter_p[None, :], size=(self.time_size, num_emitters))
            time_mask = ndimage.gaussian_filter1d(time_mask, axis=0, sigma=self.time_sigma)
            time_mask = (time_mask > .5).astype(int)
            emitter_intensity = emitter_intensity * time_mask

        #   todo: vectorize
        for t in range(self.time_size):
            emitter_field[t] = gaussian_density(self.emitter_coords, emitter_intensity[t],
                                                shape=(self.lat_size, self.lon_size), sigma=self.psf_sigma)
        self.emitter_intensity = emitter_intensity
        self.emitter_field = emitter_field

    def make_image_readout(self):
        self.image_fg = rescale_intensity(self.emitter_field)
        if self.bg_distrib is not None:
            self.image_bg = self.random_sample(self.bg_distrib, size=self.image_fg.shape)
            self.image = rescale_intensity(self.image_fg + self.image_bg)

    def make_stray_coords(self):
        assert self.lat_size
        assert self.lon_size
        assert self.stray_density

        num_stray = int(self.lat_size * self.lon_size * self.stray_density)
        if num_stray > 0:
            _coords = self.rng.uniform(low=(0, 0), high=(self.lat_size, self.lon_size), size=(num_stray, 2))
            self.stray_coords = filter_valid_coords(_coords, (self.lat_size, self.lon_size))
        else:
            self.stray_coords = numpy.empty((0, 2))


class LocalizerTrainDataItem(SimulatedDataItem):
    def __init__(self):
        super().__init__()
        self.targets = None
        self.label_target = None
        self.loc_target = None

    def make_target(self):
        self.make_image()

        assert (self.labeled_coords is not None) and len(self.labeled_coords)

        labeled_pos = self.labeled_coords.T[1]
        stray_pos = self.stray_coords.T[1]
        shape = self.image.shape[-1]

        labeled_index = labeled_pos.astype(int)
        pos_in_pixel = labeled_pos - labeled_index

        assert ((pos_in_pixel >= 0) & (pos_in_pixel <= 1) & (labeled_pos >= 0) & (labeled_pos < shape)).all()
        assert len(pos_in_pixel) == len(labeled_pos)

        self.targets = LocalizerTargets(
            label_target=numpy.full(shape, fill_value=LocalizerEnum.BG, dtype=int),
            loc_target=numpy.full(shape, fill_value=.5, dtype=float),
        )
        self.targets.label_target[stray_pos.astype(int)] = LocalizerEnum.STRAY
        self.targets.label_target[labeled_index] = LocalizerEnum.FG
        self.targets.loc_target[labeled_index] = pos_in_pixel
        self.target = numpy.stack(self.targets)


class TaskMixin:
    def __init__(self):
        self.run_name = None
        self.run_root_dir = None
        self.tqdm_enable = True
        self.paths_config = Paths()
        self.task_name = type(self).__name__
        self.task_root_dir = self.paths_config.out_path(self.task_name)

        self.seed = None
        self.axes_iter = None

        self.wandb_mode = None

        self.test_mode_enable = False
        self.log_wandb_enable = False
        self.log_debug_enable = False
        self.tqdm = None
        self.step_index = 0

    def init_task(self):
        #   order important!
        self.generate_random_task_name()
        self.init_rng()
        self.init_log()

    def task_steps(self):
        try:
            self.init_task()
            for step in self._task_steps():
                self.step_index += 1
                self.tqdm.update()
                yield step
        finally:
            if self.log_wandb_enable:
                wandb.finish()

    def generate_random_task_name(self):
        self.run_name = generate_name(self.task_name)
        if self.test_mode_enable:
            self.task_root_dir = self.task_root_dir / Config.TEST_DIR
        self.run_root_dir = (self.task_root_dir / self.run_name)

    def init_log(self):
        self.tqdm = tqdm(desc=self.run_name, disable=not self.tqdm_enable)

        if self.log_wandb_enable:
            path_mkdir(self.run_root_dir)
            wandb.init(
                name=self.run_name, dir=str(self.run_root_dir),
                mode=self.wandb_mode,
                project=Config.PROJECT_NAME,
            )
            wandb.config.update(self.task_params_dict())

    def init_rng(self):
        monai.utils.set_determinism(seed=self.seed)
        self.rng = default_rng(seed=self.seed)

    def _log_debug(self, *args):
        if not self.log_debug_enable:
            return
        func_name = currentframe().f_back.f_code.co_name
        items = (self.task_name, func_name, *args)
        if self.tqdm is None:
            print(items)
        else:
            self.tqdm.write(' '.join(map(str, items)))

    def log_wandb_figure(self, figure=pyplot, key=Config.WANDB_FIGURE):
        if self.log_wandb_enable:
            wandb.log({key: figure})

    def task_params_dict(self):
        data = asdict_recursive(self, include_modules=[self.__module__], prefix=Config.WANDB_PREFIX)
        data = nested_dict_filter_types(data)
        return data

    def _task_steps(self):
        raise NotImplementedError

    def run(self):
        more_itertools.consume(self.task_steps())

    def log_shapes(self, item: dict):
        for key, val in item.items():
            if hasattr(val, 'shape'):
                self._log_debug(f'{key}.shape={val.shape}')


class TrainerMixin(TaskMixin):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.validation_every = 2000
        self.log_images_every = 1000
        self.log_metrics_every = None
        self.checkpoint_every = 10000
        self.module = None
        self.load_checkpoint = False
        self.checkpoint_search_dir = None
        self.log_metrics_every = 1

        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.lr = 1e-3
        self.batch_size = 1
        self.log_tqdm_postfix = {}

    def init_task(self):
        super().init_task()
        self.init_ensure_module()
        self.init_validation()  # also inits the module
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = Adam(self.module.parameters(), lr=self.lr)

    def _task_steps(self):
        self.module.train()
        self.__train_data_loader = self._train_data_loader()
        for batch in self.__train_data_loader:
            yield self._training_step(batch)

    def _training_step(self, data_item: dict):
        self.optimizer.zero_grad()

        output_item = {Keys.MODULE_OUTPUT: self.module_forward(data_item[Keys.MODULE_INPUT])}
        loss_item = self.loss_func(output_item[Keys.MODULE_OUTPUT], data_item[Keys.MODULE_TARGET])

        loss_item[Keys.LOSS_TENSOR].backward()
        self.optimizer.step()

        step_item = data_item | output_item | loss_item
        self.step_events(step_item)

        return step_item

    def make_metrics(self, step_item):
        loss_metric = step_item[Keys.LOSS_TENSOR].detach().item()
        metrics_item = {Keys.LOSS: loss_metric}
        return metrics_item

    def step_events(self, step_item):
        log_metrics_every = max(1, self.log_metrics_every // self.batch_size)
        log_images_every = max(1, self.log_images_every // self.batch_size)
        validation_every = max(1, self.validation_every // self.batch_size)

        if self.step_index % log_images_every == 0:
            self.log_train_batch(step_item)

        if self.step_index % validation_every == 0:
            self.validation_step()

        if self.step_index % log_metrics_every == 0:
            metrics_item = self.make_metrics(step_item)
            self._log_update_tqdm_postfix(metrics_item)
            self.log_metrics({Keys.TRAIN: metrics_item})

        if self.checkpoint_every is not None:
            checkpoint_every = max(1, self.checkpoint_every // self.batch_size)
            if self.step_index % 100 == 0:
                self.checkpoint_save()

    def _module_build(self):
        raise NotImplementedError

    def _data_collate(self, items: list):
        items_to_batch = Compose([
            SelectItemsd(keys=[Keys.MODULE_INPUT, Keys.MODULE_TARGET], allow_missing_keys=True)
        ])(items)
        batch = list_data_collate(items_to_batch) | {Keys.BATCH_ITEMS: items}
        return batch

    def init_validation(self):
        pass

    def _checkpoint_load(self):
        if self.load_checkpoint:
            try:
                if self.checkpoint_search_dir is None:
                    search_dir = self.task_root_dir
                else:
                    search_dir = self.checkpoint_search_dir

                file = find_file(search_dir=search_dir, suffix=Config.CHECKPOINT_FILE)
            except (ValueError, RuntimeError):
                print_exc()
                raise
            else:
                print('loading checkpoint: ', file, '\n\n')
                self.module.load_state_dict(torch.load(file, map_location=self.device))

    def log_metrics(self, metrics_dict):
        if self.log_wandb_enable:
            wandb.log(metrics_dict, step=self.step_index)

    def _log_update_tqdm_postfix(self, metrics_dict):
        self.log_tqdm_postfix |= metrics_dict
        self.tqdm.set_postfix(self.log_tqdm_postfix)

    def _on_log_init_end(self):
        if self.log_wandb_enable:
            wandb.watch(models=self.module, log="all", log_graph=True)

    def checkpoint_save(self):
        checkpoint_file = self.paths_config.out_file_mkdir(self.run_root_dir, Config.CHECKPOINT_FILE)
        if checkpoint_file.exists():
            shutil.copy(checkpoint_file, checkpoint_file.with_suffix('.bkp'))
        self._log_update_tqdm_postfix({"ckpt": self.step_index})
        torch.save(self.module.state_dict(), checkpoint_file)

    def _train_data_loader(self):
        dataset = IterableDataset(itertools.count(), transform=lambda i: self.train_item() | {Keys.DATA_ITEM_INDEX: i})
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=0, collate_fn=self._data_collate)

    def train_item(self):
        raise NotImplementedError

    def loss_func(self, output, target):
        raise NotImplementedError

    def log_train_batch(self, output_batch):
        pass

    def validation_step(self):
        pass

    def _module_apply(self, data_batch: dict):
        return data_batch | {Keys.MODULE_OUTPUT: self.module_forward(data_batch[Keys.MODULE_INPUT])}

    def to_tensor(self, obj):
        return convert_to_tensor(obj, device=self.device, dtype=self.dtype)

    def module_forward(self, module_input):
        return Compose([
            self.to_tensor,
            self.module,
        ])(module_input)

    def init_ensure_module(self):
        if self.module is None:
            self.module = self._module_build()
            self._checkpoint_load()
            self.module.to(self.device)
            self.__num_module_params = num_module_params(self.module)


class SimulatedGenomeItem:
    def __init__(self):
        self.rng = default_rng()
        self.sitevec = None
        self.genome_len = None
        self.sparsity = None

    def make_genome(self):
        assert self.genome_len
        assert self.sparsity

        sitevec = numpy.flatnonzero(self.rng.binomial(n=1, p=1 / self.sparsity, size=self.genome_len))
        assert len(sitevec)
        sitevec.sort()
        self.sitevec = sitevec


class LocalizerLosses(NamedTuple):
    label_loss: Union[ndarray, Tensor]
    loc_loss: Union[ndarray, Tensor]


class LocalizerEnum:
    BG = 0
    STRAY = 1
    FG = 2


class LocalizerTargets(NamedTuple):
    label_target: Union[ndarray, Tensor]
    loc_target: Union[ndarray, Tensor]


class LocalizerOutputs(NamedTuple):
    label_bg: Union[ndarray, Tensor]
    label_stray: Union[ndarray, Tensor]
    label_fg: Union[ndarray, Tensor]
    loc_output: Union[ndarray, Tensor]


class LocalizerModule(TrainerMixin):
    def __init__(self):
        super().__init__()
        self.divisible_size = 16
        self.min_spatial_size = 32
        self.load_checkpoint = False
        self.checkpoint_search_dir = Config.CHECKPOINT_SEARCH_DIR
        self.upsample = "pixelshuffle"

        self.image_channels = 5
        self.unet_channel_divider = 1
        self.out_channels = len(LocalizerOutputs.__annotations__)

        self.image_data = LocalizerTrainDataItem()
        sparsity = 4 ** 6
        self.nominal_scale = self.image_data.nominal_scale

        self.module_output_item = LocalizerOutputItem()
        self.module_output_item.nominal_scale = self.nominal_scale

        nominal_num_labels_fragment = 16
        nominal_fragment_len = sparsity * nominal_num_labels_fragment
        self.image_data.fragment_len = nominal_fragment_len

        self.genome_data = SimulatedGenomeItem()
        self.genome_data.sparsity = sparsity
        nominal_num_labels = nominal_num_labels_fragment * 2
        self.genome_data.genome_len = sparsity * nominal_num_labels

        self.batch_image_size = int(nominal_fragment_len / self.nominal_scale)

    def init_task(self):
        super().init_task()

    def _resize_pad_or_crop_end(self, size):
        return Compose([
            SpatialPad(spatial_size=size, method=Method.END),
            SpatialCrop(roi_slices=[slice(0, s) for s in size]),
        ])

    def image_input(self, segment_3d, spatial_size=None):
        image_2d = segment_3d.reshape((-1, segment_3d.shape[-1]))
        return Compose([
            AddChannel(),
            ScaleIntensity(),
            self._resize_pad_or_crop_end((self.image_channels, spatial_size)),
            SqueezeDim(),
        ])(image_2d)

    def train_item(self):
        data = self.make_data()

        module_input = self.image_input(data.image, self.batch_image_size)

        module_target = Compose([
            self._resize_pad_or_crop_end(module_input.shape[-1:]),
        ])(data.target)

        item = {
            Keys.DATA_ITEM: data,
            Keys.MODULE_INPUT: module_input,
            Keys.MODULE_TARGET: module_target,
        }
        return item

    def make_data(self):
        self.image_data.lat_size_min = self.image_channels
        self.image_data.rng = self.rng
        self.genome_data.rng = self.rng

        for trial in itertools.count():
            try:
                genome_data = copy(self.genome_data)
                genome_data.make_genome()

                image_data = copy(self.image_data)
                image_data.sitevec = genome_data.sitevec
                image_data.make_target()
            except AssertionError:
                if trial > Config.MAX_RETRIES:
                    raise
            else:
                return image_data

    def validation_item(self):
        data = self.make_data()
        item = self.inference_item(data.image)
        item.data_item = data
        loss_item = self.loss_func(item.output_tensor, data.target)
        item.loss_item = loss_item
        item.metrics = self.make_metrics(loss_item)
        return item

    def _inference_forward(self, module_input: ndarray):
        x = Compose([
            SpatialPad(spatial_size=self.min_spatial_size, method=Method.END),
            DivisiblePad(k=self.divisible_size, method=Method.END)
        ])(module_input)

        x = self.module_forward(x[None])
        x = Compose([
            ToNumpy(),
            SqueezeDim(),
            self._resize_pad_or_crop_end(module_input.shape[-1:]),
        ])(x)

        return x

    def inference_item(self, segment_3d):
        image_input = self.image_input(segment_3d)
        # print("original image input shape"+ str(image_input.shape[-1:]))

        with inference_eval(self.module):
            output_tensor = self._inference_forward(image_input)
           
        item = copy(self.module_output_item)
        item.image = segment_3d
        item.image_input = image_input
        item.make_pred(output_tensor)
        return item

    def loss_func(self, output, target):
        if len(output.shape) == 2:
            output = output[None]

        if len(target.shape) == 2:
            target = target[None]

        assert output.shape[-1:] == target.shape[-1:]

        targets = LocalizerTargets(*torch.unbind(self.to_tensor(target), dim=1))
        outputs = LocalizerOutputs(*torch.unbind(self.to_tensor(output), dim=1))

        label_criterion = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False)
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')

        label_output = torch.stack([
            outputs.label_bg,
            outputs.label_stray,
            outputs.label_fg,
        ], dim=1)

        mask = (targets.label_target > 0).float()
        mask_mean = mask.mean()
        assert mask_mean > 0

        loss_tuple = LocalizerLosses(
            label_loss=label_criterion(label_output, targets.label_target.long()[:, None]),
            loc_loss=((mask * bce(outputs.loc_output, targets.loc_target)).mean() / mask_mean)*0,
        )
        return {
            Keys.LOSS_TENSOR: torch.stack(loss_tuple).mean(),
            Keys.LOSSES: loss_tuple._asdict(),
        }

    def make_metrics(self, step_item):
        metrics_item = super().make_metrics(step_item)
        return metrics_item | {k: v.detach().item() for k, v in step_item[Keys.LOSSES].items()}

    def _module_build(self):
        features = numpy.stack([32, 32, 64, 128, 256, 32]) // self.unet_channel_divider
        return BasicUNet(
            spatial_dims=1,
            features=tuple(features),
            in_channels=self.image_channels,
            out_channels=self.out_channels,
            upsample=self.upsample,
        )

    def run_training(self):
        self.log_wandb_enable = True
        self.batch_size = 256
        self.checkpoint_search_dir = Config.CHECKPOINT_SEARCH_DIR
        self.task_root_dir = Config.LOCALIZER_TRAINING_OUTPUT_DIR
        self.run()


class LocalizerOutputItem:
    def __init__(self):
        self.image = None
        self.image_input = None
        self.loc_pred_delta = None
        self.loc_pred = None
        self.output_tensor = None
        self.label_pred = None
        self.outputs = None

    def make_pred(self, output_tensor):
        self.output_tensor = output_tensor
        self.outputs = LocalizerOutputs(*output_tensor)
        self.label_pred = numpy.stack([
            self.outputs.label_bg,
            self.outputs.label_stray,
            self.outputs.label_fg,
        ]).argmax(axis=0)
        self.loc_pred_delta = numpy_sigmoid(self.outputs.loc_output)
        indices = numpy.flatnonzero(self.label_pred == LocalizerEnum.FG)
        self.loc_pred = indices + self.loc_pred_delta[indices]


if __name__ == '__main__':
    LocalizerModule().run_training()
