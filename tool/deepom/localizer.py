from typing import NamedTuple, Union

import numpy as np
import torch
from monai.networks.nets import BasicUNet
from monai.transforms import Compose, SpatialPad, SpatialCrop, AddChannel, ScaleIntensity, SqueezeDim, \
    DivisiblePad, ToNumpy
from monai.utils import Method, convert_to_tensor
from pydantic import BaseModel
from utils.env import ENV
from utils.pyutils import NDArray


class LocalizerEnum:
    BG = 0
    STRAY = 1
    FG = 2


class LocalizerOutputs(NamedTuple):
    label_bg: Union[np.ndarray, torch.Tensor]
    label_stray: Union[np.ndarray, torch.Tensor]
    label_fg: Union[np.ndarray, torch.Tensor]
    loc_output: Union[np.ndarray, torch.Tensor]


class DeepOMLocalizer:
    class Config(BaseModel):
        divisible_size = 16
        min_spatial_size = 32
        upsample = "pixelshuffle"
        image_channels = 5
        unet_channel_divider = 1
        out_channels = len(LocalizerOutputs.__annotations__)
        model_file = ENV.DEEPOM_MODEL_FILE
        device = 'cpu'

    class Input(BaseModel):
        image: NDArray

    class Output(BaseModel):
        localizations: NDArray
        model_input_image: NDArray = None

    class OutputInfo(Output):
        offset_pred: NDArray = None
        output_tensor: NDArray = None
        occupancy_pred: NDArray = None
        outputs: NDArray = None

    dtype = torch.float32

    def __init__(self, **config):
        self.config = self.Config(**config)
        self.module = self._module_build()
        self.state_dict = torch.load(self.config.model_file, map_location=self.config.device)
        self.module.load_state_dict(self.state_dict)
        self.module.eval()

    def compute_localizations(self, image: NDArray):
        return self._process_image(self.Input(image=image)).dict()

    def localize(self, image_item: dict):
        return self._process_image(self.Input(**image_item)).dict()

    def _process_image(self, image_item: Input):
        return self.Output(**self._inference(image_item.image).dict())

    def _crop_image_width_for_model(self, image: np.ndarray):
        image = image[0]
        assert image.ndim == 2
        target_width = self.config.image_channels
        source_width = image.shape[0] // 2 + 1
        image = image[source_width - target_width // 2: source_width + target_width // 2 + 1]
        return image

    def _inference(self, image: np.ndarray):
        image = self._crop_image_width_for_model(image)
        assert image.ndim == 2
        model_input_image = self._input_image(image)

        output_tensor = self._inference_forward_pass(model_input_image)

        outputs = LocalizerOutputs(*output_tensor)
        occupancy_pred = np.stack([
            outputs.label_bg,
            outputs.label_stray,
            outputs.label_fg,
        ]).argmax(axis=0)
        output = outputs.loc_output

        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)

        offset_pred = torch.sigmoid(output).numpy()
        predicted_fg_indices = np.flatnonzero(occupancy_pred == LocalizerEnum.FG)

        return self.OutputInfo(
            localizations=predicted_fg_indices + offset_pred[predicted_fg_indices],
            model_input_image=model_input_image,
            output_tensor=output_tensor,
            occupancy_pred=occupancy_pred,
            offset_pred=offset_pred,
        )

    def _to_tensor(self, obj):
        return convert_to_tensor(obj, device=self.config.device, dtype=self.dtype)

    def _module_forward(self, module_input):
        return Compose([
            self._to_tensor,
            self.module,
        ])(module_input)

    def _input_image(self, segment_3d, spatial_size=None):
        image_2d = segment_3d.reshape((-1, segment_3d.shape[-1]))
        return Compose([
            # TODO: Class `AddChannel` has been deprecated since version 0.8. It will be removed in version 1.3. please use MetaTensor data type and monai.transforms.EnsureChannelFirst instead with `channel_dim='no_channel'`.
            AddChannel(),
            ScaleIntensity(),
            self._resize_pad_or_crop_end((self.config.image_channels, spatial_size)),
            SqueezeDim(),
        ])(image_2d)

    def _resize_pad_or_crop_end(self, size):
        return Compose([
            SpatialPad(spatial_size=size, method=Method.END),
            SpatialCrop(roi_slices=[slice(0, s) for s in size]),
        ])

    def _inference_forward_pass(self, module_input: np.ndarray):
        x = Compose([
            SpatialPad(spatial_size=self.config.min_spatial_size, method=Method.END),
            DivisiblePad(k=self.config.divisible_size, method=Method.END)
        ])(module_input)

        x = self._module_forward(x[None])
        x = Compose([
            ToNumpy(),
            SqueezeDim(),
            self._resize_pad_or_crop_end(module_input.shape[-1:]),
        ])(x)

        return x

    def _module_build(self):
        features = np.stack([32, 32, 64, 128, 256, 32]) // self.config.unet_channel_divider
        return BasicUNet(
            spatial_dims=1,
            features=tuple(features),
            in_channels=self.config.image_channels,
            out_channels=self.config.out_channels,
            upsample=self.config.upsample,
        )
