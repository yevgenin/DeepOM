import torch
import wandb
import itertools
import shutil
from copy import copy
from inspect import currentframe
from traceback import print_exc
from typing import NamedTuple, Union
from monai.utils import convert_to_tensor,Method, set_determinism
from monai.losses import DiceCELoss
from monai.networks.nets import BasicUNet
from monai.transforms import Compose, SelectItemsd, SpatialPad, SpatialCrop, AddChannel, ScaleIntensity, SqueezeDim, \
    DivisiblePad, ToNumpy
from numpy import ndarray, stack, int_
from numpy.random import default_rng
from tqdm.auto import tqdm, trange
from torch import Tensor, from_numpy
from torch.optim import Adam
from deepom.config import Config
from deepom.aligner import Aligner
from deepom.localizer import LocalizerModule
from deepom.bionano_utils import XMAPItem, BionanoRefAlignerRun, MoleculeSelector, BNXItemCrop, BionanoFileData
from deepom.data_fetcher import DataFetcher
from deepom.my_utils import dice_loss, MetaBase
from deepom.utils import path_mkdir, generate_name, Paths, asdict_recursive, nested_dict_filter_types
from deepom.common_types import LocalizerEnum, LocalizerLosses, LocalizerGT, LocalizerOutputs, DataItem



class DataLocalizer(metaclass=MetaBase):
    def __init__(self):
        self.image_channels = 5
        self.divisible_size = 16
        self.min_spatial_size = 32
        self.out_channels = len(LocalizerOutputs.__annotations__)
        self.upsample = "pixelshuffle"
        self.unet_channel_divider = 1
        self.net = self._module_build()
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.dtype = torch.float32
        self.lr = 1e-3
        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.log_metrics_every = 1
        self.batch_size = 1
        self.epochs = 5
        self.log_images_every = 1000
        self.validation_every = 2000
        self.checkpoint_every = 10000
        
        self.top_mol_num = 100
        self.molecule_amount = 240
        
        self.data_fetcher = DataFetcher(self.top_mol_num)
        self.test_data_ratio = 0.2
        
        # Reproducability
        self.seed = 42
        self.init_rng()
        # Logging Info
        self.run_name = None
        self.run_root_dir = None
        
        self.step_index = 0
        self.tqdm = None
        self.tqdm_enable = True
        self.log_wandb_enable = True
        self.log_tqdm_postfix = {}
        
        self.paths_config = Paths()
        self.task_name = type(self).__name__
        self.task_root_dir = self.paths_config.out_path(self.task_name)
        
    
      
    def init_rng(self):
        set_determinism(seed=self.seed)
        self.rng = default_rng(seed=self.seed)    

    def _init_refs():
        cmap_file_data = BionanoFileData()
        cmap_file_data.file = Config.REF_CMAP_FILE
        cmap_file_data.read_bionano_file()

        return Series({
            ref_id: ref_df[ref_df["LabelChannel"] == 1]["Position"].values
            for ref_id, ref_df in cmap_file_data.file_df.groupby("CMapId")
            })

    def _module_build(self):
        features = stack([32, 32, 64, 128, 256, 32]) // self.unet_channel_divider
        return BasicUNet(
            spatial_dims=1,
            features=tuple(features),
            in_channels=self.image_channels,
            out_channels=self.out_channels,
            upsample=self.upsample,
        )


    def loss_func_prev(self, output, target):
        if len(output.shape) == 2:
            output = output[None]

        if len(target.shape) == 2:
            target = target[None]

        assert output.shape[-1:] == target.shape[-1:]

        targets = LocalizerGT(*torch.unbind(self.to_tensor(target), dim=1))
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
            loc_loss=(mask * bce(outputs.loc_output, targets.loc_target)).mean() / mask_mean,
        )
        return {
            "loss_tensor": torch.stack(loss_tuple).mean(),
            "losses": loss_tuple._asdict(),
        }


    def loss_func(self, output, target):

        outputs = LocalizerOutputs(*torch.unbind(self.to_tensor(output), dim=1))

        bce = torch.nn.BCEWithLogitsLoss(reduction='none')

        # label_output = torch.stack([
        #     outputs.label_output,
        # ], dim=1)
        
        # print(f"Target label target shape {target.label_target.shape}")

        mask = (target.label_target > 0).astype(float)
        mask_mean = mask.mean()
        assert mask_mean > 0

        self.__logger.debug(f"Label output shape {outputs.label_output.shape}")
        self.__logger.debug(f"{target.label_target[:, None].shape}")
        label_output = torch.sigmoid(outputs.label_output)
        loss_tuple = LocalizerLosses(
            label_loss=dice_loss(label_output, from_numpy(target.label_target)),
            loc_loss=(from_numpy(mask) * bce(torch.flatten(outputs.loc_output),
                    from_numpy(target.loc_target))).mean() / mask_mean,
        )
        return {
            "loss_tensor": torch.stack(loss_tuple).mean(),
            "losses": loss_tuple._asdict(),
        }

    def step_events(self, step_item):
        self.step_index +=1
        # self.tqdm.update()
        log_metrics_every = max(1, self.log_metrics_every // self.batch_size)
        log_images_every = max(1, self.log_images_every // self.batch_size)
        validation_every = max(1, self.validation_every // self.batch_size)

        # if self.step_index % log_images_every == 0:
        #     self.log_train_batch(step_item)

        # if self.step_index % validation_every == 0:
        #     self.validation_step()

        if self.step_index % log_metrics_every == 0:
            metrics_item = step_item
            self._log_update_tqdm_postfix({"total loss":metrics_item["loss_tensor"].item()})
            self.log_metrics({"train": metrics_item})

        if self.checkpoint_every is not None:
            checkpoint_every = max(1, self.checkpoint_every // self.batch_size)
            if self.step_index % checkpoint_every == 0:
                self.checkpoint_save()
    
    
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
    
    def generate_random_task_name(self):
        self.run_name = generate_name(self.task_name)
        # if self.test_mode_enable:
        #     self.task_root_dir = self.task_root_dir / Config.TEST_DIR
        self.run_root_dir = (self.task_root_dir / self.run_name)
        
    def task_params_dict(self):
        data = asdict_recursive(self, include_modules=[self.__module__], prefix=Config.WANDB_PREFIX)
        data = nested_dict_filter_types(data)
        return data
    
    def init_log(self):
        self.tqdm = tqdm(desc=self.run_name, disable=not self.tqdm_enable)

        if self.log_wandb_enable:
            path_mkdir(self.run_root_dir)
            wandb.init(
                name=self.run_name, dir=str(self.run_root_dir),
                mode=None,
                project=Config.PROJECT_NAME,
            )
            wandb.config.update(self.task_params_dict())

    def log_metrics(self, metrics_dict):
        if self.log_wandb_enable:
            self.__logger.debug(metrics_dict)
            wandb.log(metrics_dict, step=self.step_index)
            
    def _log_update_tqdm_postfix(self, metrics_dict):
        self.log_tqdm_postfix |= metrics_dict
        self.tqdm.set_postfix(self.log_tqdm_postfix)

    # def _on_log_init_end(self):
    #     if self.log_wandb_enable:
    #         wandb.watch(models=self.module, log="all", log_graph=True)

    def to_tensor(self, x):
        return convert_to_tensor(x, device=self.device, dtype=self.dtype)

    def forward(self, x):
        return self.net(self.to_tensor(x))

    def _train_step(self, data):
        image_input = self.image_input(data.molecule.bionano_image.segment_image[0])

        x = Compose([
            SpatialPad(spatial_size=self.min_spatial_size, method=Method.END),
            DivisiblePad(k=self.divisible_size, method=Method.END)
        ])(image_input)

        x = x[:,0:data.ground_truth.label_target.shape[0]]
        self.__logger.debug("gt :"+ str(data.ground_truth.label_target.shape))

        self.optimizer.zero_grad()

        output = self.forward(x[None])
        loss = self.loss_func(output, data.ground_truth)
        
        loss["loss_tensor"].backward()
        self.optimizer.step()

        step_item = loss
        self.step_events(step_item)

    def get_ground_truth(self,mol_index):
        return self.data_fetcher.generate_ground_truth(mol_index)

    def prepare_data(self):
        num_of_test_molecules = round(self.molecule_amount * self.test_data_ratio)
        test_molecules_ids = self.rng.choice(range(1,self.molecule_amount+1), size= num_of_test_molecules, replace = False)
        train_molecules_ids = [i for i in range(1, self.molecule_amount+1) if i not in test_molecules_ids]
        train_data = []
        test_data = []
        for mol_index in trange(self.top_mol_num,desc="Creating Ground Truth Data",unit="molecules"):
            mol, a, b = self.get_ground_truth(mol_index)
            ground_truth = LocalizerGT(a,b)
            item = DataItem(mol, ground_truth)

            if mol.xmap_item.ref_id in train_molecules_ids:
                train_data.append(item)
            else:
                test_data.append(item)

        return train_data, test_data 


    def train(self, train_data):     
        try:
            #Order important, if task name generated after rng reset all tasks will have the same name
            self.generate_random_task_name()
            self.init_rng()
            self.init_log()
            for epoch in range(self.epochs):
                with self.tqdm(train_data,unit="batch") as train_data_tqdm:
                    for batch in train_data_tqdm:
                        self._train_step(batch)
                
        finally:
            if self.log_wandb_enable:
                wandb.finish()

        
# if __name__ =='__main__':
#     localizer = DataLocalizer()
#     train_data, test_data = localizer.prepare_data()
#     localizer.train(train_data)