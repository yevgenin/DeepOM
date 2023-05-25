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
import numpy
from matplotlib.pyplot import eventplot, imshow, figure, xlim, savefig, gca
from numpy import ndarray, stack, int_
from numpy.random import default_rng
from tqdm.auto import tqdm, trange
from torch import Tensor, from_numpy
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from deepom.config import Config
from deepom.aligner import Aligner
from deepom.localizer import LocalizerModule
from deepom.bionano_utils import XMAPItem, BionanoRefAlignerRun, MoleculeSelector, BNXItemCrop, BionanoFileData
from deepom.data_fetcher import DataFetcher
from deepom.my_utils import dice_loss, MetaBase, torch_temporary_seed, fp_precentage ,fn_precentage
from deepom.utils import path_mkdir, generate_name, Paths, asdict_recursive, nested_dict_filter_types, inference_eval, numpy_sigmoid, find_file, num_module_params, pickle_dump, pickle_load
from deepom.common_types import LocalizerEnum, LocalizerLosses, LocalizerGT, LocalizerOutputs, DataItem

from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


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
        self.label_pred = torch.sigmoid(self.outputs.label_output) > 0.5
        self.loc_pred_delta = numpy_sigmoid(self.outputs.loc_output)
        indices = numpy.flatnonzero(self.label_pred.numpy())
        self.loc_pred = indices + self.loc_pred_delta[indices]


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
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.log_metrics_every = 100
        self.batch_size = 1
        self.epochs = 5
        self.log_images_every = 1000
        self.validation_every = None
        self.checkpoint_every = 100
        self.nominal_scale = Config.BIONANO_NOMINAL_SCALE

        self.module_output_item = LocalizerOutputItem()
        self.module_output_item.nominal_scale = self.nominal_scale
        
        self.top_mol_num = 1 #Amount of actual images loaded by the datafetcher
        # self.molecule_amount = 240 #Amount of samples to be used in the dataset
        self.test_data_ratio = 0 #Validation to train set ratios
        self.genome_size = 24
        self.threshold = 0.99
        self.data_fetcher = DataFetcher(self.top_mol_num, self.threshold)
        self.simulated_images = True
        
        # Reproducability
        self.seed = 42
        self.name_seed = 343
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
        self.task_name = type(self).__name__ #DataLocalizer
        self.task_root_dir = Config.LOCALIZER_TRAINING_OUTPUT_DIR
        self.data_root_dir = Config.GROUND_TRUTH_DIR
        self.checkpoint_search_dir = Config.SECOND_CHECKPOINT_SEARCH_DIR
        self.load_checkpoint = False 
        self.save_data = True
        self.best_loss = numpy.inf 
        self.early_stopping = 200
        self.num_stagnant_epochs = 0
        self.stop_training = False
        
        self.eval_localizer = self.data_fetcher.localizer #yevgeni's
        self.checkpoint_file = Config.CHECKPOINT_FILE
        self.dice_smooth = 100
        self.test_data = None 
      
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
        features = stack([32, 64, 64, 128, 256, 32]) // self.unet_channel_divider
        return BasicUNet(
            spatial_dims=1,
            features=tuple(features),
            in_channels=self.image_channels,
            out_channels=self.out_channels,
            upsample=self.upsample,
        )

    def init_ensure_module(self):
        self.net = self._module_build()
        self._checkpoint_load()
        self.net.to(self.device)
        self.__num_module_params = num_module_params(self.net)

    def _checkpoint_load(self):
        if self.load_checkpoint:
            try:
                if self.checkpoint_search_dir is None:
                    search_dir = self.task_root_dir
                    print("no checkpoint dir found")
                else:
                    search_dir = self.checkpoint_search_dir
                    print(f"search dir is {search_dir}")
                file = find_file(search_dir=search_dir, suffix=self.checkpoint_file)
            except (ValueError, RuntimeError):
                print_exc()
                raise
            else:
                print('loading checkpoint: ', file, '\n\n')
                self.net.load_state_dict(torch.load(file, map_location=self.device))


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
        # loc_weight = 0.01
        loss_tuple = LocalizerLosses(
            label_loss=dice_loss(label_output, from_numpy(target.label_target),self.dice_smooth),
            loc_loss=((from_numpy(mask) * bce(torch.flatten(outputs.loc_output),
                    from_numpy(target.loc_target))).mean() / mask_mean),
        )

        label_pred = (label_output > 0.5).type(torch.int8)
        loc_pred_delta = numpy_sigmoid(outputs.loc_output.detach())[0]
        indices = numpy.flatnonzero(label_pred.detach().numpy())
        # print(f"indices {indices}")
        prediction = indices + loc_pred_delta[indices]

        gt_indices = numpy.flatnonzero(target.label_target)
        ground_truth = from_numpy(gt_indices + target.loc_target[gt_indices])

        # print(f"prediction: {prediction} ")
        # print(f"ground truth : {ground_truth}")
        # print(f"sanity check: {prediction")

        fp = fp_precentage(prediction, ground_truth)
        fn = fn_precentage(prediction, ground_truth)
        return {
            "loss_tensor": torch.stack(loss_tuple).mean(),
            "losses": loss_tuple._asdict(),
            "false_positive": fp,
            "false_negative": fn
        }

    def step_events(self, step_item):
        self.step_index +=1
        self.tqdm.update()
        log_metrics_every = max(1, self.log_metrics_every // self.batch_size)
        log_images_every = max(1, self.log_images_every // self.batch_size)
        #validation_every = max(1, self.validation_every // self.batch_size)

        # if self.step_index % log_images_every == 0:
        #     self.log_train_batch(step_item)

           
        if self.step_index % log_metrics_every == 0:
            metrics_item = step_item
            self._log_update_tqdm_postfix({"total loss":metrics_item["loss_tensor"].item()})
            self.log_metrics({"train": metrics_item})
            self.scheduler.step(metrics_item["loss_tensor"].item()) 
            if (self.best_loss > metrics_item["loss_tensor"].item()):
                self.best_loss = metrics_item["loss_tensor"].item()
                if self.checkpoint_every is not None:
                    checkpoint_every = max(1, self.checkpoint_every // self.batch_size)
                    if self.step_index % checkpoint_every == 0:
                        self.checkpoint_save()
                    
                if self.early_stopping is not None:
                    self.num_stagnant_epochs = 0 
                
            else:
                self.num_stagnant_epochs += 1
        
            if self.early_stopping is not None and self.num_stagnant_epochs >= self.early_stopping:
                self.stop_training = True
           
    def validation_step(self,step_item,image_input,target:LocalizerGT):
        # print(len(image_input))
        our_fp = step_item["false_positive"]
        our_fn = step_item["false_negative"]
        prediction = numpy.flatnonzero(self.eval_localizer.inference_item(image_input).loc_pred)
      
        gt_indices = numpy.flatnonzero(target.label_target)
        ground_truth = (gt_indices + target.loc_target[gt_indices])
        # print(prediction)
        # print(ground_truth)
        eval_fp = fp_precentage(prediction, ground_truth)
        eval_fn = fn_precentage(prediction, ground_truth)
        
        print(f" our fp: {our_fp}, eval fp {eval_fp}")
        print(f" our fn: {our_fn}, eval fn {eval_fn}")
        
                
    def checkpoint_save(self):
        checkpoint_file = self.paths_config.out_file_mkdir(self.checkpoint_search_dir, self.checkpoint_file)
        print(f"saving model in {checkpoint_file}")
        if checkpoint_file.exists():
            shutil.copy(checkpoint_file, checkpoint_file.with_suffix('.bkp'))
        self._log_update_tqdm_postfix({"ckpt": self.step_index})
        torch.save(self.net.state_dict(), checkpoint_file)
        
    
    
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
        with torch_temporary_seed(self.name_seed):
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


    def to_tensor(self, x):
        return convert_to_tensor(x, device=self.device, dtype=self.dtype)

    def forward(self, x):
        return self.net(self.to_tensor(x))

    def _train_step(self, data):
        segment_image = data.molecule.bionano_image.segment_image[0]
        segment_image[0:3] = 0
        segment_image[-3:] = 0
        image_input = self.image_input(data.molecule.bionano_image.segment_image[0])
        x = Compose([
            SpatialPad(spatial_size=self.min_spatial_size, method=Method.END),
            DivisiblePad(k=self.divisible_size, method=Method.END)
        ])(image_input)
        #print(data.ground_truth.label_target.shape[0],x.shape[-1])
        x, data = self.crop_to_min(x, data)
        #print(data.ground_truth.label_target.shape[0],x.shape[-1])
        # data = Compose([
        #     self._resize_pad_or_crop_end(x.shape[-1:]),
        # ])(data.ground_truth)
        # print(f"ground_truth shape: {data.ground_truth.shape}, input shape: {x.shape}")
        self.__logger.debug("gt :"+ str(data.ground_truth.label_target.shape))

        self.optimizer.zero_grad()

        output = self.forward(x[None])
        loss = self.loss_func(output, data.ground_truth)
        
        loss["loss_tensor"].backward()
        self.optimizer.step()

        step_item = loss
            
        self.step_events(step_item)
        
        if self.validation_every is not None and self.step_index % self.validation_every == 0:
            self.validation_step(step_item,data.molecule.bionano_image.segment_image[0][:x.shape[-1]],data.ground_truth)

    def crop_to_min(self, x, mol):
        if x.shape[-1]< mol.ground_truth.label_target.shape[0]:
            a = mol.ground_truth.label_target[:x.shape[-1]]
            b = mol.ground_truth.loc_target[:x.shape[-1]]
            ground_truth = LocalizerGT(a,b)
            item = DataItem(mol.molecule, ground_truth)
            return x, item
        else:
            x = x[:,0:mol.ground_truth.label_target.shape[0]]
            return x, mol


    def get_ground_truth(self,mol_index):
        if self.simulated_images:
            return self.data_fetcher.generate_ground_truth_simulated(mol_index)
        else:
            return self.data_fetcher.generate_ground_truth(mol_index)

    def prepare_data(self):
        num_of_test_molecules = round(self.genome_size * self.test_data_ratio)
        molecule_index_list = range(1, self.genome_size+1) # by default, list from 1 to 24
        test_molecules_ids = self.rng.choice(molecule_index_list, size= num_of_test_molecules, replace = False)
        train_molecules_ids = [i for i in molecule_index_list if i not in test_molecules_ids]

        # saving test molecule id list for benchmark use
        self.output_test_list_pickle(test_molecules_ids)
        print(f"test_molecules_ids: {test_molecules_ids}")
        train_data = []
        test_data = []

        for mol_index in trange(self.top_mol_num,desc="Creating Ground Truth Data",unit="molecules"):
            mol, a, b  = self.get_ground_truth(mol_index)
            if mol != None:
                ground_truth = LocalizerGT(a,b)
                item = DataItem(mol, ground_truth)

                if mol.xmap_item.ref_id in train_molecules_ids:
                    train_data.append(item)
                else:
                    test_data.append(item)
                    
        print("train data size: " + str(len(train_data)))
        print("test data size: " + str(len(test_data)))
        
        if self.save_data:
            self.output_pickle_dump(train_data,f".train_data_simulated_images_{self.top_mol_num}.pickle")
            self.output_pickle_dump(test_data,f".test_data_simulated_images_{self.top_mol_num}.pickle")

        self.test_data = test_data
        return train_data, test_data 


    def train(self, train_data):     
        try:
            self.generate_random_task_name()
            self.init_rng()
            self.init_log()
            self.stop_training = False
           
            
            for epoch in trange(self.epochs):
                with tqdm(train_data,unit="batch") as train_data_tqdm:
                    for batch in train_data_tqdm:
                        train_data_tqdm.set_description(f"Epoch {epoch}")
                        self._train_step(batch)
                        if self.stop_training:
                            break
                
                
        finally:
            if self.log_wandb_enable:
                wandb.finish()

                
    def _train_data_loader(self,data):
        dataset = TensorDataset(torch.Tensor(data))
        print(len(data))
        self.train_data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0)
    
    def _data_collate(self, items: list):
        items_to_batch = Compose([
            SelectItemsd(keys=[Keys.MODULE_INPUT, Keys.MODULE_TARGET], allow_missing_keys=True)
        ])(items)
        batch = list_data_collate(items_to_batch) | {Keys.BATCH_ITEMS: items}
        return batch

    def _inference_forward(self, module_input: ndarray):
        x = Compose([
            SpatialPad(spatial_size=self.min_spatial_size, method=Method.END),
            DivisiblePad(k=self.divisible_size, method=Method.END)
        ])(module_input)

        # zeroing first and last pixel rows
        # x[0] = 0
        # x[-1] = 0

        x = self.forward(x[None])
        x = Compose([
            ToNumpy(),
            SqueezeDim(),
            self._resize_pad_or_crop_end(module_input.shape[-1:]),
        ])(x)

        return x

    def inference_item(self, segment_3d):
        segment_3d[0:3] = 0
        segment_3d[-3:] = 0
        figure(figsize=(30, 3)) 
        target_width = 5
        source_width = segment_3d.shape[0] // 2 + 1
        print_input = segment_3d[source_width - target_width // 2: source_width + target_width // 2 + 1]
        # segment_3d[source_width - target_width // 2: source_width + target_width // 2 + 1][0] = 0
        # segment_3d[source_width - target_width // 2: source_width + target_width // 2 + 1][-1] = 0
        imshow(segment_3d, aspect="auto", cmap="gray")

        print(f"shape segment: {segment_3d.shape}")


        image_input = self.image_input(segment_3d)
        figure(figsize=(30, 3)) 
        target_width = 5
        source_width = image_input.shape[0] // 2 + 1
        print_input = image_input[source_width - target_width // 2: source_width + target_width // 2 + 1]
        imshow(print_input, aspect="auto", cmap="gray")

        print(f"image_input size: {image_input.shape}")

        # image_input[0] = 0
        # image_input[-1] = 0

        print(f"image_input size: {type(image_input)}")

        with inference_eval(self.net):
            output_tensor = self._inference_forward(image_input)
           
        item = copy(self.module_output_item)
        item.image = segment_3d
        item.image_input = image_input
        item.make_pred(self.to_tensor(output_tensor))

        return item
    
    def output_pickle_dump(self, obj, suffix):
        file = Config.GROUND_TRUTH_DIR.with_suffix(suffix)
        print(file)
        pickle_dump(file, obj)
        
    def pickle_load(self,file):
        return pickle_load(file)
    
    def output_test_list_pickle(self, obj):
        file = Config.TEST_LIST_FILE.with_suffix(".pickle")
        print(f"test list file: {file}")
        pickle_dump(file, obj)

    def validate_predictions(self):
        self.tqdm = tqdm(desc="validating predictions", disable=not self.tqdm_enable)

        fp_yvg = []
        fp_our = []
        fn_yvg = []
        fn_our = []

        with tqdm(self.test_data) as test_data:
            for item in test_data:
                image = item.molecule.bionano_image.segment_image[0]
                gt_indices = numpy.flatnonzero(item.ground_truth.label_target)
                ground_truth = (gt_indices + item.ground_truth.loc_target[gt_indices])

                our_pred = self.inference_item(image).loc_pred
                yvg_pred = self.eval_localizer.inference_item(image).loc_pred      

                fp_yvg.append(fp_precentage(yvg_pred, ground_truth))
                fn_yvg.append(fn_precentage(yvg_pred, ground_truth))
                fp_our.append(fp_precentage(our_pred, ground_truth))
                fn_our.append(fn_precentage(our_pred, ground_truth))
                self.tqdm.update()
        
        
        print(f"means yevgeni: fp={numpy.array(fp_yvg).mean()}, fn={numpy.array(fn_yvg).mean()}")
        print(f"means ours: fp={numpy.array(fp_our).mean()}, fn={numpy.array(fn_our).mean()}")
