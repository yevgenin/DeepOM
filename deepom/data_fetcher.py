from scipy.interpolate import interp1d
from deepom.bionano_utils import MoleculeSelector

from deepom.config import Config
from deepom.localizer import LocalizerModule
from deepom.common_types import LocalizerEnum
import monai

from deepom.aligner import Aligner
from deepom.utils import generate_name
from deepom.my_utils import get_aligner, get_scaler_from_aligner, overlap_percentage, overlap_fraction, fp_precentage, fn_precentage
import numpy as np

from deepom.bionano_compare import BionanoCompare
from matplotlib.pyplot import eventplot, imshow, figure, xlim, savefig, close, title

from deepom.config import Config
from threading import Lock, Thread
from pathlib import Path
from numba import njit,jit
import os 


class DataFetcher():
    
    def __init__(self, num_molecules: int,plot_size = 500,threshold = 0.99):
        #Plot location and bounds data
        self.save_path = "../data/ground_truth/"+generate_name("gt")
        self.plot_size = plot_size
        
        self.selector = MoleculeSelector(read_images=False)
        self.num_molecules = num_molecules
        self.selector.top_mol_num = num_molecules
        self.loaded_molecules = False #To cache molecules
        
        self.refs = self._make_refs()
        self.localizer = self._make_localizer()
        self.low_confidence_mols = []
        self.threshold = threshold
        self.rng = np.random.default_rng()
        
    
    def get_molecules_from_file(self):
        
        if self.loaded_molecules:
            return
        self.selector.select_molecules()
        self.loaded_molecules = True
    
    @classmethod
    def _make_refs(self):
        compare = BionanoCompare()
        compare.read_cmap()
        compare.make_refs()
        return compare.refs
    
    @classmethod
    def _make_localizer(self):
        localizer_module = LocalizerModule()
        localizer_module.checkpoint_search_dir = Config.CHECKPOINT_SEARCH_DIR
        localizer_module.load_checkpoint = True
        localizer_module.init_ensure_module()
        return localizer_module
    
    def create_localization(self,mol_index):
        image_input = self.selector.selected[mol_index].bionano_image.segment_image[0]
        target_width = self.localizer.image_channels
        source_width = image_input.shape[0] // 2 + 1
        image_input = image_input[source_width - target_width // 2: source_width + target_width // 2 + 1]
        inference_item = self.localizer.inference_item(image_input)
        return inference_item

    def get_best_aligner(self,inference_item,mol_index):
        ref_id = self._get_ref_id(mol_index)

        aligner_reg = get_aligner()
        aligner_rev = get_aligner()

        qry = inference_item.loc_pred * 335
        inverted_qry = np.sort((inference_item.image.shape[-1] - inference_item.loc_pred)*335)
        
        aligner_reg.make_alignment(qry=qry, ref=self.refs[ref_id])
        aligner_rev.make_alignment(qry=inverted_qry, ref=self.refs[ref_id])
        
        regular_overlap = self._get_overlap(aligner_reg,mol_index)
        reverse_overlap = self._get_overlap(aligner_rev,mol_index)
        rev = False
        (rev,aligner) = (False,aligner_reg) if regular_overlap > reverse_overlap else (True,aligner_rev)
        # print(f"best overlap for mol number {mol_index} is: {max(regular_overlap,reverse_overlap)}")
        offset = inference_item.image.shape[-1] - inference_item.loc_pred[-1]
        aligner.add_offset(offset)
        aligner.image_len = inference_item.image.shape[-1]
        overlap = max(regular_overlap,reverse_overlap)
        return rev,aligner,overlap
        
    def interpolate_ground_truth(self,aligner,mol_index,rev):
        ref_id = self._get_ref_id(mol_index)
        scaler = get_scaler_from_aligner(aligner)
        
        z = aligner.alignment_ref
        y = (z-z[0]) / (scaler*335)
        x = aligner.alignment_qry/335
        if rev == True:
            y = np.sort(aligner.image_len-y)-aligner.offset
            x = np.sort(aligner.image_len-x)
        else:
             y = y+aligner.alignment_qry[0] /335
            
        interp_func = interp1d(x=y,y=x,bounds_error = False, fill_value = "extrapolate")
       
        ref = self.refs[ref_id]
        start, end = aligner.alignment_ref[[0, -1]]
        start_index = np.argmax(ref >= start)
        end_index = np.argmin(ref <= end)
        ref_crop = ref[start_index:end_index]

        z_hat = (ref_crop-ref_crop[0]) / (scaler*335)
        if rev == True:
            z_hat = np.sort(aligner.image_len-z_hat)-aligner.offset
        else:
            z_hat = z_hat+x[0]
        
        y_hat = interp_func(z_hat)
        return y_hat


    def generate_ground_truth_plots(self):
        self.get_molecules_from_file()
        
        for molecule_index in range(self.num_molecules):
            inference_item = self.create_localization(molecule_index)
            rev,aligner,_ = self.get_best_aligner(inference_item,molecule_index)
            ground_truth = self.interpolate_ground_truth(aligner,molecule_index,rev)
            overlap = self._get_overlap(aligner,molecule_index) 
      

            self.save_ground_truth_plot(inference_item,ground_truth,molecule_index,
                                    aligner.score,overlap)
            #print(f"generated molecule number {molecule_index}")
           
    def generate_ground_truth(self, molecule_index):
        self.get_molecules_from_file()
        # self.selector.selected[molecule_index].make_simulated_image(self.refs, self.rng)

        inference_item = self.create_localization(molecule_index)
        
        rev,aligner,overlap = self.get_best_aligner(inference_item,molecule_index)
        if overlap < self.threshold:
            return (None,None,None)
        ground_truth = self.interpolate_ground_truth(aligner,molecule_index,rev) #y_hat
        mol = self.selector.selected[molecule_index]

        shape = inference_item.image.shape[-1]
        
        labeled_index = ground_truth.astype(int)
        pos_in_pixel = ground_truth - labeled_index
   
        assert ((pos_in_pixel >= 0) & (pos_in_pixel <= 1) & (ground_truth >= 0) & (ground_truth < shape)).all()
        assert len(pos_in_pixel) == len(ground_truth)
        
        a = np.full(shape, fill_value=LocalizerEnum.BG, dtype=int)
        b = np.full(shape, fill_value=.5, dtype=float)
       
        a[labeled_index] = LocalizerEnum.FG
        b[labeled_index] = pos_in_pixel
        # print(f"GT for mol number {molecule_index}: is of length {shape}")
        return mol,a,b
       
    def generate_ground_truth_simulated(self, molecule_index):
        self.get_molecules_from_file()
        self.selector.selected[molecule_index].make_simulated_image(self.refs, self.rng)
        mol = self.selector.selected[molecule_index]
        locations = mol.simulated.emitter_coords[...,1]
        labeled_index = locations.astype(int)
        pos_in_pixel = locations - labeled_index
        shape = mol.bionano_image.segment_image[0].shape[-1]
       
        a = np.full(shape, fill_value=LocalizerEnum.BG, dtype=int)
        b = np.full(shape, fill_value=.5, dtype=float)
       
        a[labeled_index] = LocalizerEnum.FG
        b[labeled_index] = pos_in_pixel
        # print(f"GT for mol number {molecule_index}: is of length {shape}")
        return mol,a,b

    def generate_ground_truth_plots_simulated(self):
        self.get_molecules_from_file()
        for molecule_index in range(self.num_molecules):
            figure(figsize=(30, 3))
            mol, a, b = self.generate_ground_truth_simulated(molecule_index)
            image_input = mol.bionano_image.segment_image[0]
            target_width = self.localizer.image_channels
            source_width = image_input.shape[0] // 2 + 1
            image_input = image_input[source_width - target_width // 2: source_width + target_width // 2 + 1]
            imshow(image_input, aspect="auto", cmap="gray")
            locations = mol.simulated.emitter_coords[...,1]
            print(locations.shape)
            eventplot([locations], colors=["r"])

            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
            savefig(Path(self.save_path) / (f"simulated_ground_truth_{molecule_index}.png"))
            close()


    def save_ground_truth_plot(self,inference_item,ground_truth,mol_index,score,overlap):
        print(mol_index)
        figure(figsize=(30, 3))

        fp = fp_precentage(ground_truth, inference_item.loc_pred, 0.5)
        fn = fn_precentage(ground_truth, inference_item.loc_pred, 0.5)

        image_input = self.selector.selected[mol_index].bionano_image.segment_image[0]
        target_width = self.localizer.image_channels
        source_width = image_input.shape[0] // 2 + 1
        image_input = image_input[source_width - target_width // 2: source_width + target_width // 2 + 1]
        imshow(image_input, aspect="auto", cmap="gray")
        eventplot([inference_item.loc_pred,ground_truth], colors=["b", "r"])
        color = 'black'
        if overlap < self.threshold : #conservative estimate
            color= 'red'
            self.low_confidence_mols.append(mol_index)
        title(f"mol index: {mol_index} score: {score}, overlap: {overlap}, fp: {fp}, fn: {fn}",color = color)
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        savefig(Path(self.save_path) / (f"ground_truth_{mol_index}.png"))
        # savefig(f'ground_truth_{mol_index}.png')
        # (Path("../../DeepOM-Paper/figures") / ("simulation_figure" + ext), bbox_inches='tight')
        close() # Don't want to have too many figure instances open
            
    def _get_ref_id(self,mol_index):
        return self.selector.selected[mol_index].xmap_item.ref_id
    
    def _get_overlap(self,aligner,mol_index):
        return overlap_percentage(self.selector.selected[mol_index].xmap_item.ref_lims, aligner.alignment_ref[[0,-1]])
