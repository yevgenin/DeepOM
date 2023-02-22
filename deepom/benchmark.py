from deepom.bionano_compare import DataPrep, BionanoCompare, QryItem, AlignmentItem, AccuracyItem
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy
import pandas
from numpy import ndarray
from numpy.random import default_rng
from pandas import Series, DataFrame
from tqdm.auto import tqdm

from deepom import bionano_utils, localizer, aligner
from deepom.aligner import Aligner
from deepom.bionano_utils import XMAPItem, BionanoRefAlignerRun, MoleculeSelector, BNXItemCrop, BionanoFileData
from deepom.localizer import LocalizerModule, LocalizerOutputItem
from deepom.utils import Config, Paths, asdict_recursive, nested_dict_filter_types, pickle_dump, \
    pickle_load, git_commit, timestamp_str_iso_8601

from matplotlib import pyplot
from matplotlib.pyplot import eventplot, imshow, figure, xlim, savefig, gca
from matplotlib.ticker import PercentFormatter
import matplotlib

from statsmodels.stats.proportion import proportion_confint
from IPython.display import display
# compares two localizers on the bionano dataset
class LocalizerBenchmark:
    nominal_scale = Config.BIONANO_NOMINAL_SCALE
    confidence_alpha = 0.05
    # cmap_filepath = Config.REF_CMAP_FILE
    parallel = True
    def __init__(self,first_localizer,second_localizer):
        self.localizers = [first_localizer,second_localizer]
        self.test_molecules_ids = self._load_test_molecule_list()
        self.data_prep = DataPrep(num_crops_per_size=5,top_mol_num=10, test_id_list=self.test_molecules_ids) # 512 x 100
        self.executor = ThreadPoolExecutor()
        self.refs = self._make_refs()
        self.run_name = "benchmark"
        self.output_file_base = Paths().out_file_mkdir("bionano_compare", self.run_name, self.run_name)
        

    @classmethod
    def _make_refs(self):
        compare = BionanoCompare()
        compare.read_cmap()
        compare.make_refs()
        return compare.refs   

    @classmethod
    def _load_test_molecule_list(self):
        try:
            return pickle_load(Config.TEST_LIST_FILE.with_suffix(".pickle"))
        except FileNotFoundError:
            return None


    def run_aligner(self, localizer_index):
        self.aligner_items = list(self.aligner_alignment_items_top(self.localizer_qry_items(localizer_index)))
        print(f"{len(self.aligner_items)=}")


    def aligner_alignment_items_top(self, qry_items):
        items = self.aligner_alignment_items(qry_items)
        alignment_df = DataFrame(map(vars, items))
        yield from (
            alignment_df["alignment_item"].loc[index] for index in
            alignment_df.groupby("molecule_id")["score"].idxmax()
        )

    def aligner_alignment_items(self, qry_items):
        tasks = []
        for qry_item in tqdm(qry_items, desc="qry items", total=len(self.data_prep.crop_items) * 2):
            try:
                for ref_id, ref in self.refs.items():
                    if self.parallel:
                        tasks.append(self.executor.submit(self.align, qry_item, ref, ref_id))
                    else:
                        yield self.align(qry_item, ref, ref_id)
            except AssertionError:
                print("task failed")

        if self.parallel:
            for task in tqdm(as_completed(tasks), desc="alignments",
                             total=len(self.data_prep.crop_items) * len(self.refs) * 2):
                try:
                    yield task.result()
                except AssertionError:
                      print("task failed")


    def localizer_qry_items(self, localizer_index):
        inference_items = self.executor.map(lambda crop_item: self.crop_inference(crop_item, localizer_index), self.data_prep.crop_items)
        first = True
        for inference_item in inference_items:
            scale = self.nominal_scale

            locvec = numpy.sort(inference_item.loc_pred)
            image_len = inference_item.image_input.shape[-1]

            #print for check
            # if first:
            figure(figsize=(30, 3)) 
            target_width = 5
            source_width = inference_item.image_input.shape[0] // 2 + 1
            image_input = inference_item.image_input[source_width - target_width // 2: source_width + target_width // 2 + 1]
            imshow(inference_item.image_input, aspect="auto", cmap="gray")
            eventplot([inference_item.loc_pred], colors=["r"])
                # first = False

            def _qry_item(orientation, locs):
                qry_item = QryItem()
                qry_item.locs = numpy.sort(locs)
                qry_item.qry = qry_item.locs * scale
                qry_item.scale = scale
                qry_item.orientation = orientation
                qry_item.crop_item = inference_item.crop_item
                #qry_item.inference_item = inference_item
                return qry_item

            yield _qry_item(orientation="+", locs=locvec)
            yield _qry_item(orientation="-", locs=numpy.sort(image_len - locvec))

    def crop_inference(self, crop_item,localizer_index):     
        image_input = crop_item.segment_image[0]
        target_width = self.localizers[localizer_index].image_channels
        source_width = image_input.shape[0] // 2 + 1
        image_input = image_input[source_width - target_width // 2: source_width + target_width // 2 + 1]
        # if localizer_index == 0:
        #     image_input = crop_item.segment_image[0]
        inference_item = self.localizers[localizer_index].inference_item(image_input)
        inference_item.crop_item = crop_item
        return inference_item

    def run_aligners(self):
        print("starting to run aligners")
        with self.executor:
            self.run_aligner(0) #ours
            self.output_pickle_dump(self.aligner_items, ".aligner1.pickle")
            # self.run_aligner(1) #yevgeni's
            # self.output_pickle_dump(self.aligner_items, ".aligner2.pickle") 

    def run_benchmark(self):
        self.data_prep.selector.run_ids = None
        self.data_prep.make_crops()
        self.run_aligners()
        
    
    def align(self, qry_item, ref, ref_id):
        
        aligner = Aligner()
        aligner.align_params = {}
        aligner.make_alignment(qry=qry_item.qry, ref=ref)
        alignment_item = AlignmentItem()
        alignment_item.alignment = aligner.path
        alignment_item.molecule_id = qry_item.crop_item.molecule_id
        alignment_item.ref_id = ref_id
        alignment_item.orientation = qry_item.orientation
        alignment_item.score = aligner.score
        alignment_item.alignment_ref = aligner.alignment_ref
        alignment_item.alignment_qry = aligner.alignment_qry
        alignment_item.ref_lims = alignment_item.alignment_ref[[0, -1]]
        alignment_item.qry_lims = alignment_item.alignment_qry[[0, -1]]
        alignment_item.crop_item = qry_item.crop_item
        alignment_item.qry_item = qry_item
        alignment_item.alignment_item = alignment_item
        return alignment_item
    
    def output_pickle_dump(self, obj, suffix):
        file = self.output_file_base.with_suffix(suffix)
        print(file)
        pickle_dump(file, obj)



##################################################


    def read_compute_results(self):
        file_base = Paths().out_path("bionano_compare", self.run_name, self.run_name)
        try:
            self.aligner_items1 = pickle_load(file_base.with_suffix(".aligner1.pickle"))
        except FileNotFoundError:
            pass
        try:
            self.aligner_items2 = pickle_load(file_base.with_suffix(".aligner2.pickle"))
        except FileNotFoundError:
            pass
        try:
            self.bionano_items = pickle_load(file_base.with_suffix(".bionano.pickle"))
        except FileNotFoundError:
            pass
        
#################################################

    def plot_a(self):
        self.plot_compare_init()
        self.plot_aligner_accuracy()
    
    def plot_compare_init(self):
        ax = pyplot.gca()
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        pyplot.xlabel("Fragment Length (kb)")
        pyplot.ylabel("Success Rate")
        pyplot.legend()

        pyplot.xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, y: int(x // 1000)))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10000))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        pyplot.grid(which="both")
        pyplot.xticks(numpy.stack([20, 50, 100, 200, 400]) * 1000)
         
        
    def plot_aligner_accuracy(self):
        self.aligner_accuracy_items1 = list(self.accuracy_items(self.aligner_items1))
        self.aligner_accuracy_items2 = list(self.accuracy_items(self.aligner_items2))
        self.aligner_accuracy1 = self.plot_accuracy(self.aligner_accuracy_items1, label="Aligner1")
        self.aligner_accuracy2 = self.plot_accuracy(self.aligner_accuracy_items2, label="Aligner2")
        
        
    def accuracy_items(self, alignment_items):
        alignment_item: AlignmentItem

        for alignment_item in alignment_items:
            crop_item = alignment_item.crop_item

            accuracy_item = AccuracyItem()
            accuracy_item.len_bp = crop_item.crop_size_bp
            accuracy_item.num_bnx_labels = len(crop_item.original_locs_crop)
            if alignment_item.qry_item:
                accuracy_item.num_qry_labels = len(alignment_item.qry_item.qry)
            accuracy_item.parent_orientation = crop_item.parent_bnx_item.xmap_item.orientation
            accuracy_item.parent_ref_id = int(crop_item.parent_bnx_item.xmap_item.ref_id)
            accuracy_item.correct_seq = (
                    (alignment_item.ref_id == accuracy_item.parent_ref_id) and
                    (alignment_item.orientation == accuracy_item.parent_orientation)
            )

            accuracy_item.parent_ref_lims = alignment_item.crop_item.parent_bnx_item.xmap_item.ref_lims
            accuracy_item.overlap_fraction = self.overlap_fraction(parent_seg=accuracy_item.parent_ref_lims, crop_seg=alignment_item.ref_lims)
            
            accuracy_item.correct = accuracy_item.correct_seq and (accuracy_item.overlap_fraction > 0)
            accuracy_item.alignment_item = alignment_item
            accuracy_item.accuracy_item = accuracy_item
            yield accuracy_item
            
            
    def overlap_fraction(self, parent_seg, crop_seg):
        (x, y), (a, b) = sorted(parent_seg), sorted(crop_seg)
        assert (a < b) and (x < y)
        intersection = max(0, min(b, y) - max(a, x))
        overlap = intersection / (b - a)
        return overlap
    
    
    def plot_accuracy(self, accuracy_items, label):
        df = DataFrame(map(vars, accuracy_items))
        # print(df)
        correct = df.groupby("len_bp")["correct"]
        num_successes = correct.sum()
        # print(f"num_successes: {num_successes}")
        num_observarions = correct.count()
        acc = (num_successes / num_observarions)
        len_bp = acc.index.values
        pyplot.plot(len_bp, acc.values, marker='.', label=label)

        ci = numpy.stack([
            proportion_confint(count=count, nobs=nobs, alpha=self.confidence_alpha, method='beta')
            for count, nobs in zip(num_successes, num_observarions)
        ])
        pyplot.fill_between(len_bp, ci.T[0], ci.T[1], alpha=0.2)
        return acc