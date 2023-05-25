from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count

from matplotlib.ticker import PercentFormatter
from statsmodels.stats.proportion import proportion_confint
import matplotlib
import numpy
import pandas
import yaml
from matplotlib import pyplot
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


class DataPrep:
    # crop_size_range_bp = 10 * 1000, 450 * 1000
    # num_crops_per_size = 512
    # num_sizes = 24
    nominal_scale = Config.BIONANO_NOMINAL_SCALE

    def __init__(self, crop_size_range_bp = (400 * 1000, 450 * 1000), num_crops_per_size=512, num_sizes=1, top_mol_num=512, test_id_list=[]):
        # crop_size_range_bp = (10 * 1000, 450 * 1000), num_sizes=24
        self.rng = default_rng(seed=0)
        self.selector = MoleculeSelector(filter_data=True, filter_ids=test_id_list)
        self.crop_size_range_bp = crop_size_range_bp
        self.num_crops_per_size = num_crops_per_size
        self.num_sizes = num_sizes
        self.top_mol_num = top_mol_num

    def crops_df(self):
        return DataFrame(map(vars, self.crop_items))

    def make_crops(self):
        self.selector.top_mol_num = self.top_mol_num
        self.selector.select_molecules()

        self.crop_sizes_bp = numpy.geomspace(*self.crop_size_range_bp, self.num_sizes)[::-1]
        print("crop_sizes", self.crop_sizes_bp)

        def _crop_items():
            molecule_ids = count(1)
            for crop_size_bp in self.crop_sizes_bp:
                bnx_items = self.rng.choice(self.selector.selected, size=self.num_crops_per_size, replace=True)
                for bnx_item in bnx_items:
                    try:
                        yield self.generate_crop(bnx_item=bnx_item, molecule_id=next(molecule_ids),
                                                 crop_size_bp=crop_size_bp)
                    except AssertionError:
                        pass

        self.crop_items = list(tqdm(_crop_items(), desc="crops"))

    def generate_crop(self, bnx_item, molecule_id, crop_size_bp):

        segment_image = bnx_item.bionano_image.segment_image
        image_len = segment_image.shape[-1]

        crop_size = int(crop_size_bp / self.nominal_scale)

        assert crop_size < image_len

        start = self.rng.integers(0, image_len - crop_size)
        stop = start + crop_size
        segment_image = segment_image[..., start: stop]
        crop_lims = (start, stop)

        crop = BNXItemCrop(
            parent_bnx_item=bnx_item, crop_lims=crop_lims, molecule_id=molecule_id
        )
        crop.crop_size_pixels = crop_size
        crop.crop_size_bp = crop_size_bp
        crop.segment_image = segment_image
        return crop

    def print_crops_report(self):
        assert len(self.crop_items)
        df = DataFrame([
            {**vars(item), **item.bnx_item.bnx_record}
            for item in self.crop_items
        ])
        report = {
            "num_original_molecules": df["OriginalMoleculeId"].nunique(),
        }
        print(report)

    def bnx_items_to_text(self, bnx_items):
        return "\n".join([
            *self.selector.bnx_file_data.header_lines,
            *[_.bnx_to_text() for _ in bnx_items],
        ]).replace("\n\n", "\n") + "\n"

    def make_crops_bnx(self):
        self.bnx_text = self.bnx_items_to_text([_.bnx_item for _ in self.crop_items])


class QryItem:
    orientation: str
    qry: ndarray
    scale: float
    locs: ndarray
    crop_item: BNXItemCrop
    inference_item: LocalizerOutputItem


class AlignmentItem:
    alignment: ndarray
    molecule_id: int
    orientation: str
    ref_id: int
    ref_lims: ndarray
    qry_lims: ndarray
    crop_item: BNXItemCrop
    qry_item: QryItem = None
    xmap_item: XMAPItem = None
    alignment_item: 'AlignmentItem'

    def plot_alignment_item(self):
        pyplot.figure(figsize=(20, 5))
        im = self.qry_item.inference_item.image_input
        pyplot.gca().pcolorfast((0, im.shape[-1]), (0, 1), im, cmap="gray")
        pyplot.eventplot([self.qry_item.locs], colors="r")


class AccuracyItem:
    overlap_fraction: float
    len_pixels: float
    len_bp: float
    num_bnx_labels: int
    num_qry_labels: int
    parent_orientation: str
    parent_ref_id: int
    correct_seq: bool
    correct: bool
    alignment_item: 'AlignmentItem'
    accuracy_item: 'AccuracyItem'


class BionanoCompare:
    nominal_scale = Config.BIONANO_NOMINAL_SCALE
    cmap_filepath = Config.REF_CMAP_FILE
    aligner_use_bnx_locs = False
    parallel = True

    def __init__(self):
        self.cmap_file_data = BionanoFileData()
        self.bionano_ref_aligner_run = BionanoRefAlignerRun()
        self.data_prep = DataPrep()
        self.localizer_module = LocalizerModule()
        self.run_name = timestamp_str_iso_8601()
        self.report = BionanoCompareReport()
        self.executor = ThreadPoolExecutor()

    @classmethod
    def get_output_file_base(cls, run_name):
        return Paths().out_file_mkdir("bionano_compare", run_name, run_name)

    def read_cmap(self):
        self.cmap_file_data.file = self.cmap_filepath
        self.cmap_file_data.read_bionano_file()

    def run_bionano_on_bnx(self):
        self.bionano_ref_aligner_run.bnx_text = self.data_prep.bnx_text
        self.bionano_ref_aligner_run.run_refaligner()

    def read_result_xmap(self):
        xmap_file_data = BionanoFileData()
        xmap_file_data.file = self.bionano_ref_aligner_run.xmap_file
        xmap_file_data.read_bionano_file()
        self.result_xmap_file_data = xmap_file_data

    def localizer_qry_items(self):
        self.localizer_module.checkpoint_search_dir = Config.CHECKPOINT_SEARCH_DIR
        self.localizer_module.load_checkpoint = True
        self.localizer_module.init_ensure_module()

        inference_items = self.executor.map(self.crop_inference, self.data_prep.crop_items)
        for inference_item in inference_items:
            scale = self.nominal_scale

            locvec = numpy.sort(inference_item.loc_pred)
            image_len = inference_item.image_input.shape[-1]

            def _qry_item(orientation, locs):
                qry_item = QryItem()
                qry_item.locs = numpy.sort(locs)
                qry_item.qry = qry_item.locs * scale
                qry_item.scale = scale
                qry_item.orientation = orientation
                qry_item.crop_item = inference_item.crop_item
                qry_item.inference_item = inference_item
                return qry_item

            yield _qry_item(orientation="+", locs=locvec)
            yield _qry_item(orientation="-", locs=numpy.sort(image_len - locvec))

    def crop_inference(self, crop_item):
        image_input = crop_item.segment_image[0]
        target_width = self.localizer_module.image_channels
        source_width = image_input.shape[0] // 2 + 1
        image_input = image_input[source_width - target_width // 2: source_width + target_width // 2 + 1]
        inference_item = self.localizer_module.inference_item(image_input)
        inference_item.crop_item = crop_item
        return inference_item

    def bnx_qry_items(self):
        scale = self.nominal_scale / Config.BIONANO_BNX_SCALE

        for crop_item in self.data_prep.crop_items:
            def _qry_item(orientation, locs):
                qry_item = QryItem()
                qry_item.orientation = orientation
                qry_item.locs = numpy.sort(locs)
                qry_item.qry = qry_item.locs * scale
                qry_item.crop_item = crop_item
                return qry_item

            yield _qry_item(orientation="+", locs=crop_item.original_locs_crop - crop_item.crop_bnx_lims[0])
            yield _qry_item(orientation="-", locs=numpy.sort(crop_item.crop_bnx_lims[1] - crop_item.original_locs_crop))

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
                pass

        if self.parallel:
            for task in tqdm(as_completed(tasks), desc="alignments",
                             total=len(self.data_prep.crop_items) * len(self.refs) * 2):
                try:
                    yield task.result()
                except AssertionError:
                    pass

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

    def make_refs(self):
        self.refs = Series({
            ref_id: ref_df[ref_df["LabelChannel"] == 1]["Position"].values
            for ref_id, ref_df in self.cmap_file_data.file_df.groupby("CMapId")
        })

    def aligner_alignment_items_top(self, qry_items):
        items = self.aligner_alignment_items(qry_items)
        alignment_df = DataFrame(map(vars, items))

        yield from (
            alignment_df["alignment_item"].loc[index] for index in
            alignment_df.groupby("molecule_id")["score"].idxmax()
        )

    def bionano_alignment_items(self):
        self.crops_df = DataFrame([{
            "crop_item": crop_item,
            "molecule_id": crop_item.molecule_id,
        } for crop_item in self.data_prep.crop_items
        ])

        self.xmap_df = self.result_xmap_file_data.file_df

        crops_xmap_join_df = pandas.merge(
            self.crops_df,
            self.xmap_df,
            left_on="molecule_id",
            right_on="QryContigID"
        )

        for _, xmap_record in crops_xmap_join_df.iterrows():
            crop_item: BNXItemCrop = xmap_record["crop_item"]

            xmap_item = XMAPItem()
            xmap_item.xmap_record = xmap_record
            xmap_item.parse_values()

            alignment_item = AlignmentItem()

            alignment_item.molecule_id = crop_item.molecule_id

            alignment_item.ref_id = xmap_item.ref_id
            alignment_item.orientation = xmap_item.orientation
            alignment_item.ref_lims = xmap_item.ref_lims
            alignment_item.qry_lims = xmap_item.qry_lims

            alignment_item.crop_item = crop_item
            alignment_item.xmap_item = xmap_item
            alignment_item.alignment_item = alignment_item
            yield alignment_item

    def run_aligner(self):
        self.aligner_items = list(self.aligner_alignment_items_top(self.localizer_qry_items()))
        print(f"{len(self.aligner_items)=}")

    def run_aligner_bnx(self):
        self.aligner_bnx_items = list(self.aligner_alignment_items_top(self.bnx_qry_items()))
        print(f"{len(self.aligner_bnx_items)=}")

    def run_bionano_refaligner(self):
        self.run_bionano_on_bnx()
        self.read_result_xmap()
        self.bionano_items = list(self.bionano_alignment_items())
        print(f"{len(self.bionano_items)=}")
        self.output_pickle_dump(self.bionano_items, ".bionano.pickle")

    def run_aligners(self):
        with self.executor:
            self.run_aligner()
            self.output_pickle_dump(self.aligner_items, ".aligner.pickle")
            self.run_aligner_bnx()
            self.output_pickle_dump(self.aligner_bnx_items, ".aligner_bnx.pickle")

    def output_pickle_dump(self, obj, suffix):
        file = self.output_file_base.with_suffix(suffix)
        print(file)
        pickle_dump(file, obj)

    def init_run(self):
        self.report.run_name = self.run_name
        self.output_file_base = self.get_output_file_base(self.run_name)
        print(self.output_file_base)
        self.output_file_base.with_suffix(".params.yml").write_text(yaml.dump(self.get_params()))

    def run_bionano_compare_a(self):
        self.init_run()
        self.read_cmap()
        self.make_refs()

        self.data_prep.selector.run_ids = None
        self.data_prep.make_crops()
        self.data_prep.print_crops_report()
        self.run_aligners()

    def run_bionano_compare_b(self):
        self.data_prep.selector.top_mol_num = 8
        self.data_prep.selector.run_ids = numpy.arange(8) + 1
        self.init_run()
        self.read_cmap()
        self.make_refs()

        self.data_prep.make_crops()
        self.data_prep.print_crops_report()
        self.data_prep.make_crops_bnx()
        self.run_bionano_refaligner()
        self.run_aligners()

    def get_params(self):
        params = asdict_recursive(self, include_modules=[self.__module__, bionano_utils, localizer, aligner])
        params = {"commit": git_commit()} | nested_dict_filter_types(params)
        return params


class BionanoCompareReport:
    run_name: str
    confidence_alpha = 0.05
    aligner_items = None
    aligner_bnx_items = None
    bionano_items = None

    def read_compute_results(self):
        file_base = Paths().out_path("bionano_compare", self.run_name, self.run_name)
        try:
            self.aligner_items = pickle_load(file_base.with_suffix(".aligner.pickle"))
        except FileNotFoundError:
            pass
        try:
            self.aligner_bnx_items = pickle_load(file_base.with_suffix(".aligner_bnx.pickle"))
        except FileNotFoundError:
            pass
        try:
            self.bionano_items = pickle_load(file_base.with_suffix(".bionano.pickle"))
        except FileNotFoundError:
            pass

    def overlap_fraction(self, parent_seg, crop_seg):
        (x, y), (a, b) = sorted(parent_seg), sorted(crop_seg)
        assert (a < b) and (x < y)
        intersection = max(0, min(b, y) - max(a, x))
        overlap = intersection / (b - a)
        return overlap

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
            accuracy_item.overlap_fraction = (
                self.overlap_fraction(parent_seg=accuracy_item.parent_ref_lims, crop_seg=alignment_item.ref_lims)
                if accuracy_item.correct_seq else None
            )
            accuracy_item.correct = accuracy_item.correct_seq and (accuracy_item.overlap_fraction > 0)
            accuracy_item.alignment_item = alignment_item
            accuracy_item.accuracy_item = accuracy_item
            yield accuracy_item

    def plot_accuracy(self, accuracy_items, label):
        df = DataFrame(map(vars, accuracy_items))
        correct = df.groupby("len_bp")["correct"]
        num_successes = correct.sum()
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

    def plot_bionano_accuracy(self):
        self.bionano_accuracy_items = list(self.accuracy_items(self.bionano_items))
        self.bionano_accuracy = self.plot_accuracy(self.bionano_accuracy_items, label="Bionano Localizer + Aligner")

    def plot_aligner_bnx_accuracy(self):
        self.aligner_bnx_accuracy_items = list(self.accuracy_items(self.aligner_bnx_items))
        self.aligner_bnx_accuracy = self.plot_accuracy(self.aligner_bnx_accuracy_items,
                                                       label="Bionano Localizer + DeepOM Aligner")

    def plot_aligner_accuracy(self):
        self.aligner_accuracy_items = list(self.accuracy_items(self.aligner_items))
        self.aligner_accuracy = self.plot_accuracy(self.aligner_accuracy_items, label="DeepOM")

    def plot_a(self):
        self.plot_compare_init()
        self.plot_aligner_accuracy()
        self.plot_aligner_bnx_accuracy()

    def plot_b(self):
        self.plot_compare_init()
        self.plot_aligner_accuracy()
        self.plot_aligner_bnx_accuracy()
        self.plot_bionano_accuracy()


if __name__ == '__main__':
    BionanoCompare().run_bionano_compare_a()
    BionanoCompare().run_bionano_compare_b()
