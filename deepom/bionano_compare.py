from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
import subprocess

from matplotlib.ticker import PercentFormatter
from sqlalchemy import create_engine
import napari
from scipy.interpolate import interp1d
from statsmodels.stats.proportion import proportion_confint

from deepom.aligner import SpAligner
from deepom.localizer import LocalizerModule, LocalizerOutputItem
from om_decoder.utils import *
from om_decoder.optical_mapper import *
from om_decoder.utils_cached import read_jxr, read_jxr_segment


def parse_bionano_file_line(row):
    return [
        _.strip() for _ in row.strip().split("\t")
    ]


class XMAPItem:
    xmap_record: dict
    xmap_row: tuple
    bnx_scale: float = Config.BIONANO_BNX_SCALE

    def parse_values(self):
        self.ref_id = self.xmap_record["RefContigID"]
        self.orientation = self.xmap_record["Orientation"]
        self.alignment = self.xmap_record["Alignment"]
        self.alignment_df = DataFrame(re.findall("\((.*?),(.*?)\)", self.alignment),
                                      columns=["Ref", "Query"]).astype(int)
        self.qry_lims = self.xmap_record[["QryStartPos", "QryEndPos"]].astype(float).sort_values().values
        self.ref_lims = self.xmap_record[["RefStartPos", "RefEndPos"]].astype(float).sort_values().values

        self.scale = (self.ref_lims[1] - self.ref_lims[0]) / ((self.qry_lims[1] - self.qry_lims[0]) / self.bnx_scale)

    def query_to_ref_interp(self, query_locs):
        self.alignment_intep_func = interp1d(
            self.qry_lims,
            self.ref_lims,
            fill_value="extrapolate"
        )
        ref_sites = self.alignment_intep_func(numpy.stack(query_locs)).astype(int)
        return ref_sites


class BionanoRefAlignerRun:
    bnx_text: str
    ref_file: str = "/home/ynogin/data/bionano_data/refaligner_data/hg38_DLE1_0kb_0labels.cmap"
    refaligner_exe = "/home/ynogin/bionano_sw/tools/pipeline/Solve3.7_03302022_283/RefAligner/1.0/RefAligner"
    """    
    -T <Pvalue> : only use alignments with Pvalue below this threshold [Default 0.0001]
    -A <Aligned-Sites> : only use alignments with at least this many sites [Default 8]
    -L <Aligned-Length> : only use alignments with at least this length (in kb) aligned [Default 0]. NOTE: Does not yet apply to pairwise alignment
    -BestRef <0/1> : 0 means allow Molecules to align to multiple -ref contigs, 1 means allow Molecules to align to best -ref contig only. 
    -usecolor <N> : If -i input data has 2 colors only the <N> th color is used as if only that one color were present (but output BNX or _q.cmap will include both colors).
    -FP <False Positives per 100kb, one per color>[Default 1.0]
    -FN <False Negative Probability, one per color>[Default 0.15]
    -sd <sizing error in root-kb, one per color>[Default 0.15]
    -sf <fixed sizing error in kb, one per color>[Default 0.10]
    -sr <relative sizing error, one per color>[Default 0.00]
    -se <resolution limited sizing error, one per color>[Default 0.00]
    -res <r> : r is Resolution in pixels, one value per color [Default 3.5] : NOTE : Depends on bpp value
    -resSD <rs> : rs is standard deviation of resolution in pixels, one value per color [Default 0.75] : NOTE : Depends on bpp value
    """
    options = "-f -BestRef 1 -usecolor 1 -A 2 -T 1 -S -1000"  # -FP 2.0 -FN 0.10 -sf 0.25 -sd 0.11 -sr 0.03 -res 3.1 -nosplit 2"

    def run_refaligner(self):
        self.run_name = timestamp_str_iso_8601()
        self.run_dir = Paths().out_path_mkdir(f"refaligner_out/{self.run_name}")

        self.bnx_file = self.run_dir / f"{self.run_name}.bnx"
        self.bnx_file.write_bytes(self.bnx_text.encode())

        self.out_prefix = self.run_dir / self.run_name
        self.xmap_file = self.out_prefix.with_suffix(".xmap")

        assert Path(self.bnx_file).exists()
        assert Path(self.refaligner_exe).exists()
        assert Path(self.ref_file).exists()

        self.cmd = f"{self.refaligner_exe} -i {self.bnx_file} -ref {self.ref_file} -o {self.out_prefix} {self.options}"
        print(self.cmd)

        progress = tqdm(desc="refaligner", total=1)
        self.completed_process = subprocess.run(self.cmd.split(), capture_output=True)
        progress.update()

        self.stdout = self.completed_process.stdout
        self.stdout_file = self.out_prefix.with_suffix(".stdout.txt").write_bytes(self.stdout)
        assert self.xmap_file.exists()


class BNXItem:
    bnx_record: Series
    xmap_item: XMAPItem = None
    bionano_image: 'BionanoImage'
    locs: ndarray
    bnx_scale = Config.BIONANO_BNX_SCALE
    meta_fields: list
    block_fields: list

    def __init__(self):
        self.molecule_id = None

    def plot_bnx_item(self):
        pyplot.subplots(figsize=(30, 3))
        im, = self.bionano_image.segment_image
        pyplot.gca().pcolorfast((0, im.shape[-1]), (-.5, .5), im, cmap='gray')
        pyplot.eventplot(self.locs / self.bnx_scale,
                         lineoffsets=-.2, linelengths=.4, colors='r', alpha=.5)
        pyplot.grid(False)

    def bnx_to_text(self):
        lines_data = [
            bnx_values
            for line_id, bnx_values in self.bnx_record[self.block_fields].items()
        ]
        return "\n".join(lines_data)

    def parse_bnx(self, row):
        self.molecule_id = row["MoleculeID"]
        self.locs = Series(parse_bionano_file_line(row["1"])[1:-1]).astype(float).values
        self.bnx_record = row

        self.bionano_image = BionanoImage()
        self.bionano_image.bnx_item = self
        self.bionano_image.read_bionano_image()

        self.xmap_item = XMAPItem()
        self.xmap_item.xmap_record = self.bnx_record
        self.xmap_item.parse_values()


class BionanoImage:
    bnx_item: BNXItem
    bnx_read: 'BNXFileData'
    fov_size = 2048
    segment_width = 17

    def parse_segment_endpoints(self):
        start_fov, stop_fov = self.bnx_item.bnx_record[["StartFOV", "EndFOV"]].astype(int).values
        start_y, start_x = self.bnx_item.bnx_record[["StartY", "StartX"]].astype(float).values
        start_y = start_y + (start_fov - 1) * self.fov_size
        stop_y, stop_x = self.bnx_item.bnx_record[["EndY", "EndX"]].astype(float).values
        stop_y = stop_y + (stop_fov - 1) * self.fov_size

        self.endpoints = numpy.stack([
            [start_y, start_x],
            [stop_y, stop_x],
        ])

    def read_jxr_image(self):
        self.fov_image = cached_func(read_jxr)(self.bnx_item.bnx_record["JXRFile"])

    def read_segment_image(self):
        self.segment_image = cached_func(read_jxr_segment)(file=self.bnx_item.bnx_record["JXRFile"],
                                                           endpoints=self.endpoints,
                                                           segment_width=self.segment_width)

    def read_bionano_image(self):
        self.parse_segment_endpoints()
        self.read_segment_image()

    def napari_show_segment(self):
        viewer = napari.Viewer()
        viewer.add_image(self.fov_image)
        viewer.add_vectors(
            numpy.stack([
                self.endpoints[0],
                self.endpoints[1] - self.endpoints[0]
            ])[None, :],
            opacity=.2,
            edge_width=11,
        )
        viewer.add_image(self.segment_image)


class MoleculeSelector:
    xmap_file = "/home/ynogin/data/bionano_data/bionano_run_data/exp_refineFinal1.xmap"
    jxr_root = "/home/ynogin/mnt/Q/Yevgeni/bionano_jxr/"
    xmap_file_data: 'BionanoFileData'
    bnx_file_data: 'BNXFileData'
    top_mol_num: int = 512
    top_mols_by: str = "Confidence"
    same_fov = True
    min_qry_len = 450 * 1000

    molecule_ids = None
    min_len = None
    min_confidence = None
    run_ids = None
    min_num_labels = None
    min_snr = None
    max_id = None
    scan_ids = None
    jxr_channel = 3

    def __init__(self):
        self.bnx_file_data = BNXFileData()
        self.xmap_file_data = BionanoFileData()

    def read_files(self):
        self.bnx_file_data.parse_header()
        self.xmap_file_data.file = self.xmap_file
        self.xmap_file_data.read_bionano_file()

    def select_molecules_df(self):
        self.read_files()
        xmap_df = self.xmap_file_data.file_df

        #
        # if self.min_confidence is not None:
        #     xmap_df = xmap_df[xmap_df["Confidence"].astype(float) >= self.min_confidence]
        #
        if self.min_qry_len is not None:
            xmap_df = xmap_df[xmap_df["QryLen"].astype(float) >= self.min_qry_len]

        ids = xmap_df["QryContigID"].tolist()
        bnx_df = self.bnx_file_data.read_db(ids=ids)
        df = pandas.merge(
            bnx_df, xmap_df,
            left_on="MoleculeID", right_on="QryContigID"
        )

        meta = df["0"].str.split("\t", expand=True)
        meta.columns = self.bnx_file_data.names
        meta = meta.astype(dict(zip(meta.columns, self.bnx_file_data.dtypes)))

        df = pandas.merge(
            meta, df, how="inner",
            left_on="MoleculeID",
            right_on="MoleculeID",
            suffixes=("", "_XMAP")
        )

        df = pandas.merge(
            df.astype({"RunId": int}),
            self.bnx_file_data.runs_df.astype({"RunId": int}),
            how="inner",
            left_on="RunId",
            right_on="RunId",
            suffixes=("", "_RUNDATA")
        )

        if self.same_fov:
            df = df[df["StartFOV"] == df["EndFOV"]]

        if self.scan_ids is not None:
            df = df[df["Scan"].astype(int).isin(self.scan_ids)]

        if self.run_ids is not None:
            df = df[df["RunId"].astype(int).isin(self.run_ids)]

        if self.molecule_ids is not None:
            df = df[df["MoleculeID"].astype(int).isin(self.molecule_ids)]

        if self.min_len is not None:
            df = df[df["Length"].astype(float) >= self.min_len]

        if self.max_id is not None:
            df = df[df["MoleculeID"].astype(int) <= self.max_id]

        if self.min_num_labels is not None:
            df = df[df["NumberofLabels"].astype(int) >= self.min_num_labels]

        if self.min_snr is not None:
            df = df[df["SNR"].astype(float) >= self.min_snr]

        if self.top_mols_by is not None:
            df = df.nlargest(n=self.top_mol_num, columns=self.top_mols_by)

        return df

    def parse_jxr_files(self, df):
        def _jxr_file(row):
            Channel = self.jxr_channel
            Scan = row["Scan"]
            Bank = row["Bank"]
            C_digits = int(row["Column"])
            ChipId = row["ChipId"].split(",")[-2].lstrip("Run_")
            jxr_file = Path(self.jxr_root,
                            f"{ChipId}/FC{row['Flowcell']}/Scan{Scan}/Bank{Bank}/B{Bank}_CH{Channel}_C{C_digits:03d}.jxr")
            return jxr_file

        return df.apply(_jxr_file, axis=1)

    def select_molecules(self):
        df = self.select_molecules_df()
        df["JXRFile"] = self.parse_jxr_files(df)
        df["JXRFileExists"] = df["JXRFile"].apply(Path.exists)
        df = df[df["JXRFileExists"]]
        self.selected_df = df

        def _items():
            for _, row in df.iterrows():
                bnx_item = BNXItem()
                bnx_item.meta_fields = self.bnx_file_data.names
                bnx_item.block_fields = self.bnx_file_data.block_fields
                bnx_item.parse_bnx(row)
                yield bnx_item

        self.selected = list(tqdm(_items(), desc="selected"))


class BNXItemCrop:
    bnx_scale = Config.BIONANO_BNX_SCALE
    segment_image: ndarray
    crop_size_pixels: int
    crop_size_bp: int
    min_locs: int = 3

    def __init__(self,
                 parent_bnx_item: 'BNXItem',
                 crop_lims: tuple[int, int],
                 molecule_id: int,
                 ):
        self.parent_bnx_item = copy(parent_bnx_item)

        self.crop_lims = crop_lims
        self.molecule_id = molecule_id

        self.original_locs_crop = locs = self.parent_bnx_item.locs
        QX11_values = parse_bionano_file_line(self.parent_bnx_item.bnx_record["QX11"])[1:]
        QX12_values = parse_bionano_file_line(self.parent_bnx_item.bnx_record["QX12"])[1:]
        Length = self.parent_bnx_item.bnx_record["Length"]

        if self.crop_lims is not None:
            self.crop_bnx_lims = numpy.stack(self.crop_lims) * self.bnx_scale

            assert len(locs) >= self.min_locs
            assert is_sorted(locs)

            slice_start, slice_stop = locs.searchsorted(self.crop_bnx_lims)

            self.original_locs_crop = locs = locs[slice_start: slice_stop]
            QX11_values = QX11_values[slice_start: slice_stop]
            QX12_values = QX12_values[slice_start: slice_stop]

            locs = locs - self.crop_bnx_lims[0]
            Length = self.crop_bnx_lims[1] - self.crop_bnx_lims[0]

        assert len(locs) >= self.min_locs

        self.bnx_item = copy(self.parent_bnx_item)
        self.bnx_item.locs = locs
        self.bnx_item.bnx_record = self.parent_bnx_item.bnx_record.copy().astype(object)
        meta = {**self.bnx_item.bnx_record} | {
            "Length": int(Length),
            "NumberofLabels": len(locs),
            "MoleculeID": int(self.molecule_id),
            "OriginalMoleculeId": int(self.parent_bnx_item.molecule_id),
        }
        bnx_block = {
            "0": Series(meta)[self.parent_bnx_item.meta_fields].astype(str)[1:],
            "1": [f"{_:.2f}" for _ in [*locs, float(Length)]],
            "2": [f"{float(Length):.2f}"],
            "QX11": QX11_values,
            "QX21": "",
            "QX12": QX12_values,
            "QX22": "",
        }
        bnx_block = {k: "\t".join([k, *v]) for k, v in bnx_block.items()}
        self.bnx_item.bnx_record.update(meta | bnx_block)

        self.pixel_locs = self.bnx_item.locs / self.bnx_scale

    def plot_crop(self):
        pyplot.subplots(figsize=(20, 5))
        im, = self.parent_bnx_item.bionano_image.segment_image
        pyplot.gca().pcolorfast((0, im.shape[-1]), (0, 1), im, cmap='gray')
        im, = self.segment_image
        pyplot.gca().pcolorfast(self.crop_lims, (1, 2), im, cmap='gray')
        pyplot.eventplot([self.parent_bnx_item.locs / self.bnx_scale, self.pixel_locs + self.crop_lims[0]],
                         lineoffsets=[.5, 1.5], linelengths=.5, colors=['r', 'g'], alpha=.5)
        pyplot.grid(False)


class BNXFileData:
    bnx_file: str = "/home/ynogin/data/bionano_data/bionano_run_data/T1_chip2_channels_swapped.bnx"
    db_file = "/home/ynogin/data/bionano_data/bnx.db"
    block_size = 7
    chunk_size = 10 ** 4
    index_name = "MoleculeID"
    block_fields = ["0", "1", "2", "QX11", "QX21", "QX12", "QX22"]

    def parse_header(self):
        self.header_lines = []
        run_data_lines = []
        for i, line in enumerate(open(self.bnx_file)):
            if line.startswith("#"):
                self.header_lines.append(line)
                if line.startswith("#rh"):
                    self.runs_cols = parse_bionano_file_line(line.lstrip("#rh"))
                elif line.startswith("#0h"):
                    self.names = parse_bionano_file_line(line.lstrip("#0h"))
                elif line.startswith("#0f"):
                    self.dtypes = parse_bionano_file_line(line.lstrip("#0f"))
                elif line.startswith("# Run Data"):
                    run_data_lines.append(parse_bionano_file_line(line)[1:])
            else:
                self.num_header_lines = i
                break

        runs_df = DataFrame(run_data_lines, columns=self.runs_cols)
        runs_df = runs_df.join(runs_df["SourceFolder"].str.extract(r"Cohort(?P<Scan>\d\d)(?P<Bank>\d)(?P<Cohort>\d)"))
        self.runs_df = runs_df

    def write_db(self, limit_chunks=None):
        lines = itertools.islice(open(self.bnx_file), self.num_header_lines, None)
        blocks = more_itertools.chunked(lines, self.block_size)
        chunks = more_itertools.chunked(blocks, self.chunk_size)
        chunks = itertools.islice(chunks, limit_chunks)

        engine = self._sql_engine()
        engine.execute("DROP TABLE IF EXISTS bnx")

        for blocks in tqdm(chunks):
            df = DataFrame(iter(blocks))
            df.insert(0, self.index_name, df[0].str.extract("0\t(\d*?)\t")[0].astype(int).values)
            df.to_sql("bnx", con=engine, index=False, if_exists="append")
        engine.execute(f"CREATE UNIQUE INDEX {self.index_name} ON bnx({self.index_name})")

    def read_db(self, ids):
        q = ','.join('?' * len(ids))
        query = f"SELECT * FROM bnx WHERE {self.index_name} in ({q})"
        params = ids
        df = self.read_sql_query(query, params=params)
        df = df.rename(columns={str(k): v for k, v in enumerate(self.block_fields)})
        return df

    def read_sql_query(self, *args, **kwargs):
        df = pandas.read_sql_query(*args, **kwargs, con=self._sql_engine())
        return df

    def read_ids_range(self):
        df = pandas.read_sql_query(
            f"SELECT min({self.index_name}), max({self.index_name}), COUNT(DISTINCT  {self.index_name}) FROM bnx")
        return df

    def _sql_engine(self):
        engine = create_engine(f'sqlite:///{self.db_file}', echo=False)
        return engine


class DataPrep:
    rng = default_rng(seed=0)
    crop_size_range_bp = 15 * 1000, 450 * 1000
    num_crops_per_size = 128
    num_sizes = 24
    nominal_scale = Config.BIONANO_NOMINAL_SCALE

    def __init__(self):
        self.selector = MoleculeSelector()

    def crops_df(self):
        return DataFrame(map(vars, self.crop_items))

    def make_crops(self):
        self.selector.select_molecules()
        assert self.num_crops_per_size <= len(self.selector.selected)

        self.crop_sizes_bp = numpy.geomspace(*self.crop_size_range_bp, self.num_sizes)[::-1]
        print("crop_sizes", self.crop_sizes_bp)

        def _crop_items():
            molecule_ids = count(1)
            for crop_size_bp in self.crop_sizes_bp:
                bnx_items = self.rng.choice(self.selector.selected, size=self.num_crops_per_size, replace=False)
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

        if crop_size < image_len:
            start = self.rng.integers(0, image_len - crop_size)
            stop = start + crop_size
            segment_image = segment_image[..., start: stop]
            crop_lims = (start, stop)
        else:
            crop_lims = None

        crop = BNXItemCrop(
            parent_bnx_item=bnx_item, crop_lims=crop_lims, molecule_id=molecule_id
        )
        crop.crop_size_pixels = crop_size
        crop.crop_size_bp = int(segment_image.shape[-1] * self.nominal_scale)
        crop.segment_image = segment_image
        return crop

    def print_crops_report(self):
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


class BionanoFileData:
    file: str
    nrows = None

    def read_bionano_header(self):
        for row in open(self.file):
            if row.startswith("#"):
                if row.startswith("#h"):
                    self.names = parse_bionano_file_line(row.lstrip("#h"))
                elif row.startswith("#f"):
                    self.dtypes = parse_bionano_file_line(row.lstrip("#f"))
            else:
                break

    def read_bionano_file(self):
        self.read_bionano_header()
        self.file_df = cached_func(pandas.read_csv, verbose=False)(
            self.file, sep="\t", comment="#",
            names=self.names,
            dtype=dict(zip(self.names, self.dtypes)),
            nrows=self.nrows,
        )


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
    cmap_filepath = "/home/ynogin/data/bionano_data/refaligner_data/hg38_DLE1_0kb_0labels.cmap"
    spaligner_use_bnx_locs = False
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
        self.localizer_module.checkpoint_search_dir = "LocalizerModule"
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

    def spaligner_alignment_items(self, qry_items):
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
        aligner = SpAligner()
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

    def spaligner_alignment_items_top(self, qry_items):
        items = self.spaligner_alignment_items(qry_items)
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

    def run_spaligner(self):
        self.spaligner_items = list(self.spaligner_alignment_items_top(self.localizer_qry_items()))
        print(f"{len(self.spaligner_items)=}")

    def run_spaligner_bnx(self):
        self.spaligner_bnx_items = list(self.spaligner_alignment_items_top(self.bnx_qry_items()))
        print(f"{len(self.spaligner_bnx_items)=}")

    def run_bionano(self):
        self.run_bionano_on_bnx()
        self.read_result_xmap()
        self.bionano_items = list(self.bionano_alignment_items())
        print(f"{len(self.bionano_items)=}")
        pickle_dump(self.output_file_base.with_suffix(".bionano.pickle"), self.bionano_items)

    def run_aligners(self):
        with self.executor:
            # self.executor.submit(self.run_bionano)
            self.run_spaligner()
            pickle_dump(self.output_file_base.with_suffix(".spaligner.pickle"), self.spaligner_items)
            self.run_spaligner_bnx()
            pickle_dump(self.output_file_base.with_suffix(".spaligner_bnx.pickle"), self.spaligner_bnx_items)

    def run_data_prep(self):
        self.read_cmap()
        self.make_refs()
        self.data_prep.make_crops()
        # self.write_plot_crops()
        pickle_dump(self.output_file_base.with_suffix(".selected.pickle"), self.data_prep.selector.selected)
        pickle_dump(self.output_file_base.with_suffix(".crop_items.pickle"), self.data_prep.crop_items)
        self.data_prep.print_crops_report()
        self.data_prep.make_crops_bnx()

    def init_run(self):
        self.report.run_name = self.run_name
        self.output_file_base = self.get_output_file_base(self.run_name)
        print(self.output_file_base)
        self.output_file_base.with_suffix(".params.yml").write_text(yaml.dump(self.get_params()))

    def run(self):
        self.init_run()
        self.run_data_prep()
        self.run_aligners()
        self.report.plot_results()

    def get_params(self):
        params = asdict_recursive(self, include_modules=[self.__module__])
        params = {"commit": git_commit()} | nested_dict_filter_types(params)
        return params


class BionanoCompareReport:
    run_name: str
    confidence_alpha = 0.05

    def __init__(self):
        self.spaligner_bnx_items = None
        self.bionano_items = None

    def plot_results(self):
        self.read_compute_results()
        self.compute_accuracy()
        self.plot_compare()
        self.write_report()

    def read_report(self):
        self.output_file_base = BionanoCompare.get_output_file_base(self.run_name)
        return pandas.read_csv(self.output_file_base.with_suffix(".csv"))

    def write_report(self):
        pyplot.tight_layout()
        self.output_file_base = BionanoCompare.get_output_file_base(self.run_name)
        figure_file = self.output_file_base.with_suffix(".png")
        print(figure_file)

        pyplot.savefig(figure_file)

        # df = self.spaligner_accuracy.to_frame("spaligner_accuracy")
        # df["bionano_accuracy"] = self.bionano_accuracy
        # df.to_csv(self.output_file_base.with_suffix(".csv"))

    def write_plot_crops(self):
        self.output_file_base = BionanoCompare.get_output_file_base(self.run_name)
        for i, crop_item in enumerate(tqdm(self.data_prep.crop_items, desc="plot crops")):
            crop_item.plot_crop()
            figure_file = (self.output_file_base.parent / "crops" / str(i)).with_suffix(".jpg")
            figure_file = Paths().out_file_mkdir(figure_file)
            pyplot.title(crop_item.molecule_id)
            pyplot.savefig(figure_file)
            pyplot.close()

    def read_compute_results(self):
        file_base = Paths().out_file_mkdir("bionano_compare", self.run_name, self.run_name)
        self.spaligner_items = pickle_load(file_base.with_suffix(".spaligner.pickle"))
        try:
            self.spaligner_bnx_items = pickle_load(file_base.with_suffix(".spaligner_bnx.pickle"))
        except FileNotFoundError:
            pass
        try:
            self.bionano_items = pickle_load(file_base.with_suffix(".bionano.pickle"))
        except FileNotFoundError:
            pass

    # def compute_accuracy(self):
    # self.spaligner_accuracy_items = self._filter_items()

    #
    # def _filter_items(self):
    #     spaligner_accuracy_items = [
    #         item for item in self.spaligner_accuracy_items
    #         if item.alignment_item.crop_item.molecule_id in
    #            {item.alignment_item.crop_item.molecule_id for item in self.bionano_accuracy_items}
    #     ]
    #     return spaligner_accuracy_items

    def overlap_fraction(self, parent_seg, crop_seg):
        (x, y), (a, b) = sorted(parent_seg), sorted(crop_seg)
        assert (a < b) and (x < y)
        intersection = max(0, min(b, y) - max(a, x))
        iou = intersection / (b - a)
        return iou

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
        pyplot.plot(len_bp, acc.values, '.-', label=label)

        ci = numpy.stack([
            proportion_confint(count=count, nobs=nobs, alpha=self.confidence_alpha, method='beta')
            for count, nobs in zip(num_successes, num_observarions)
        ])
        pyplot.fill_between(len_bp, ci.T[0], ci.T[1], alpha=0.2)
        return acc

    def plot_compare(self):
        # pyplot.figure(figsize=(8, 5), dpi=400, facecolor="w")
        ax = pyplot.gca()
        #
        # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20000))
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, y: int(x // 1000)))
        #
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        #
        # pyplot.grid()

        self.spaligner_accuracy_items = list(self.accuracy_items(self.spaligner_items))
        if self.spaligner_bnx_items is not None:
            self.spaligner_bnx_accuracy_items = list(self.accuracy_items(self.spaligner_bnx_items))
            self.spaligner_accuracy = self.plot_accuracy(self.spaligner_accuracy_items,
                                                         label="DeepOM")
            self.spaligner_bnx_accuracy = self.plot_accuracy(self.spaligner_bnx_accuracy_items,
                                                             label="Bionano Localizer")
        else:
            self.spaligner_accuracy = self.plot_accuracy(self.spaligner_accuracy_items, label="DeepOM")
            if self.bionano_items is not None:
                self.bionano_accuracy_items = list(self.accuracy_items(self.bionano_items))
                self.bionano_accuracy = self.plot_accuracy(self.bionano_accuracy_items, label="Bionano")

        pyplot.xlabel("Fragment Length (kb)")
        pyplot.ylabel("Success Rate")
        pyplot.legend()


if __name__ == '__main__':
    fire.Fire()
