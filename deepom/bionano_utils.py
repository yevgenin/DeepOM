import itertools
import re
import subprocess
from copy import copy
from pathlib import Path

import more_itertools
import napari
import numpy
import pandas
from matplotlib import pyplot
from numpy import ndarray
from pandas import DataFrame, Series
from sqlalchemy import create_engine
from tqdm.auto import tqdm
from IPython.display import display

from deepom.config import Config
from deepom.utils import timestamp_str_iso_8601, Paths, cached_func, is_sorted
from deepom.utils_cached import read_jxr, read_jxr_segment
from deepom.localizer import SimulatedDataItem


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


class BionanoRefAlignerRun:
    bnx_text: str
    ref_file: str = Config.REF_CMAP_FILE
    refaligner_exe = Config.REFALIGNER
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
    options = "-f -BestRef 1 -usecolor 1 -A 2 -T 1 -S -1000"

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

    def parse_bnx(self, row, read_image=True):
        self.molecule_id = row["MoleculeID"]
        self.locs = Series(parse_bionano_file_line(row["1"])[1:-1]).astype(float).values
        self.bnx_record = row

        self.bionano_image = BionanoImage()
        self.bionano_image.bnx_item = self
        self.bionano_image.read_bionano_image()

        self.xmap_item = XMAPItem()
        self.xmap_item.xmap_record = self.bnx_record
        self.xmap_item.parse_values()

        self.bionano_image = BionanoImage()
        self.bionano_image.bnx_item = self

        if read_image:
            self.bionano_image.read_bionano_image()
    
    def plot_simulated(self):
        pyplot.subplots(figsize=(30, 3))
        im, = self.bionano_image.segment_image
        pyplot.gca().pcolorfast((0, im.shape[-1]), (-.5, .5), im, cmap='gray')
        pyplot.eventplot(self.simulated.fragment_sitevec / self.simulated.scale + self.simulated.strand_image_offset[0],
                         lineoffsets=-.2, linelengths=.4, colors='r', alpha=.5)
        pyplot.grid(False)

        
    def make_simulated_image(self, refs: dict, rng):
        xmap_item = self.xmap_item
        simulated = SimulatedDataItem()
        simulated.rng = rng
        simulated.label_eff = 1
        simulated.lat_size_min = 9
        simulated.stray_density = 1e-9
        simulated.scale = Config.BIONANO_NOMINAL_SCALE
        reference_lims = xmap_item.ref_lims
        reference_positions = refs[xmap_item.ref_id]
        start, stop = reference_positions.searchsorted(reference_lims)
        reference_positions = reference_positions[start:stop + 1]
        if xmap_item.orientation == "+":
            simulated.fragment_sitevec = reference_positions - reference_positions[0]
        elif xmap_item.orientation == "-":
            simulated.fragment_sitevec = numpy.sort(reference_positions[-1] - reference_positions)
        else:
            raise ValueError
        simulated.xmap_item = xmap_item
        simulated.make_params()
        simulated.make_fragment_image()
        self.simulated = simulated
        self.bionano_image.segment_image = simulated.image

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
    xmap_file = Config.XMAP_FILE
    bionano_images_dir = Config.BIONANO_IMAGES_DIR
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
    

    def __init__(self, filter_data=False, filter_ids=[],read_images=True):
        self.bnx_file_data = BNXFileData()
        self.xmap_file_data = BionanoFileData()
        self.filter_data = filter_data
        self.filter_ids = filter_ids
        self.simulated_mode = False
        self.read_images = read_images

    def read_files(self):
        self.bnx_file_data.parse_header()
        self.xmap_file_data.file = self.xmap_file
        self.xmap_file_data.read_bionano_file()

    def select_molecules_df(self):
        self.read_files()
        xmap_df = self.xmap_file_data.file_df

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
            jxr_file = Path(self.bionano_images_dir,
                            f"{ChipId}/FC{row['Flowcell']}/Scan{Scan}/Bank{Bank}/B{Bank}_CH{Channel}_C{C_digits:03d}.jxr")
            return jxr_file

        return df.apply(_jxr_file, axis=1)

    def select_molecules(self):
        df = self.select_molecules_df()
        df["JXRFile"] = self.parse_jxr_files(df)
        df["JXRFileExists"] = df["JXRFile"].apply(Path.exists)

        assert df["JXRFileExists"].all()

        df = df[df["JXRFileExists"]]

        self.selected_df = df

        def _items():
            for _, row in df.iterrows():
                bnx_item = BNXItem()
                bnx_item.meta_fields = self.bnx_file_data.names
                bnx_item.block_fields = self.bnx_file_data.block_fields
                bnx_item.parse_bnx(row,read_image=self.read_images)
                if self.filter_data is False:
                    yield bnx_item
                elif (bnx_item.xmap_item.ref_id in self.filter_ids):
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
        assert is_sorted(locs)

        QX11_values = parse_bionano_file_line(self.parent_bnx_item.bnx_record["QX11"])[1:]
        QX12_values = parse_bionano_file_line(self.parent_bnx_item.bnx_record["QX12"])[1:]
        Length = self.parent_bnx_item.bnx_record["Length"]

        if self.crop_lims is not None:
            self.crop_bnx_lims = numpy.stack(self.crop_lims) * self.bnx_scale

            assert len(locs) >= self.min_locs

            slice_start, slice_stop = locs.searchsorted(self.crop_bnx_lims)

            self.original_locs_crop = locs = locs[slice_start: slice_stop]
            QX11_values = QX11_values[slice_start: slice_stop]
            QX12_values = QX12_values[slice_start: slice_stop]

            locs = locs - self.crop_bnx_lims[0]
            Length = self.crop_bnx_lims[1] - self.crop_bnx_lims[0]
        else:
            self.crop_bnx_lims = locs[[0, -1]]

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
    bnx_file: str = Config.BNX_FILE
    db_file = Config.BNXDB_FILE
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
