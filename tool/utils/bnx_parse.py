from typing import TextIO

import more_itertools
import numpy as np
import pandas as pd
from pydantic import BaseModel

from utils.pyutils import NDArray


class BNXParser:
    BNX_LINES_PER_MOLECULE = 7
    RUN_DATA = "# Run Data"

    class BNXRun(BaseModel):
        RunId: str
        SourceFolder: str
        InstrumentSerial: str
        Time: str
        NanoChannelPixelsPerScan: str
        StretchFactor: float
        BasesPerPixel: str
        NumberofScans: int
        ChipId: str
        FlowCell: str
        SNRFilterType: str
        MinMoleculeLength: float
        MinLabelSNR1: float
        MinLabelSNR2: float

        Scan: str = None
        Bank: str = None
        Cohort: str = None

    class BNXMolecule(BaseModel):
        BNXLocalizations: NDArray

        #  BNX file format:
        AvgIntensity: float
        ChipId: str
        Column: int
        EndFOV: int
        EndX: int
        EndY: int
        Flowcell: str
        GlobalScanNumber: int
        LabelChannel: int
        Length: float
        MoleculeID: int
        NumberofLabels: int
        OriginalMoleculeId: int
        RunId: str
        SNR: float
        ScanDirection: str
        ScanNumber: int
        StartFOV: int
        StartX: int
        StartY: int

    def parse_block(self, block: list[str]):
        metadata, data = [line.strip().split() for line in block[:2]]
        return self.BNXMolecule(
            BNXLocalizations=np.asarray(data)[1:].astype(float),
            **dict(zip(self.bnx_columns, metadata)),
        ).dict()

    def iter_blocks(self, bnx_buffer: TextIO):
        yield from more_itertools.chunked(self._lines(bnx_buffer), self.BNX_LINES_PER_MOLECULE)

    def iter_molecules(self, bnx_buffer: TextIO):
        yield from map(self.parse_block, self.iter_blocks(bnx_buffer))

    def parse_runs(self, bnx_buffer: TextIO) -> dict:
        next(iter(self.iter_blocks(bnx_buffer)))
        return pd.DataFrame(self.run_data_lines, columns=self.run_columns).to_dict()

    def _lines(self, reader: TextIO):
        self.run_data_lines = []
        for line in reader:
            name, *data = line.strip().split()
            if name.startswith("#"):
                if line.startswith(self.RUN_DATA):
                    self.run_data_lines.append(line.lstrip(self.RUN_DATA).strip().split('\t'))
                elif name == "#rh":
                    self.run_columns = data
                elif name == "#0h":
                    self.bnx_columns = data
                elif name == "#0f":
                    self.bnx_dtypes = data
            else:
                self.dtypes = dict(zip(self.bnx_columns, self.bnx_dtypes))
                assert self.run_columns is not None
                assert self.bnx_columns is not None
                assert self.bnx_dtypes is not None
                yield line
                yield from reader
