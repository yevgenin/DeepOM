from typing import Optional

import imagecodecs
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from devtools import debug
from pydantic import BaseModel

from utils.env import ENV, joblib_memory
from utils.bnx_parse import BNXParser
from utils.gcs_utils import any_file_read_bytes
from utils.pyutils import NDArray
from utils.image_utils import extract_segment_from_endpoints


class ImageReader:
    class Config(BaseModel):
        bionano_images_dir = ENV.BIONANO_IMAGES_DIR
        bnx_channel = 3
        fov_size = 2048
        segment_width = 11

    class Input(BNXParser.BNXMolecule):
        pass

    class Output(BaseModel):
        # multi-channel image with dimensions (channels, height, width)
        image: Optional[NDArray]
        fov_file: str = None
        error: str = None

    class Info(Output):
        fov: NDArray
        endpoints: NDArray

    class Error(Output):
        image: NDArray = None

    def __init__(self, runs: dict, **config):
        self.config = self.Config(**config)
        debug(self.config)
        self.runs = self._parse_runs(runs)

    def _parse_runs(self, runs):
        df = pd.DataFrame(runs)
        # parse the SourceFolder column into Scan, Bank, Cohort
        df = df.join(df["SourceFolder"].str.extract(r"Cohort(?P<Scan>\d\d)(?P<Bank>\d)(?P<Cohort>\d)"))
        data = df.set_index("RunId", drop=False).T.to_dict()
        # return a dict of BNXRun objects indexed by RunId
        return {key: BNXParser.BNXRun(**value).dict() for key, value in data.items()}

    def read_image(self, molecule: dict):
        return self.Output(**self._read_image(self.Input(**molecule)).dict()).dict()

    def _read_image(self, molecule: BNXParser.BNXMolecule):
        fov_file = self._fov_relpath(molecule)
        try:
            file = AnyPath(self.config.bionano_images_dir) / fov_file
            endpoints = self._parse_segment_endpoints(molecule)
            fov = read_jxr_image(str(file))
            image = extract_segment_from_endpoints(fov[None], endpoints=endpoints,
                                                   segment_width=self.config.segment_width)
        except Exception as e:
            return self.Error(
                fov_file=fov_file,
                error=str(e),
            )
        else:
            return self.Info(
                image=image,
                fov=fov,
                endpoints=endpoints,
                fov_file=fov_file,
            )

    def _parse_segment_endpoints(self, molecule: BNXParser.BNXMolecule):
        start_y, start_x = molecule.StartY, molecule.StartX
        start_y = start_y + (molecule.StartFOV - 1) * self.config.fov_size

        stop_y, stop_x = molecule.EndY, molecule.EndX
        stop_y = stop_y + (molecule.EndFOV - 1) * self.config.fov_size

        return np.stack([
            [start_y, start_x],
            [stop_y, stop_x],
        ])

    def _fov_relpath(self, molecule: BNXParser.BNXMolecule):
        Channel = self.config.bnx_channel
        run = BNXParser.BNXRun(**self.runs[molecule.RunId])
        C_digits = molecule.Column
        ChipId = molecule.ChipId.split(",")[-2].lstrip("Run_")
        return f"{ChipId}/FC{molecule.Flowcell}/Scan{run.Scan}/Bank{run.Bank}/B{run.Bank}_CH{Channel}_C{C_digits:03d}.jxr"


@joblib_memory.cache(mmap_mode='r')
def read_jxr_image(file: str):
    return imagecodecs.jpegxr_decode(any_file_read_bytes(file))
