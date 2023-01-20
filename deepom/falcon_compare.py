import matlab.engine
import numpy as np
from pathlib import Path
from typing import List, Tuple
from matplotlib import pyplot as plt



class FalconLocalizations:
    x: np.ndarray
    y: np.ndarray

class Falcon:
    def __enter__(self):
        self.start()
        
    def start(self):
        self.engine = matlab.engine.start_matlab()
        falcon_path = Path(__file__).parent.parent.parent.joinpath("FALCON2D").resolve()
        print(falcon_path)
        self.engine.addpath(str(falcon_path))
        self.engine.addpath(str(falcon_path / "functions"))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.quit()
    
    def __call__(self, image_file):
        table = self.engine.falcon(str(image_file))
        table = np.asarray(table)
        _, x, y, _, _ = table.T
        locs = FalconLocalizations()
        locs.x = x
        locs.y = y
        return locs


if __name__ == "__main__":
    from deepom.bionano_compare import BionanoCompare
    BionanoCompare(simulated_mode=True).run_falcon_compare()
    BionanoCompare().run_falcon_compare()