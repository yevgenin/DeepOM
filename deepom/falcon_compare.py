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
        self.engine.addpath("../../FALCON2D/functions")
        self.engine.addpath("../../FALCON2D")
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
