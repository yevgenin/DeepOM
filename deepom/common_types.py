from typing import NamedTuple, Union
from torch import Tensor
from numpy import ndarray

class LocalizerEnum:
    BG = 0
    FG = 1
    
      
class LocalizerLosses(NamedTuple):
    label_loss: Union[ndarray, Tensor]
    loc_loss: Union[ndarray, Tensor]


class LocalizerGT(NamedTuple):
    label_target: Union[ndarray, Tensor]
    loc_target: Union[ndarray, Tensor]

class LocalizerOutputs(NamedTuple):
    label_output: Union[ndarray, Tensor]
    loc_output: Union[ndarray, Tensor]

class DataItem(NamedTuple):
    molecule: Union[ndarray, Tensor]
    ground_truth: LocalizerGT