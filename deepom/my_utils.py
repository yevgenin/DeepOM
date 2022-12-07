from deepom.aligner import Aligner

import numpy
from numba import njit
from numpy import ndarray

from deepom.utils import is_sorted, ndargmax


def get_aligner():
    aligner = Aligner()
    aligner.align_params = {}
    return aligner


def get_scaler_from_aligner(aligner):
    return (aligner.alignment_ref[-1]-aligner.alignment_ref[0]) /\
    (aligner.alignment_qry[-1]-aligner.alignment_qry[0])

def plot_sequences(image,ref,qry,max_index,save=False,name = "plot"):
    figure(figsize=(30, 3))
    imshow(image, aspect="auto", cmap="gray")
    eventplot([inference_item.loc_pred, selector.selected[i].locs / 375], colors=["b", "r"])
    eventplot([ref,qry], colors=["b", "r"])
    max_index = max(max_index, len(image))
           
def overlap_percentage(xlist,ylist):
                    
    assert is_sorted(xlist) and is_sorted(ylist)  
                    
    min1 = xlist[0]
    max1 = xlist[-1]
    min2 = ylist[0]
    max2 = ylist[-1]

    overlap = max(0, min(max1, max2) - max(min1, min2))
    length = max1-min1 + max2-min2
    lengthx = max1-min1
    lengthy = max2-min2

#     return 2*overlap/length , overlap/lengthx  , overlap/lengthy
    return 2*overlap/length
    

def generateGroundTruth(mol):
    pass





