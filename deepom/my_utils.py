from deepom.aligner import Aligner

import numpy
from numba import njit
from numpy import ndarray
import torch
from deepom.utils import is_sorted, ndargmax
import logging
import sys
from contextlib import contextmanager

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
                   
    # print(f"xlist :{len(xlist)}")
    # print(f"ylist :{len(ylist)}")
    min1 = xlist[0]
    max1 = xlist[-1]
    min2 = ylist[0]
    max2 = ylist[-1]

    overlap = max(0, min(max1, max2) - max(min1, min2))
    length = max1-min1 + max2-min2
    lengthx = max1-min1
    lengthy = max2-min2

    return 2*overlap/length
    
def overlap_fraction(parent_seg, crop_seg):
    (x, y), (a, b) = sorted(parent_seg), sorted(crop_seg)
    assert (a < b) and (x < y)
    intersection = max(0, min(b, y) - max(a, x))
    overlap1 = intersection / (b - a)
    overlap2 = intersection / (y - x)
    overlap = min(overlap1,overlap2)
    return overlap

def fp_precentage(prediction, gt, delta=1):
    try:
        misaligned = 0
        for x in prediction:
            if x > gt[-1]:
                # print(x)
                misaligned += 1
                continue
            for y in gt:
                if y > x+delta:
                    # print(x)
                    misaligned += 1
                    break
                if y>=x-delta and y<=x+delta:
                    break
        return (misaligned)/(len(prediction))
    except Exception as e:
        print("got exception in fp percentage ")
        return 1

def fn_precentage(prediction, gt, delta=1):
    return fp_precentage(gt, prediction, delta)

'''
Loss functions and Neural Network Related Utils
'''
def dice_loss(y_pred,y_true,smooth=100):
    return 1 - dice_coef(y_pred,y_true,smooth)

def dice_coef(y_pred,y_true, smooth=100):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return dice


# def tversky(prediction, ground_truth, weight_map=None, alpha=0.5, beta=0.5):

#     prediction = tf.to_float(prediction)
#     if len(ground_truth.shape) == len(prediction.shape):
#         ground_truth = ground_truth[..., -1]
#     one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
#     one_hot = tf.sparse_tensor_to_dense(one_hot)

#     p0 = prediction
#     p1 = 1 - prediction
#     g0 = one_hot
#     g1 = 1 - one_hot

#     if weight_map is not None:
#         num_classes = prediction.shape[1].value
#         weight_map_flattened = tf.reshape(weight_map, [-1])
#         weight_map_expanded = tf.expand_dims(weight_map_flattened, 1)
#         weight_map_nclasses = tf.tile(weight_map_expanded, [1, num_classes])
#     else:
#         weight_map_nclasses = 1

#     tp = tf.reduce_sum(weight_map_nclasses * p0 * g0)
#     fp = alpha * tf.reduce_sum(weight_map_nclasses * p0 * g1)
#     fn = beta * tf.reduce_sum(weight_map_nclasses * p1 * g0)

#     EPSILON = 0.00001
#     numerator = tp
#     denominator = tp + fp + fn + EPSILON
#     score = numerator / denominator
#     return 1.0 - tf.reduce_mean(score)
'''
Logger meta class
'''

@contextmanager
def torch_temporary_seed(seed: int):
    """
    A context manager which temporarily sets torch's random seed, then sets the random
    number generator state back to its previous state.
    :param seed: The temporary seed to set.
    """

    default_state = torch.random.get_rng_state()
 
    try:
        torch.random.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(default_state)



class MetaBase(type):
    def __init__(cls, *args):
        super().__init__(*args)

        # Explicit name mangling
        logger_attribute_name = '_' + cls.__name__ + '__logger'

        # Logger name derived accounting for inheritance for the bonus marks
        logger_name = '.'.join([c.__name__ for c in cls.mro()[-2::-1]])

        setattr(cls, logger_attribute_name, logging.getLogger(logger_name))

        