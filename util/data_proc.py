"""
Contains functions for processing data dictionaries.

Kipp Freud
07/11/2019
"""

# ------------------------------------------------------------------

from typing import Union, NamedTuple
from torch import Tensor
import numpy as np

import util.utilities as ut
from util.message import message

# ------------------------------------------------------------------

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

def getSubDict(dic, keys):
    """
    Will return a sub dictionary of dict that only contains keys given in :param:`keys`.
    """
    ret = {}
    for k in keys:
        if k in dic.keys():
            ret[k] = dic[k]
    return ret

def compute_accuracy(labels: Union[Tensor, np.ndarray], preds: Union[Tensor, np.ndarray]) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)