"""
Contains functions for processing data dictionaries.

Kipp Freud
07/11/2019
"""

# ------------------------------------------------------------------

from typing import Union, NamedTuple, Dict
from torch import Tensor
import numpy as np

import util.utilities as ut
from util.message import message

# ------------------------------------------------------------------

LABEL_NAMES = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}


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

def compute_per_class_accuracy(
    labels: Union[Tensor, np.ndarray], preds: Union[Tensor, np.ndarray],
) -> Dict[str, float]:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    label_counts = np.bincount(labels, minlength=len(LABEL_NAMES))
    pred_counts = [np.sum((labels == i) & (preds == i)) for i in LABEL_NAMES]
    accuracies = {}
    for label, (label_count, pred_count) in enumerate(zip(label_counts, pred_counts)):
        accuracies[LABEL_NAMES[label]] = float(pred_count) / label_count
    return accuracies
