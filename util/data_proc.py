"""
Contains functions for processing data dictionaries.

Kipp Freud
07/11/2019
"""

# ------------------------------------------------------------------

import torch
from typing import Union, NamedTuple, Dict
from torch import Tensor
import numpy as np

import util.utilities as ut
from util.message import message
from networks.late_fusion_CNN import CLateFusionCNN

# ------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# data & data types
#-----------------------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------------------
# public functions
#-----------------------------------------------------------------------------------------

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

def validate(model, test_loader):
    results = {}
    model.eval()

    # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for batch, labels, filenames in test_loader:
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(batch)
            preds = logits.cpu().numpy()
            for (pred, label, filename) in zip(list(preds), list(labels.cpu().numpy()), filenames):
                file_res = results.setdefault(filename, {"preds": [], "labels": []})
                file_res["preds"].append(pred)
                file_res["labels"].append(label)
    results = _combine_file_results(results)
    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    per_class_acc = compute_per_class_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )

    message.logDebug(f"validation accuracy: {accuracy * 100:2.2f}",
                     "data_proc::validate")
    for label, acc in per_class_acc.items():
        message.logDebug(f"Accuracy for class '{label}': {acc * 100:2.2f}",
                         "data_proc::validate")

#-----------------------------------------------------------------------------------------
# private functions
#-----------------------------------------------------------------------------------------

def _combine_file_results(results):
    new_res = {"preds": [], "labels": []}
    for res in results.values():
        new_res["preds"].append(np.argmax(np.mean(res["preds"], axis=0)))
        new_res["labels"].append(np.round(np.mean(res["labels"])).astype(int))
    return new_res
