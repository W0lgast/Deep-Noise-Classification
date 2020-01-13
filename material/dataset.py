"""
Reads pkl datasets.
"""
# --------------------------------------------------------------

from util.message import message
import util.utilities as ut

import torch
from torch.utils import data
import numpy as np
import pickle

# --------------------------------------------------------------

# Define feature key names
FEAT_DICT = 'features'
LOGMELSPEC = 'logmelspec'
MFCC = 'mfcc'
CHROMA = 'chroma'
SPEC_CONT = 'spectral_contrast'
TONNETZ = 'tonnetz'

# --------------------------------------------------------------

class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        ind_dict = self.dataset[index]
        feature_dict = ind_dict[FEAT_DICT]
        if self.mode == 'LMC':
            # Log-mel spectrogram, chroma, spectral contrast
            # and tonnetz are aggregated to form the LMC feature sets
            feature = self._makeLMC(feature_dict)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            # MFCC is combined with chroma,
            # spectral contrast and tonnetz to form the MC feature sets
            feature = self._makeMC(feature_dict)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            # Combined LM, MFCC and CST together to form MLMC feature.
            feature = self._makeMLMC(feature_dict)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == "TSCNN":
            # This mode is to be used with a fusion model, we return a list of two features,
            # the MC and LMC features
            mc_feature = self._makeMC(feature_dict)
            mc_feature = torch.from_numpy(mc_feature.astype(np.float32)).unsqueeze(0)
            lmc_feature = self._makeLMC(feature_dict)
            lmc_feature = torch.from_numpy(lmc_feature.astype(np.float32)).unsqueeze(0)
            feature = [mc_feature, lmc_feature]
            feature = torch.stack(feature)
            #feature = torch.from_numpy(feature)
            #feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        else:
            message.logError("Unknown mode.", "UrbanSound8KDataset::__getitem__")
            ut.exit(0)
       
        label = ind_dict['classID']
        fname = ind_dict['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)

    # ------------------------------------------------------------------
    # 'public' members
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 'private' members
    # ------------------------------------------------------------------

    def get_feature_dict(self, index):
        ind_dict = self.dataset[index]
        return ind_dict[FEAT_DICT]

    def _makeLMC(self, feat_dict):
        """
        Gets LMC feature from feature dict.

        :return: ndarray of LMC feature instance.
        """
        LMC = np.concatenate((feat_dict[LOGMELSPEC],
                              feat_dict[CHROMA],
                              feat_dict[SPEC_CONT],
                              feat_dict[TONNETZ]),
                             axis=0)
        return LMC

    def _makeMC(self, feat_dict):
        """
        Gets MC feature from feature dict.

        :return: ndarray of MC feature instance.
        """
        MC = np.concatenate((feat_dict[MFCC],
                            feat_dict[CHROMA],
                            feat_dict[SPEC_CONT],
                            feat_dict[TONNETZ]),
                            axis=0)
        return MC

    def _makeMLMC(self, feat_dict):
        """
        Gets MLMC feature from feature dict.

        :return: ndarray of MLMC feature instance.
        """
        MLMC =  np.concatenate((feat_dict[MFCC],
                                feat_dict[LOGMELSPEC],
                               feat_dict[CHROMA],
                               feat_dict[SPEC_CONT],
                               feat_dict[TONNETZ]),
                               axis=0)
        return MLMC