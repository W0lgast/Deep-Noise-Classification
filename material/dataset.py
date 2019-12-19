import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'LMC':
            # Log-mel spectrogram, chroma, spectral contrast
            # and tonnetz are aggregated to form the LMC feature sets

            # Edit here to load and concatenate the neccessary features to 
            # create the LMC feature
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            # MFCC is combined with chroma,
            # spectral contrast and tonnetz to form the MC feature sets

            # Edit here to load and concatenate the neccessary features to 
            # create the MC feature
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            # Combined LM, MFCC and CST together to form MLMC feature.

            # Edit here to load and concatenate the neccessary features to 
            # create the MLMC feature
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
       
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)

    # ------------------------------------------------------------------
    # 'public' members
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 'private' members
    # ------------------------------------------------------------------
