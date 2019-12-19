"""
This is the main file, maybe it will do stuff eventually

Kipp McAdam Freud, Stoil Ganev
19/12/2019
"""
# --------------------------------------------------------------

import torch

from util.message import message
import util.utilities as ut
from networks.four_fold_CNN import CFourFoldCNN
from material.dataset import UrbanSound8KDataset

# --------------------------------------------------------------

TRAIN_DATA_LOCATION = "material/UrbanSound8K_train.pkl"
TEST_DATA_LOCATION = "material/UrbanSound8K_test.pkl"

#mode = 'LMC'
#mode = 'MC'
mode = 'MLMC'

# --------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --------------------------------------------------------------

train_loader = torch.utils.data.DataLoader(
      UrbanSound8KDataset(TRAIN_DATA_LOCATION, mode),
      batch_size=32, shuffle=True,
      num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
     UrbanSound8KDataset(TEST_DATA_LOCATION, mode),
     batch_size=32, shuffle=False,
     num_workers=8, pin_memory=True)

#FFCNN = CFourFoldCNN()

for i, (input, target, filename) in enumerate(train_loader):
    print("ooo")

for i, (input, target, filename) in enumerate(val_loader):
    print("oo")

