"""
Loads a pickled model, then does stuff to it I guess.

Kipp McAdam Freud
12/01/2019
"""
# --------------------------------------------------------------

import torch
from torch.utils.tensorboard import SummaryWriter
import pickle

from networks.late_fusion_CNN import CLateFusionCNN
from util.message import message
import util.utilities as ut
from material.dataset import UrbanSound8KDataset
from util.data_proc import validate

# --------------------------------------------------------------

TEST_DATA_LOCATION = "material/UrbanSound8K_test.pkl"

FILE_PATH_PREF = "models/"
FILE_PATH_END = "_four_fold_cnn.pkl"
MODES = ["LMC", "MC", "MLMC"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

models = {}
for mode in MODES:
    file_path = FILE_PATH_PREF + mode + FILE_PATH_END
    infile = open(file_path,'rb')
    models[mode] = pickle.load(infile)
    infile.close()

# --------------------------------------------------------------

for mode in MODES:

    message.logDebug("Testing model trained for mode " + mode + ".",
                     "main_test::__main__")

    model = models[mode]

    test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset(TEST_DATA_LOCATION, mode),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True
    )

    validate(model, test_loader)

message.logDebug("Testing model trained for mode TSCNN.",
                 "main_test::__main__")

TSCNN = CLateFusionCNN([models["MC"], models["LMC"]])
test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset(TEST_DATA_LOCATION, "TSCNN"),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True
    )

validate(TSCNN, test_loader)

ut.exit(1)