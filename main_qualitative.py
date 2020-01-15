"""
Loads pickled models, loops through dataset until we find some data index which satisfies
an exit condition. This script was used to find the indexes of interestingly classified
data points in section 'Qualitative Results'.

Kipp McAdam Freud
12/01/2019
"""
# --------------------------------------------------------------

import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np

from networks.late_fusion_CNN import CLateFusionCNN
from util.message import message
import util.utilities as ut
from material.dataset import UrbanSound8KDataset

# --------------------------------------------------------------

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

TEST_DATA_LOCATION = "material/UrbanSound8K_test.pkl"

FILE_PATH_PREF = "models/"
FILE_PATH_END = "_four_fold_cnn.pkl"
MODES = ["LMC", "MC", "MLMC"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
DEVICE = torch.device("cpu")

file_path = FILE_PATH_PREF + "LMC" + FILE_PATH_END
infile = open(file_path,'rb')
lmcnet = pickle.load(infile)
infile.close()

file_path = FILE_PATH_PREF + "MC" + FILE_PATH_END
infile = open(file_path,'rb')
mcnet = pickle.load(infile)
infile.close()

file_path = FILE_PATH_PREF + "MLMC" + FILE_PATH_END
infile = open(file_path,'rb')
mlmcnet = pickle.load(infile)
infile.close()

tscnnnet = CLateFusionCNN([mcnet, lmcnet])

lmcnet.eval()
lmcnet.to(DEVICE)
mcnet.eval()
mcnet.to(DEVICE)
mlmcnet.eval()
mlmcnet.to(DEVICE)

# --------------------------------------------------------------

mc_test_loader = torch.utils.data.DataLoader(
    UrbanSound8KDataset(TEST_DATA_LOCATION, "MC"),
    batch_size=32, shuffle=False,
    num_workers=0, pin_memory=True
)
mc_dataset = mc_test_loader.dataset

lmc_test_loader = torch.utils.data.DataLoader(
    UrbanSound8KDataset(TEST_DATA_LOCATION, "LMC"),
    batch_size=32, shuffle=False,
    num_workers=0, pin_memory=True
)
lmc_dataset = lmc_test_loader.dataset

mlmc_test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset(TEST_DATA_LOCATION, "MLMC"),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True
    )
mlmc_dataset = mlmc_test_loader.dataset

tscnn_test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset(TEST_DATA_LOCATION, "TSCNN"),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True
    )
tscnn_dataset = tscnn_test_loader.dataset


num_data = len(lmc_test_loader.dataset)

for i in range(1000, num_data):

    if lmc_dataset[i][1] != mc_dataset[i][1]:
        message.logError("Weird datset problem!",
                         "main_qual::__main__")
        ut.exit(0)

    correct_pred = lmc_dataset[i][1]

    message.logDebug("Testing index " + str(i) + ": " + LABEL_NAMES[correct_pred],
                     "main_qual::__main__")

    lmc = lmc_dataset[i][0].unsqueeze(0)
    lmc.to(DEVICE)
    mc = mc_dataset[i][0].unsqueeze(0)
    mc.to(DEVICE)
    mlmc = mlmc_dataset[i][0].unsqueeze(0)
    mlmc.to(DEVICE)
    tscnn = tscnn_dataset[i][0].unsqueeze(0)
    tscnn.to(DEVICE)

    lmc_logits = lmcnet(lmc)
    mc_logits = mcnet(mc)
    mlmc_logits = mlmcnet(mlmc)
    tscnn_logits = tscnnnet(tscnn)

    results = {}
    for mod in [(lmcnet, lmc, "LMC net"),
                (mcnet, mc, "MC net"),
                (mlmcnet, mlmc, "MLMC Net"),
                (tscnnnet, tscnn, "TSCNN Net")]:
        logits = mod[0](mod[1])
        with torch.no_grad():
            preds = logits.detach().numpy()
            pred_ind = np.argmax(preds)
            pred_name = LABEL_NAMES[pred_ind]
            msg = mod[2] + " predicts " + pred_name
            print(msg)
            results[mod[0]] = pred_ind == correct_pred

    # Define your exit condition here
    if not results[lmcnet]:
        if not results[mcnet]:
            if not results[mlmcnet]:
                if not results[tscnnnet]:
                    message.logDebug("Index " + str(i) + " generates stop condition",
                                     "main_qual::__main__")
                    ut.exit(1)

# If exit condition has not been generated, we have failed.
ut.exit(0)