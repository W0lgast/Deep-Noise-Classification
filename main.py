"""
This is the main file, maybe it will do stuff eventually

Kipp McAdam Freud, Stoil Ganev
19/12/2019
"""
# --------------------------------------------------------------

import torch
from torch.utils.tensorboard import SummaryWriter

from util.message import message
import util.utilities as ut
from networks.four_fold_CNN import CFourFoldCNN
from material.dataset import UrbanSound8KDataset
from trainers.four_fold_trainer import CFFTrainer

# --------------------------------------------------------------

TRAIN_DATA_LOCATION = "material/UrbanSound8K_train.pkl"
TEST_DATA_LOCATION = "material/UrbanSound8K_test.pkl"
LOG_DIR = "logs/"

mode = 'LMC'
#mode = 'MC'
#mode = 'MLMC'

# --------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Set this to 0 for debugging, 8 otherwise.
NUM_WORKERS = 0

LEARNING_RATE = 0.001

# --------------------------------------------------------------

train_loader = torch.utils.data.DataLoader(
      UrbanSound8KDataset(TRAIN_DATA_LOCATION, mode),
      batch_size=32, shuffle=True,
      num_workers=NUM_WORKERS, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
     UrbanSound8KDataset(TEST_DATA_LOCATION, mode),
     batch_size=32, shuffle=False,
     num_workers=NUM_WORKERS, pin_memory=True)

FFCNN = CFourFoldCNN(
    height=85,
    width=41,
    channels=1,
    class_count=10,
    dropout=0
)

# Define the criterion to be softmax cross entropy
criterion = torch.nn.CrossEntropyLoss()

# Define the optimizer
# todo:: They claim a momentum of 0.9, and that they user Adam... (?)
# todo:: They use L2 regularization, but dont give a value, this is the Adam 'weight decay' param.
optimizer = torch.optim.Adam(FFCNN.parameters(),
                             lr=LEARNING_RATE)

summary_writer = SummaryWriter(
            str(LOG_DIR),
            flush_secs=5
)

trainer = CFFTrainer(
    model=FFCNN,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    summary_writer=summary_writer,
    device=DEVICE
)

trainer.train(
        epochs=20,
        val_frequency=5,
        print_frequency=10,
        log_frequency=10,
    )

summary_writer.close()
