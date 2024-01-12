from sklearn.model_selection import train_test_split

from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject, Extract_block_data_from_subject, Extract_report
from Python_Processing.Utilitys import filterCondition, selectElectrodes
from Python_Processing.Data_processing import Select_time_window

from braindecode.models.util import to_dense_prediction_model
from braindecode.classifier import EEGClassifier
from braindecode.datasets import create_from_X_y, BaseDataset


datatype = "eeg"  # Data Type {eeg, exg, baseline}
N_S = 1           # Subject number [1 to 10]
fs  = 256         # Sampling freq

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
X, Y = filterCondition(X, Y, Config.idInnerCondition, discardNonEssentialCols=False)

X = selectElectrodes(X, ["D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"])
X = Select_time_window(X)
y = Y[:, Config.classColumn]

#dataset = create_from_X_y(X, y, drop_last_window=False, sfreq=fs)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

import torch
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds


cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_chans = X_train.shape[1]
n_times = X_train.shape[2]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    n_times=n_times,
    final_conv_length='auto',
)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    model = model.cuda()

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier

# We found these values to be good for the shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 4

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(X_test),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
    classes=classes,
)

_ = clf.fit(X_train, y_train, epochs=n_epochs)

