from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject
from Python_Processing.Utilitys import filterCondition, selectElectrodes
from Python_Processing.Data_processing import Select_time_window

from braindecode.datasets import create_from_X_y


datatype = "eeg"  # Data Type {eeg, exg, baseline}
N_S = 1           # Subject number [1 to 10]
fs  = 256         # Sampling freq

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
# Keep only inner speech trials (since dataset consists of inner, pronounced and imagined)
X, Y = filterCondition(X, Y, Config.idInnerCondition, discardNonEssentialCols=False)

X = selectElectrodes(X, ["D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"])
X = Select_time_window(X)
y = Y[:, Config.classColumn]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

normalizeData = True
if normalizeData:
    X_train_reshaped = X_train.reshape((X_train.shape[0], -1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], -1))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    X_train = X_train_scaled.reshape(X_train.shape)
    X_test = X_test_scaled.reshape(X_test.shape)


# Create datasets
train_set = create_from_X_y(X_train, y_train, drop_last_window=False, sfreq=fs)
val_set = create_from_X_y(X_test, y_test, drop_last_window=False, sfreq=fs)


import torch
from braindecode.models import ShallowFBCSPNet, EEGConformer
from braindecode.util import set_random_seeds


cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

classes = np.unique(y_train)
n_classes = classes.size

# Extract number of chans and time steps from dataset
n_chans = X_train.shape[1]
n_times = X_train.shape[2]

model = EEGConformer(
    n_outputs=n_classes,
    n_chans=n_chans,
    n_times=n_times,
    sfreq=fs,
    final_fc_length=1480
)


# Send model to GPU
if cuda:
    model = model.cuda()

from braindecode.classifier import EEGClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler

# Define training parameters
lr = 0.001  # Adjust the learning rate as needed
weight_decay = 1e-4
batch_size = 32
n_epochs = 10

# Define the EEGClassifier
clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(val_set),  # using val_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
    classes=classes
)

# Train the model
_ = clf.fit(train_set, y=None, epochs=n_epochs)

"""
PLOTTING RESULTS
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'valid_acc']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns, index=clf.history[:, 'epoch'])

# Convert lists to numpy arrays for arithmetic operations
train_loss_array = np.array(clf.history[:, 'train_loss'])
valid_acc_array = np.array(clf.history[:, 'valid_acc'])

# get percent of misclass for better visual comparison to loss
df = df.assign(
    train_misclass=100 - 100 * train_loss_array,  # Assuming train_loss as a proxy for train misclassification
    valid_misclass=100 - 100 * valid_acc_array
)

fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14
)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['train_misclass', 'valid_misclass']].plot(
    ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False
)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
plt.ion()

handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()
plt.show()

plt.ioff()




from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix

# generate confusion matrices
# get the targets
y_true = val_set.get_metadata().target
y_pred = clf.predict(val_set)

# generating confusion matrix
confusion_mat = confusion_matrix(y_true, y_pred)

labels = [v for k, v in sorted(Config.labelToName.items(), key=lambda kv: kv[1])]

# plot the basic conf. matrix
cnfMat = plot_confusion_matrix(confusion_mat, class_names=classes)


plt.show()


