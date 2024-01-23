from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject
from Python_Processing.Utilitys import filterCondition, selectElectrodes
from Python_Processing.Data_processing import Select_time_window
import numpy as np
import matplotlib.pyplot as plt
from pyriemann.embedding import SpectralEmbedding
from pyriemann.transfer import (
    decode_domains,
    TLCenter,
    TLRotate,
    TLStretch
)
from pyriemann.estimation import Covariances
from sklearn import manifold



datatype = "eeg"  # Data Type {eeg, exg, baseline}
N_S = 1           # Subject number [1 to 10]
fs  = 256         # Sampling freq

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
X, Y = filterCondition(X, Y, Config.idInnerCondition, discardNonEssentialCols=False)

electrodes = ["D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20", "D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28", "D29", "D30", "D31", "D32"]
# electrodes = ["A5","A6","A7","A8","A9","A10","A11","A14","A16","A17","A18","A19","A21","A24","A25","A28","A29","B3","B5","B6","B7","B8","B9","B10","B11","B12","B13","B16","B21","B26","C10","C13","C15","C16","C18","C28","C29","D5","D17","D18","D19","D20","D25","D27","D28","D29","D30"]
X = selectElectrodes(X, electrodes)
X = Select_time_window(X)
y = Y[:, Config.classColumn]  # labels of each sample
domain = Y[:, Config.sessionColumn]  # session# of each sample
y_enc = np.core.defchararray.add(np.core.defchararray.add('session', domain.astype(str)), np.core.defchararray.add('/', y.astype(str)))  # Concat to format: 'session#/label'

numOfTrials = X.shape[0]
numOfChannels = X.shape[1]
#X = X*1e7

# Define bandpass filter parameters
lowcut = 8
highcut = 40
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(4, [low, high], btype='band')
# Apply bandpass filter to each trial in X
X = np.array([filtfilt(b, a, trial, axis=1) for trial in X])

normalizeData = True
if normalizeData:
    X_reshaped = X.reshape((X.shape[0], -1))
    scaler = StandardScaler()
    X_scaled  = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X.shape)

c = Covariances(estimator='lwf')
X_org = c.fit_transform(X)

# Fix seed for reproducible results
seed = 66

# instantiate object for doing spectral embeddings
emb = SpectralEmbedding(n_components=2, metric='riemann')

# create dict to store the embedding after each step of RPA
embedded_points = {}

# embed the source and target datasets after recentering
rct = TLCenter(target_domain='session1')
X_rct = rct.fit_transform(X_org, y_enc)

stretch = TLStretch(target_domain='session1', centered_data=True)
X_stretched = stretch.fit_transform(X_rct, y_enc)

# embed the source and target datasets after rotating
rot = TLRotate(target_domain='session1', metric='riemann')
X_rot = rot.fit_transform(X_stretched, y_enc)

points = np.concatenate([X_org, X_rct, X_stretched, X_rot, np.eye(numOfChannels)[None, :, :]])
S = emb.fit_transform(points)
#S = S - S[-1]
embedded_points['origin'] = S[:numOfTrials]
embedded_points['rct'] = S[numOfTrials:2*numOfTrials]
embedded_points['stretched'] = S[2*numOfTrials:3*numOfTrials]
embedded_points['rot'] = S[3*numOfTrials:-1]

# mds = manifold.MDS(2, normalized_stress=False)
# embedded_points['origin'] = mds.fit_transform(S[:numOfTrials])
# embedded_points['rct'] = mds.fit_transform(S[numOfTrials:2*numOfTrials])
# embedded_points['stretched'] = mds.fit_transform(S[2*numOfTrials:3*numOfTrials])
# embedded_points['rot'] = mds.fit_transform(S[3*numOfTrials:-1])


###############################################################################
# Plot the results, reproducing the Figure 1 of [1]_.

# fig, ax = plt.subplots(figsize=(13.5, 4.4), ncols=3, sharey=True)
# plt.subplots_adjust(wspace=0.10)
# steps = ['origin', 'rct', 'rot']
# titles = ['original', 'after recentering', 'after rotation']


fig, ax = plt.subplots(figsize=(13.5, 4.4), ncols=4, sharey=True)
plt.subplots_adjust(wspace=0.10)
steps = ['origin', 'rct', 'stretched', 'rot']
titles = ['original', 'after re-centering', 'after stretching', 'after rotation']
for axi, step, title in zip(ax, steps, titles):

    data_all = embedded_points[step]
    data_session1 = data_all[domain == 1]
    y_session1 = y[domain == 1]
    data_session2 = data_all[domain == 2]
    y_session2 = y[domain == 2]
    data_session3 = data_all[domain == 3]
    y_session3 = y[domain == 3]

    # Session 1
    axi.scatter(
        data_session1[y_session1 == 0][:, 0],
        data_session1[y_session1 == 0][:, 1],
        c='C0', s=50, alpha=0.50)
    axi.scatter(
        data_session1[y_session1 == 1][:, 0],
        data_session1[y_session1 == 1][:, 1],
        c='C1', s=50, alpha=0.50)
    axi.scatter(
        data_session1[y_session1 == 2][:, 0],
        data_session1[y_session1 == 2][:, 1],
        c='C2', s=50, alpha=0.50)
    axi.scatter(
        data_session1[y_session1 == 3][:, 0],
        data_session1[y_session1 == 3][:, 1],
        c='C3', s=50, alpha=0.50)

    # Session 2
    axi.scatter(
        data_session2[y_session2 == 0][:, 0],
        data_session2[y_session2 == 0][:, 1],
        c='C0', marker="^", s=50, alpha=0.50)
    axi.scatter(
        data_session2[y_session2 == 1][:, 0],
        data_session2[y_session2 == 1][:, 1],
        c='C1', marker="^", s=50, alpha=0.50)
    axi.scatter(
        data_session2[y_session2 == 2][:, 0],
        data_session2[y_session2 == 2][:, 1],
        c='C2', marker="^", s=50, alpha=0.50)
    axi.scatter(
        data_session2[y_session2 == 3][:, 0],
        data_session2[y_session2 == 3][:, 1],
        c='C3', marker="^", s=50, alpha=0.50)

    # Session 3
    axi.scatter(
        data_session3[y_session3 == 0][:, 0],
        data_session3[y_session3 == 0][:, 1],
        c='C0', marker="s", s=50, alpha=0.50)
    axi.scatter(
        data_session3[y_session3 == 1][:, 0],
        data_session3[y_session3 == 1][:, 1],
        c='C1', marker="s", s=50, alpha=0.50)
    axi.scatter(
        data_session3[y_session3 == 2][:, 0],
        data_session3[y_session3 == 2][:, 1],
        c='C2', marker="s", s=50, alpha=0.50)
    axi.scatter(
        data_session3[y_session3 == 3][:, 0],
        data_session3[y_session3 == 3][:, 1],
        c='C3', marker="s", s=50, alpha=0.50)


    axi.scatter(S[-1, 0], S[-1, 1], c='k', s=80, marker="*")
    #axi.set_xlim(-0.60, +1.60)
    #axi.set_ylim(-1.10, +1.25)
    #axi.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    #axi.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axi.set_title(title, fontsize=14)
ax[0].scatter([], [], c="C0", label="session1 - class 0")
ax[0].scatter([], [], c="C1", label="session1 - class 1")
ax[0].scatter([], [], c="C2", label="session1 - class 2")
ax[0].scatter([], [], c="C3", label="session1 - class 3")

ax[0].scatter([], [], marker="^", c="C0", label="session2 - class 0")
ax[0].scatter([], [], marker="^", c="C1", label="session2 - class 1")
ax[0].scatter([], [], marker="^", c="C2", label="session2 - class 2")
ax[0].scatter([], [], marker="^", c="C3", label="session2 - class 3")

ax[0].scatter([], [], marker="s", c="C0", label="session2 - class 0")
ax[0].scatter([], [], marker="s", c="C1", label="session2 - class 1")
ax[0].scatter([], [], marker="s", c="C2", label="session2 - class 2")
ax[0].scatter([], [], marker="s", c="C3", label="session2 - class 3")

ax[0].legend(loc="upper right")

plt.show()
