from pyriemann.utils.distance import distance_riemann
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
#sessionsToKeep = None  # If None we keep all sessions
sessionsToKeep = [1, 3]
X, Y = filterCondition(X, Y, Config.idInnerCondition, sessionsToKeep, discardNonEssentialCols=False)

# filter here data if we want to test it on binary
binaryProblem = True
if binaryProblem:
    labelsToKeep = [0, 1]  # 0:up, 1:down, 2:right, 3:left
    idxToKeep = np.isin(Y[:, Config.classColumn], labelsToKeep)
    X = X[idxToKeep]
    Y = Y[idxToKeep]


electrodes = ["D5", "D6", "D7", "D8"]
electrodes = ["D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20", "D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28", "D29", "D30", "D31", "D32"]
# electrodes = ["A5","A6","A7","A8","A9","A10","A11","A14","A16","A17","A18","A19","A21","A24","A25","A28","A29","B3","B5","B6","B7","B8","B9","B10","B11","B12","B13","B16","B21","B26","C10","C13","C15","C16","C18","C28","C29","D5","D17","D18","D19","D20","D25","D27","D28","D29","D30"]
X = selectElectrodes(X, electrodes)
X = Select_time_window(X)
y = Y[:, Config.classColumn]  # labels of each sample
domain = Y[:, Config.sessionColumn]  # session# of each sample
y_enc = np.core.defchararray.add(np.core.defchararray.add('session', domain.astype(str)), np.core.defchararray.add('/', y.astype(str)))  # Concat to format: 'session#/label'

numOfTrials = X.shape[0]
numOfChannels = X.shape[1]

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

seed = 2024

startingMatrices = {}
embedded_points = {}  # Embedded points will either be spectral embedding or MDS result
targetDomain = 'session3'

c = Covariances(estimator='lwf')
multConstant = 1  # constant to multiply the cov matrices to avoid dealing with small numbers

startingMatrices['origin'] = multConstant*c.fit_transform(X)

rct = TLCenter(target_domain=targetDomain)
startingMatrices['rct'] = multConstant*rct.fit_transform(startingMatrices['origin'], y_enc)  # (samples, 28, 28) cov matrices after re-centering

stretch = TLStretch(target_domain=targetDomain, centered_data=True)
startingMatrices['stretched'] = multConstant*stretch.fit_transform(startingMatrices['rct'], y_enc)  # (samples, 28, 28) cov matrices after stretching

rot = TLRotate(target_domain=targetDomain, metric='riemann')
startingMatrices['rot'] = multConstant*rot.fit_transform(startingMatrices['stretched'], y_enc)  # (samples, 28, 28) cov matrices after rotating

covMatricesAll = np.concatenate([startingMatrices['origin'], startingMatrices['rct'], startingMatrices['stretched'], startingMatrices['rot'], np.eye(numOfChannels)[None, :, :]])

spectralEmbedding = False
if spectralEmbedding:
    # Embedding in 2D space
    emb = SpectralEmbedding(n_components=2, metric='riemann')
    S = emb.fit_transform(covMatricesAll)  # (4*samples+1, 2) coordinates in embedded space
    S = S - S[-1]
    embedded_points['origin'] = S[:numOfTrials]
    embedded_points['rct'] = S[numOfTrials:2*numOfTrials]
    embedded_points['stretched'] = S[2*numOfTrials:3*numOfTrials]
    embedded_points['rot'] = S[3*numOfTrials:-1]
else:
    # MDS in 2D space
    for k in startingMatrices.keys():  # k = ['origin', 'rct', 'stretched', 'rot']
        cov_matrices = startingMatrices[k]
        # Compute pairwise Riemannian distances
        num_trials = cov_matrices.shape[0]
        distances = np.zeros((num_trials, num_trials))
        for i in range(num_trials):
            for j in range(i + 1, num_trials):
                distances[i, j] = distance_riemann(cov_matrices[i], cov_matrices[j])
                distances[j, i] = distances[i, j]

        # Perform MDS to reduce dimensionality for visualization
        mds = manifold.MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, random_state=42)
        mds_xy = mds.fit_transform(distances)
        embedded_points[k] = mds_xy

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
markers = ['o', '^', 's']  # Define markers for each session

for axi, step, title in zip(ax, steps, titles):
    data_all = embedded_points[step]

    for session in np.unique(domain):  # For all sessions
        thisSessionData = data_all[domain == session]
        thisSessionLabels = y[domain == session]

        for cls in np.unique(y):
            mask = thisSessionLabels == cls
            axi.scatter(
                thisSessionData[mask, 0],
                thisSessionData[mask, 1],
                c=f'C{cls}', marker=markers[session-1], s=50, alpha=0.50,
                label=f'Session {session} - Class {cls}'
            )
    # Put text for single class
    classToPlotTxt = np.unique(y)[0]
    for i in range(len(data_all)):
        if y[i]==classToPlotTxt:
            axi.text(data_all[i, 0], data_all[i, 1], str(i+1), fontsize=8, ha='center', va='center', color='black')


    #axi.set_aspect('equal')
    axi.set_title(title, fontsize=14)
    axi.axis('tight')

ax[0].legend(loc="upper right")
if spectralEmbedding:
    fig.suptitle('Spectral Embedding visualization', fontsize=16)
else:
    fig.suptitle('MDS visualization', fontsize=16)
plt.show()
