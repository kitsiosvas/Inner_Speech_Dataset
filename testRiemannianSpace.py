from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject
from Python_Processing.Utilitys import filterCondition, selectElectrodes
from Python_Processing.Data_processing import Select_time_window

import numpy as np
from matplotlib import pyplot as plt
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from pyriemann.utils.distance import distance_riemann

from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder

datatype = "eeg"  # Data Type {eeg, exg, baseline}
N_S = 1           # Subject number [1 to 10]
fs  = 256         # Sampling freq

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
X, Y = filterCondition(X, Y, Config.idInnerCondition, sessionNum=None, discardNonEssentialCols=False)

#X = selectElectrodes(X, ["D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"])
X = selectElectrodes(X, ["D5","D6","D7","D8","D9","D10"])
X          = Select_time_window(X)
y          = Y[:, Config.classColumn]
sessionNum = Y[:, Config.sessionColumn]

normalizeData = True
if normalizeData:
    X_reshaped = X.reshape((X.shape[0], -1))
    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X_reshaped)
    X          = X_scaled.reshape(X.shape)

# Define bandpass filter parameters
lowcut = 8
highcut = 40
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(4, [low, high], btype='band', output='ba')
# Apply bandpass filter to each trial in X
X = np.array([filtfilt(b, a, trial, axis=1) for trial in X])


plot = False
if plot:
    # Calculate Covariance Matrices
    cov_est = Covariances()
    cov_matrices = cov_est.fit_transform(X)

    # Compute pairwise Riemannian distances
    num_trials = cov_matrices.shape[0]
    distances = np.zeros((num_trials, num_trials))
    for i in range(num_trials):
        for j in range(i + 1, num_trials):
            distances[i, j] = distance_riemann(cov_matrices[i], cov_matrices[j])
            distances[j, i] = distances[i, j]

    # Perform MDS to reduce dimensionality for visualization
    mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, random_state=42)
    mds_xy = mds.fit_transform(distances)

    # Set up colors and markers
    class_colors = ['r', 'g', 'b', 'purple']
    session_markers = ['o', 's', '^']

    plt.figure(figsize=(8, 6))

    unique_classes = np.unique(y)
    unique_sessions = np.unique(sessionNum)

    for session, session_marker in zip(unique_sessions, session_markers):
        for cls, class_color in zip(unique_classes, class_colors):
            mask = (sessionNum == session) & (y == cls)
            plt.scatter(
                mds_xy[mask, 0],
                mds_xy[mask, 1],
                label=f'Session {session} - Class {cls}',
                color=class_color,
                marker=session_marker
            )
    plt.title('Visualization of Riemannian Distances')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.legend()
    plt.show()

runClassification = True
if runClassification:
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # For type of covariance matrices estimators see: https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.utils.covariance.covariances.html#pyriemann.utils.covariance.covariances
    clf = make_pipeline(
        Covariances(estimator='lwf'),
        TangentSpace(metric="riemann"),
        LogisticRegression(),
    )

    preds = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X):
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X[train_idx], y_train)
        preds[test_idx] = clf.predict(X[test_idx])

    # Printing the results
    acc = np.mean(preds == y)
    print("Classification accuracy: %f " % (acc))

    names = ["up", "down", "right", "left"]
    cm = confusion_matrix(y, preds)
    ConfusionMatrixDisplay(cm, display_labels=names).plot()
    plt.show()
