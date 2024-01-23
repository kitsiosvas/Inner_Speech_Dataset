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



datatype = "eeg"  # Data Type {eeg, exg, baseline}
N_S = 1           # Subject number [1 to 10]
fs  = 256         # Sampling freq

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
X, Y = filterCondition(X, Y, Config.idInnerCondition, sessionNum=1, discardNonEssentialCols=False)

#X = selectElectrodes(X, ["D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"])
X = Select_time_window(X)
y = Y[:, Config.classColumn]


# Define bandpass filter parameters
lowcut = 13
highcut = 30
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(4, [low, high], btype='band', output='ba')
# Apply bandpass filter to each trial in X
X = np.array([filtfilt(b, a, trial, axis=1) for trial in X])



# Define a monte-carlo cross-validation generator (reduce variance):
cv = KFold(n_splits=10, shuffle=True, random_state=42)


# For type of covariance matrices estimators seeL https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.utils.covariance.covariances.html#pyriemann.utils.covariance.covariances
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
