from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject
from Python_Processing.Utilitys import filterCondition, selectElectrodes
from Python_Processing.Data_processing import Select_time_window

import numpy as np
from mne.time_frequency import psd_array_welch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold

import xgboost as xgb


# This scripts performs binary classification between the action and rest intervals.


# Data Type {eeg, exg, baseline}
datatype = "eeg"

# Subject number
N_S = 1   # [1 to 10]

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
#X, Y = filterCondition(X, Y, Config.idInnerCondition, discardNonEssentialCols=False)

electrodes = ["D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"]
#X = selectElectrodes(X, electrodes)

Xrest = Select_time_window(X, t_start=3.5, t_end=4.5)  # Keep only action interval
Xaction = Select_time_window(X)

psdRest, freqRest = psd_array_welch(Xrest, 256, fmin=8, fmax=100, verbose=False)
psdAction, freqAction = psd_array_welch(Xaction, 256, fmin=8, fmax=100, verbose=False)
X = np.concatenate((psdAction, psdRest), axis=0)
X = X.reshape(X.shape[0], -1)  # Flatten data into 2D

# Create labels for binary classification
y_action = np.ones((psdAction.shape[0],))
y_rest = np.zeros((psdRest.shape[0],))
y = np.concatenate((y_action, y_rest))


normalizeData = False
if normalizeData:
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

applyPCA = True
if applyPCA:
    pca = PCA(n_components=0.99)
    X   = pca.fit_transform(X)

modelType = "xgb"
if modelType == "rf":
    model = RandomForestClassifier(n_estimators=100, random_state=42)

elif modelType == "svm":
    model = SVC(kernel='poly', degree=4)

elif modelType == "xgb":
    model = xgb.XGBClassifier()

# Perform K-fold cross-validation
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# Print the cross-validation scores
print(f'Cross-validation scores: {scores}')
print(f'Mean accuracy: {np.mean(scores)}')

