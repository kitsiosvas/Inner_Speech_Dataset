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

# Data Type {eeg, exg, baseline}
datatype        = "eeg"
overallAccuracy = 0

for iSubject in Config.subjectsList:

    # Load all trials for a single subject
    X, Y = Extract_data_from_subject(Config.datasetDir, iSubject, datatype)
    X, Y = filterCondition(X, Y, Config.idInnerCondition, discardNonEssentialCols=False)

    electrodes = ["D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"]
    #electrodes = ["A5","A6","A7","A8","A9","A10","A11","A14","A16","A17","A18","A19","A21","A24","A25","A28","A29","B3","B5","B6","B7","B8","B9","B10","B11","B12","B13","B16","B21","B26","C10","C13","C15","C16","C18","C28","C29","D5","D17","D18","D19","D20","D25","D27","D28","D29","D30"]
    #X = selectElectrodes(X, electrodes)
    X = Select_time_window(X)  # Keep only action interval
    y = Y[:, Config.classColumn]  # Keep only labels

    psds, freqs = psd_array_welch(X, 256, fmin=8, fmax=100, verbose=False)
    X = psds.reshape(psds.shape[0], -1)  # Flatten data into 2D

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
        model = SVC()

    elif modelType == "xgb":
        model = xgb.XGBClassifier()


    # Perform K-fold cross-validation
    cv     = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Print the cross-validation scores
    print(f'Subject {iSubject} Cross-validation scores: {scores}')
    print(f'Subject {iSubject} Mean accuracy: {np.mean(scores)} \n')

    overallAccuracy += np.mean(scores)

overallAccuracy = overallAccuracy/Config.numOfSubjects
print(f"Accuracy: {overallAccuracy}")

