from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject, Extract_block_data_from_subject, Extract_report
from Python_Processing.Utilitys import filterCondition, selectElectrodes
from Python_Processing.Data_processing import Select_time_window

from mne.time_frequency import psd_array_welch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Data Type {eeg, exg, baseline}
datatype = "eeg"

# Subject number
N_S = 1   # [1 to 10]

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
X, Y = filterCondition(X, Y, Config.idInnerCondition)

X = selectElectrodes(X, ["D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"])
X = Select_time_window(X)
y = Y[:,0]

psds, freqs = psd_array_welch(X, 256, fmin=8, fmax=100)

X_flatten = psds.reshape(psds.shape[0], -1)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flatten, y, test_size=0.2)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized  = scaler.transform(X_test)


#rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#rf_classifier.fit(X_train, y_train)
#y_pred = rf_classifier.predict(X_test)

svm_classifier = SVC(kernel='poly', degree=4)
svm_classifier.fit(X_train_normalized, y_train)
y_pred = svm_classifier.predict(X_test_normalized)


# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)



