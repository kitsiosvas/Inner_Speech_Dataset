from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject
from Python_Processing.Utilitys import filterCondition, selectElectrodes
from Python_Processing.Data_processing import Select_time_window
import numpy as np
from mne.time_frequency import psd_array_welch
from sklearn.decomposition import PCA
from tensorflow import keras
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import SGD, Adam


# Data Type {eeg, exg, baseline}
datatype = "eeg"

# Subject number
N_S = 2   # [1 to 10]

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
X, Y = filterCondition(X, Y, Config.idInnerCondition, discardNonEssentialCols=False)


electrodes = ["D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"]
#electrodes = ["A5","A6","A7","A8","A9","A10","A11","A14","A16","A17","A18","A19","A21","A24","A25","A28","A29","B3","B5","B6","B7","B8","B9","B10","B11","B12","B13","B16","B21","B26","C10","C13","C15","C16","C18","C28","C29","D5","D17","D18","D19","D20","D25","D27","D28","D29","D30"]
#X = selectElectrodes(X, electrodes)
X = Select_time_window(X)  # Keep only action interval
y = Y[:, Config.classColumn]

useSpectralDomainData = False
if useSpectralDomainData:
    X, freqs = psd_array_welch(X, 256, fmin=8, fmax=100, verbose=False)

applyPCA = False
if applyPCA:
    pca = PCA(n_components=0.99)
    X   = pca.fit_transform(X)

X = np.transpose(X, (0, 2, 1))  # Reshape to (number_of_samples, number_of_time_steps, number_of_features) for LSTM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

normalizeData = True
if normalizeData:

    X_train_reshaped = X_train.reshape((X_train.shape[0], -1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], -1))

    scaler = StandardScaler()
    X_train_scaled  = scaler.fit_transform(X_train_reshaped)
    X_test_scaled   = scaler.transform(X_test_reshaped)

    X_train = X_train_scaled.reshape(X_train.shape)
    X_test = X_test_scaled.reshape(X_test.shape)

bidirectinal = True
if not bidirectinal:
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(np.unique(y)), activation='softmax'))
else:
    model = Sequential()
    model.add(Bidirectional(LSTM(units=1024, input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(units=len(np.unique(y)), activation='softmax'))

sgd = SGD(learning_rate=0.001)
adam = Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=40, batch_size=16, validation_split=0.25)


#print(model.predict(X_test))
# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
