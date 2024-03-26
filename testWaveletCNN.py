from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject
from Python_Processing.Utilitys import filterCondition
from Python_Processing.Data_processing import Select_time_window
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from scipy.signal import cwt, ricker

"""
    Training of a CNN network using wavelet transform for EEG signal classification
"""

# Data Type {eeg, exg, baseline}
datatype = "eeg"

# Subject number
N_S = 2  # [1 to 10]

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
X, Y = filterCondition(X, Y, Config.idInnerCondition, discardNonEssentialCols=False)


electrodes = ["D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20",
              "D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28", "D29", "D30", "D31", "D32"]
# electrodes = ["A5","A6","A7","A8","A9","A10","A11","A14","A16","A17","A18","A19","A21","A24","A25","A28","A29","B3","B5","B6","B7","B8","B9","B10","B11","B12","B13","B16","B21","B26","C10","C13","C15","C16","C18","C28","C29","D5","D17","D18","D19","D20","D25","D27","D28","D29","D30"]
# X = selectElectrodes(X, electrodes)
X = Select_time_window(X)  # Keep only action interval
y = Y[:, Config.classColumn]


def apply_wavelet_transform(data):
    num_of_trials, num_of_sensors, num_of_samples = data.shape
    widths = np.arange(1, 128)  # Adjust the range based on your preference
    transformed_data = np.zeros((num_of_trials, len(widths), num_of_sensors, num_of_samples))

    # Apply the wavelet transform for each trial and sensor
    for trial in range(num_of_trials):
        for sensor in range(num_of_sensors):
            # Apply the 1D ricker wavelet along the sample axis
            transformed_data[trial, :, sensor, :] = cwt(data[trial, sensor, :], ricker, widths)

    return transformed_data


X = apply_wavelet_transform(X)

y = Y[:, Config.classColumn]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation='softmax')  # Assuming 4 classes, adjust accordingly
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint_cnn.h5', verbose=1, save_best_only=True)

fittedModel = model.fit(X_train, y_train, batch_size=16, epochs=30, validation_split=0.25, verbose=2, callbacks=[checkpointer])
model.load_weights('/tmp/checkpoint_cnn.h5')

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == y_test.argmax(axis=-1))
print(f"Classification accuracy: {acc}")


