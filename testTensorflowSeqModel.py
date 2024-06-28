import os
import numpy as np
from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject
from Python_Processing.Utilitys import filterCondition
from Python_Processing.Data_processing import Select_time_window
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, Flatten, Dense, InputLayer
import tensorflow as tf

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Data Type {eeg, exg, baseline}
datatype = "eeg"

# Subject number
N_S = 2  # [1 to 10]

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)
X, Y = filterCondition(X, Y, Config.idInnerCondition, discardNonEssentialCols=False)

# Select time window
X = Select_time_window(X)
y = Y[:, Config.classColumn]

# Print data shapes for debugging
print(f"X shape before split: {X.shape}")
print(f"y shape before split: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Print data shapes for debugging
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Normalize data
normalizeData = True
if normalizeData:
    X_train_reshaped = X_train.reshape((X_train.shape[0], -1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], -1))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    X_train = X_train_scaled.reshape(X_train.shape)
    X_test = X_test_scaled.reshape(X_test.shape)

# Convert data to NHWC format
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Print data shapes for debugging
print(f"X_train shape after reshape: {X_train.shape}")
print(f"X_test shape after reshape: {X_test.shape}")

# Define a simplified EEGNet model
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    Conv2D(8, (1, 64), padding='same', use_bias=False),
    DepthwiseConv2D((2, 1), use_bias=False, depth_multiplier=2),
    AveragePooling2D((1, 4)),
    Flatten(),
    Dense(4, activation='softmax')  # Adjust the number of classes as needed
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Add model checkpoint callback
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.keras', verbose=1, save_best_only=True)

# Train the model for a few epochs
fittedModel = model.fit(X_train, y_train, batch_size=4, epochs=5, validation_split=0.25, verbose=2, callbacks=[checkpointer])

# Load the best model
model.load_weights('/tmp/checkpoint.keras')

# Evaluate the model
probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == y_test.argmax(axis=-1))
print(f"Classification accuracy: {acc}")

# Print model summary
model.summary()

import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(fittedModel.history['accuracy'], label='Training Accuracy')
plt.plot(fittedModel.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(fittedModel.history['loss'], label='Training Loss')
plt.plot(fittedModel.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()