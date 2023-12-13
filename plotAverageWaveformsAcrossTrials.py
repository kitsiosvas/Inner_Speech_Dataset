import numpy as np
import matplotlib.pyplot as plt
from Python_Processing.Data_extractions import Extract_data_from_subject


# Hyper parameters:

# The root dir have to point to the folder that contains the database
root_dir = "..\\Dataset\\"

# Data Type {eeg, exg, baseline}
datatype = "EEG"

# Sampling rate
fs = 256

# Subject number
N_S = 1   # [1 to 10]

# Load all trials for a single subject
X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

# Calculate the average across all trials
average_waveforms = np.mean(X, axis=0)

# Define the time axis from 0 to 4.5 seconds
time_axis = np.linspace(0, 4.5, average_waveforms.shape[1])

# Plot the average waveforms for each electrode
fig, ax = plt.subplots(figsize=(12, 6))
for electrode in range(128):
    ax.plot(time_axis, average_waveforms[electrode, :])

# Set x-axis limits
ax.set_xlim(0, 4.5)
# Remove x-axis labels
ax.set_xticks(np.arange(0, 4.6, 0.5))

# Set title and y-axis label
ax.set_title('Average Waveforms Across All Trials')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()
plt.show()
