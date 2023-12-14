from scipy.io import savemat
from Python_Processing.Data_extractions import Extract_data_from_subject
import os

# NOTES:
#   Takes ~2 minutes to run locally (my personal laptop)


# Hyper parameters

# The root dir have to point to the folder that contains the database
root_dir = "../Dataset/"

# Data Type {eeg, exg, baseline}
datatype = "EEG"

# Sampling rate
fs = 256

# Subject number
N_S = 1   # [1 to 10]

# Load all trials for a single subject
X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

# Keep only trials for inner speech (see https://www.nature.com/articles/s41597-022-01147-2 Table 5)
idxInnerSpeechTrials = (Y[:, 2] == 1)
# Keep only essential columns
columnsToKeep = [1, 3]  # Class(left, right, up, down) and session number(1, 2 or 3) column

X = X[idxInnerSpeechTrials]
Y = Y[idxInnerSpeechTrials]
Y = Y[:, columnsToKeep]

# Load baseline data
datatype = "baseline"
xBaseline, yBaseline = Extract_data_from_subject(root_dir, N_S, datatype)


data_dict = {
    'X': X,
    'Y': Y,
    'fs': fs,
    'numSubject': N_S,
    'baseline': xBaseline
}

# Specify the full path where you want to save the .mat file
savePath = '../matFiles/'
fileName = 'subject'+str(N_S)+'.mat'

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(savePath), exist_ok=True)

# Save the dictionary to a .mat file
savemat(savePath+fileName, data_dict)


