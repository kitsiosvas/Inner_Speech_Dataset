from config import Config
from Python_Processing.Data_extractions import Extract_data_from_subject


# Hyper parameters

# Data Type {eeg, exg, baseline}
datatype = "eeg"

# Subject number
N_S = 1   # [1 to 10]
# Session number
N_B = 1

# Load all trials for a single subject
X, Y = Extract_data_from_subject(Config.datasetDir, N_S, datatype)

print(X.shape)
print(Y.shape)


