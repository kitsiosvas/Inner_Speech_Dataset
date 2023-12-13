from Python_Processing.Data_extractions import Extract_data_from_subject, Extract_block_data_from_subject


# Hyper parameters

# The root dir have to point to the folder that contains the database
root_dir = "..\\Dataset\\"

# Data Type {eeg, exg, baseline}
datatype = "exg"

# Sampling rate
fs = 256

# Subject number
N_S = 1   # [1 to 10]

# Load all trials for a single subject
# X, Y = Extract_data_from_subject(root_dir, N_S, datatype)
X, Y = Extract_block_data_from_subject(root_dir, N_S, datatype, 1)  # block data means for single session

print(X.shape)
print(Y.shape)
