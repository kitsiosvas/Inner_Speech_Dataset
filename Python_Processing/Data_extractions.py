# -*- coding: utf-8 -*-

"""
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar

Utilitys from extract, read and load data from Inner Speech Dataset
"""
from config import Config
import mne
import gc
import numpy as np
from Python_Processing.Utilitys import getFullNameFromSbjNumber, unify_names
import pickle


def Extract_subject_from_BDF(root_dir, sbjNumber, sessionNumber):
    # name correction if N_Subj is less than 10
    Num_s = getFullNameFromSbjNumber(sbjNumber)

    #  load data
    file_name = root_dir + '/' + Num_s + '/ses-0' + str(sessionNumber) + '/eeg/' + Num_s + '_ses-0' + str(sessionNumber) + '_task-innerspeech_eeg.bdf'
    rawdata = mne.io.read_raw_bdf(input_fname=file_name, preload=True, verbose='WARNING')
    return rawdata, Num_s


def Extract_data_from_subject(root_dir, sbjNumber, datatype):
    """
    Load all blocks for one subject and stack the results in X
    """

    # name correction if sbjNumber is less than 10
    sbjName = getFullNameFromSbjNumber(sbjNumber)
    data    = dict()
    y       = dict()
    datatype = datatype.lower()

    fileSuffix = Config.getFileSuffixFromShort(datatype)

    for thisSession in Config.sessionsList:

        y[thisSession] = load_events(root_dir, sbjNumber, thisSession)
        file_name = root_dir + '/derivatives/' + sbjName + '/ses-0' + str(thisSession) + '/' + sbjName + '_ses-0' + str(thisSession) + fileSuffix
        X = mne.read_epochs(file_name, verbose='WARNING')
        data[thisSession] = X.get_data(copy=True)
        if datatype == "baseline":
            # NOTE: When loading baseline .fif file, it returns an (1, 137, 3841) array instead of
            #       (1, 136, 3841) as mentioned in the paper (136 = 128+8). The last row seems to be
            #       an error in the .fif file. We discard it.
            data[thisSession] = data[thisSession][:, :-1, :]

    X = np.vstack((data.get(1), data.get(2), data.get(3)))
    Y = np.vstack((y.get(1), y.get(2), y.get(3)))

    return X, Y


def Extract_block_data_from_subject(root_dir, N_S, datatype, N_B):
    """
    Load selected block from one subject
    """

    # Get subject name
    Num_s = getFullNameFromSbjNumber(N_S)
    datatype = datatype.lower()

    # Get events
    Y = load_events(root_dir, N_S, N_B)

    sub_dir = root_dir + '/derivatives/' + Num_s + '/ses-0' + str(N_B) + '/' + Num_s + '_ses-0' + str(N_B)
    if datatype == "eeg":
        file_name = sub_dir + '_eeg-epo.fif'
        X = mne.read_epochs(file_name, verbose='WARNING')
        X = X._data

    elif datatype == "exg":
        file_name = sub_dir + '_exg-epo.fif'
        X = mne.read_epochs(file_name, verbose='WARNING')
        X = X._data

    elif datatype == "baseline":
        # NOTE: When loading baseline .fif file, it returns an (1, 137, 3841) array instead of
        #       (1, 136, 3841) as mentioned in the paper (136 = 128+8). The last row seems to be
        #       an error in the .fif file. We discard it.
        file_name = sub_dir + '_baseline-epo.fif'
        X = mne.read_epochs(file_name, verbose='WARNING')
        X = X._data[:, :-1, :]
    else:
        raise Exception("Invalid Datatype")

    return X, Y


def Extract_report(root_dir, N_B, N_S):
    # Get subject name
    Num_s = getFullNameFromSbjNumber(N_S)

    # Save report
    sub_dir = root_dir + '/derivatives/' + Num_s + '/ses-0' + str(N_B) + '/' + Num_s + '_ses-0' + str(N_B)
    file_name = sub_dir + Config.fileSuffixReport

    with open(file_name, 'rb') as input:
        report = pickle.load(input)

    return report


def Extract_TFR(TRF_dir, Cond, Class, TFR_method, TRF_type):
    # Unify names as stored
    Cond, Class = unify_names(Cond, Class)

    fname = TRF_dir + TFR_method + "_" + Cond + "_" + Class + "_" + TRF_type + "-tfr.h5"

    TRF = mne.time_frequency.read_tfrs(fname)[0]

    return TRF


def Extract_data_multisubject(root_dir, N_S_list, datatype='EEG'):
    """
    Load all blocks for a list of subject and stack the results in X
    """

    tmp_list_X = []
    tmp_list_Y = []
    rows = []
    total_elem = len(N_S_list) * 3  # assume 3 sessions per subject
    S = 0
    datatype = datatype.lower()
    for N_S in N_S_list:
        print("Iteration ", S)
        print("Subject ", N_S)
        for N_B in Config.sessionsList:

            Num_s = getFullNameFromSbjNumber(N_S)

            base_file_name = root_dir + '/derivatives/' + Num_s + '/ses-0' + str(N_B) + '/' + Num_s + '_ses-0' + str(
                N_B)
            events_file_name = base_file_name + '_events.dat'
            data_tmp_Y = np.load(events_file_name, allow_pickle=True)
            tmp_list_Y.append(data_tmp_Y)
            print("Inner iteration ", N_B)
            if datatype == "eeg":
                # load data and events
                eeg_file_name = base_file_name + '_eeg-epo.fif'
                data_tmp_X = mne.read_epochs(eeg_file_name, verbose='WARNING')._data
                rows.append(data_tmp_X.shape[0])
                if S == 0 and N_B == 1:  # assume same number of channels, time steps, and column labels in every subject and session
                    chann = data_tmp_X.shape[1]
                    steps = data_tmp_X.shape[2]
                    columns = data_tmp_Y.shape[1]
                tmp_list_X.append(data_tmp_X)

            elif datatype == "exg":
                exg_file_name = base_file_name + '_exg-epo.fif'
                data_tmp_X = mne.read_epochs(exg_file_name, verbose='WARNING')._data
                rows.append(data_tmp_X.shape[0])
                if S == 0 and N_B == 1:
                    chann = data_tmp_X.shape[1]
                    steps = data_tmp_X.shape[2]
                    columns = data_tmp_Y.shape[1]
                tmp_list_X.append(data_tmp_X)

            elif datatype == "baseline":
                baseline_file_name = base_file_name + '_baseline-epo.fif'
                data_tmp_X = mne.read_epochs(baseline_file_name, verbose='WARNING')._data
                rows.append(data_tmp_X.shape[0])
                if S == 0 and N_B == 1:
                    chann = data_tmp_X.shape[1]
                    steps = data_tmp_X.shape[2]
                    columns = data_tmp_Y.shape[1]
                tmp_list_X.append(data_tmp_X)

            else:
                raise Exception("Invalid Datatype")
                return None, None

        S += 1

    X = np.empty((sum(rows), chann, steps))
    Y = np.empty((sum(rows), columns))
    offset = 0
    # put elements of list into numpy array
    for i in range(total_elem):
        print("Saving element {} into array ".format(i))
        X[offset:offset + rows[i], :, :] = tmp_list_X[0]
        if datatype == "eeg" or datatype == "exg":
            Y[offset:offset + rows[i], :] = tmp_list_Y[0]  # only build Y for the datatypes that uses it
        offset += rows[i]
        del tmp_list_X[0]
        del tmp_list_Y[0]
        gc.collect()
    print("X shape", X.shape)
    print("Y shape", Y.shape)

    if datatype == "eeg" or datatype == "exg":
        # for eeg and exg types, there is a predefined label that is returned
        return X, Y
    else:
        # for baseline datatypes, there's no such label (rest phase)
        return X


def load_events(root_dir, sbjNumber, sessionNumber):
    Num_s = getFullNameFromSbjNumber(sbjNumber)
    # Create file Name
    file_name = root_dir + "/derivatives/" + Num_s + "/ses-0" + str(sessionNumber) + "/" + Num_s + "_ses-0" + str(
        sessionNumber) + Config.fileSuffixEvents
    # Load Events
    events = np.load(file_name, allow_pickle=True)

    return events
