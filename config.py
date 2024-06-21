class Config:
    """
    Project configuration variables. Use this file to change all parameters (f.e paths) according to your set-up

    """

    # Paths can be either relative or absolute

    datasetDir = "../Dataset/" # This basically means: "Go one directory back (the '..') and look for the 'Dataset' folder"

    # File suffices
    fileSuffixEEG      = "_eeg-epo.fif"
    fileSuffixEXG      = "_exg-epo.fif"
    fileSuffixBaseline = "_baseline-epo.fif"
    fileSuffixReport   = "_report.pkl"
    fileSuffixEvents   = "_events.dat"

    # General variables (#subjects, #sessions etc.)
    numOfSubjects    = 10
    subjectsList     = range(1, numOfSubjects+1)
    numOfSessions    = 3
    sessionsList     = range(1, numOfSessions+1)
    numOfConditions  = 3  # Pronounced (0) | Inner (1) | Visualized (2)
    idInnerCondition = 1

    # Y Matrix columns: sample# | class | condition | session#
    classColumn     = 1
    conditionColumn = 2
    sessionColumn   = 3

    # Mapping of numerical labels to corresponding names
    labelToName = {
        0: 'up',
        1: 'down',
        2: 'right',
        3: 'left'
    }

    @staticmethod
    def getFileSuffixFromShort(datatype):
        if datatype == "eeg":
            return Config.fileSuffixEEG
        elif datatype == "exg":
            return Config.fileSuffixEXG
        elif datatype == "baseline":
            return Config.fileSuffixBaseline
        else:
            raise Exception("Invalid Datatype")



