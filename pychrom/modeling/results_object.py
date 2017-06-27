

class ResultsDataSet(object):

    def __init__(self):
        """
        Class to store modeling results in pychrom
        """

        self.C = None
        self.Cp = None
        self.Q = None
        self.times = None
        self.col_locs = None
        self.par_locs = None
        self.components = None

        # to add sensitivities if available

