
import numpy as np
import statsmodels as smt

from .r2_coxsnell import r2_coxsnell

# Nagelkerke/Cragg & Uhler's R^2
def r2_nagelkerke(self):
    """
    Nagelkerke/Cragg & Uhler's R^2
    ------------------------------

    Parameters
    ----------
    self : an instance of class Logit, MNLogit or OrderedModel

    Return
    ------
    value : float
    """
    max_r2coxsnell = 1 - np.exp(self.llnull)**(2/self.nobs)
    return r2_coxsnell(self)/max_r2coxsnell