
import numpy as np
import statsmodels as smt

# Cox and Snell R^2
def r2_coxsnell(self):
    """
    Cox and Snell R^2
    -----------------

    Parameters
    ----------
    self : an instance of class Logit, MNLogit or OrderedModel

    Return
    ------
    value : float
    """
    return 1 - (np.exp(self.llnull)/np.exp(self.llf))**(2/self.nobs)