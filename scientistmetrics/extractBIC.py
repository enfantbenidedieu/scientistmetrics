# -*- coding: utf-8 -*-

def extractBIC(self):
    """
    Bayesian information criterion
    ------------------------------

    Parameters
    ----------
    self : an instance of statsmodels model class.

    Returns
    -------
    bic : float
    """
    return self.bic