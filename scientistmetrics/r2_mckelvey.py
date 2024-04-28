# -*- coding: utf-8 -*-
import numpy as np

# MCKelvey & Zavoina R^2
def r2_mckelvey(self=None,y_prob=None):
    """
    McKelvey & Zavoina R^2
    ----------------------

    Parameters
    ----------
    self : an object of class Logit

    y_prob : array of float
            The predicted probabilities for binary outcome
    
    Return
    ------
    value : float
    """
    if self is None:
        yprob = np.array(y_prob)
    else:
        yprob = self.predict()
    return np.var(yprob) / (np.var(yprob) + (np.power(np.pi, 2.0) / 3.0) )