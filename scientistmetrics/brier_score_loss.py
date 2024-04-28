# -*- coding: utf-8 -*-
import numpy as np
import statsmodels as smt
from sklearn import metrics
 
def brier_score_loss(self=None,y_true=None,y_prob=None):
    """
    Compute the Brier score loss
    ----------------------------

    Parameters
    ----------
    self : an instance of class Logit

    y_true : array-like of shape (n_samples,) , default = None.
            True binary labels or binary label indicators
    
    y_prob : array-like of shape (n_samples,) , default =None.
            Probabilities of the positive class.

    Returns:
    -------
    score : float.
            Brier score loss.
    """
    if self is None:
        n_label = len(np.unique(y_true))
        if n_label != 2:
            raise TypeError("'brier_score_loss' only applied for binary classification")
        ytrue = y_true
        yprob = y_prob
    elif self is not None:
        if self.model.__class__ != smt.discrete.discrete_model.Logit:
            raise TypeError("'self' must be an object of class Logit")
        ytrue = self.model.endog
        yprob = self.predict()
    return metrics.brier_score_loss(y_true=ytrue,y_prob=yprob)