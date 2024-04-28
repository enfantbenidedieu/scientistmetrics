# -*- coding: utf-8 -*-
import numpy as np
import statsmodels as smt
from sklearn import metrics

def average_precision_score(self=None,y_true=None, y_prob = None):
    """
    Compute average precision (AP) from prediction scores
    -----------------------------------------------------

    Parameters
    ----------
    self : an instance of class Logit

    y_true : array-like of shape (n_samples,) , default = None.
            True binary labels or binary label indicators

    y_prob : array-like of shape (n_samples,) , default =None.
            Probabilities of the positive class..

    Returns
    -------
    average_precision : float.
                        Average precision score.
    """
    if self is None:
        n_label = len(np.unique(y_true))
        if n_label != 2:
            raise TypeError("'average_precision_score' only applied for binary classification")
        ytrue = y_true
        yprob = y_prob
    elif self is not None:
        if self.model.__class__ != smt.discrete.discrete_model.Logit:
            raise TypeError("'self' must be an object of class Logit")
        ytrue = self.model.endog
        yprob = self.predict()
    return metrics.average_precision_score(y_true=ytrue,y_prob=yprob)