

import numpy as np
import statsmodels as smt
from sklearn import metrics
# Balance accuracy score
def balanced_accuracy_score(self=None,y_true=None,y_pred=None, threshold=0.5):
    """
    Compute the balanced accuracy
    -----------------------------

    Parameters
    ---------- 
    self : an instance of class Logit

    y_true : 1d array-like, or label indicator array, default = None
            Ground thuth (correct) labels

    y_pred : 1d array-like, or label indicator array, default =None.
            Predicted labels, as returned by a classifier.
    
    threshold : float,  default = 0.5.
            The threshold value is used to make a binary classification decision based on the probability of the positive class.
           
    Returns
    --------
    balanced_accuracy : float
                        Balanced accuracy score.
    """
    if self is None:
        n_label = len(np.unique(y_true))
        if n_label != 2:
            raise TypeError("'balanced_accuracy_score' only applied for binary classification")
        ytrue = y_true
        ypred = y_pred
    elif self is not None:
        if self.model.__class__ != smt.discrete.discrete_model.Logit:
            raise TypeError("'self' must be an object of class Logit")
        ytrue = self.model.endog
        ypred = np.where(self.predict() < threshold,0,1)
    return metrics.balanced_accuracy_score(y_true=ytrue,y_pred=ypred)
