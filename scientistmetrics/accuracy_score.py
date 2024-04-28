

import numpy as np
from sklearn import metrics
import statsmodels as smt

def accuracy_score(self=None,y_true=None,y_pred=None,threshold=0.5):
    """
    Accuracy classification score
    -----------------------------

    Read more in the [User Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score).

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
    -------
    score : float
    """
    if self is None:
        n_label = len(np.unique(y_true))
        if n_label != 2:
            raise TypeError("'accuracy_score' only applied for binary classification")
        ytrue = y_true
        ypred = y_pred
    else:
        if self.model.__class__ != smt.discrete.discrete_model.Logit:
            raise TypeError("'self' must be an object of class Logit")
        ytrue = self.model.endog
        ypred = np.where(self.predict() < threshold,0,1)
    return metrics.accuracy_score(y_true=ytrue,y_pred=ypred)