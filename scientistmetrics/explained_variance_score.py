
import statsmodels as smt
from sklearn import metrics

# Explained Variance Score
def explained_variance_score(self=None,y_true=None,y_pred=None):
    """
    Explained Variance Ratio regression score function
    --------------------------------------------------

    Best possible score is 1.0, lower values are worse.

    Parameters
    ----------
    self : an instance of class OLS.

    y_true : array-like of shape (n_samples,)
            Ground truth (correct) target values.
    
    y_pred : array-like of shape (n_samples,)
            Estimated target values.
    
    Return:
    ------
    score : float
    """

    if self is None:
        ytrue = y_true
        ypred = y_pred
    else:
        if self.model.__class__ != smt.regression.linear_model.OLS:
            raise ValueError("'self' must be an object of class OLS")
        ytrue = self.model.endog
        ypred = self.predict()
    return metrics.explained_variance_score(y_true=ytrue,y_pred=ypred)