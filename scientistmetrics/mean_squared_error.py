# -*- coding: utf-8 -*-
import statsmodels as smt
from sklearn import metrics

# Mean Squared Error/ Root Mean Squared Error
def mean_squared_error(self=None, y_true=None, y_pred=None,squared=True):
    """
    (Root)Mean Squared Error ((R)MSE) regression loss
    -------------------------------------------------

    Read more in the [User Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error).

    Parameters
    ----------
    self : an instance of class OLS

    y_true : array-like of shape (n_samples,)
            Ground truth (correct) target values

    y_pred : array-like of shape (n_samples,)
            Estimated target values.
    
    squared : bool, default = True
              if True returns MSE value, if False returns RMSE value
             
    Returns
    -------
    loss : float
            A non-negative floating point value (the best value is 0.0)
    """
    if self is None:
        ytrue = y_true
        ypred = y_pred
    else:
        if self.model.__class__ != smt.regression.linear_model.OLS:
            raise TypeError("'self' must be an object of class OLS")
        ytrue = self.model.endog
        ypred = self.predict()
    return metrics.mean_squared_error(y_true=ytrue,y_pred=ypred,squared=squared)
