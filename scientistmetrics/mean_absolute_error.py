# -*- coding: utf-8 -*-
import statsmodels as smt
from sklearn import metrics
 
# Mean Absolute Error
def mean_absolute_error(self=None, y_true=None, y_pred=None, percentage=False):
    """
    Mean Absolute (Percentage) Error regression loss
    ------------------------------------------------

    Read more in the [User Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error).

    Parameters:
    -----------
    self : an instance of class OLS.

    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) target values.
    
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.
    
    percentage : bool, default = False;
                if True returns MAPE, il False returns MAE
    
    Returns:
    ------
    loss : float
           MA(P)E output is non-negative floating point. The best value is 0.0.
    """
    if self is None:
        ytrue = y_true
        ypred = y_pred
    else:
        if self.model.__class__ != smt.regression.linear_model.OLS:
            raise TypeError("'self' must be an object of class OLS")
        ytrue = self.model.endog
        ypred = self.predict()
    
    if percentage:
        return metrics.mean_absolute_percentage_error(y_true=ytrue,y_pred=ypred)
    else:
        return metrics.mean_absolute_error(y_true=ytrue,y_pred=ypred)