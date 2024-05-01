# -*- coding: utf-8 -*-
import statsmodels as smt
from sklearn import metrics
 
def mae(self=None, y_true=None, y_pred=None):
    """
    Mean Absolute Error (MAE) regression loss
    -----------------------------------------

    Read more in the [User Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error).

    Parameters:
    -----------
    self : an instance of class OLS.

    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) target values.
    
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.
    
    Return
    ------
    loss : float
           MAE output is non-negative floating point. The best value is 0.0.
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self is None:
        ytrue = y_true
        ypred = y_pred
    else:
        if self.model.__class__ != smt.regression.linear_model.OLS:
            raise TypeError("'self' must be an object of class OLS")
        ytrue = self.model.endog
        ypred = self.predict()
    return metrics.mean_absolute_error(y_true=ytrue,y_pred=ypred)