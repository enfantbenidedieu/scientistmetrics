# -*- coding: utf-8 -*-
import statsmodels as smt
from sklearn import metrics

def mse(self=None, y_true=None, y_pred=None):
    """
    Mean Squared Error (MSE) regression loss
    ----------------------------------------

    Description
    -----------
    The mean square error is the mean of the sum of squared residuals, i.e. it measures the average of the squares of the errors. 
    Less technically speaking, the mean square error can be considered as the variance of the residuals, i.e. 
    the variation in the outcome the model doesn't explain. Lower values (closer to zero) indicate better fit.

    Read more in the [User Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error).

    Parameters
    ----------
    self : an instance of class OLS

    y_true : array-like of shape (n_samples,)
            Ground truth (correct) target values

    y_pred : array-like of shape (n_samples,)
            Estimated target values.
             
    Return
    ------
    loss : float
            A non-negative floating point value (the best value is 0.0)
    
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
    return metrics.mean_squared_error(y_true=ytrue,y_pred=ypred,squared=True)
