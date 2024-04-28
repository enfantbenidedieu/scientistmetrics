# -*- coding: utf-8 -*-
import numpy as np
import statsmodels as smt
from statsmodels.stats.outliers_influence import OLSInfluence, GLMInfluence

from .residuals import residuals
# Studentized residuals    
def rstudent(self):
    """
    Studentized residuals
    ---------------------

    Parameters
    ----------
    self : an object of class OLS, Logit

    Returns
    -------
    resid : pd.series of float.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Studentized_residual
    """
    if self.model.__class__ == smt.discrete.discrete_model.MNLogit:
        raise TypeError("no applicable method for 'rstudent' applied to an object of class MNLogit")
    elif self.model.__class__ == smt.miscmodels.ordinal_model.OrderedModel:
        raise TypeError("no applicable method for 'rstudent' applied to an object of class OrderedModel")
    
    # Studentized residuals for Ordinary Least Squares
    if self.model.__class__ == smt.regression.linear_model.OLS:
        influ = OLSInfluence(self)
        return influ.resid_studentized_external
    # Studentized residuals for logistic model
    elif self.model.__class__ == smt.discrete.discrete_model.Logit:
        influ = GLMInfluence(self)
        hii = influ.hat_matrix_exog_diag
        dev_res = residuals(self,choice="deviance")
        pear_res = residuals(self,choice="pearson")
        stud_res = np.sign(dev_res)*np.sqrt(dev_res**2 + (hii*pear_res**2)/(1 - hii))
        return stud_res