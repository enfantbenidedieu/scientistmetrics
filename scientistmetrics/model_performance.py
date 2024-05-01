# -*- coding: utf-8 -*-
import pandas as pd
import statsmodels as smt

from .extractAIC import extractAIC
from .extractAICC import extractAICC
from .extractBIC import extractBIC
from .r2_score import r2_score
from .explained_variance_score import explained_variance_score
from .mean_absolute_error import mean_absolute_error
from .median_absolute_error import median_absolute_error
from .mean_squared_error import mean_squared_error
from .accuracy_score import accuracy_score
from .r2_mcfadden import r2_mcfadden
from .r2_coxsnell import r2_coxsnell
from .r2_nagelkerke import r2_nagelkerke
from .r2_efron import r2_efron
from .r2_mckelvey import r2_mckelvey
from .r2_count import r2_count
from .r2_count_adj import r2_count_adj
from .r2_tjur import r2_tjur

def model_performance(self, metrics = "common"):
    """
    Performance of Regression or Classification Models
    --------------------------------------------------

    Parameters:
    -----------
    self : an instance of class OLS, Logit, MNLOgit

    metrics : {"common","all"}, default = "common"

    Return
    ------
    metrics
    """

    if metrics not in ["all","common"]:
        raise ValueError("'metrics' should be one of 'all', 'common'")

    # Common metrics
    res = {"AIC" :extractAIC(self),
           "AICC":extractAICC(self),
           "BIC" :extractBIC(self)}

    if metrics == "all":
        if self.model.__class__  == smt.regression.linear_model.OLS:
            res["r2 score"] = r2_score(self)
            res["r2 score adj."] = r2_score(self,adjust=True)
            res["expl. var. score"] = explained_variance_score(self)
            res["mean abs. error"] = mean_absolute_error(self)
            res["median abs. error"] = median_absolute_error(self)
            res["mean sq. error"] = mean_squared_error(self)
            res["root mean sq. error"] = mean_squared_error(self,squared=False)
            res["mean abs. percentage error"] = mean_absolute_error(self,percentage=True)
        elif self.model.__class__ in [smt.discrete.discrete_model.Logit,smt.discrete.discrete_model.MNLogit]:
            res["accuracy"] = accuracy_score(self)
            res["r2 mcfadden"] = r2_mcfadden(self)
            res["r2 mcfadden adj."] = r2_mcfadden(self,adjust=True)
            res["r2 coxsnell"] = r2_coxsnell(self)
            res["r2 naglekerke"] = r2_nagelkerke(self)
        elif self.model.__class__ == smt.discrete.discrete_model.Poisson:
            res["pseudo r2"] = 1 - (-2*self.llf)/(-2*self.llnull)
        
        if self.model.__class__ == smt.discrete.discrete_model.Logit:
            res["r2 efron"] = r2_efron(self)
            res["r2 mckelvey"] = r2_mckelvey(self)
            res["r2 count"] = r2_count(self)
            res["r2 count adj."] = r2_count_adj(self)
            res["r2 tjur"] = r2_tjur(self)
        
    result = pd.Series(res.values(),index=[*res.keys()],name="statistics")
    return result