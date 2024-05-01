# -*- coding: utf-8 -*-
import collections
import scipy as sp
import numpy as np
import pandas as pd
import statsmodels as smt

def MannWhitneyTest(self=None, y_true=None, y_prob=None):
    """
    Mann - Whitney Test
    -------------------

    Parameters:
    -----------
    self : an instance of class Logit, default=None.

    ytrue : array of int, default = None.
            The outcome label (e.g. 1 or 0)

    yprob : array of float, default = None.
            The predicted outcome probability

    Return:
    -------
    statistic : float
                Mann Whitney statistic
    
    pvalue : float
            The normal critical probability
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self is None:
        n_label = len(np.unique(y_true))
        if n_label != 2:
            raise TypeError("'MannWhitneyTest' only applied for binary classification")
        ytrue = y_true
        yprob = y_prob
    else:
        if self.model.__class__ != smt.discrete.discrete_model.Logit:
            raise TypeError("'self' must be an object of class Logit")
        ytrue = self.model.endog
        yprob = self.predict()
        
    df = pd.DataFrame({'y' : ytrue,'score' : yprob})
    n_moins, n_plus = df.y.value_counts()
    df = df.sort_values(by="score",ascending=True)
    df["rang"] = np.arange(1,len(ytrue)+1)
    # Somme des rangs de chaque groupe
    srang_moins, srang_plus = df.pivot_table(index='y',values="rang",aggfunc="sum").values[:,0]
    # Statistiques
    u_moins = srang_moins - (n_moins*(n_moins+1)/2)
    u_plus = srang_plus - (n_plus*(n_plus+1)/2)
    U = min(u_moins,u_plus)
    # Statistique de Mann - Whitney
    mn_statistic = (U - (n_plus*n_moins)/2)/(np.sqrt((1/12)*(n_moins*n_plus+1)*(n_moins*n_plus)))
    # Pvalue
    pvalue = sp.stats.norm.sf(mn_statistic)
    # Store all informations
    Result = collections.namedtuple("MannWhitneyResult",["statistic","pvalue"],rename=False)
    return Result(statistic=mn_statistic,pvalue=pvalue) 