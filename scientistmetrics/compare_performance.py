
import numpy as np
import pandas as pd
import statsmodels as smt
from sklearn import metrics

from .extractAIC import extractAIC
from .extractAICC import extractAICC
from .extractBIC import extractBIC

# Compare performance
def compare_performance(model=list()):
    """
    
    Parameters
    ----------
    model : list of training model to compare

    Returns
    -------
    DataFrame
    """

    if not isinstance(model,list):
        raise TypeError("'model' must be a list of model.")

    def evaluate(i,name):
        res = pd.DataFrame({"AIC" : extractAIC(name), # Akaike information criterion.
                            "AICC":extractAICC(name), # 
                             "BIC" : extractBIC(name), # Bayesian information criterion.
                             "Log-Likelihood" : name.llf}, # Log-likelihood of model
                             index=["Model " + str(i+1)])
        if name.model.__class__  == smt.regression.linear_model.OLS:
            res["R-squared"] = name.rsquared
            res["Adj. rsquared"] = name.rsquared_adj
            ytrue, ypred= name.model.endog, name.predict()
            res["RMSE"] = metrics.mean_squared_error(y_true=ytrue,y_pred=ypred,squared=True)
            res["sigma"] = np.sqrt(name.scale)
            res.insert(0,"Name","ols")
        elif name.model.__class__ == smt.discrete.discrete_model.Logit:
            res["Pseudo R-squared"] = name.prsquared  # McFadden's pseudo-R-squared.
            ytrue, yprob = name.model.endog, name.predict()
            ypred = np.where(yprob > 0.5, 1, 0)
            res["log loss"] = metrics.log_loss(y_true=ytrue,y_pred=ypred)
            res.insert(0,"Name","logit")
        elif name.model.__class__ == smt.tsa.arima.model.ARIMA:
            res["MAE"] = name.mae
            res["RMSE"] = np.sqrt(name.mse)
            res["SSE"] = name.sse
            res.insert(0,"Name","arima")
        elif name.model.__class__ == smt.discrete.discrete_model.Poisson:
            res.insert(0,"Name","poisson")
        elif name.model.__class__ == smt.discrete.discrete_model.MNLogit:
            res.insert(0,"Name","multinomial")
        elif name.model.__class__ == smt.miscmodels.ordinal_model.OrderedModel:
            res.insert(0,"Name","ordinal")
        return res
    res1 = pd.concat(map(lambda x : evaluate(x[0],x[1]),enumerate(model)),axis=0)
    return res1
        