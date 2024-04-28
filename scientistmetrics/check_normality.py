
import scipy as sp
import statsmodels as smt
import collections

from .rstandard import rstandard

def check_normality(self, test="shapiro", effects = "fixed"):
    """
    Check model for (non-)normality of residuals
    --------------------------------------------

    Parameters:
    ----------
    self : an instance of class OLS

    test : {'shapiro','jarque-bera','agostino'}, default = 'shapiro'
            - 'shapiro' : Perform the Shapiro-Wilk test for normality.
            - 'jarque-bera' : Perform the Jarque-Bera goodness of fit test on sample data.
            - 'agostino' : It is based on D'Agostino and Pearson's, test that combines skew and kurtosis to produce an omnibus test of normality.
    
    effects : {'fixed','random'}
            Should normality for residuals ("fixed") or random effects ("random") be tested? Only applies to mixed-effects models. May be abbreviated.

    Returns:
    -------
    results : nametuple
                statistic : flaot
                    The test statistic
                pvalue : float
                    The p - value for the hypothseis test

    Notes:
    ------
    check_normality()  checks the standardized residuals (or studentized residuals for mixed models) for normal distribution. 

    References:
    ----------
    D'Agostino, R. B. (1971), “An omnibus test of normality for moderate and large sample size”, Biometrika, 58, 341-348
    D'Agostino, R. and Pearson, E. S. (1973), “Tests for departure from normality”, Biometrika, 60, 613-622
    Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3/4), 591-611.
    Jarque, C. and Bera, A. (1980) “Efficient tests for normality, homoscedasticity and serial independence of regression residuals”, 6 Econometric Letters 255-259.
    """

    if self.model.__class__ != smt.regression.linear_model.OLS:
        raise TypeError("Checking normality of residuals is only appropriate for linear models.")
    
    resid = rstandard(self,choice="sd_1")
    
    if test == 'shapiro':
        stat = sp.stats.shapiro(resid)
    elif test == 'jarque-bera':
        stat = sp.stats.jarque_bera(resid)
    elif test == 'agostino':
        stat = sp.stats.normaltest(resid)
    Result = collections.namedtuple("NormalityTest",["statistic","pvalue"],rename=False)
    return Result(statistic=stat.statistic,pvalue=stat.pvalue)