# -*- coding: utf-8 -*-

from scipy.stats.contingency import association
from scipy.stats import chi2_contingency
from pandas import DataFrame, crosstab
from numpy import eye
from itertools import combinations

def scientistmetrics(X,method="cramer",correction=False,lambda_ = None):
    """Compute the degree of association between two nomila variables and return a DataFrame

    Parameters
    ----------
    X : DataFrame.
        Observed values
    
    method : {"chi2","cramer","tschuprow","pearson"} (default = "cramer")
        The association test statistic.
    
    correction : bool, optional
        Inherited from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    
    lambda_ : float or str, optional
        Inherited from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

    Returns:
    --------
    statistic : DataFrame
        value of the test statistic   
    """
    # Chack if X is an instance of class
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with "
                        "pd.DataFrame. For more information see: "
                        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    if method not in ["chi2","cramer","tschuprow","pearson"]:
        raise ValueError("Error : Valid method are 'chi2', 'cramer','tschuprow' or pearson.")
    
    # Extract catehorical columns
    cat_columns = X.select_dtypes(include=["category","object"]).columns

    if len(cat_columns)==0:
        raise KeyError("No categorical variables found")

    # get all possible pair-wise combinations in the columns list
    # this assumes that A-->B equals B-->A so we don't need to
    # calculate the same thing twice
    # we also never get "A --> A"
    all_combinations = combinations(cat_columns, r=2)

    # fill matrix with zeros, except for the main diag (which will
    # be always equal to one)
    matrix = DataFrame(eye(len(cat_columns)),columns=cat_columns,index=cat_columns)

    # note that because we ignore redundant combinations,
    # we perform half the calculations, so we get the results
    # twice as fast
    for comb in all_combinations:
        i = comb[0]
        j = comb[1]

        # make contingency table
        input_tab = crosstab(X[i],X[j])

        # Chi2 contingency
        if method == "chi2":
            res_association = chi2_contingency(input_tab,correction=correction,lambda_=lambda_)[0]
        else:
            res_association = association(input_tab, method=method,correction=correction,lambda_=lambda_)

        matrix[i][j], matrix[j][i] = res_association, res_association
    
    return matrix

        