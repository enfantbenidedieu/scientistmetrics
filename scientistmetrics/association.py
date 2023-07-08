# -*- coding: utf-8 -*-

from pandas import DataFrame, crosstab
from scipy.stats.contingency import association
from pandas import DataFrame, crosstab
from numpy import eye
from itertools import combinations


class catassociation:
    """
    
    
    """
    
    def __init__(self,method="cramer", correction=False,lambda_ = None):
        self.method = method
        self.correction = correction
        self.lambda_ = lambda_
    
    def fit(self,X):
        """
        
        """

        if not isinstance(X, DataFrame):
            raise TypeError("Error : 'X' must be an instance of a pd.DataFrame")
        
        self.matrix = None

        # Compute statistics
        self._compute_stats(X=X)

        return self.matrix
    
    def _compute_stats(self,X):
        """
        
        
        """

        cat_columns = X.select_dtypes(include=['category']).columns

        if len(cat_columns)==0:
            raise KeyError("No categorical variables found")
        
        n = len(cat_columns)

        # get all possible pair-wise combinations in the columns list
        # this assumes that A-->B equals B-->A so we don't need to
        # calculate the same thing twice
        # we also never get "A --> A"
        all_combinations = combinations(cat_columns, r=2)

        '''
        init a square matrix n x n fill with zeros,
        where n is the total number of categorical variables
        found in the pd.DataFrame

        Returns
        -------
        None.

        '''
        # fill matrix with zeros, except for the main diag (which will
        # be always equal to one)
        self.matrix = DataFrame(eye(len(self.cat_columns)),columns=self.cat_columns,index=self.cat_columns)

        # note that because we ignore redundant combinations,
        # we perform half the calculations, so we get the results
        # twice as fast
        for comb in all_combinations:
            i = comb[0]
            j = comb[1]

            # make contingency table
            input_tab = crosstab(X[i],X[j])

            # find the resulting categorical association value using scipy's association method
            res_association = association(input_tab, method=self.method,correction=self.correction,lambda_=self.lambda_)
            self.matrix[i][j], self.matrix[j][i] = res_association, res_association
