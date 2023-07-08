

from pandas import DataFrame, crosstab
from scipy.stats.contingency import association
from pandas import DataFrame, crosstab
from numpy import sqrt, zeros, eye
from itertools import combinations

class PairWisemetrics:
    
    def __init__(self, dataframe, copy = True):
        '''
        Class initialization, it serves as a base class for all the metrics

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Pandas dataframe containing the variables
            of interest to measure the degree of association.

        Returns
        -------
        None.
        '''
        if isinstance(dataframe, DataFrame):
        
            if copy:
                self.data = dataframe.copy()
            else:
                self.data = dataframe
                
        else:
            raise TypeError("dataframe must be an instance of a pd.DataFrame")
    
class catassociation(PairWisemetrics):
    """
    
    
    """
    
    def __init__(self, dataframe, method="cramer"):
        PairWisemetrics.__init__(self, dataframe)
        self.matrix = None
        self.method = method

    def select_variables(self):
        '''
        Selects all category variables

        Returns
        -------
        None.

        '''
        self.cat_columns = self.data.select_dtypes(include=['category']).columns

        if len(self.cat_columns)==0:
            raise KeyError("No categorical variables found")

    def init_pairwisematrix(self):
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
        self.matrix = DataFrame(
            eye(len(self.cat_columns)),
            columns=self.cat_columns,
            index=self.cat_columns
        )

    def fill_pairwisematrix(self):
        '''
        fills the square matrix using scipy's association method

        Returns
        -------
        None.

        '''
        
        n = len(self.cat_columns)
        # get all possible pair-wise combinations in the columns list
        # this assumes that A-->B equals B-->A so we don't need to
        # calculate the same thing twice
        # we also never get "A --> A"
        all_combinations = combinations(self.cat_columns, r=2)

        # note that because we ignore redundant combinations,
        # we perform half the calculations, so we get the results
        # twice as fast
        for comb in all_combinations:
            i = comb[0]
            j = comb[1]

            # make contingency table
            input_tab = crosstab(self.data[i], self.data[j])

            # find the resulting categorical association value using scipy's association method
            res_association = association(input_tab, method=self.method)
            self.matrix[i][j], self.matrix[j][i] = res_association, res_association

    def fit(self):
        '''
        Creates a pairwise matrix filled with categories association
        where columns and index are the categorical
        variables of the passed pandas.DataFrame

        Returns
        -------
        statistic matrix.
        '''
        self.select_variables()
        self.init_pairwisematrix()
        self.fill_pairwisematrix()

        return self.matrix