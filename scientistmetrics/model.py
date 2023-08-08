
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from more_itertools import powerset
import statsmodels.formula.api as smf
from sklearn.metrics import (
    # Regression metrics
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    # Classification metrics
    accuracy_score,
    roc_auc_score
)

def powersetmodel(DTrain=pd.DataFrame,
                  DTest=None,
                  model_type ="linear",
                  target=str,
                  test_size=0.3,
                  random_state=None,
                  shuffle=True,
                  stratity=None):
    """

    Parameters
    ----------
    DTrain : DataFrame
            Training sample
    
    DTest : DataFrame, default = None
            Test sample
        
    model_type : str
    target : target name
    
    
    
    """

    if not isinstance(DTrain,pd.DataFrame):
        raise TypeError(f"{type(DTrain)} is not supported. Please convert to a DataFrame with "
                        "pd.DataFrame. For more information see: "
                        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    if DTest is not None:
        if not isinstance(DTest,pd.DataFrame):
            raise TypeError(f"{type(DTest)} is not supported. Please convert to a DataFrame with "
                            "pd.DataFrame. For more information see: "
                            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
    else:
        DTrain, DTest = train_test_split(DTrain,test_size=test_size,random_state=random_state,shuffle=shuffle,stratify=stratity)
    
    # Create formula : https://stackoverflow.com/questions/35518477/statsmodels-short-way-of-writing-formula
    def create_formula(y=str,x=list[str]):
        return y + ' ~ ' + ' + '.join(x)
    
    # List of features
    features = list(DTrain.drop(columns=target).columns)
    # Powerset features and Remove first element
    list_features = list(map(set, powerset(features)))[1:]

    #################################################################################
    #  Linear regression 
    #################################################################################
    # linear regression metrics
    def ols_metrics(formula,ytrue,ypred):
        res = {"formula":formula,
               "expl. var. score" : explained_variance_score(ytrue,ypred),
               "max error" : max_error(ytrue,ypred),
               "mean abs. error" : mean_absolute_error(ytrue,ypred),
               "mean sq. error" : mean_squared_error(ytrue,ypred),
               "mean sq. log error" : mean_squared_log_error(ytrue,ypred),
               "median abs. error" : median_absolute_error(ytrue,ypred),
               "r2 score" : r2_score(ytrue,ypred),
               "mean abs. percentage error" : mean_absolute_percentage_error(ytrue,ypred)}
        return pd.DataFrame(res,index=["metrics"])
    
    # Estimation of ols model
    def ols_estimated(y,x,df1,df2):
        # Create formula
        formula = create_formula(y=y,x=x)
        # Train the model
        model = smf.ols(formula=formula,data=df1).fit()
        # Predict under Test Dataset
        predict = model.predict(df2)
        # Metrics under test sampling
        metrics = ols_metrics(formula,df2[y],predict)
        return metrics
    
    # Store ols model
    def ols_model(y,x,df1):
        # Create formula
        formula = create_formula(y=y,x=x)
        # Train the model
        model = smf.ols(formula=formula,data=df1).fit()
        return model
    
    ############################################################################################
    #  Logistic regression model
    ############################################################################################

    # logistic metric
    def glm_metrics(formula,ytrue,ypred):
        res = {"formula":formula,
               "accuracy score" : accuracy_score(ytrue,ypred),
               "auc" : roc_auc_score(ytrue,ypred)}
        return pd.DataFrame(res,index=["stat"])
    
    def glm_estimated(y,x,df1,df2):
        # Create formula
        formula = create_formula(y=y,x=x)
        # Train the model
        model = smf.logit(formula=formula,data=df1).fit(disp=False)
        # Predict under Test dataset
        predict = np.where(model.predict(df2)>0.5,1,0)
        # Metrics under test sampling
        metrics = glm_metrics(formula,df2[y],predict)
        return metrics
    
    # Store ols model
    def glm_model(y,x,df1):
        # Create formula
        formula = create_formula(y=y,x=x)
        # Train the model
        model = smf.logit(formula=formula,data=df1).fit(disp=False)
        return model

    if model_type == "linear":
        list_model = list(map(lambda x : ols_model(target,x,DTrain),list_features))
        res = pd.concat(map(lambda x : ols_estimated(target,x,DTrain,DTest),list_features),axis=0,ignore_index=True)
    elif model_type == "logistic":
        list_model = list(map(lambda x : glm_model(target,x,DTrain),list_features))
        res = pd.concat(map(lambda x : glm_estimated(target,x,DTrain,DTest),list_features),axis=0,ignore_index=True)

    return list_model, res
    


    

