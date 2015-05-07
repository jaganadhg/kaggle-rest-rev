#!/usr/bin/env python
"""
Feature selection based on scikit-learn
"""

import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

def refcv_feature_select(data_frame,target_name,n_feats=20):
    """
    :param data_frame: a apndas DataFrame containing the data
    :param target_name: Header of the target variable name 
    :param n_feats: Number of features to be selected
    :returns scored: pandas DataFrame containing feature scoring
    Identify the number of features based Recursive Feature Elimination
    Cross Validated method in scikit-learn.
    """
    estimator = SVR(kernel='linear')
    selector = RFECV(estimator, step = 1, cv = 5)
    _ = selector.fit(data_frame.drop(target_name,axis = 1),\
    data_frame[target_name])
    
    scores = pd.DataFrame()
    scores["Attribute Name"] = data_frame.drop(target_name,axis = 1).columns
    scores["Ranking"] = selector.ranking_
    scores["Support"] = selector.support_
    
    return scores


if __name__ =="__main__":
    data = pd.read_csv("data/train.csv",index_col="Id")
    data = data.drop(['City', 'City Group', 'Type',"Open Date"], axis = 1)
    selections = refcv_feature_select(data,"revenue")
    selections.to_csv("refcv_feats.csv",index=False)