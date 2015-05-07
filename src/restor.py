#!/usr/bin/env python
from datetime import  datetime, timedelta
from datetime import date

import numpy as np
import pandas as pd


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor


def open_till_days(data_frame):
    """
    The restorant is open for how many days
    """
    today = pd.to_datetime(date.today())

    data_frame["Open Date"] = data_frame["Open Date"].apply(lambda x: \
    pd.to_datetime(x, format='%m/%d/%Y'))
    data_frame["Operational"] = data_frame["Open Date"].apply(lambda x:\
    (today - x).days / 365)
    
    city_class_mapping = {label:idx for idx, label in enumerate(\
    set(data_frame['City']))}
    data_frame["City"] = data_frame["City"].map(city_class_mapping)
    
    city_gp_class_mapping = {label:idx for idx, label in enumerate(\
    set(data_frame['City Group']))}
    data_frame["City Group"] = data_frame["City Group"].map(city_gp_class_mapping)
    
    city_gp_class_mapping = {label:idx for idx, label in enumerate(\
    set(data_frame['Type']))}
    data_frame["Type"] = data_frame["Type"].map(city_gp_class_mapping)
    return data_frame

def preprocess_data(data):
    """
    Preprocess the traininfg data to features and target
    """
    to_reatin = ['City', 'City Group', 'Type', 'P1', 'P2', 'P3', 'P4', \
    'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', \
    'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', \
    'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', \
    'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'Operational']
    features = data[to_reatin]
    target = data.revenue
    return features, target

def preprocess_data_test(data):
    """
    Preprocess the traininfg data to features and target
    """
    data = open_till_days(data)
    to_reatin = ['City', 'City Group', 'Type', 'P1', 'P2', 'P3', 'P4', \
    'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', \
    'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', \
    'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', \
    'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'Operational']
    features = data[to_reatin]

    return features

def gb_regressor(features,target,test):
    """
    Create GradiantBoostingRegressor
    """
    sweep_params = {'loss':['ls','huber'],'n_estimators':[250,300,400]}
    gb_reg = GridSearchCV(GradientBoostingRegressor(),sweep_params,\
    cv=3, verbose=10, refit=True, n_jobs=-1)
    _ = gb_reg.fit(features, target)
    print_grid_scores(gb_reg)
    results = gb_reg.predict(test)
    return results



def rf_regressor(features,target,test):
    """
    Create GradiantBoostingRegressor
    """
    sweep_params = {'criterion':['mse'],'n_estimators':[1000,1250,1500],\
    'max_features':['auto','sqrt','log2'],'random_state':[0,1],\
    'oob_score':[True,False],'max_depth':[7,10,13]}
    gb_reg = GridSearchCV(RandomForestRegressor(),sweep_params,\
    cv=3, verbose=10, refit=True, n_jobs=-1,scoring='mean_squared_error')
    #cv=3, verbose=10, refit=True, n_jobs=-1,scoring='mean_squared_error')
    _ = gb_reg.fit(features, target)
    print_grid_scores(gb_reg)
    results = gb_reg.predict(test)
    return results


def sv_regressor(features,target,test):
    """
    Support vector regressor
    """
    sweep_params = {'kernel':['linear','poly','rbf','sigmoid'],\
    'max_iter':[-1],'random_state':[0,1]}
    svr = GridSearchCV(SVR(),sweep_params,cv=3,verbose=10,refit=True,\
    n_jobs=-1,scoring='mean_squared_error')
    _ = svr.fit(features,target)
    print_grid_scores(svr)
    results = svr.predict(test)
    return results

def sgd_regressor(features,target,test):
    """
    SGD Regressor
    """
    sweep_params = {'loss':['squared_loss','huber','epsilon_insensitive',\
    'squared_epsilon_insensitive'],'penalty':['l2','l1','elasticnet'],\
    'n_iter':[5,10],'shuffle':[True,False],'random_state':[0,1],\
    'warm_start':[True,False]}
    sgd = GridSearchCV(SGDRegressor(),sweep_params,cv=3,verbose=10,refit=True,\
    n_jobs=-1,scoring='mean_squared_error')
    _ = sgd.fit(features,target)
    print_grid_scores(sgd)
    results = sgd.predict(test)
    return results

def print_grid_scores(grid_clf):
    print "BE", grid_clf.best_estimator_
    print "BS", grid_clf.best_score_
    print "BP", grid_clf.best_params_
    return {"BE":grid_clf.best_estimator_, \
    "BS":grid_clf.best_score_, "BP":grid_clf.best_params_}
    
def prepare_submission(preds):
    header = "Prediction"
    #print preds
    # class_preds = pd.Panel(preds)
    class_preds = pd.DataFrame(preds)
    class_preds.to_csv("HotelRev_Regressor_RF_1000_plus.csv")

if __name__ == "__main__":
    train_data = pd.read_csv("data/train.csv", index_col="Id") 
    test_data = pd.read_csv("data/test.csv", index_col="Id")
    test_feat = preprocess_data_test(test_data)
    df = open_till_days(train_data)
    feat,target = preprocess_data(df)
    res = rf_regressor(feat,target,test_feat)
    prepare_submission(res)
    
"""
Best score 24th 
BE RandomForestRegressor(bootstrap=True, compute_importances=None,
           criterion='mse', max_depth=None, max_features='log2',
           max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
           min_samples_split=2, n_estimators=400, n_jobs=1,
           oob_score=False, random_state=1, verbose=0)
BS 0.123911898903
BP {'max_features': 'log2', 'n_estimators': 400, 'random_state': 1, 'criterion':
 'mse'}
[ 4174150.5275  3595683.3975  4239505.2275 ...,  4298095.175   4388396.055
  4952115.615 ]
5_csv
BE RandomForestRegressor(bootstrap=True, compute_importances=None,
           criterion='mse', max_depth=7, max_features='log2',
           max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
           min_samples_split=2, n_estimators=600, n_jobs=1, oob_score=True,
           random_state=0, verbose=0)
MSE BS -5.8004263e+12
BP {'oob_score': True, 'n_estimators': 600, 'random_state': 0, 'criterion': 'mse', 'max_features': 'log2', 'max_depth': 7}

6_csv
MSE BS -5.79063870727e+12
BP {'oob_score': True, 'n_estimators': 900, 'random_state': 1, 'criterion': 'mse', 'max_features': 'log2', 'max_depth': 13}
SVC
MSE
BE SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.0,
  kernel='poly', max_iter=-1, probability=False, random_state=0,
  shrinking=True, tol=0.001, verbose=False)
BS -6.88167471309e+12
BP {'kernel': 'poly', 'max_iter': -1, 'random_state': 0}

SGD
BS -2.64088013062e+13
BP {'warm_start': True, 'loss': 'epsilon_insensitive', 'shuffle': True, 'n_iter': 10, 'penalty': 'l1', 'random_state': 0}
#HotelRev_Regressor_SGD.csv


====
GridSearch score options
['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']

38th reclaim 24th
BE RandomForestRegressor(bootstrap=True, compute_importances=None,
           criterion='mse', max_depth=13, max_features='log2',
           max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
           min_samples_split=2, n_estimators=1000, n_jobs=1,
           oob_score=True, random_state=1, verbose=0)
BS -5.76490183787e+12
BP {'oob_score': True, 'n_estimators': 1000, 'random_state': 1, 'criterion': 'mse', 'max_features': 'log2', 'max_depth': 13}

"""
