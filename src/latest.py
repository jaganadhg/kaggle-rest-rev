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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest,f_regression


def cmap(city):
    city_map = {'\xc4\xb0zmir': 0, 'Mu\xc4\x9fla': 1, 'Tekirda\xc4\x9f': 2,\
    'Trabzon': 18, 'Afyonkarahisar': 4, 'Adana': 5, 'K\xc4\xb1rklareli': 6, \
    'Elaz\xc4\xb1\xc4\x9f': 7, 'Ayd\xc4\xb1n': 8, 'Gaziantep': 9, \
    'Ankara': 10, 'Kayseri': 11, 'Sakarya': 31, 'Tokat': 13, \
    'U\xc5\x9fak': 21, 'Bolu': 16, '\xc5\x9eanl\xc4\xb1urfa': 17, \
    'K\xc3\xbctahya': 3, 'Kocaeli': 19, 'Samsun': 20, 'Eski\xc5\x9fehir': 15,\
    'Kastamonu': 22, 'Antalya': 32, 'Karab\xc3\xbck': 23, 'Isparta': 24,\
    'Denizli': 25, 'Diyarbak\xc4\xb1r': 33, 'Osmaniye': 27, 'Amasya': 28, \
    'Bal\xc4\xb1kesir': 29, 'Bursa': 30, 'Edirne': 12,\
    '\xc4\xb0stanbul': 14, 'Konya': 26}
    if city_map.has_key(city):
        return city_map[city]
    else:
        return 0


def new_preprocessing_test(test_data,data_frame):
    """
    Apply LabelEncoder and SelectKBest feature based pre-processing
    """
    today = pd.to_datetime(date.today())
    encoder = LabelEncoder()
    for colum in test_data.columns:
        if colum != "revenue" and colum != "Open Date":
            encoded = encoder.fit(data_frame[colum]).transform(test_data[colum])
            test_data[colum] = encoded
    test_data["Open Date"] = test_data["Open Date"].apply(lambda x: \
    pd.to_datetime(x, format='%m/%d/%Y'))
    test_data["Operational"] = test_data["Open Date"].apply(lambda x:\
    (today - x).days)
    test_data["Age"] = test_data["Open Date"].apply(lambda x:\
    (today - x).days / 365)
    test_data["New"] = test_data["Age"].apply(lambda x:\
    x <= 1.5 and 0 or 1)
    test_data["Established"] = test_data["Age"].apply(lambda x:\
    x > 3 and 0 or 1)
    test_data["Old"] = test_data["Age"].apply(lambda x:\
    x > 5 and 0 or 1)
    test_data = test_data.drop("Open Date",axis=1)
    retain_list = ['City', 'City Group', 'Type', 'P1', 'P2', 'P6', 'P7', 'P8', \
    'P10', 'P11', 'P12', 'P13', 'P17', 'P20', 'P21', 'P22', 'P28', 'P29', \
    'Operational', 'Age', 'revenue']
    return test_data[retain_list]


def new_preprocessing(data_frame,test_df_ = False):
    """
    Apply LabelEncoder and SelectKBest feature based pre-processing
    """
    omit_cols = ["revenue","Open Date"]
    today = pd.to_datetime(date.today())
    encoder = LabelEncoder()
    """for colum in data_frame.columns:
        if colum not in omit_cols:
            encoded = encoder.fit(data_frame[colum]).transform(data_frame[colum])
            data_frame[colum] = encoded"""
    data_frame["Open Date"] = data_frame["Open Date"].apply(lambda x: \
    pd.to_datetime(x, format='%m/%d/%Y'))
    data_frame["Operational"] = data_frame["Open Date"].apply(lambda x:\
    (today - x).days)
    data_frame["Age"] = data_frame["Open Date"].apply(lambda x:\
    (today - x).days / 365)
    data_frame["New"] = data_frame["Age"].apply(lambda x:\
    x <= 1.5 and 0 or 1)
    data_frame["Established"] = data_frame["Age"].apply(lambda x:\
    x > 3 and 0 or 1)
    data_frame["Old"] = data_frame["Age"].apply(lambda x:\
    x > 5 and 0 or 1)
    data_frame = data_frame.drop(["Open Date",'City', 'City Group', 'Type'],axis=1)
    retain_train = ['P2', 'P6', 'P13', 'P21', 'P28', 'P29', 'Operational', 'Age','revenue']
    retain_test = ['P2', 'P6', 'P13', 'P21', 'P28', 'P29', 'Operational', 'Age']
    #retain_list_test = ['City', 'City Group', 'Type', 'P1', 'P2', 'P6', 'P7', 'P8', \
    """retain_list_test = ['P1', 'P2', 'P6', 'P7', 'P8', \
    'P10', 'P11', 'P12', 'P13', 'P17', 'P20', 'P21', 'P22', 'P28', 'P29', \
    'Operational', 'Age']
    #retain_list = ['City', 'City Group', 'Type', 'P1', 'P2', 'P6', 'P7', 'P8', \
    retain_list = ['P1', 'P2', 'P6', 'P7', 'P8', \
    'P10', 'P11', 'P12', 'P13', 'P17', 'P20', 'P21', 'P22', 'P28', 'P29', \
    'Operational', 'Age', 'revenue']"""
    if test_df_ == True:
        #return data_frame[retain_list_test]
        return data_frame[retain_test]
    else:
        #return data_frame[retain_list]
        return data_frame[retain_train]

def open_till_days(data_frame):
    """
    The restorant is open for how many days
    """
    city_map = {'\xc4\xb0zmir': 0, 'Mu\xc4\x9fla': 1, 'Tekirda\xc4\x9f': 2,\
    'Trabzon': 18, 'Afyonkarahisar': 4, 'Adana': 5, 'K\xc4\xb1rklareli': 6, \
    'Elaz\xc4\xb1\xc4\x9f': 7, 'Ayd\xc4\xb1n': 8, 'Gaziantep': 9, \
    'Ankara': 10, 'Kayseri': 11, 'Sakarya': 31, 'Tokat': 13, \
    'U\xc5\x9fak': 21, 'Bolu': 16, '\xc5\x9eanl\xc4\xb1urfa': 17, \
    'K\xc3\xbctahya': 3, 'Kocaeli': 19, 'Samsun': 20, 'Eski\xc5\x9fehir': 15,\
    'Kastamonu': 22, 'Antalya': 32, 'Karab\xc3\xbck': 23, 'Isparta': 24,\
    'Denizli': 25, 'Diyarbak\xc4\xb1r': 33, 'Osmaniye': 27, 'Amasya': 28, \
    'Bal\xc4\xb1kesir': 29, 'Bursa': 30, 'Edirne': 12,\
    '\xc4\xb0stanbul': 14, 'Konya': 26}
    today = pd.to_datetime(date.today())
    column_names = data_frame.columns
    for name in column_names:
        if name.startswith("P"):
            mapping = {label:idx for idx, label in enumerate(\
            set(data_frame[name]))}
            data_frame[name] = data_frame[name].map(mapping)
    #print column_names

    data_frame["Open Date"] = data_frame["Open Date"].apply(lambda x: \
    pd.to_datetime(x, format='%m/%d/%Y'))
    data_frame["Operational"] = data_frame["Open Date"].apply(lambda x:\
    (today - x).days)
    data_frame["Age"] = data_frame["Open Date"].apply(lambda x:\
    (today - x).days / 365)
    data_frame["New"] = data_frame["Age"].apply(lambda x:\
    x <= 1.5 and 0 or 1)
    data_frame["Established"] = data_frame["Age"].apply(lambda x:\
    x > 3 and 0 or 1)
    data_frame["Old"] = data_frame["Age"].apply(lambda x:\
    x > 5 and 0 or 1)
    
    city_class_mapping = {label:idx for idx, label in enumerate(\
    set(data_frame['City']))}
    print city_class_mapping
    data_frame["City"] = data_frame["City"].apply(cmap)
    #data_frame["City"] = data_frame["City"].map(city_class_mapping)
    
    city_gp_class_mapping = {label:idx for idx, label in enumerate(\
    set(data_frame['City Group']))}
    data_frame["City Group"] = data_frame["City Group"].map(city_gp_class_mapping)
    
    city_gp_class_mapping = {label:idx for idx, label in enumerate(\
    set(data_frame['Type']))}
    data_frame["Type"] = data_frame["Type"].map(city_gp_class_mapping)
    data_frame = data_frame.drop("Open Date",axis=1)
    return data_frame

def preprocess_data(data):
    """
    Preprocess the traininfg data to features and target
    """
    #to_reatin = ['P1', 'P2', 'P3', 'P4', \
    to_reatin = ['City', 'City Group', 'Type', 'P1', 'P2', 'P3', 'P4', \
    'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', \
    'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', \
    'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', \
    'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'Operational',\
    'Age','Established','New','Old']
    features = data[to_reatin]
    target = data.revenue
    return features, target

def preprocess_data_test(data):
    """
    Preprocess the traininfg data to features and target
    """
    data = open_till_days(data)
    #to_reatin = ['P1', 'P2', 'P3', 'P4', \
    to_reatin = ['City', 'City Group', 'Type', 'P1', 'P2', 'P3', 'P4', \
    'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', \
    'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', \
    'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', \
    'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'Operational',\
    'Age','Established','New','Old']
    features = data[to_reatin]

    return features

def gb_regressor(features,target,test):
    """
    Create GradiantBoostingRegressor
    """
    sweep_params = {'loss':['ls','huber'],'n_estimators':[1000,1250]}
    gb_reg = GridSearchCV(GradientBoostingRegressor(),sweep_params,\
    cv=3, verbose=10, refit=True, n_jobs=-1)
    _ = gb_reg.fit(features, target)
    print_grid_scores(gb_reg)
    results = gb_reg.predict(test)
    return results

def isotonic_reg(features,target,test):
    """
    Istonic Regression
    """
    sweep_params = {'increasing':[True,False,'auto']}
    iso_reg = GridSearchCV(IsotonicRegression(),sweep_params,cv=5,\
    verbose = 14,refit=True,n_jobs=-1)
    _ = iso_reg.fit(features,target)
    print_grid_scores(iso_reg)
    results = iso_reg.predict(test)
    return results

def rf_regressor(features,target,test):
    """
    Create GradiantBoostingRegressor
    """
    sweep_params = {'criterion':['mse'],'n_estimators':[1000,1250,500],\
    'max_features':['auto','sqrt','log2'],'random_state':[0,1],\
    'max_depth':[2,6],'min_samples_split':[1,3],\
    'min_samples_leaf':[1,3],'bootstrap':[True,False]}
    gb_reg = GridSearchCV(RandomForestRegressor(),sweep_params,\
    cv=5, verbose=14, refit=True, n_jobs=-1,scoring='mean_squared_error')
    #cv=3, verbose=10, refit=True, n_jobs=-1,scoring='mean_squared_error')
    _ = gb_reg.fit(features, target)
    print_grid_scores(gb_reg)
    results = gb_reg.predict(test)
    return results


def radious_neighbour_regressor(features,target,test):
    """
    Knn Regressr
    """
    sweep_params = {'weights':['uniform','distance'],\
    'algorithm':['ball_tree','kd_tree','brute','auto'],'leaf_size':[30,35,40],\
    'metric':['minkowski','euclidean','chebyshev','manhattan'],'p':[1,2]}
    rnn_reg = GridSearchCV(RadiusNeighborsRegressor(),sweep_params,cv=5,\
    verbose=12,refit=True,n_jobs=-1,scoring='mean_squared_error')
    _ = rnn_reg.fit(features,target)
    print_grid_scores(rnn_reg)
    results = knn_reg.predict(test)
    return results



def knn_regressor(features,target,test):
    """
    Knn Regressr
    """
    sweep_params = {'n_neighbors':[5,7,9,12],'weights':['uniform','distance'],\
    'algorithm':['ball_tree','kd_tree','brute','auto'],'leaf_size':[30,35,40],\
    'metric':['minkowski','euclidean','chebyshev','manhattan'],'p':[1,2]}
    knn_reg = GridSearchCV(KNeighborsRegressor(),sweep_params,cv=5,\
    verbose=12,refit=True,n_jobs=-1,scoring='mean_squared_error')
    _ = knn_reg.fit(features,target)
    print_grid_scores(knn_reg)
    results = knn_reg.predict(test)
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
    class_preds.to_csv("RF_regression_with_RFECV_30_April.csv")

if __name__ == "__main__":
    """train_data = pd.read_csv("data/train.csv", index_col="Id") 
    test_data = pd.read_csv("data/test.csv", index_col="Id")
    #test_feat = preprocess_data_test(test_data)
    #df = open_till_days(train_data)
    #feat,target = preprocess_data(df)
    prep_train = new_preprocessing(train_data,test_df_=False)
    prep_test = new_preprocessing(test_data,test_df_=True)
    target,feat = prep_train.revenue, prep_train.drop('revenue',axis=1)
    res = rf_regressor(feat,target,prep_test)"""
    train_data = pd.read_csv("train_refcv.csv")
    test_data = pd.read_csv("test_refcv.csv")
    res = rf_regressor(train_data.drop("revenue",axis=1),train_data.revenue,test_data)
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

All Categorical
BE RandomForestRegressor(bootstrap=True, compute_importances=None,
           criterion='mse', max_depth=13, max_features='log2',
           max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
           min_samples_split=2, n_estimators=1500, n_jobs=1,
           oob_score=True, random_state=1, verbose=0)
BS -5.92128527144e+12
BP {'oob_score': True, 'n_estimators': 1500, 'random_state': 1, 'criterion': 'mse', 'max_features': 'log2', 'max_depth': 13}
==================
Remove city BE RandomForestRegressor(bootstrap=True, compute_importances=None,
           criterion='mse', max_depth=13, max_features='log2',
           max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
           min_samples_split=2, n_estimators=1250, n_jobs=1,
           oob_score=True, random_state=1, verbose=0)
BS -6.00415153775e+12
BP {'oob_score': True, 'n_estimators': 1250, 'random_state': 1, 'criterion': 'mse', 'max_features': 'log2', 'max_depth': 13}
April 22 Random Forest
BS -5.85249444275e+12
BP {'oob_score': True, 'n_estimators': 2000, 'random_state': 0, 'criterion': 'mse', 'max_features': 'log2', 'max_depth': 13}
KS 1807131.05746
April 27
BE RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
           max_features='log2', max_leaf_nodes=None, min_samples_leaf=1,
                      min_samples_split=3, min_weight_fraction_leaf=0.0,
                                 n_estimators=1250, n_jobs=1, oob_score=False, random_state=1,
                                            verbose=0, warm_start=False)
                                            BS -5.91117826954e+12
                                            BP {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 1250, 'min_samples_split': 3, 'random_state': 1, 'criterion': 'mse', 'max_features': 'log2', 'max_depth': 6}
                                            
LB Score : 1730743.25768
Run 2 with city city type etc ...
BE RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
           max_features='sqrt', max_leaf_nodes=None, min_samples_leaf=3,
                      min_samples_split=1, min_weight_fraction_leaf=0.0,
                                 n_estimators=1250, n_jobs=1, oob_score=False, random_state=1,
                                            verbose=0, warm_start=False)
                                            BS -5.86463938891e+12
                                            BP {'bootstrap': True, 'min_samples_leaf': 3, 'n_estimators': 1250, 'min_samples_split': 1, 'random_state': 1, 'criterion': 'mse', 'max_features': 'sqrt', 'max_depth': 6}
LB Score 1729848.44690
April 28
New Pre-processing
BE RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
           max_features='sqrt', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=3, min_weight_fraction_leaf=0.0,
           n_estimators=1000, n_jobs=1, oob_score=False, random_state=0,
           verbose=0, warm_start=False)
BS -5.52776247923e+12
BP {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 1000, 'min_samples_split': 3, 'random_state': 0, 'criterion': 'mse', 'max_features': 'sqrt', 'max_depth': 6}
LB Score :1871827.00004,

Without city city group type
BE RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
           max_features='sqrt', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=3, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=1, oob_score=False, random_state=0,
           verbose=0, warm_start=False)
BS -5.78001007645e+12
BP {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 500, 'min_samples_split': 3, 'random_state': 0, 'criterion': 'mse', 'max_features': 'sqrt', 'max_depth': 6}
LB Score :1800677.58047

Features :
['P2', 'P6', 'P13', 'P21', 'P28', 'P29', 'Operational', 'Age']

BE RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
           max_features='sqrt', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=1, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=1, oob_score=False, random_state=0,
           verbose=0, warm_start=False)
BS -5.56575454719e+12
BP {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 500, 'min_samples_split': 1, 'random_state': 0, 'criterion': 'mse', 'max_features': 'sqrt', 'max_depth': 6}

LB Score : 1856400.02407
April 30 1
BE RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
           max_features='sqrt', max_leaf_nodes=None, min_samples_leaf=3,
           min_samples_split=1, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=1, oob_score=False, random_state=1,
           verbose=0, warm_start=False)
BS -6.22451257752e+12
BP {'bootstrap': True, 'min_samples_leaf': 3, 'n_estimators': 500, 'min_samples_split': 1, 'random_state': 1, 'criterion': 'mse', 'max_features': 'sqrt', 'max_depth': 6}
REFCV Features
LB Score:1961905.35339
"""
