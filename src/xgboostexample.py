import xgboost as xgb
from sklearn.datasets import load_iris, load_digits, load_boston
boston = load_boston()
boston["data"]
import pandas as pd
bos = pd.DataFrame(boston["data"])
bos.describe()
boston["data"]
bos.as_matrix()
xgb_model = xgb.XGBRegressor().fit(boston["data"],boston["target'])
xgb_model = xgb.XGBRegressor().fit(boston["data"],boston["target"])
xgb_model
train_data = pd.read_csv("data/train.csv", index_col="Id")
from latest import open_till_days,preprocess_data
df = open_till_days(train_data)
feat,target = preprocess_data(df)
feat.head()
feat = feat.as_matrix()
feat
target = target.as_matrix()
xgb_model = xgb.XGBRegressor().fit(feat,target)
help(xgb_model)
test_data = pd.read_csv("data/test.csv", index_col="Id")
from latest import open_till_days,preprocess_data,preprocess_data_test
test_feat = preprocess_data_test(test_data)
predictions = xgb_model.predict(test_data.as_matrix())
test_data.as_matrix()
test_data.describe()
test_data.head()
test_data_a = test_data.drop(['Open Date'])
test_data_a = test_data.drop(['Open Date'],axis=1)
test_data_a.head()
feat.head()
predictions = xgb_model.predict(test_data.as_matrix())
test_data_a.head()
test_data.describe()
predictions = xgb_model.predict(test_data_a.as_matrix())
predictions
test_data.head()
res = pd.DataFrame(predictions)
res.head()
res.to_csv("Xgboost_res_April_25.csv")
xgb_model
xgb_model = xgb.XGBRegressor(n_estimators=1000).fit(feat,target)
predictions = xgb_model.predict(test_data_a.as_matrix())
res = pd.DataFrame(predictions)
res.to_csv("Xgboost_res_April_25_2.csv")
test_data.head()
train_data.head()
feat,target = preprocess_data(df)
feat.head()
test_feat = preprocess_data_test(test_data)
test_feat.head()
import keras.models
help(keras.models)
train_data = pd.read_csv("data/train.csv", index_col="Id")
train_data.summary()
train_data.describe()
%history
