#!/usr/bin/env python
import pandas as pd
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.preprocessing import LabelEncoder


def find_kbest_features(csv_file,k=20):
    """
    Select K-Best features fro the gicen data
    :param csv_file: csv file containing the data
    :param k: number of fatures to be selected
    :returns feat_imp: Feature importance with Annova F value, P values
    and support. Th greater the Annova F Value (Score) the better the 
    feature, the lesser the p-value the feature is good. If the support
    is True it means that the feature is important
    Steps: First all the categorical values are converted to numeric 
    using the sklearn LabelEncoder (Because SelectKBest assumes that 
    the data is numeric in nature). Then the slect k best model is fited.
    """
    data = pd.read_csv(csv_file)
    data.astype(object).fillna(0) #Fill the NA values with 0
    
    encoder = LabelEncoder()
    
    for column in data.columns:
        _ = encoder.fit(data[column])
        data[column] = encoder.transform(data[column])
    #Transformed string attributes/categorical to int using LabelEncoder
    
    kbest_selector = SelectKBest(f_classif,k=20).fit(\
    data.drop("DROP",axis=1),data.DROP)
    #Filtted the SelectKBest model
    
    feat_imp = pd.DataFrame()
    feat_imp["Annova F Score"] = kbest_selector.scores_
    feat_imp["P Valu"] = kbest_selector.pvalues_
    feat_imp["Support"] = kbest_selector.get_support()
    feat_imp["Attribute Name"] = data.drop("DROP",axis=1).columns
    #Extracted the feature importnace values and storing it as pandas
    DataFrame
    
    feat_imp.to_csv("feature_importance.csv",index=False)
    #Writing the DataFrame to CSV file
    return feat_imp

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    feats = find_kbest_features(filename)
    