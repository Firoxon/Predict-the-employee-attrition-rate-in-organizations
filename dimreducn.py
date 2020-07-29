# code to find the variables of importance to reduce the number of variables
import pandas as pd
import numpy as np
# loading the data
train_df = pd.read_csv('Dataset/Train.csv')
test_df = pd.read_csv('Dataset/Test.csv')
# showing a little dataset
train_df.head()
# importing the modules
from sklearn.preprocessing import MinMaxScaler
# defining the func for data prep
def prepare_data(train_df, test_df, id, target_col):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in train_df.drop(id, axis=1).columns:
        if train_df[col].dtype == object:
            dummies = pd.get_dummies(train_df[col], drop_first=True)
            for _col in dummies:
                train_df["%s_%s" % (col, _col)] = dummies[_col]
                test_df["%s_%s" % (col, _col)] = dummies[_col]
            train_df.drop(col, axis=1, inplace=True)
            test_df.drop(col, axis=1, inplace=True)
    X_train = train_df.drop([id, target_col], axis=1).fillna(train_df.mean())
    columns = train_df.drop([id, target_col], axis=1).fillna(train_df.mean()).columns
    X_test = test_df.drop([id], axis=1).fillna(0).values
    y = train_df[target_col].values

    return X_train, X_test, y, columns

id = 'Employee_ID'
target = 'Attrition_rate'

X_train, X_test, y, columns = prepare_data(train_df, test_df, id, target)
# using the random forest classifier for better results
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y)

feature_list = list(zip(list(rf.feature_importances_), list(columns)))
sorted(feature_list)
# features to keep
to_keep = list(filter(lambda x: x[0] > 0.015, feature_list))
# printing the list
list(map(lambda x: x[1], to_keep))
