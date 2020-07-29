import pandas as pd
import numpy as np
import seaborn as sns
# First step: read the data

train_df = pd.read_csv('Dataset/Train.csv')
test_df = pd.read_csv('Dataset/Test.csv')

train_df.head()

# Second step: fill missing values with the mean of its respective column.
# Luckily no categorical columns have missing values in our data. Peace!!

nan_cols = []

for col in train_df.columns:
    if train_df[col].isnull().values.any():
        nan_cols.append(col)

for col in nan_cols:
        train_df[col] = train_df[col].fillna(train_df[col].mean())
        test_df[col] = test_df[col].fillna(test_df[col].mean())

# Function prepare_data transforms the dataframes by turning categorical columns into one-hot encoded array-columns
# and then proceeds to scale the data using a minmaxscaler. the train data is split into train data, used to train
# the network and dev data, used to assess the network's performance. Test data is the data which predictions will
# be submitted to HackerEarth's system

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(train_df, test_df, id, target_col, one_hot_vars=False):
    train_df = train_df.copy()
    test_df = test_df.copy()


    for col in train_df.drop(id, axis=1).columns:
        if train_df[col].dtype == object or (one_hot_vars and col in ['VAR5', 'VAR6']):
            dummies = pd.get_dummies(train_df[col], drop_first=True)
            for _col in dummies:
                train_df["%s_%s" % (col, _col)] = dummies[_col]
                test_df["%s_%s" % (col, _col)] = dummies[_col]
            train_df.drop(col, axis=1, inplace=True)
            test_df.drop(col, axis=1, inplace=True)

    X_train, X_dev, y_train, y_dev = train_test_split(train_df.drop([id, target], axis=1), train_df[[target_col]])
    cols_outliers = []

    X_train_ = X_train.fillna(X_train.mean()).values
    X_dev = X_dev.fillna(X_dev.mean()).values

    X_test = test_df.drop([id], axis=1).values
    y = train_df[target_col].values

    sc = MinMaxScaler().fit(X_train_)

    return sc.transform(X_train_), \
           sc.transform(X_dev), \
           sc.transform(X_test),  \
           y_train.loc[X_train.index].values.reshape(len(X_train)), \
           y_dev.values.reshape(len(y_dev)), \
           train_df.drop([id, target_col], axis=1).columns

id = 'Employee_ID'
target = 'Attrition_rate'


# These columns/features were selected using a random forest regressor, trained with all the data. Later I printed
# feature importances and selected features which importance was above a certain threshold
# (file 'Dimensionality reduction.ipynb')
features_keep = ['Age',
 'Education_Level',
 'Time_of_service',
 'Time_since_promotion',
 'growth_rate',
 'Travel_Rate',
 'Post_Level',
 'Pay_Scale',
 'Work_Life_balance',
 'VAR1',
 'VAR2',
 'VAR3',
 'VAR4',
 'VAR5',
 'VAR6',
  'VAR7']

cols = features_keep + [id]

X_train, X_dev, X_test, y_train, y_dev, features = prepare_data(train_df[cols + [target]], test_df[cols], id, target, one_hot_vars=True)

# Useful imports

import math

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam

from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

from sklearn.metrics import mean_squared_error

# Function that defines the neural network. It uses dropout regularization and Adam optimizer, with a very small
# learning rate. I have observed that neural networks tend to overfit pretty fast to the provided data, so I had
# to reduce the learning rate as much as I could without having to train the model for too long

features = len(X_train[0])

def linear_model():
    model = Sequential()

    model.add(Dense(math.ceil(1.2 * features / 2), kernel_initializer='random_normal', activation='relu', input_dim=features))
    model.add(Dropout(0.4))
    model.add(Dense(math.ceil(0.6 * features),  kernel_initializer='random_normal', activation='relu', kernel_constraint=maxnorm(5)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', kernel_constraint=maxnorm(5)))
    model.compile(optimizer=Adam(lr=0.0001), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])


    return model


def print_estimations(ids, predictions, filename='submissions.csv'):
    with open(filename, 'w') as file:
        file.write("Employee_ID,Attrition_rate\n")
        for pred in zip(ids, predictions):
            file.write("%s,%s\n" % pred)

ln_model = linear_model()
best_weighted_error = None
best_total_error = None

sup = np.argwhere(y_train >= 0.3)
sup = sup.reshape(1,len(sup))[0]
X_train_sup = X_train[sup]
y_sup = y_train[sup]


best = None


for i in range(0,3):
    ln_model.fit(X_train_sup, y_sup, epochs=1, batch_size=256, verbose=0)

    ln_model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=0)

    total_error = np.sqrt(mean_squared_error(y_dev, ln_model.predict(X_dev)))
    print(total_error)
    if best is None or best > total_error:
        best_total_error = total_error
        preds = ln_model.predict(X_test).reshape((1, len(X_test)))[0]
        print_estimations(test_df[id].values, preds)


preds = ln_model.predict(X_test).reshape((1, len(X_test)))[0]
print_estimations(test_df[id].values, preds)

total_error = np.sqrt(mean_squared_error(y_dev, ln_model.predict(X_dev)))
print(total_error)

sns.distplot(preds)
