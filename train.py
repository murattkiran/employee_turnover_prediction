# Importing Libraries and Datasets

import pickle
import xgboost as xgb
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


df = pd.read_csv("employee.csv")
df.columns = df.columns.str.lower()

leaveornot_values = {
    1: 'Leave',
    0: 'Stay'
}

df.leaveornot = df.leaveornot.map(leaveornot_values)


# Data Preparation

from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = (df_train.leaveornot == 'Leave').astype('int').values
y_val = (df_val.leaveornot == 'Leave').astype('int').values
y_test = (df_test.leaveornot == 'Leave').astype('int').values


del df_train['leaveornot']
del df_val['leaveornot']
del df_test['leaveornot']



# We see that **XGBoost** is the best model.

print('training the final model')

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = (df_full_train.leaveornot == 'Leave').astype(int).values

del df_full_train['leaveornot']

dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

#xgb
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names_out().tolist())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out().tolist())


xgb_params = {
    'eta': 0.1,
    'max_depth': 4,
    'min_child_weight': 10,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=70)
y_pred = model.predict(dtest)
auc_score = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {auc_score}')


# Save the model
output_file = "model.bin"

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')






