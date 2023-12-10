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



# Exploratory Data Analysis (EDA)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")

    print("\n##################### Types #####################")
    print(dataframe.dtypes)

    print("\n##################### Head #####################")
    print(dataframe.head(head))

    print("\n##################### Tail #####################")
    print(dataframe.tail(head))

    print("\n##################### NA #####################")
    print(dataframe.isnull().sum())

    print("\n##################### Quantiles #####################")
    print(dataframe.describe().T)


check_df(df)

leaveornot_values = {
    1: 'Leave',
    0: 'Stay'
}

df.leaveornot = df.leaveornot.map(leaveornot_values)


##################################################
# Capturing numerical and Categorical variables
##################################################

# This function analyzes the columns in a dataframe and determines categorical, numerical, and other columns.

def grab_col_names(dataframe, cat_th=8, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() >= car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#########################################
# Analysis of Categorical Variables
#########################################

# This function performs the analysis and visualization of categorical variables.

def cat_summary(dataframe, col_name, plot=True):

    value_counts = dataframe[col_name].value_counts()
    ratio = 100 * value_counts / len(dataframe)

    if plot:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=dataframe[col_name], data=dataframe, order=value_counts.index)
        plt.title(f"{col_name} Distribution")
        plt.xlabel("Count")
        plt.ylabel(col_name)
        plt.xticks(rotation=0)

        for i, v in enumerate(value_counts.values):
            plt.text(v + 1, i, f"{v} ({ratio.iloc[i]:.2f}%)", va='center')

        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)

########################################
# Analysis of Numerical Variables
########################################

# This function performs the analysis of numerical variables and, if desired, visualizes them.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    summary = dataframe[numerical_col].describe(quantiles)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(18, 4))

        # Plot histogram
        dataframe[numerical_col].hist(bins=20, ax=axes[0], color="lightgreen")
        axes[0].set_xlabel(numerical_col)
        axes[0].set_title(numerical_col)

        # Displaying summary statistics as text
        summary_text = "\n".join([f'{col}: {value:.3f}' for col, value in summary.items()])
        axes[1].text(0.5, 0.5, summary_text, fontsize=12, va='center', ha='left', linespacing=1.5)
        axes[1].axis('off')

        plt.show()
        print("################################################################")


for col in num_cols:
    num_summary(df, col, plot=True)

########################################
#   Target Variable Analysis
########################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
    print("##########################################")

for col in num_cols:
    target_summary_with_num(df, "leaveornot", col)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"leaveornot_count": dataframe.groupby(categorical_col)[target].count()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "leaveornot", col)



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






