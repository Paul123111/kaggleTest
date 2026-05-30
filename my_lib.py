#
# SETUP
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from IPython.display import display, Markdown
pd.set_option('display.max_columns', None)
sns.set_style('darkgrid')

SEED = 2026

#
# IMPORTS
#
def import_data(location, weather_agg, df_zika, df_dengue, df_score, df_weather, debug=False):
    criteria = "location == @location"
    df_z = df_zika.query(criteria)
    df_d = df_dengue.query(criteria)
    df_s = df_score.query(criteria)
    df_w = df_weather.query(criteria)

    df_i = pd.concat([df_z, df_s], ignore_index=True)
    df_w = pd.merge_asof(df_w, df_i[['EW_start_date']],
                         left_on='date', right_on='EW_start_date', direction='backward')
    
    cols = df_w.columns.to_list()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    df_w = df_w[cols].copy()

    df_w_agg = df_w.groupby("EW_start_date").agg(weather_agg).reset_index()
    cols = [(f"{f}__{a}" if a else f"{f}") for f,a in df_w_agg.columns]
    df_w_agg.columns = cols
    
    if debug:
        print(f"Data imported for location {location}")

    criteria = "EW_start_date >= '2016-01-01'"
    df_z.query(criteria).to_feather(f"data/zika_{location}.feather")
    df_d.query(criteria).to_feather(f"data/dengue_{location}.feather")
    df_s.query(criteria).to_feather(f"data/score_{location}.feather")
    df_w_agg.to_feather(f"data/weather_{location}.feather")

#
# ML IMPORTS
#
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import randint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error

#
# MODEL BUILDING
#
def prepare_data(df_train, df_test, df_d, df_w, z_features, target, weather_lag=3, debug=False):
    y_train = df_train[target].values
    y_test = df_test[target].values if target in df_test.columns else None

    # zika
    df_train_tmp = df_train.copy()
    df_train_tmp['Month'] = df_train_tmp['EW_start_date'].dt.month
    df_train_tmp = df_train_tmp[z_features].copy()

    df_test_tmp = df_test.copy()
    df_test_tmp['Month'] = df_test_tmp['EW_start_date'].dt.month
    df_test_tmp = df_test_tmp[z_features].copy()

    # dengue
    df_d_tmp = df_d[['EW_start_date', 'cases']].copy()
    df_d_tmp.rename(columns={'cases': 'dengue'}, inplace=True)

    # weather
    df_w_tmp = df_w.copy()
    for k in range(1, weather_lag+1):
        df_tmp = df_w_tmp.shift(k)
        df_tmp.EW_start_date = df_w_tmp.EW_start_date
        df_w_tmp = pd.merge(df_w_tmp, df_tmp, suffixes=('', f'__lag{k}'), on='EW_start_date', how='left')

    # merge
    df_train_tmp = pd.merge(df_train_tmp, df_d_tmp, on='EW_start_date', how='left')
    df_train_tmp = pd.merge(df_train_tmp, df_w_tmp, on='EW_start_date', how='left')
    #
    df_test_tmp = pd.merge(df_test_tmp, df_d_tmp, on='EW_start_date', how='left')
    df_test_tmp = pd.merge(df_test_tmp, df_w_tmp, on='EW_start_date', how='left')

    # features
    cat_features = ['Month']
    num_features = [f for f in df_train_tmp.columns[2:] if f not in cat_features]
    if debug:
        print(f"{cat_features = }")
        print(f"{num_features = }")

    # preprocessing
    ss = StandardScaler()
    X_train_num = ss.fit_transform(df_train_tmp[num_features])
    X_test_num = ss.transform(df_test_tmp[num_features])

    one = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = one.fit_transform(df_train_tmp[cat_features])
    X_test_cat = one.transform(df_test_tmp[cat_features])

    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    pipeline = Pipeline([
        ('pca', PCA()),
        ('classifier', LinearRegression())
    ])

    param_distributions = {
        'pca__n_components': range(1, 100),
    }

    param_range = np.logspace(-4,4,20)
    param_grid = {'clf__C': param_range,
    ' clf__gamma': param_range}

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=100,
        cv=5,
        random_state=SEED,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    print(f"Best Score: {random_search.best_score_:.4f}")
    print(f"Best Parameters: {random_search.best_params_}")

    best_pipeline = random_search.best_estimator_

    X_train = best_pipeline.named_steps['pca'].transform(X_train)
    X_test = best_pipeline.named_steps['pca'].transform(X_test)

    return X_train, X_test, y_train, y_test

def prepare_mining_data(df_train, df_test, df_d, df_w, z_features, target, preprocessor: ColumnTransformer, weather_lag=3, debug=False, random_state=SEED):
    y_train = df_train[target].values
    y_test = df_test[target].values if target in df_test.columns else None

    # zika
    df_train_tmp = df_train.copy()
    df_train_tmp['Month'] = df_train_tmp['EW_start_date'].dt.month
    df_train_tmp = df_train_tmp[z_features].copy()

    df_test_tmp = df_test.copy()
    df_test_tmp['Month'] = df_test_tmp['EW_start_date'].dt.month
    df_test_tmp = df_test_tmp[z_features].copy()

    # dengue
    df_d_tmp = df_d[['EW_start_date', 'cases']].copy()
    df_d_tmp.rename(columns={'cases': 'dengue'}, inplace=True)

    # weather
    df_w_tmp = df_w.copy()
    for k in range(1, weather_lag+1):
        df_tmp = df_w_tmp.shift(k)
        df_tmp.EW_start_date = df_w_tmp.EW_start_date
        df_w_tmp = pd.merge(df_w_tmp, df_tmp, suffixes=('', f'__lag{k}'), on='EW_start_date', how='left')

    # merge
    df_train_tmp = pd.merge(df_train_tmp, df_d_tmp, on='EW_start_date', how='left')
    df_train_tmp = pd.merge(df_train_tmp, df_w_tmp, on='EW_start_date', how='left')

    df_test_tmp = pd.merge(df_test_tmp, df_d_tmp, on='EW_start_date', how='left')
    df_test_tmp = pd.merge(df_test_tmp, df_w_tmp, on='EW_start_date', how='left')

    X_train = preprocessor.fit_transform(df_train_tmp)
    X_test = preprocessor.transform(df_test_tmp)

    return X_train, X_test, y_train, y_test


