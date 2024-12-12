import numpy as np 
import pandas as pd 
import logging
import requests
import glob
import os
from os.path import join
from datetime import datetime
import joblib
import gradio as gr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_selector
import io

def create_folders(**kwargs):
    date = kwargs.get('ds')
    os.makedirs(join('.', 'dags', date), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "raw"), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "preprocessed"), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "splits"), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "models"), exist_ok=True)

def download_data(**kwargs):
    date = kwargs.get('ds')
    i = -1
    raw_path = join('.', 'dags', date, 'raw')
    for i, file in enumerate(glob.glob(join(raw_path, 'X_t*.parquet'))):
        pass
    while True:
        if i == -1:
            i += 1
            file = f"X_t{i}.parquet"
        url = f"https://gitlab.com/mds7202-2/proyecto-mds7202/-/raw/main/competition_files/{file}"
        response = requests.get(url)
        if response!= 200:
            break
        data = pd.read_parquet(io.BytesIO(response.content))
        data.to_csv()
        i += 1

    response = requests.get(url)

def load_and_merge(**kwargs):
    date = kwargs.get('ds')
    
    df1 = pd.read_csv(join('.', 'dags', date, 'raw', 'data_1.csv'))
    if os.path.exists(join('.', 'dags', date, 'raw', 'data_2.csv')):
        df2 = pd.read_csv(join('.', 'dags', date, 'raw', 'data_2.csv'))
        data = pd.concat([df1, df2], axis=0)
    else:
        data = df1.copy()

    data.to_csv(join('.', 'dags', date, 'preprocessed', 'data.csv'))

def preprocess_data(**kwargs):
    first_transformer = ColumnTransformer([
    ('scale_data', MinMaxScaler(), make_column_selector(dtype_include='number',)),
    ('object', 'drop', ['wallet_address']),
    ('categorical', 'passthrough', make_column_selector(dtype_include='category'))
    ],
    remainder='passthrough',
    verbose_feature_names_out=False)
    first_transformer.set_output(transform='pandas')
    transf_pipe = Pipeline([
                    ('add_borrow', borrow_times()),
                    ('diff_tranf', tx_diff()),
                    ('binary_cols', search_binary()),
                    ('cols_transf', first_transformer),
                    ])
    
def split_data(**kwargs):
    date = kwargs.get('ds')

    df = pd.read_csv(join('.', 'dags', date, 'preprocessed', 'data.csv'))
    X = df.drop(columns=['HiringDecision'])
    y = df['HiringDecision']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y,
                                                        random_state=29)
    
    X_train.to_csv(join('.', 'dags', date, 'splits', 'X_train.csv'), index=False)
    X_test.to_csv(join('.', 'dags', date, 'splits', 'X_test.csv'), index=False)
    y_train.to_csv(join('.', 'dags', date, 'splits', 'y_train.csv'), index=False)
    y_test.to_csv(join('.', 'dags', date, 'splits', 'y_test.csv'), index=False)

def train_model(model, **kwargs):
    date = kwargs.get('ds')
    X_train = pd.read_csv(join('.', 'dags', date, 'splits', 'X_train.csv'))
    y_train = pd.read_csv(join('.', 'dags', date, 'splits', 'y_train.csv'))

    clasico = ColumnTransformer([
        ('minmas', MinMaxScaler(), X_train.select_dtypes(include='number').columns),
        ('nada', 'passthrough', X_train.select_dtypes(include='category').columns)
    ],
    verbose_feature_names_out=True)
    clasico.set_output(transform='pandas')

    model_pipe = Pipeline([
        ('col_trans', clasico),
        ('random', model)
    ])

    model_pipe.fit(X_train, y_train)

    with open(join('.', 'dags', date, 'models', f'{model.__class__.__name__}.zlib'), 'wb') as modelfile:
        joblib.dump(model_pipe, modelfile)

def evaluate_models(**kwargs):
    date = kwargs.get('ds')

    max_acc = 0.
    best_model = None
    for file in os.listdir(join('.', 'dags', date, 'models')):
        model_name = file.split('.')[0]
        with open(join('.', 'dags', date, 'models', str(file)), 'rb') as modelfile:
            model_pipe = joblib.load(modelfile)

        X_test = pd.read_csv(join('.', 'dags', date, 'splits', 'X_test.csv'))
        y_test = pd.read_csv(join('.', 'dags', date, 'splits', 'y_test.csv'))
        y_pred = model_pipe.predict(X_test)
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        logging.info(f"Model: {model_name}, Accuracy: {acc}")
        if acc > max_acc:
            max_acc = acc
            best_model_name = model_name
            best_model = model_pipe

    with open(join('.', 'dags', date, 'models', 'besto_'+str(file)), 'wb') as modelfile:
            model_pipe = joblib.dump(best_model, modelfile)
            
    logging.info(f"Besto model: {best_model_name}, Accuracy: {max_acc}")

class borrow_times(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.borrow_per_cli = None

    def fit(self, X, y=None):
        data = X.copy()
        object_var = data.select_dtypes(include='object').columns.to_list()
        vars = object_var + ['borrow_timestamp']

        self.borrow_per_cli = data[vars].groupby(*object_var).count()
        self.borrow_per_cli.rename(columns={'borrow_timestamp': 'borrow_times'}, inplace=True)
        self.borrow_per_cli.reset_index(inplace=True)
        return self

    def transform(self, X, y=None):
        data = X.copy()
        new_X = pd.merge(data, self.borrow_per_cli, on='wallet_address', how='left').fillna(0)
        new_X = new_X.sort_index(axis=1)
        return new_X

    def set_output(self,transform='default'):
        #No modificar este método
        return self

class tx_diff(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ts_diff_tx = None

    def fit(self, X, y=None):
        data = X.copy()
        self.data = data
        return self

    def transform(self, X, y=None):
        data = X.copy()
        data['ts_diff_tx'] = data['last_tx_timestamp'] - data['first_tx_timestamp']
        data.rename(columns={'risky_first_last_tx_timestamp_diff':'ts_diff_risky_tx'}, inplace=True)
        data.drop(columns=['last_tx_timestamp',
                           'first_tx_timestamp',
                           'risky_last_tx_timestamp',
                           'risky_first_tx_timestamp',
                        #    'borrow_timestamp'
                           ], inplace=True)

        new_data = data.sort_index(axis=1)
        return new_data

    def set_output(self,transform='default'):
        return self

class search_binary(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_cols = []

    def fit(self, X, y=None):
        data = X.copy()
        for col in data.columns:
            diff_values = len(data[col].value_counts())
            is_binary = diff_values == 2
            if is_binary:
                self.binary_cols.append(col)
        return self

    def transform(self, X, y=None):
        data = X.copy()
        if self.binary_cols:
            binary_col = self.binary_cols[0] if isinstance(self.binary_cols, list) else self.binary_cols
            data[binary_col] = data[binary_col].astype('category')
        return data

    def set_output(self, transform='default'):
        return self
    

if __name__=='__main__':
    print(os.listdir("."))
