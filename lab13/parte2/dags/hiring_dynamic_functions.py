import numpy as np 
import pandas as pd 
import logging
import os
from os.path import join
from datetime import datetime
import joblib
import gradio as gr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split



def create_folders(**kwargs):
    date = kwargs.get('ds')
    print(type(date), date)

    os.makedirs(join('.', 'dags', date), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "raw"), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "preprocessed"), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "splits"), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "models"), exist_ok=True)

def load_and_merge(**kwargs):
    date = kwargs.get('ds')
    
    df1 = pd.read_csv(join('.', 'dags', date, 'raw', 'data_1.csv'))
    if os.path.exists(join('.', 'dags', date, 'raw', 'data_2.csv')):
        df2 = pd.read_csv(join('.', 'dags', date, 'raw', 'data_2.csv'))
        data = pd.concat([df1, df2], axis=0)
    else:
        data = df1.copy()

    data.to_csv(join('.', 'dags', date, 'preprocessed', 'data.csv'))

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


if __name__=='__main__':
    print(os.listdir("."))
