import numpy as np 
import pandas as pd 
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

# /usr/bin/bash -c curl -o ./dags/2024-12-05/raw/data_1.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv
# /usr/bin/bash -c curl -o dags/2024-12-04/raw/ https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv
def create_folders(**kwargs):
    date = kwargs.get('ds')

    os.makedirs(join('.', 'dags', date), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "raw"), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "splits"), exist_ok=True)
    os.makedirs(join('.', 'dags', date, "models"), exist_ok=True)

def split_data(**kwargs):
    date = kwargs.get('ds')

    df = pd.read_csv(join('.', 'dags', date, 'raw', 'data_1.csv'))
    X = df.drop('HiringDecision')
    y = df['HiringDecision']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=True,
                                                        random_state=29)

    X_train.to_csv(join('.', 'dags', date, 'splits', 'X_train.csv'))
    X_test.to_csv(join('.', 'dags', date, 'splits', 'X_test.csv'))
    y_train.to_csv(join('.', 'dags', date, 'splits', 'y_train.csv'))
    y_test.to_csv(join('.', 'dags', date, 'splits', 'y_test.csv'))

def preprocess_and_train(**kwargs):
    date = kwargs.get('ds')
    X_train = pd.read_csv(join('.', 'dags', date, 'splits', 'X_train.csv'))
    y_train = pd.read_csv(join('.', 'dags', date, 'splits', 'y_train.csv'))

    X_test = pd.read_csv(join('.', 'dags', date, 'splits', 'X_test.csv'))
    y_test = pd.read_csv(join('.', 'dags', date, 'splits', 'y_test.csv'))

    clasico = ColumnTransformer([
        ('minmas', MinMaxScaler(), X_train.select_dtypes(include='number').columns),
        ('nada', 'passthrough', X_train.select_dtypes(include='category').columns)
    ],
    verbose_feature_names_out=True)
    clasico.set_output('pandas')

    rf = RandomForestClassifier(random_state=29)

    pipe_clasica = Pipeline([
        ('col_trans', clasico),
        ('random', rf)
    ])
    rf_pipe = pipe_clasica
    rf_pipe.fit(X_train, y_train)
    y_pred = rf_pipe.predict(X_test)
    
    print("F1-Score:", f1_score(y_true=y_test, y_pred=y_pred, average='weighted')) 
    print("Accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred)) 

    with open(join('.', 'dags', date, 'models','randomforest.zlib' 'wb')) as randomfile:
        joblib.dump(rf_pipe, randomfile)

def predict(file, model_path):
    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}

def gradio_interface(**kwargs):
    date = kwargs.get('ds')
    model_path = join('.', 'dags', date, 'models', 'randomforest.zlib')

    interface = gr.Interface(
        fn = lambda file: predict(file, model_path),
        inputs = gr.File(label="Sube un archivo JSON"),
        outputs = "json",
        title = "Hiring Decision Prediction",
        description = "Sube un archivo JSON con las características de entrada para predecir si Nico será contratado o no."
    )
    
    interface.launch(share=True)

if __name__=='__main__':
    # create_folders()
    ds = datetime(2020,10,10)
    create_folders(ds=ds)
    