
import numpy as np
import pandas as pd
import mlflow, optuna, os, pickle
# El submodulo de matplotlib, es más rapida la descarga de archivos
from optuna.visualization.matplotlib import (plot_optimization_history, plot_param_importances, 
                                    plot_parallel_coordinate)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# Data
data = pd.read_csv('water_potability.csv')
data.head()

# Preprocesamiento
X = data.drop(columns='Potability').copy()
y = data.Potability.copy()

# Funcion a optimizar
def objective_function(trial):

    # Split into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
                                            X, y, test_size=0.3, random_state=29, 
                                            shuffle=True)

    # Hyperparameters to tune
    xgb_params = {
            "objective": "binary:logistic",
            "n_estimators": trial.suggest_int("n_estimators", 10, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            'max_leaves': trial.suggest_int("max_leaves", 3, 30),
            "grow_policy": trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "n_jobs": trial.suggest_int('n_jobs', 1, 3),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }


    # Train model
    model = XGBClassifier(seed=29, **xgb_params)
    model.fit( X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],)
    
    # Predict and evaluate the model
    yhat = model.predict(X_valid)
    f1 = f1_score(y_valid, yhat, average='weighted')
   
    run_name = f"XGB_con_lr_{xgb_params['learning_rate']:.5f}_n_estimators_{xgb_params['n_estimators']}\
                _Mdepth_{xgb_params['max_depth']}_Mleaves_{xgb_params['max_leaves']}"
    
    with mlflow.start_run(run_name=run_name):
        # cargo los parametros y metricas a mlflow 
        mlflow.log_params(xgb_params)
        mlflow.log_metric('valid_f1', f1)
        mlflow.sklearn.log_model(model, "model")

    return f1

def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")
    return best_model

# Busca el id del experimento según su nombre
def search_id(exp_name):
    return dict(mlflow.get_experiment_by_name(exp_name))['experiment_id']

def optimize_model():
    
    exp_name='XGB_Clf'
    mlflow.set_experiment(exp_name)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_function, n_trials=50)
    
    with mlflow.start_run():
        os.makedirs('plots', exist_ok=True)
        # optimization_history
        # el submodulo de matplotlib retorna un Axesplot, no una figura
        opti_history = plot_optimization_history(study).figure 
        opti_history.tight_layout()

        opti_history.savefig('plots/opti_history.png')

        # # parallel coordinate 
        parallel = plot_parallel_coordinate(study).figure
        parallel.tight_layout()
        parallel.savefig('plots/parallel_coordinate.png')

        # # params importances
        importances = plot_param_importances(study).figure
        importances.tight_layout()
        importances.savefig('plots/param_importances.png')
            
        # subo la carpeta plots (local) completa a mlflow
        mlflow.log_artifacts('plots', artifact_path='plots')

    best_model = study.best_trial.params
    print(study.best_trial)
    print('Best model params:', best_model)

    exp_id = search_id(exp_name)
    best_model = get_best_model(exp_id)

    os.makedirs('model', exist_ok=True)
    with open('model/best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
        

if __name__ == "__main__":
    optimize_model()