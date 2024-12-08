from datetime import datetime
from os.path import join
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from hiring_dynamic_functions import create_folders, load_and_merge, split_data, train_model, evaluate_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

def choose_branch(**kwargs):
    date = kwargs.get('ds')
    threshold_date = datetime.strptime('2024-11-01', "%Y-%m-%d")
    date = datetime.strptime(date, "%Y-%m-%d")
    if date < threshold_date:
        return 'dl_data_1'
    else:
        return ['dl_data_1', 'dl_data_2']
    

start_date = datetime(2024, 10, 1)
default_args = {
    'owner': 'airflow',
    # 'depends_on_past': False,  # Evita backfill
    'start_date': start_date,
    'retries': 0,  # No intentos adicionales
}

with DAG(
    dag_id='hiring_dynamic',  # Cambiar por un ID que desees
    default_args=default_args,
    schedule_interval='0 15 5 * *',  
    catchup=True,  # Realiza backfill
    description='DAG lineal para contratación',
    tags=['example', 'lineal', 'contratación']  # Opcional, etiquetas
) as dag:

    # Definimos tareas dummy como placeholders
    tarea_1 = EmptyOperator(task_id='inicio')
    
    tarea_2 = PythonOperator(
        task_id='crear_carpetas',
        python_callable=create_folders,
        provide_context=True  # Proveer contexto para acceso a ds (execution_date)
    )
    branch_1 = BranchPythonOperator(
        task_id='branch_1',
        python_callable=choose_branch, 
        provide_context=True, 
        dag=dag
    )
    dl_data_1 = BashOperator(
        task_id='dl_data_1',
        bash_command="curl -o $AIRFLOW_HOME/dags/{{ execution_date.strftime('%Y-%m-%d') }}/raw/data_1.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv",
        dag=dag
    )
    dl_data_2 = BashOperator(
        task_id='dl_data_2',
        bash_command="curl -o $AIRFLOW_HOME/dags/{{ execution_date.strftime('%Y-%m-%d') }}/raw/data_2.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv",
        dag=dag
    )
    loadmerge = PythonOperator(
        task_id='load_and_merge',
        python_callable=load_and_merge,
        provide_context=True,
        trigger_rule='one_success'
    )
    split = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        provide_context=True
    )
    def train_operator(model):
        train_op = PythonOperator(
            task_id=f'train_model_{model.__class__.__name__}',
            python_callable=train_model,
            provide_context=True,
            op_kwargs={'model':model}
        )
        return train_op
    
    eval = PythonOperator(
        task_id='eval_model',
        python_callable=evaluate_models,
        provide_context=True,
        trigger_rule='all_success'
    )
    fin = EmptyOperator(task_id='fin')

    # Definimos la estructura lineal de las tareas
    tarea_1 >> tarea_2 >>  branch_1 # Tronco
    branch_1 >> [dl_data_1, dl_data_2] >> loadmerge >> split
    split >> [train_operator(RandomForestClassifier(random_state=29)), train_operator(DecisionTreeClassifier(random_state=29)), train_operator(DummyClassifier(random_state=29)) ] >> eval >> fin