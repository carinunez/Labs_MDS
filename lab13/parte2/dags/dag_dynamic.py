from datetime import datetime
from os.path import join
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from hiring_dynamic_functions import create_folders, load_and_merge, split_data, train_model, evaluate_models
from sklearn.tree import DecisionTreeClassifier

def choose_branch(**kwargs):
    date = kwargs.get('ds')
    threshold_date = '2024-11-01'

    if date < threshold_date:
        return 'load_and_merge'
    else:
        return 'raw_files_df2'



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
    tarea_3 = BashOperator(
        task_id='raw_files',
        bash_command="curl -o $AIRFLOW_HOME/dags/{{ execution_date.strftime('%Y-%m-%d') }}/raw/data_1.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv",
        dag=dag
    )
    tarea_4 = BranchPythonOperator(
        task_id='search_df2',
        python_callable=choose_branch, 
        provide_context=True, 
        dag=dag
    )
    tarea_4b = BashOperator(
        task_id='raw_files_df2',
        bash_command="curl -o $AIRFLOW_HOME/dags/{{ execution_date.strftime('%Y-%m-%d') }}/raw/data_1.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv",
        dag=dag
    )
    tarea_5 = PythonOperator(
        task_id='load_and_merge',
        python_callable=load_and_merge,
        provide_context=True
    )
    tarea_6 = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        provide_context=True
    )
    tarea_7 = PythonOperator(
        task_id='train_model',
        python_callable=train_model(DecisionTreeClassifier(random_state=29)),
        provide_context=True
    )

    fin = EmptyOperator(task_id='fin')

    # Definimos la estructura lineal de las tareas
    tarea_1 >> tarea_2 >> tarea_3 >> tarea_4
    tarea_4 >> tarea_5 >> tarea_6 >> tarea_7 >> fin
    tarea_4 >> tarea_4b >> tarea_5 >> tarea_6 >> tarea_7 >> fin

    #data 2: https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv