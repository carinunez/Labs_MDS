from datetime import datetime
from os.path import join
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from hiring_functions import create_folders, split_data, preprocess_and_train, predict, gradio_interface


start_date = datetime(2024, 10, 1)
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,  # Evita backfill
    'start_date': start_date,
    'retries': 0,  # No intentos adicionales
}

with DAG(
    dag_id='hiring_lineal',  # Cambiar por un ID que desees
    default_args=default_args,
    schedule_interval=None,  # Ejecución manual
    catchup=False,  # No realizar backfill
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
    tarea_4 = PythonOperator(
        task_id='train_test_split',
        python_callable=split_data,
        provide_context=True
    )
    tarea_5 = PythonOperator(
        task_id='preprocess_train',
        python_callable=preprocess_and_train,
        provide_context=True
    )
    tarea_6 = PythonOperator(
        task_id='to_gradio_app',
        python_callable=gradio_interface,
        provide_context=True
    )
    fin = EmptyOperator(task_id='fin')

    # Definimos la estructura lineal de las tareas
    tarea_1 >> tarea_2 >> tarea_3 >> tarea_4 >> tarea_5 >> tarea_6 >> fin