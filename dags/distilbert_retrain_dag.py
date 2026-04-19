"""
Runs every 3 months. Linear pipeline:
  1. Load & clean data
  2. Detect data drift
  3. Retrain DistilBERT
  4. Evaluate & promote
"""

from __future__ import annotations

from datetime import datetime, timedelta


from airflow import DAG
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv
import sys
import os

load_dotenv()

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}
sys.path.insert(0, os.path.join(os.getcwd(), "src", "workflow"))

with DAG(
    dag_id="distilbert_mental_health_quarterly_retrain",
    description="Quarterly retrain of DistilBERT mental-health classifier",
    schedule_interval="0 0 1 */3 *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["nlp", "distilbert", "mental-health", "mlops"],
) as dag:
    # Load & clean data
    from src.workflow.step1 import load_and_clean_data
    load_task = PythonOperator(
        task_id="load_and_clean_data",
        python_callable=load_and_clean_data,
    )

    # Detect data drift
    from src.workflow.step2 import detect_drift
    drift_task = PythonOperator(
        task_id="detect_data_drift",
        python_callable=detect_drift,
    )

    # Retrain DistilBERT
    from src.workflow.step3 import retrain_model
    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
        execution_timeout=timedelta(hours=4),
    )

    # Evaluate & promote
    from src.workflow.step4 import evaluate_and_promote
    evaluate_task = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote,
        execution_timeout=timedelta(hours=2),
    )

    load_task >> drift_task >> retrain_task >> evaluate_task
