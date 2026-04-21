def evaluate_and_promote(**ctx):
    import os

    import mlflow
    import pandas as pd
    import sys
    sys.path.insert(0, "/opt/airflow/dags/src/utility")
    from testing import score_model

    MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"]
    REGISTERED_MODEL_NAME = os.environ["REGISTERED_MODEL_NAME"]
    TEST_CSV = os.environ["TEST_PATH"]
    REF_PATH = os.environ["REFERENCE_PATH"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    new_model_version = ctx["ti"].xcom_pull(
        key="new_model_version", task_ids="retrain_model"
    )
    new_run_id = ctx["ti"].xcom_pull(key="new_run_id", task_ids="retrain_model")
    test_df = pd.read_csv(TEST_CSV)

    new_model = mlflow.pyfunc.load_model(
        f"models:/{REGISTERED_MODEL_NAME}/{str(new_model_version)}"
    )
    new_f1 = score_model(new_model, test_df)
    print(f"New model test macro-F1: {new_f1:.4f}")

    prod_f1 = 0.0
    prod_model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@production")
    prod_f1 = score_model(prod_model, test_df)
    print(f"Production model test macro-F1: {prod_f1:.4f}")

    with mlflow.start_run(run_id=new_run_id):
        mlflow.log_metrics(
            {
                "test_macro_f1": new_f1,
                "prod_macro_f1": prod_f1,
            }
        )

    if new_f1 >= prod_f1:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME, alias="production", version=new_model_version
        )
        update_deployment_endpoint()

        pd.read_parquet(
            ctx["ti"].xcom_pull(key="cleaned_path", task_ids="load_and_clean_data")
        ).to_parquet(REF_PATH, index=False)
        print(f"Reference data updated → {REF_PATH}")
    else:
        print(f"New F1 ({new_f1:.4f}) < production ({prod_f1:.4f}) — no promotion.")
