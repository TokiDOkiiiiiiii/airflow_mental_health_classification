def update_deployment(**ctx):
    from src.utility.update import update_deployment_endpoint
    new_model_version = ctx["ti"].xcom_pull(
        key="new_model_version", task_ids="retrain_model"
    )
    update_deployment_endpoint(new_model_version)
