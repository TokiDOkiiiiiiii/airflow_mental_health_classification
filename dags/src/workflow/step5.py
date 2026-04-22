def update_deployment(**ctx):
    from src.utility.update import update_deployment_endpoint
    if ctx["ti"].xcom_pull(key="deploy", task_ids="evaluate_and_promote"):
        update_deployment_endpoint()
    else:
        print("skipped")
