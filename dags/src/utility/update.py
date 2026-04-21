import os

import mlflow
from dotenv import load_dotenv
from mlflow.deployments import get_deploy_client


def update_deployment_endpoint():
    load_dotenv()

    TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
    ROLE_ARN = os.environ["ROLE_ARN"]
    REGION_NAME = os.environ["REGION_NAME"]
    ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
    MODEL_REGISTRY_NAME = os.environ["REGISTERED_MODEL_NAME"]

    mlflow.set_tracking_uri(TRACKING_URI)

    config = dict(
        assume_role_arn=ROLE_ARN,
        region_name=REGION_NAME,
        mode="replace",
    )
    client = get_deploy_client("sagemaker")
    client.update_deployment(
        ENDPOINT_NAME,
        model_uri=f"models:/{MODEL_REGISTRY_NAME}@production",
        flavor="python_function",
        config=config,
    )


if __name__ == "__main__":
    update_deployment_endpoint()
