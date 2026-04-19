import os

import mlflow
from dotenv import load_dotenv
from mlflow.deployments import get_deploy_client


def create_deployment_endpoint():
    load_dotenv()

    TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
    ROLE_ARN = os.environ["ROLE_ARN"]
    MLFLOW_IMAGE_URI = os.environ["MLFLOW_IMAGE_URI"]
    REGION_NAME = os.environ["REGION_NAME"]
    INSTANCE_TYPE = os.environ["INSTANCE_TYPE"]
    ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
    REGISTERED_MODEL_NAME = os.environ["REGISTERED_MODEL_NAME"]

    mlflow.set_tracking_uri(TRACKING_URI)

    config = dict(
        assume_role_arn=ROLE_ARN,
        execution_role_arn=ROLE_ARN,
        image_url=MLFLOW_IMAGE_URI,
        region_name=REGION_NAME,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
    )
    client = get_deploy_client("sagemaker")
    client.create_deployment(
        ENDPOINT_NAME,
        model_uri=f"models:/{REGISTERED_MODEL_NAME}@production",
        flavor="python_function",
        config=config,
    )


if __name__ == "__main__":
    create_deployment_endpoint()
