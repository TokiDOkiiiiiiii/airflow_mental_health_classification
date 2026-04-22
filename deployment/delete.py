import os

from dotenv import load_dotenv
from mlflow.deployments import get_deploy_client


def delete_deployment_endpoint():
    load_dotenv()

    ROLE_ARN = os.environ["ROLE_ARN"]
    REGION_NAME = os.environ["REGION_NAME"]
    ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]

    config = dict(
        assume_role_arn=ROLE_ARN,
        region_name=REGION_NAME,
    )
    client = get_deploy_client("sagemaker")
    client.delete_deployment(
        ENDPOINT_NAME,
        config=config,
    )


if __name__ == "__main__":
    delete_deployment_endpoint()
