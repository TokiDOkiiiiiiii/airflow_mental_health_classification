import json
import os

import boto3
from dotenv import load_dotenv

load_dotenv()

REGION_NAME = os.environ["REGION_NAME"]
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]

# Initialize the runtime client
sm = boto3.client("sagemaker", region_name=REGION_NAME)
runtime = boto3.client("sagemaker-runtime", region_name=REGION_NAME)

# Check endpoint status
endpoint = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
print(endpoint["EndpointStatus"])

# Prepare your payload
payload = json.dumps(
    {"dataframe_records": [{"text": "I want to kill myself everyday after i met you"}]}
)

# Invoke the endpoint
response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME, ContentType="application/json", Body=payload
)

# Parse result
result = json.loads(response["Body"].read().decode())
print(result)
