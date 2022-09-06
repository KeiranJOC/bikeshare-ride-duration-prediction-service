import io
import json
import boto3
import botocore


def test_handler():
    # Set "running_locally" flag if you are running the integration test locally
    running_locally = True

    if running_locally:
        # Create Lambda SDK client to connect to appropriate Lambda endpoint
        lambda_client = boto3.client('lambda',
            region_name='ap-southeast-2',
            endpoint_url='http://127.0.0.1:3001',
            use_ssl=False,
            verify=False
        )

    else:
        lambda_client = boto3.client('lambda')

    test_event_path = './deployment/events/test.json'
    with open(test_event_path, 'rb') as f_in:
        test_event = json.load(f_in)
    
    response = lambda_client.invoke(FunctionName='PredictFunction', Payload=json.dumps(test_event))
    payload = json.load(response['Payload'])

    # Verify the response
    assert response['StatusCode'] == 200