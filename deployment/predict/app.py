import os
import warnings

import model

# prevent multiprocessing warnings in AWS logs
warnings.filterwarnings(action='ignore')

MODEL_NAME = os.getenv('MODEL_NAME', 'bikeshare-ride-duration-regressor')
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI',
    'http://ec2-54-79-228-176.ap-southeast-2.compute.amazonaws.com:5000/',
)


model_service = model.init(model_name=MODEL_NAME, tracking_uri=MLFLOW_TRACKING_URI)


def lambda_handler(event, context):
    return model_service.lambda_handler(event)


# if __name__=='__main__':
#     lambda_handler({
#         "queryStringParameters": {
#             "start_station_id": 31646.0,
#             "end_station_id": 31248.0,
#             "rideable_type": "classic_bike"
#         }
#     })
