import os
import model
import warnings

# prevent multiprocessing warnings in AWS logs
warnings.filterwarnings(action='ignore')

RUN_ID = os.getenv('RUN_ID', '302789306678475c9603fd64714919e1')

model_service = model.init(run_id=RUN_ID)


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