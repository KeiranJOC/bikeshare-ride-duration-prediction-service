import os
import json
import uuid
import mlflow
import pickle
import logging
import warnings

from typing import Dict


# prevent multiprocessing warnings in AWS logs
warnings.filterwarnings(action='ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_RUN_ID = os.getenv('MODEL_RUN_ID', '302789306678475c9603fd64714919e1')

model_path = f's3://mlflow-artifacts-remote-2/1/{MODEL_RUN_ID}/artifacts/models'
dv_path = f's3://mlflow-artifacts-remote-2/1/{MODEL_RUN_ID}/artifacts/preprocessor/preprocessor.bin'
dv_path = mlflow.artifacts.download_artifacts(dv_path)

with open(dv_path, 'rb') as f_in:
    dv = pickle.load(f_in)

model = mlflow.pyfunc.load_model(model_path)


def prepare_features(ride: Dict):
    record = ride.copy()
    features = {}
    features['start_end'] = '%s_%s' % (record['start_station_id'], record['end_station_id'])
    features['rideable_type'] = record['rideable_type']
    return features

def predict(features: Dict):
    x = dv.transform(features)
    preds = model.predict(x)
    print(preds)
    return float(preds[0])


def lambda_handler(event, context):

    # get input
    try:
        body = event['queryStringParameters']
        logger.info(f"Received request body: {body}")
    except Exception as e:
        logger.error(e)

    # prepare features
    try:
        features = prepare_features(body)
        logger.info(f"Prepared input features: {features}")
    except Exception as e:
        logger.error(e)

    # calculate prediction
    try:
        prediction = predict(features)
        ride_id = str(uuid.uuid4())
        prediction_payload = {
            'ride_id': ride_id,
            'start_end': features['start_end'],
            'rideable_type': features['rideable_type'],
            'duration': prediction,
            'model_version': MODEL_RUN_ID
        }

        logger.info(f"Predicted duration: {prediction}")
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Success",
                    "prediction": prediction_payload
                }
            ),
        }

    except Exception as e:
        logger.error(e)
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Unhandled error",
                }
            ),
        }
