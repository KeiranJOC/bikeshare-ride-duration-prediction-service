import json
import logging
import os
import pickle
import uuid
from typing import Dict

import mlflow


def get_prod_run_id(client, model_name: str):
    return client.get_latest_versions(model_name, ["Production"])[0].run_id


def get_model_preprocessor_paths(run_id: str):
    model_path = os.getenv("MODEL_PATH")
    if model_path is not None:
        return model_path

    model_bucket = os.getenv("MODEL_BUCKET", "mlflow-artifacts-remote-2")
    experiment_id = os.getenv("EXPERIMENT_ID", "1")
    model_path = f"s3://{model_bucket}/{experiment_id}/{run_id}/artifacts/model"
    preprocessor_path = f"s3://{model_bucket}/{experiment_id}/{run_id}/artifacts/preprocessor/preprocessor.bin"

    return model_path, preprocessor_path


def load_model_preprocessor(run_id: str):
    model_path, preprocessor_path = get_model_preprocessor_paths(run_id)
    model = mlflow.pyfunc.load_model(model_path)
    preprocessor_artifact = mlflow.artifacts.download_artifacts(preprocessor_path)

    with open(preprocessor_artifact, "rb") as f_in:
        preprocessor = pickle.load(f_in)

    return model, preprocessor


class ModelService:
    def __init__(self, model, preprocessor, logger, model_version=None) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.logger = logger
        self.model_version = model_version

    def prepare_features(self, ride: Dict):
        record = ride.copy()
        features = {}
        features["start_end"] = "%s_%s" % (
            record["start_station_id"],
            record["end_station_id"],
        )
        features["rideable_type"] = record["rideable_type"]
        return features

    def predict(self, features: Dict):
        x = self.preprocessor.transform(features)
        preds = self.model.predict(x)
        return float(preds[0])

    def lambda_handler(self, event):

        # get input
        try:
            body = event["queryStringParameters"]
            self.logger.info(f"Received request body: {body}")
        except Exception as e:
            self.logger.error(e)

        # prepare features
        try:
            features = self.prepare_features(body)
            self.logger.info(f"Prepared input features: {features}")
        except Exception as e:
            self.logger.error(e)

        # calculate prediction
        try:
            prediction = self.predict(features)
            ride_id = str(uuid.uuid4())
            prediction_payload = {
                "ride_id": ride_id,
                "start_end": features["start_end"],
                "rideable_type": features["rideable_type"],
                "duration": prediction,
                "model_version": self.model_version,
            }

            self.logger.info(f"Predicted duration: {prediction}")
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": "Success",
                        "prediction": prediction_payload,
                    }
                ),
            }

        except Exception as e:
            self.logger.error(e)
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "Unhandled error",
                    }
                ),
            }


def init(model_name: str, tracking_uri: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run_id = get_prod_run_id(client, model_name)

    model, preprocessor = load_model_preprocessor(run_id)
    model_service = ModelService(
        model=model, preprocessor=preprocessor, logger=logger, model_version=run_id
    )

    return model_service
