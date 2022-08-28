import os
import mlflow
import pickle

from typing import Dict
from flask import Flask, request, jsonify


MODEL_RUN_ID = os.getenv('MODEL_RUN_ID')

logged_model = f's3://mlflow-artifacts-remote-2/1/{MODEL_RUN_ID}/artifacts/models'
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride: Dict):
    features = {}
    features['start_end'] = ride['start_station_id'] + '_' + ride['end_station_id']
    features['rideable_type'] = ride['rideable_type']
    return features


def predict(features: Dict):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    
    result = {
        'duration': pred,
        'model_version': MODEL_RUN_ID
    }

    return jsonify(result)


if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)