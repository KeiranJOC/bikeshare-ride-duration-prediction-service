import mlflow
import pickle
import zipfile
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict, List
from prefect import flow, task
from hyperopt.pyll import scope
from mlflow.tracking import MlflowClient
from prefect.deployments import Deployment
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from prefect.flow_runners import SubprocessFlowRunner
from sklearn.feature_extraction import DictVectorizer
from prefect.orion.schemas.schedules import IntervalSchedule
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error



@task
def read_data(url: str):
    """
    Capital Bikeshare datasets are zipped
    We need to download then extract the csv
    """
    zip_path = url.split('/')[-1] 
    file_name = zip_path.split('.')[0] + '.csv'

    req = requests.get(url)

    with open(zip_path, 'wb') as f_out:
        f_out.write(req.content)

    with zipfile.ZipFile(zip_path) as z:
        with z.open(file_name) as f:
            df = pd.read_csv(f, parse_dates=True)
            
    categorical_cols = ['rideable_type', 'start_station_id', 'end_station_id']
    date_cols = ['started_at', 'ended_at']
    
    df[categorical_cols] = df[categorical_cols].astype(str)
    df[date_cols] = df[date_cols].apply(pd.to_datetime, format='%Y/%m/%d %H:%M:%S')
    
    df['duration'] = df['ended_at'] - df['started_at']
    df['duration'] = df['duration'].apply(lambda x: round(x.total_seconds() / 60, 0))
    df['start_end'] = df['start_station_id'] + '_' + df['end_station_id']
    

    df = df[df['duration'] <= 120] # drop rides longer than 2 hours
    categorical_cols = ['rideable_type', 'start_end']
    target = 'duration'
    
    return df, categorical_cols, target


@task
def create_train_val_sets(df: pd.DataFrame, categorical_cols: List, target: str):
    dv = DictVectorizer()
    dicts = df[categorical_cols].to_dict(orient='records')
    
    x = dv.fit_transform(dicts)
    y = df[target].values

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    print(dv)

    return x_train, x_val, y_train, y_val, dv


@task
def model_search(x_train, y_train, x_val, y_val):
    
    def objective(params):
        
        with mlflow.start_run():
            mlflow.set_tag('model', 'Ridge')
            mlflow.log_params(params)
            
            lr = Ridge(**params)
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric('RMSE', rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'alpha': scope.int(hp.uniform('alpha', 0.1, 1))
    }

    rstate=np.random.default_rng(0)
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials(),
        rstate=rstate
    )

    return best_result


@task
def train_best_model(x_train, y_train, x_val, y_val, dv, best_result: Dict):
    
    with mlflow.start_run():
        
        print(f"Best params: {best_result}")
        mlflow.log_params(best_result)
        lr = Ridge(**best_result)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE: {rmse}")
        mlflow.log_metric('RMSE', rmse)

        with open('preprocessor.bin', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('preprocessor.bin', artifact_path='preprocessor')

        mlflow.sklearn.log_model(lr, artifact_path='models')


@flow
def apply_model(data_path='https://s3.amazonaws.com/capitalbikeshare-data/202204-capitalbikeshare-tripdata.zip'):
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('bikeshare-ride-duration-prediction')

    df, categorical_cols, target = read_data(data_path)
    x_train, x_val, y_train, y_val, dv = create_train_val_sets(df, categorical_cols, target)

    best_result = model_search(x_train, y_train, x_val, y_val)
    train_best_model(x_train, y_train, x_val, y_val, dv, best_result)


apply_model()