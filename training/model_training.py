import datetime
import os
import pickle
import zipfile
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd
import requests
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = os.getenv(
    "DATA_PATH",
    "https://s3.amazonaws.com/capitalbikeshare-data/202204-capitalbikeshare-tripdata.zip",
)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://ec2-3-26-219-211.ap-southeast-2.compute.amazonaws.com:5000/",
)
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "bikeshare-ride-duration-prediction")
MODEL_NAME = os.getenv("MODEL_NAME", "bikeshare-ride-duration-regressor")
MAX_EVALS = os.getenv("MAX_EVALS", 10)


@task
def read_data(url: str):
    """
    Capital Bikeshare datasets are zipped
    We need to download then extract the csv
    """
    zip_path = url.split("/")[-1]
    file_name = zip_path.split(".")[0] + ".csv"
    save_path = f"./datasets/{zip_path}"

    req = requests.get(url)

    with open(save_path, "wb") as f_out:
        f_out.write(req.content)

    with zipfile.ZipFile(save_path) as z:
        with z.open(file_name) as f:
            df = pd.read_csv(f, parse_dates=True)

    categorical_cols = ["rideable_type", "start_station_id", "end_station_id"]
    date_cols = ["started_at", "ended_at"]

    df[categorical_cols] = df[categorical_cols].astype(str)
    df[date_cols] = df[date_cols].apply(pd.to_datetime, format="%Y/%m/%d %H:%M:%S")

    df["duration"] = df["ended_at"] - df["started_at"]
    df["duration"] = df["duration"].apply(lambda x: round(x.total_seconds() / 60, 0))
    df["start_end"] = df["start_station_id"] + "_" + df["end_station_id"]

    df = df[df["duration"] <= 120]  # drop rides longer than 2 hours
    categorical_cols = ["rideable_type", "start_end"]
    target = "duration"

    return df, categorical_cols, target


@task
def create_train_val_sets(df: pd.DataFrame, categorical_cols: List, target: str):
    dv = DictVectorizer()
    dicts = df[categorical_cols].to_dict(orient="records")

    x = dv.fit_transform(dicts)
    y = df[target].values

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=42
    )
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    print(dv)

    return x_train, x_val, y_train, y_val, dv


@task
def model_search(x_train, y_train, x_val, y_val):
    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag("model", "Ridge")
            mlflow.log_params(params)

            lr = Ridge(**params)
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("RMSE", rmse)

        return {"loss": rmse, "status": STATUS_OK}

    search_space = {"alpha": hp.uniform("alpha", 0.1, 1)}

    rstate = np.random.default_rng(0)
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=Trials(),
        rstate=rstate,
    )

    return best_result


@task
def train_best_model(x_train, y_train, x_val, y_val, dv, best_result: Dict):

    with mlflow.start_run() as run:

        print(f"Best params: {best_result}")
        mlflow.log_params(best_result)
        lr = Ridge(**best_result)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE: {rmse}")
        mlflow.log_metric("RMSE", rmse)

        with open("preprocessor.bin", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor.bin", artifact_path="preprocessor")
        mlflow.sklearn.log_model(lr, artifact_path="model")

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
            tags={"run_id": run_id},
        )


def transition_model_stage(client, model_name, model_version, stage):
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage,
        archive_existing_versions=False,
    )

    date = datetime.date.today()
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=f"Version {model_version} of {model_name} was transitioned to {stage} on {date}.",
    )


def promote_model(client):
    model_version = client.get_latest_versions(MODEL_NAME)[-1].version
    transition_model_stage(client, MODEL_NAME, model_version, "Production")


@flow
def apply_model():
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, categorical_cols, target = read_data(DATA_PATH)
    x_train, x_val, y_train, y_val, dv = create_train_val_sets(
        df, categorical_cols, target
    )

    best_result = model_search(x_train, y_train, x_val, y_val)
    train_best_model(x_train, y_train, x_val, y_val, dv, best_result)
    promote_model(client)


deployment = Deployment.build_from_flow(
    flow=apply_model,
    name="model_training",
    schedule=CronSchedule(cron="0 6 1 * *", timezone="Australia/Sydney"),
    work_queue_name="bikeshare-ride-duration-prediction-model-training",
)


if __name__ == "__main__":
    deployment.apply()
