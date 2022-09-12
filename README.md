# bikeshare-ride-duration-prediction-service

[Capital Bikeshare](https://capitalbikeshare.com/) is a bikeshare service operating in Washington, D.C., with 5,000 bikes and 600+ stations.
The company provides a mobile app for users to rent bikes and manage their rides.
The app has a map screen on which a user can search for the station nearest to their destination, however no estimated trip time is provided.
Capital Bikeshare would like to build a ride duration prediction service that will enable them to add an estimated trip time feature to the app.

This repo contains code for the training and deployment of a duration prediction regression model.

## Instructions

#### Training

#### Deployment

#### Inference
The model is deployed as a Lambda behind in API Gateway. To get a prediction from the model, run the following command in a terminal window:
- `curl -X POST https://cr1876invh.execute-api.ap-southeast-2.amazonaws.com/Prod/predict?start_station_id=31646.0&end_station_id=31248.0&rideable_type=classic_bike`
You may get a timeout error the first time you run it. If so, try again. You should receive a duration prediction and a 200 status code.
