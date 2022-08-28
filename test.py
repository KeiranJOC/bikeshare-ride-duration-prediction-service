import requests


ride = {
    'start_station_id': '31646.0',
    'end_station_id': '31248.0',
    'rideable_type': 'classic_bike'
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())