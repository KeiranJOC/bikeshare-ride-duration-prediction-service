import deployment.predict.model as model


def test_prepare_features():
    model_service = model.ModelService(model=None, preprocessor=None, logger=None)

    ride = {
        'start_station_id': 31646.0,
        'end_station_id': 31248.0,
        'rideable_type': 'classic_bike',
    }

    actual_features = model_service.prepare_features(ride)
    expected_features = {
        'start_end': '31646.0_31248.0',
        'rideable_type': 'classic_bike',
    }

    assert actual_features == expected_features
