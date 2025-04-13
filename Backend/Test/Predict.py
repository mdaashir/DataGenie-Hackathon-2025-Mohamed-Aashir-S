import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from Backend.Schemas.Predict import TimeSeriesData
from Backend.run import app
import logging


@pytest.fixture()
def client():
    return TestClient(app)

mock_time_series_data = [
    {"Date": "2025-04-01T00:00:00", "Value": 100},
    {"Date": "2025-04-02T00:00:00", "Value": 150},
    {"Date": "2025-04-03T00:00:00", "Value": 200},
]

mock_forecast_result = {
    "forecast": [{"date": "2025-04-04", "value": 250}],
    "mape_value": 0.1
}

# Unit test for the /predict endpoint
@patch("Backend.Models.TimeSeries.forecast", return_value=(mock_forecast_result['forecast'], mock_forecast_result['mape_value']))
@patch("joblib.load")
@patch("Backend.Utils.Extract_Feature.extract_features")
def test_predict(mock_extract_features, mock_joblib_load, client):
    mock_extract_features.return_value = [0.1] * 96
    mock_joblib_load.return_value = MagicMock(predict=MagicMock(return_value=[[0.2, 0.3, 0.5]]))

    # Prepare the input data
    time_series_data = [TimeSeriesData(data=mock_time_series_data)]
    from_date = "2025-04-01T00:00:00"
    to_date = "2025-04-05T00:00:00"
    period = 1
    frequency = "D"
    model = ""

    # Make a POST request to the /predict endpoint
    logging.debug("Making a POST request to /predict endpoint")
    response = client.post(
        "/predict?from_date=" + from_date + "&to_date=" + to_date + "&period=" + str(period) + "&frequency=" + frequency + "&model=" + model,
        json=time_series_data[0].dict()
    )

    # Assert response status
    logging.debug(f"Response status: {response.status_code}")
    assert response.status_code == 200

    # Check if required fields are in response data
    response_data = response.json()
    logging.debug(f"Response Data: {response_data}")

    assert "model" in response_data
    assert "forecast" in response_data
    assert "mape_value" in response_data
    assert len(response_data["forecast"]) > 0
    assert response_data["mape_value"] == mock_forecast_result["mape_value"]


# Test logging output with the mock
@patch("Backend.Models.TimeSeries.forecast", return_value=(mock_forecast_result['forecast'], mock_forecast_result['mape_value']))
@patch("joblib.load")
@patch("Backend.Utils.Extract_Feature.extract_features")
def test_logging(mock_extract_features, mock_joblib_load, client, caplog):
    mock_extract_features.return_value = [0.1] * 96
    mock_joblib_load.return_value = MagicMock(predict=MagicMock(return_value=[[0.2, 0.3, 0.5]]))

    # Prepare the input data
    time_series_data = [TimeSeriesData(data=mock_time_series_data)]
    from_date = "2025-04-01T00:00:00"
    to_date = "2025-04-05T00:00:00"
    period = 1
    frequency = "D"
    model = ""

    # Make a POST request to the /predict endpoint
    response = client.post(
        "/predict?from_date=" + from_date + "&to_date=" + to_date + "&period=" + str(period) + "&frequency=" + frequency + "&model=" + model,
        json=time_series_data[0].dict()
    )

    # Check the log messages
    assert "Making a POST request to /predict endpoint" in caplog.text
    assert "Response status: 200" in caplog.text
    assert "Response Data" in caplog.text  # Verifying that response data is logged


# Handling errors in the predict endpoint
@patch("Backend.Models.TimeSeries.forecast", side_effect=Exception("Model prediction failed"))
@patch("joblib.load")
@patch("Backend.Utils.Extract_Feature.extract_features")
def test_predict_error(mock_extract_features, mock_joblib_load, client):
    mock_extract_features.return_value = [0.1] * 96
    mock_joblib_load.return_value = MagicMock(predict=MagicMock(return_value=[[0.2, 0.3, 0.5]]))

    # Prepare the input data
    time_series_data = [TimeSeriesData(data=mock_time_series_data)]
    from_date = "2025-04-01T00:00:00"
    to_date = "2025-04-05T00:00:00"
    period = 1
    frequency = "D"
    model = ""

    # Make a POST request to the /predict endpoint
    logging.debug("Making a POST request to /predict endpoint")
    response = client.post(
        "/predict?from_date=" + from_date + "&to_date=" + to_date + "&period=" + str(period) + "&frequency=" + frequency + "&model=" + model,
        json=time_series_data[0].dict()
    )

    # Assert response status for error
    logging.debug(f"Response status: {response.status_code}")
    assert response.status_code == 400
    assert "Error during model prediction" in response.json().get("detail")
