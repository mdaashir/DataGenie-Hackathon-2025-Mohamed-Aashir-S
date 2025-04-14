import os
import joblib
import numpy as np
import pandas as pd
from dateutil import parser
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from Backend import MODELS_DIR, MODEL_LIST
from Backend.Models.TimeSeries import forecast
from Backend.Schemas.Predict import TimeSeriesData
from Backend.Utils.Extract_Feature import extract_features
from Backend.Utils.Logger import setup_logging

log_file, logging = setup_logging(log_name="predict_route.log")

try:
    os.makedirs(MODELS_DIR, exist_ok=True)
    transform = joblib.load(f"{MODELS_DIR}/feature_extractor.pkl")
except FileNotFoundError:
    joblib.dump(extract_features, f"{MODELS_DIR}/feature_extractor.pkl")

# Load pre-trained model
try:
    loaded_model = joblib.load(f"{MODELS_DIR}/best_model.pkl")
except FileNotFoundError:
    raise RuntimeError("Model file 'best_model.pkl' not found.")

predict_router = APIRouter()
@predict_router.post("/predict")
async def classify(
    data: TimeSeriesData,
    from_date: str,
    to_date: str = "",
    period: int = 0,
    frequency: str = "",
    model: str = "",
):
    logging.info(
        f"Received request with from_date={from_date}, to_date={to_date}, period={period}, frequency={frequency}"
    )
    logging.info(f"Data received: {data}")

    print("query", from_date, to_date, period, frequency)
    print("data", data)

    time_series_data = data.data
    if not time_series_data:
        raise HTTPException(status_code=400, detail="No time series data provided.")

    print("time_series_data", time_series_data)

    logging.info(f"Time series data: {time_series_data}")
    mapper = dict(enumerate(MODEL_LIST))

    # Default model prediction if no model specified
    if model == "":
        try:
            print("inside", data)
            print("inside data", time_series_data)
            df = pd.DataFrame(
                [(entry.Date, entry.Value) for entry in time_series_data],
                columns=["Date", "Value"],
            )
            print(df)
            feature_data = extract_features(df)
            print(feature_data)
            feature_data.replace(np.nan, 0, inplace=True)
            feature_data.replace([np.inf, -np.inf], 1e10, inplace=True)
            feature_data = np.array(feature_data).reshape(1, 46)
            print(feature_data)
            y_pred = loaded_model.predict(feature_data)
            print(y_pred)
            print(type(y_pred))
            predicted_label = list(y_pred)[0]
            print(predicted_label)
            model = mapper[predicted_label]
        except Exception as e:
            logging.error(f"Error during model prediction: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Error during model prediction: {str(e)}"
            )

    # Convert time series data to pandas Series
    try:
        data_tuples = [(entry.Date, entry.Value) for entry in time_series_data]
        ts_data = pd.Series(
            data=[value for _, value in data_tuples],
            index=[
                parser.parse(date).strftime("%Y-%m-%dT%H:%M:%S")
                for date, _ in data_tuples
            ],
        )
        ts_data.index = pd.to_datetime(ts_data.index)
    except Exception as e:
        logging.error(f"Error processing time series data: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Error processing time series data: {str(e)}"
        )

    # Forecast
    try:
        print("inside try", ts_data, from_date, to_date, frequency, period, model)
        forecast_json, mape_value = forecast(
            ts_data, from_date, to_date, frequency, period, model
        )
    except Exception as e:
        logging.error(f"Error during forecasting: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Error during forecasting: {str(e)}"
        )

    logging.info(f"Model: {model}, MAPE: {mape_value}, Forecast: {forecast_json}")
    json = {"model": model, "mape_value": mape_value, "result": forecast_json}
    print(json)
    return JSONResponse(
        content=json,
        status_code=status.HTTP_200_OK,
    )
