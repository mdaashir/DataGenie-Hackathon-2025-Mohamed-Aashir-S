import os
import joblib
import numpy as np
import pandas as pd
from dateutil import parser
from fastapi import APIRouter
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from Backend import MODELS_DIR, MODEL_LIST
from Backend.Models.TimeSeries import forecast
from Backend.Schemas.Predict import TimeSeriesData
from Backend.Utils.Extract_Feature import extract_features

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
    time_series_data = data.data
    mapper = dict(enumerate(MODEL_LIST))

    # Default model prediction if no model specified
    if model == "":
        try:
            feature_data = transform(data)
            feature_data = np.array(feature_data).reshape(1, 96)
            y_pred = loaded_model.predict(feature_data)
            predicted_label = list(y_pred[0]).index(max(y_pred[0]))
            model = mapper[predicted_label]
        except Exception as e:
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
        raise HTTPException(
            status_code=400, detail=f"Error processing time series data: {str(e)}"
        )

    # Forecast
    try:
        forecast_json, mape_value = forecast(
            ts_data, from_date, to_date, frequency, period, model
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error during forecasting: {str(e)}"
        )

    return JSONResponse(
        content={"model": model, "mape_value": mape_value, "forecast": forecast_json},
        status_code=status.HTTP_200_OK,
    )