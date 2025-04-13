import os
import joblib
import numpy as np
import pandas as pd
from fastapi import status
from dateutil import parser
from fastapi import APIRouter
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

loaded_model = joblib.load(f"{MODELS_DIR}/best_model.pkl")

predict_router = APIRouter()
@predict_router.post("/predict")
async def classify(data: TimeSeriesData, from_date: str, to_date: str = "", period: int = 0, frequency: str = "", model: str = ""):

    time_series_data = data.data
    mapper = dict(enumerate(MODEL_LIST))
    if model == "":
        feature_data = transform(data)
        feature_data = np.array(feature_data)
        feature_data = feature_data.reshape(1, 96)
        y_pred = loaded_model.predict(feature_data)
        predicted_label = list(y_pred[0]).index(max(y_pred[0]))
        model = mapper[predicted_label]

    data_tuples = [(entry.Date, entry.Value) for entry in time_series_data]

    ts_data = pd.Series(data=[value for _, value in data_tuples],
                        index=[parser.parse(date).strftime("%Y-%m-%dT%H:%M:%S") for date, _ in data_tuples])
    ts_data.index = pd.to_datetime(ts_data.index)

    forecast_json, mape_value = forecast(ts_data, from_date, to_date, frequency, period, model)

    return JSONResponse(
        content={
            "model": model,
            "mape_value": mape_value,
            "forecast": forecast_json
        },
        status_code=status.HTTP_200_OK
    )