import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from Backend.Utils.Data_Processor import parse_date, calculate_forecast_period, get_actual_values, create_result_json

def predict(data, date_from, date_to, frequency, period):
    date_from_ts = parse_date(date_from)
    date_to_ts = parse_date(date_to)
    training_data = data[:date_from_ts]

    x, y = [], []
    for i in range(10, len(training_data) - 1):
        x.append(training_data.values[i-10:i])
        y.append(training_data.values[i])
    x, y = np.array(x), np.array(y)

    model = XGBRegressor(n_estimators=100)
    model.fit(x, y)

    forecast_period = calculate_forecast_period(date_from_ts, date_to_ts, frequency, period)
    actual_values = get_actual_values(data, date_from_ts, frequency, forecast_period)

    input_seq = training_data.values[-10:].tolist()
    forecast = []

    for _ in range(forecast_period):
        pred = model.predict(np.array([input_seq[-10:]]))[0]
        forecast.append(pred)
        input_seq.append(pred)

    forecast_index = pd.date_range(start=date_from_ts, periods=forecast_period, freq=frequency)
    forecast = pd.Series(forecast, index=forecast_index)

    result_json = create_result_json(forecast, actual_values)
    mape = mean_absolute_percentage_error(actual_values, forecast)
    return result_json, mape
