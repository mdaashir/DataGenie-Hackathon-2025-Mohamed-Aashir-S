import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_absolute_percentage_error
from Backend.Utils.Data_Processor import parse_date, calculate_forecast_period, get_actual_values, create_result_json

def predict(data, date_from, date_to, frequency, period):
    date_from_ts = parse_date(date_from)
    date_to_ts = parse_date(date_to)
    training_data = data[:date_from_ts][:-1]

    model = arch_model(training_data, vol='GARCH', p=1, q=1)
    model_fit = model.fit(disp='off')

    forecast_period = calculate_forecast_period(date_from_ts, date_to_ts, frequency, period)
    actual_values = get_actual_values(data, date_from_ts, frequency, forecast_period)

    volatility_forecast = model_fit.forecast(horizon=forecast_period).variance.values[-1, :]
    random_noise = np.random.normal(0, 1, forecast_period)
    forecast_values = random_noise * np.sqrt(volatility_forecast)

    forecast_index = pd.date_range(start=date_from_ts, periods=forecast_period, freq=frequency)
    forecast = pd.Series(forecast_values, index=forecast_index)

    result_json = create_result_json(forecast, actual_values)
    mape = mean_absolute_percentage_error(actual_values, forecast)
    return result_json, mape
