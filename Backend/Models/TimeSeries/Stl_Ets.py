import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
from Backend.Utils.Data_Processor import parse_date, calculate_forecast_period, get_actual_values, create_result_json

def predict(data, date_from, date_to, frequency, period):
    date_from_ts = parse_date(date_from)
    date_to_ts = parse_date(date_to)
    training_data = data[:date_from_ts][:-1]

    decomposition = seasonal_decompose(training_data, model='additive', period=12)
    # trend = decomposition.trend.dropna()
    seasonality = decomposition.seasonal.dropna()

    detrended = training_data - seasonality
    model = ExponentialSmoothing(detrended.dropna(), trend='add', seasonal=None)
    model_fit = model.fit()

    forecast_period = calculate_forecast_period(date_from_ts, date_to_ts, frequency, period)
    actual_values = get_actual_values(data, date_from_ts, frequency, forecast_period)

    trend_forecast = model_fit.forecast(forecast_period)

    seasonal_pattern = seasonality[-12:]
    seasonal_forecast = np.tile(seasonal_pattern, forecast_period // 12 + 1)[:forecast_period]
    seasonal_index = pd.date_range(start=date_from_ts, periods=forecast_period, freq=frequency)

    forecast = trend_forecast + pd.Series(seasonal_forecast, index=seasonal_index)

    result_json = create_result_json(forecast, actual_values)
    mape = mean_absolute_percentage_error(actual_values, forecast)
    return result_json, mape
